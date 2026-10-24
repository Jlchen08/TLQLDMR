#!/usr/bin/env python3
"""Leakage-controlled, staged validation selection for TL-QLDMR.

Selection v2 is a new experiment.  It never rewrites selection-v1 scripts or
outputs.  By default this runner performs Q1 validation selection and creates
an immutable selection lock, then exits without calculating a Q2/Q3-Q4
prediction or metric.  Held-out evaluation requires a later explicit
``--phase evaluate --resume`` invocation.

The primary breadth tasks are six within-farm normal-to-extreme transfers.
Six leave-one-farm-out tasks are retained as transfer-stress tests.  All
preprocessing, landmark construction, and fitting use 2019 rows only; Q1 is
used only for selection, Q2 only for interval calibration, and Q3-Q4 only for
locked evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from review.convex_refinement import refine_pinball
from review.selection_data import digest
from review.selection_data_v2 import LOADER_VERSION, load_all_selection_data_v2
from review.selection_sampling import stratified_target_indices


CAPACITY = {1: 99.0, 2: 200.0, 3: 99.0, 4: 66.0, 5: 36.0, 6: 96.0}
METHODS = ("Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR", "TargetSVQR", "HistGBR")
KERNEL_METHODS = METHODS[:-1]
POOLED_KERNEL_METHODS = ("Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR")
ACTIVE_REGULARIZERS = {
    "Full": ("lambda1", "lambda2"),
    "NoVariance": ("lambda2",),
    "NoDiscrepancy": ("lambda1",),
    "WeightedSVQR": (),
    "TargetSVQR": (),
}


def save_json(path: Path, value: object) -> None:
    """Atomically write strict JSON so a killed process cannot create a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path: Path, **arrays) -> None:
    """Atomically store an NPZ, including when disjoint workers share features."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def point_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    y, prediction = np.asarray(y), np.asarray(prediction)
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(y),
        "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        "mae": float(np.mean(np.abs(y - prediction))),
        "r2": float(1 - np.sum((y - prediction) ** 2) / denominator) if denominator else None,
    }


def interval_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     alpha: float = 0.05) -> dict:
    if np.any(lower > upper):
        raise ValueError("Unordered interval.")
    width = upper - lower
    return {
        "n": len(y),
        "picp": float(np.mean((y >= lower) & (y <= upper))),
        "mpiw": float(np.mean(width)),
        "winkler": float(np.mean(width + 2 / alpha *
                                 (np.maximum(lower - y, 0) + np.maximum(y - upper, 0)))),
    }


def conformal_quantile(scores: np.ndarray, alpha: float = 0.05) -> tuple[float, int]:
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if not 1 <= rank <= len(scores):
        raise ValueError("Insufficient calibration scores for the finite-sample rank.")
    return float(np.partition(scores, rank - 1)[rank - 1]), rank


def rolling_expansions(calibration_scores: np.ndarray, test_scores: np.ndarray,
                       window: int = 256, alpha: float = 0.05) -> np.ndarray:
    """Past-only rolling calibration; score t is appended after forecasting t."""
    history = list(np.asarray(calibration_scores, dtype=float)[-window:])
    expansion = []
    for observed_score in test_scores:
        value, _ = conformal_quantile(np.asarray(history), alpha)
        expansion.append(value)
        history.append(float(observed_score))
        history = history[-window:]
    return np.asarray(expansion)


def preprocess(x: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...] | None = None,
               source_n: int = 0):
    """Fit domain-balanced imputation/scaling on model-fit rows only.

    For pooled fits, source and target rows each receive one half of the total
    fitting mass.  This prevents thousands of source rows from determining the
    geometry used for a small target sample.  Target-only fits use equal row
    weights.  Every pooled method receives the same transformed matrix.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("Preprocessing requires a nonempty two-dimensional X and aligned y.")
    if not isinstance(source_n, (int, np.integer)) or isinstance(source_n, bool):
        raise ValueError("source_n must be an integer.")
    if not 0 <= source_n < len(y):
        raise ValueError("source_n must leave at least one target fitting row.")
    target_n = len(y) - source_n
    if source_n:
        row_weight = np.r_[np.full(source_n, 0.5 / source_n),
                           np.full(target_n, 0.5 / target_n)]
        weighting = "domain_balanced_source_0.5_target_0.5"
    else:
        row_weight = np.full(target_n, 1.0 / target_n)
        weighting = "target_only_equal_row_weight"
    valid = np.isfinite(x)
    if source_n:
        source_count = valid[:source_n].sum(axis=0)
        target_count = valid[source_n:].sum(axis=0)
        source_mean = np.divide(np.where(valid[:source_n], x[:source_n], 0).sum(axis=0),
                                source_count, out=np.zeros(x.shape[1]),
                                where=source_count > 0)
        target_mean = np.divide(np.where(valid[source_n:], x[source_n:], 0).sum(axis=0),
                                target_count, out=np.zeros(x.shape[1]),
                                where=target_count > 0)
        both = (source_count > 0) & (target_count > 0)
        mean = np.where(both, .5 * (source_mean + target_mean),
                        np.where(source_count > 0, source_mean,
                                 np.where(target_count > 0, target_mean, 0.)))
        finite_mass = .5 * (source_count / source_n) + .5 * (target_count / target_n)
    else:
        finite_mass = valid.mean(axis=0)
        mean = np.divide(np.where(valid, x, 0).sum(axis=0), valid.sum(axis=0),
                         out=np.zeros(x.shape[1]), where=valid.sum(axis=0) > 0)
    filled = np.where(valid, x, mean)
    scale = np.sqrt(((filled - mean) ** 2 * row_weight[:, None]).sum(axis=0))
    scale[scale == 0] = 1
    label_mean = float(row_weight @ y)
    label_scale = float(np.sqrt(row_weight @ ((y - label_mean) ** 2))) or 1.0
    state = {
        "mean": mean.tolist(), "scale": scale.tolist(),
        "label_mean": label_mean, "label_scale": label_scale,
        "finite_fit_count": valid.sum(axis=0).tolist(),
        "finite_fit_weight_mass": finite_mass.tolist(),
        "feature_names_per_lag": list(feature_names or ()),
        "scope": "actual 2019 model-fitting source and target windows only",
        "weighting": weighting, "source_fit_rows": source_n,
        "target_fit_rows": target_n,
    }

    def transform(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return (np.where(np.isfinite(values), values, mean) - mean) / scale

    return transform(x), (y - label_mean) / label_scale, transform, state


def regularizer_components(z: np.ndarray, y: np.ndarray,
                           source_n: int) -> tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, dict]:
    """Return the manuscript's unmodified variance and discrepancy terms.

    Raw traces are recorded to make scale differences visible.  They are not
    used to renormalize the matrices: normalizing the rank-one discrepancy by
    its own trace would create a unit penalty even as the empirical domain
    difference approaches zero, and rank-dependent normalization would change
    the objective while rank is being selected.
    """
    z, y = np.asarray(z, dtype=float), np.asarray(y, dtype=float)
    if not 0 < source_n < len(z) or y.shape != (len(z),):
        raise ValueError("Regularizer components require nonempty source and target rows.")
    centered = z[:source_n] - z[:source_n].mean(axis=0)
    centered_y = y[:source_n] - y[:source_n].mean()
    covariance = centered.T @ centered / source_n
    cross = centered.T @ centered_y / source_n
    discrepancy_vector = z[:source_n].mean(axis=0) - z[source_n:].mean(axis=0)
    discrepancy = np.outer(discrepancy_vector, discrepancy_vector)
    audit = {
        "normalization": "none_manuscript_raw_objective",
        "variance_trace_raw": float(np.trace(covariance)),
        "discrepancy_trace_raw": float(np.trace(discrepancy)),
    }
    return covariance, cross, discrepancy, audit


def _kernel_ranks(method: str) -> tuple[int, int, int]:
    if method not in KERNEL_METHODS:
        raise ValueError(f"Unknown kernel method: {method}")
    return (32, 64, 128) if method == "TargetSVQR" else (128, 256, 512)


def environment_versions() -> dict[str, str]:
    """Version identity checked before creating or resuming any run."""
    return {"python": platform.python_version(), **{
        package: package_version(package)
        for package in ("numpy", "scipy", "pandas", "scikit-learn", "joblib",
                        "threadpoolctl", "openpyxl")}}


def _canonical_candidate(method: str, candidate: dict) -> dict:
    """Represent inactive coordinates exactly as used by the fitted method."""
    candidate = dict(candidate)
    if method in KERNEL_METHODS:
        if method == "TargetSVQR":
            candidate.pop("target_fraction", None)  # fitting always uses p = 1
        for name in ("lambda1", "lambda2"):
            if name in candidate and name not in ACTIVE_REGULARIZERS[method]:
                candidate[name] = 0.0
    return candidate


def effective_model_signature(method: str, candidate: dict) -> tuple:
    """Ignore provenance and inactive coordinates, never active fit parameters."""
    candidate = _canonical_candidate(method, candidate)
    if method == "HistGBR":
        keys = ("learning_rate", "max_leaf_nodes", "min_samples_leaf",
                "l2_regularization", "max_iter", "target_fraction")
    else:
        keys = ("gamma", "total_loss", "rank")
        if method != "TargetSVQR":
            keys += ("target_fraction",)
        keys += ACTIVE_REGULARIZERS[method]
        candidate = {**_base_regularizer(method), **candidate}
    return tuple((key, candidate[key]) for key in keys)


def _assert_unique_configs(method: str, candidates: list[dict], count: int = 18) -> None:
    signatures = [effective_model_signature(method, candidate) for candidate in candidates]
    if len(candidates) != count or len(set(signatures)) != count:
        raise ValueError(f"{method} requires {count} effective unique configurations per stage.")


def _complete_unique_candidates(method: str, candidates: list[dict], seed: int,
                                count: int = 18, *, excluded: list[dict] = (),
                                regularizer_center: dict | None = None) -> list[dict]:
    """Keep the first effective occurrence and deterministically replenish gaps.

    The reserve is a fixed, global Latin-hypercube design. It is consumed only
    when anchors or clipping collapse two nominal candidates to one model.
    Target-only reserves use only the two active continuous coordinates.
    """
    unique = []
    seen = {effective_model_signature(method, candidate) for candidate in excluded}
    def append(candidate):
        candidate = _canonical_candidate(method, candidate)
        signature = effective_model_signature(method, candidate)
        if signature not in seen:
            seen.add(signature)
            unique.append(candidate)
    for candidate in candidates:
        append(candidate)
        if len(unique) == count:
            return unique
    if regularizer_center is not None:
        active = ACTIVE_REGULARIZERS[method]
        if not active:
            raise ValueError("Regularizer-only replenishment needs active regularizers.")
        design = qmc.LatinHypercube(d=len(active), seed=seed).random(128)
        for row in design:
            candidate = {**regularizer_center, "origin": "deterministic_unique_regularizer_reserve"}
            for offset, name in enumerate(active):
                candidate[name] = float(10 ** (-5 + 8 * row[offset]))
            append(candidate)
            if len(unique) == count:
                return unique
    elif method == "HistGBR":
        design = qmc.LatinHypercube(d=6, seed=seed).random(128)
        for row in design:
            append({
                "learning_rate": float(10 ** (-2 + math.log10(30) * row[0])),
                "max_leaf_nodes": (7, 15, 31, 63)[min(int(4 * row[1]), 3)],
                "min_samples_leaf": (5, 10, 20, 40, 80)[min(int(5 * row[2]), 4)],
                "l2_regularization": (0., .1, 1., 10., 100.)[min(int(5 * row[3]), 4)],
                "max_iter": (100, 200, 400, 600)[min(int(4 * row[4]), 3)],
                "target_fraction": float(.05 + .945 * row[5]),
                "origin": "deterministic_unique_global_reserve",
            })
            if len(unique) == count:
                return unique
    else:
        active = ACTIVE_REGULARIZERS[method]
        base_dimensions = 2 if method == "TargetSVQR" else 3
        design = qmc.LatinHypercube(d=base_dimensions + len(active), seed=seed).random(128)
        for index, row in enumerate(design):
            candidate = {**_base_regularizer(method),
                "gamma": float(10 ** (-4 + math.log10(800) * row[0])),
                "total_loss": float(10 ** (2 + math.log10(24000) * row[1])),
                "rank": _kernel_ranks(method)[index % 3],
                "origin": "deterministic_unique_global_reserve"}
            if method != "TargetSVQR":
                candidate["target_fraction"] = float(.05 + .945 * row[2])
            for offset, name in enumerate(active, start=base_dimensions):
                candidate[name] = float(10 ** (-5 + 8 * row[offset]))
            append(candidate)
            if len(unique) == count:
                return unique
    raise RuntimeError("Deterministic unique-candidate reserve was exhausted.")


def _kernel_base_candidates(method: str = "Full") -> list[dict]:
    """Historical anchors plus a fixed design over active base coordinates."""
    ranks = _kernel_ranks(method)
    candidates = []
    for rank in ranks:
        candidates.append({"gamma": .003, "total_loss": 1000.,
                           "target_fraction": .5, "rank": rank,
                           "origin": "selection_v1_anchor"})
    for rank in ranks:
        candidates.append({"gamma": .002, "total_loss": 562500.,
                           "target_fraction": .99413, "rank": rank,
                           "origin": "historical_point_anchor"})
    if method != "TargetSVQR":
        candidates.append({"gamma": .002, "total_loss": 562500.,
                           "target_fraction": .9808, "rank": ranks[1],
                           "origin": "historical_cross_ratio_anchor"})
    candidates.extend([
        {"gamma": .002, "total_loss": 2393000., "target_fraction": .99273,
         "rank": ranks[1], "origin": "historical_interval_ratio_anchor"},
        {"gamma": 1e-4, "total_loss": 100., "target_fraction": .05,
         "rank": ranks[0], "origin": "lower_boundary_anchor"},
        {"gamma": .08, "total_loss": 2.4e6, "target_fraction": .995,
         "rank": ranks[2], "origin": "upper_boundary_anchor"},
    ])
    dimension = 2 if method == "TargetSVQR" else 3
    design = qmc.LatinHypercube(d=dimension, seed=20261023).random(18 - len(candidates))
    for index, row in enumerate(design):
        candidate = {
            "gamma": float(10 ** (-4 + 3 * row[0])),
            "total_loss": float(10 ** (2 + math.log10(24000) * row[1])),
            "rank": ranks[index % 3], "origin": "deterministic_space_fill"}
        if method != "TargetSVQR":
            candidate["target_fraction"] = float(.05 + .945 * row[2])
        candidates.append(candidate)
    return _complete_unique_candidates(method, candidates, 20261200 + KERNEL_METHODS.index(method))


def _tree_candidates(count: int = 18) -> list[dict]:
    grid = [
        {"learning_rate": learning_rate, "max_leaf_nodes": leaves,
         "min_samples_leaf": minimum, "l2_regularization": l2,
         "max_iter": iterations, "target_fraction": fraction,
         "origin": "deterministic_tree_grid"}
        for learning_rate in (0.03, 0.1)
        for leaves in (7, 15, 31)
        for minimum in (10, 20, 40)
        for l2 in (0.0, 1.0, 10.0)
        for iterations in (100, 200, 400)
        for fraction in (0.1, 0.5, 0.9, 0.98, 0.995)
    ]
    rng = np.random.default_rng(20261023)
    selected = [grid[index] for index in rng.choice(len(grid), count - 2, replace=False)]
    candidates = [
        {"learning_rate": 0.03, "max_leaf_nodes": 7, "min_samples_leaf": 10,
         "l2_regularization": 0.0, "max_iter": 200, "target_fraction": 0.5,
         "origin": "conservative_tree_anchor"},
        {"learning_rate": 0.03, "max_leaf_nodes": 15, "min_samples_leaf": 20,
         "l2_regularization": 1.0, "max_iter": 400, "target_fraction": 0.995,
         "origin": "high_target_tree_anchor"},
        *selected,
    ]

    return _complete_unique_candidates("HistGBR", candidates, 20261210, count)


def _regularizer_candidates(method: str) -> list[dict]:
    scalar = [1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-2,
              1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    if method == "NoVariance":
        return [{"lambda1": 0.0, "lambda2": value} for value in scalar]
    if method == "NoDiscrepancy":
        return [{"lambda1": value, "lambda2": 0.0} for value in scalar]
    if method == "Full":
        anchors = [
            {"lambda1": 1e-5, "lambda2": 1e-5},
            {"lambda1": 5e-4, "lambda2": 1e-4},
            {"lambda1": 1.0, "lambda2": 1.0},
            {"lambda1": 1000.0, "lambda2": 1000.0},
            {"lambda1": 1e-4, "lambda2": 100.0},
            {"lambda1": 100.0, "lambda2": 1e-4},
        ]
        design = qmc.LatinHypercube(d=2, seed=20261024).random(12)
        space_fill = [{"lambda1": float(10 ** (-5 + 8 * row[0])),
                       "lambda2": float(10 ** (-5 + 8 * row[1]))}
                      for row in design]
        return anchors + space_fill
    if method in ("WeightedSVQR", "TargetSVQR"):
        return []
    raise ValueError(f"No kernel regularizer definition for {method}.")


def _base_regularizer(method: str) -> dict:
    return {
        "Full": {"lambda1": 1.0, "lambda2": 1.0},
        "NoVariance": {"lambda1": 0.0, "lambda2": 1.0},
        "NoDiscrepancy": {"lambda1": 1.0, "lambda2": 0.0},
        "WeightedSVQR": {"lambda1": 0.0, "lambda2": 0.0},
        "TargetSVQR": {"lambda1": 0.0, "lambda2": 0.0},
    }[method]


def _joint_kernel_candidates(method: str, selected: dict, *, excluded: list[dict] = ()) -> list[dict]:
    """Predeclared local joint refinement around a sequentially selected point."""
    active = ACTIVE_REGULARIZERS[method]
    base_dimensions = 2 if method == "TargetSVQR" else 3
    dimension = base_dimensions + len(active)
    method_offset = KERNEL_METHODS.index(method)
    design = qmc.LatinHypercube(d=dimension, seed=20261100 + method_offset).random(17)
    candidates = [{**selected, "origin": "joint_refinement_center"}]
    base_log_gamma = math.log10(selected["gamma"])
    base_log_total = math.log10(selected["total_loss"])
    base_fraction = min(max(selected.get("target_fraction", 1.0), 1e-9), 1 - 1e-9)
    base_log_odds = math.log(base_fraction / (1 - base_fraction))
    for index, row in enumerate(design):
        candidate = dict(selected)
        candidate.update({
            "gamma": float(np.clip(10 ** (base_log_gamma + 1.5 * (row[0] - .5)),
                                   1e-4, .08)),
            "total_loss": float(np.clip(10 ** (base_log_total + 2 * (row[1] - .5)),
                                        100., 2.4e6)),
            "rank": _kernel_ranks(method)[index % 3],
            "origin": "deterministic_local_joint_refinement",
        })
        if method != "TargetSVQR":
            candidate["target_fraction"] = float(np.clip(
                1 / (1 + math.exp(-(base_log_odds + 4.6 * (row[2] - .5)))), .05, .995))
        for offset, name in enumerate(active, start=base_dimensions):
            base_value = max(float(selected[name]), 1e-5)
            candidate[name] = float(np.clip(
                10 ** (math.log10(base_value) + 4 * (row[offset] - .5)), 1e-5, 1000.))
        candidates.append(candidate)
    return _complete_unique_candidates(method, candidates, 20261220 + method_offset, excluded=excluded)


def _middle_kernel_candidates(method: str, selected: dict, *, excluded: list[dict] = ()) -> list[dict]:
    """Second 18-fit base search for kernel methods without active lambdas."""
    if ACTIVE_REGULARIZERS[method]:
        raise ValueError("Active-regularizer methods use their lambda search in the middle stage.")
    dimension = 2 if method == "TargetSVQR" else 3
    design = qmc.LatinHypercube(d=dimension, seed=20261060 + KERNEL_METHODS.index(method)).random(17)
    candidates = [{**selected, "origin": "middle_base_refinement_center"}]
    base_log_gamma = math.log10(selected["gamma"])
    base_log_total = math.log10(selected["total_loss"])
    fraction = min(max(selected.get("target_fraction", 1.0), 1e-9), 1 - 1e-9)
    log_odds = math.log(fraction / (1 - fraction))
    for index, row in enumerate(design):
        candidate = dict(selected)
        candidate.update({
            "gamma": float(np.clip(10 ** (base_log_gamma + 2 * (row[0] - .5)),
                                   1e-4, .08)),
            "total_loss": float(np.clip(10 ** (base_log_total + 3 * (row[1] - .5)),
                                        100., 2.4e6)),
            "rank": _kernel_ranks(method)[index % 3],
            "origin": "deterministic_middle_base_refinement",
        })
        if method != "TargetSVQR":
            candidate["target_fraction"] = float(np.clip(
                1 / (1 + math.exp(-(log_odds + 6 * (row[2] - .5)))), .05, .995))
        candidates.append(candidate)
    return _complete_unique_candidates(method, candidates, 20261230 + KERNEL_METHODS.index(method), excluded=excluded)


def _local_tree_candidates(selected: dict, seed: int, origin: str, *, excluded: list[dict] = ()) -> list[dict]:
    design = qmc.LatinHypercube(d=5, seed=seed).random(17)
    candidates = [{**selected, "origin": origin + "_center"}]
    fraction = min(max(selected["target_fraction"], 1e-9), 1 - 1e-9)
    log_odds = math.log(fraction / (1 - fraction))
    leaves = (7, 15, 31, 63)
    minimum_leaf = (5, 10, 20, 40, 80)
    iterations = (100, 200, 400, 600)
    l2_values = (0.0, 0.1, 1.0, 10.0, 100.0)
    for row in design:
        candidates.append({
            "learning_rate": float(np.clip(
                selected["learning_rate"] * 10 ** (row[0] - .5), .01, .3)),
            "max_leaf_nodes": leaves[min(int(row[1] * len(leaves)), len(leaves) - 1)],
            "min_samples_leaf": minimum_leaf[min(int(row[2] * len(minimum_leaf)),
                                                   len(minimum_leaf) - 1)],
            "l2_regularization": l2_values[min(int(row[3] * len(l2_values)),
                                                  len(l2_values) - 1)],
            "max_iter": iterations[min(int((row[1] + row[3]) * .5 * len(iterations)),
                                        len(iterations) - 1)],
            "target_fraction": float(np.clip(
                1 / (1 + math.exp(-(log_odds + 4.6 * (row[4] - .5)))), .05, .995)),
            "origin": origin,
        })
    return _complete_unique_candidates("HistGBR", candidates, seed + 1000, excluded=excluded)


def _middle_tree_candidates(selected: dict, *, excluded: list[dict] = ()) -> list[dict]:
    """Regularizer-equivalent middle search for the tree comparator."""
    return _local_tree_candidates(selected, 20261099,
                                  "deterministic_middle_tree_refinement", excluded=excluded)


def _joint_tree_candidates(selected: dict, *, excluded: list[dict] = ()) -> list[dict]:
    """Give the tree comparator the same deterministic 18-fit joint budget."""
    return _local_tree_candidates(selected, 20261199,
                                  "deterministic_local_joint_refinement", excluded=excluded)


def protocol() -> dict:
    tasks = ([{"name": f"within_site{site}", "site": site, "mode": "within",
               "role": "primary_breadth"} for site in range(1, 7)] +
             [{"name": f"cross_site{site}", "site": site, "mode": "cross",
               "role": "leave_one_farm_out_stress"} for site in range(1, 7)])
    return {
        "version": "selection_v2",
        "design_seed": 20261023,
        "feature_loader_version": LOADER_VERSION,
        "feature_track": "common meteorology: wind, direction, temperature, pressure, humidity; 360 lagged inputs",
        "test_status": "2020 Q3-Q4 has been inspected previously; this remains a reused-test follow-up",
        "fit_period": "2019",
        "selection_period": "2020 Q1",
        "calibration_period": "2020 Q2",
        "test_period": "2020 Q3-Q4",
        "default_execution": "validation selection and lock only; no calibration/test prediction",
        "parallel_rule": "optional workers may partition base/regularizer/joint phases by disjoint task names; only an unfiltered coordinator may create the full lock",
        "methods": list(METHODS),
        "pooled_kernel_methods": list(POOLED_KERNEL_METHODS),
        "kernel_base_candidates": _kernel_base_candidates(),
        "target_kernel_base_candidates": _kernel_base_candidates("TargetSVQR"),
        "environment_versions": environment_versions(),
        "tree_candidates": _tree_candidates(),
        "regularizer_candidates": {method: _regularizer_candidates(method)
                                   for method in ACTIVE_REGULARIZERS},
        "base_regularizer_anchors": {method: _base_regularizer(method)
                                     for method in KERNEL_METHODS},
        "selection_stages": [
            "select gamma, total loss mass, target loss fraction, and requested Nystrom rank from a shared kernel design with pool-valid method-specific ranks",
            "use 18 Q1 queries for active raw-lambda tuning or a deterministic regularizer-equivalent base refinement",
            "perform one predeclared 18-candidate local joint refinement for every method; kernel variants also refine active lambdas",
        ],
        "candidate_uniqueness": "each stage contains 18 newly queried effective unique models, disjoint from all earlier stages, after ignoring origin and inactive coordinates; deterministic Latin-hypercube reserves replenish collisions caused by anchors or clipping; reserves span the global search region except active-lambda middle searches, whose reserves keep the selected base coordinates fixed; TargetSVQR omits p and uses a two-dimensional continuous design with p=1 at fitting",
        "validation_query_budget": {
            "base": 18, "middle": 18, "joint": 18,
            "total_per_task_method": 54,
            "cross_stage_rule": "18 effective unique new configurations per stage, excluding every earlier-stage signature, for 54 unique queried configurations per task and method; the previous eligible incumbent is compared using its cached validation score and fit folders, without refitting or counting it as a new query",
            "rule": "every method receives exactly the same number of Q1 configuration queries; kernel queries average seeds 0,1,2; deterministic HistGBR uses seed 0 once",
        },
        "selection_rule": "minimum mean target-Q1 RMSE across method-specific seeds (kernel 0,1,2; deterministic HistGBR 0) among candidates meeting the common solver certificate; ties retain a carried incumbent, then use new candidate index",
        "nonconvergence_rule": "retry once from the first incumbent with 12000 ADMM iterations; retain diagnostics but mark a candidate ineligible if any selection seed still misses the stated duality-gap tolerance",
        "refit_rule": "locked configuration is refitted using 2019 fitting rows only; Q1 selection labels are not added to fitting",
        "source_cap_per_other_site": 1500,
        "within_site_source_cap": 7500,
        "within_target_fit_budget": 128,
        "cross_target_fit_budget": 128,
        "target_sampling": "month x disjoint event stratum; deterministic time-even allocation using training covariates only",
        "landmark_ranks": [128, 256, 512],
        "landmark_ranks_by_method": {method: list(_kernel_ranks(method)) for method in KERNEL_METHODS},
        "landmark_pool_source_cap": 2000,
        "landmark_pool_target_cap": 2000,
        "landmark_rule": "fit-only, label-free KMeans; pooled source and target each receive total sample weight 0.5; target-only uses target rows only; effective rank is transparently capped by landmark-pool size",
        "selection_seeds": [0, 1, 2],
        "evaluation_seeds": [0, 1, 2],
        "selection_seeds_by_method": {method: ([0] if method == "HistGBR" else [0, 1, 2]) for method in METHODS},
        "evaluation_seeds_by_method": {method: ([0] if method == "HistGBR" else [0, 1, 2]) for method in METHODS},
        "taus": [0.5, 0.025, 0.975],
        "solver_attempts": [
            {"max_iter": 3000, "gap_rtol": 0.001, "gap_atol": 1e-7,
             "check_every": 25},
            {"max_iter": 12000, "gap_rtol": 0.001, "gap_atol": 1e-7,
             "check_every": 25},
        ],
        "regularizer_normalization": "none; preserve the manuscript's raw covariance, cross, and projected discrepancy terms; record raw traces for audit",
        "alpha": 0.05,
        "rolling_window": 256,
        "threads": 1,
        "label_accounting": "every task uses 128 target fitting labels plus all Q1 selection and Q2 calibration labels; 128 is the fit budget, not the total supervised-label count",
        "analysis_plan": {
            "status": "declared before selection-v2 fitting; Q3-Q4 is a reused test period",
            "primary_scope": "six within-site normal-to-extreme tasks, all six sites",
            "secondary_scope": "six leave-one-farm-out cross-site transfer-stress tasks",
            "primary_estimands": [
                "site-equal macro normalized RMSE: mean across six sites of mean-seed RMSE divided by site capacity",
                "site-equal mean method rank: rank each site's mean-seed RMSE with average ranks for ties, then average across six sites",
            ],
            "primary_reference": "Full",
            "comparators": [method for method in METHODS if method != "Full"],
            "complete_reporting": "report every predeclared task, site and method; no outcome-based exclusion or best-seed selection",
            "test_use": "one locked-setting Q3-Q4 evaluation after the full selection lock; failed executions may resume unchanged; no test-guided retuning",
            "seed_aggregation": "site RMSE averages separate seed RMSEs; inference averages squared errors across each method's declared seeds at each timestamp; no seed ensembling",
            "paired_inference": {
                "loss": "paired squared-error difference Full minus comparator at matching timestamps",
                "blocks": "non-overlapping Monday-Sunday seven-calendar-day blocks within each site; fixed 1970-01-05 Monday origin",
                "bootstrap_rule": "equal-probability resampling of whole blocks; each replicate divides sampled block loss sums by sampled observation counts",
                "sign_flip_rule": "two-sided Monte Carlo paired block sign flips with plus-one correction",
                "random_seed": 20261025,
                "bootstrap_replicates": 10000,
                "confidence_level": 0.95,
                "sign_flip_replicates": 99999,
                "holm_families": "separate within and cross families, each containing all 6 sites x 5 Full comparisons = 30 tests",
                "interpretation": "exploratory paired block inference on reused test data; non-rejection is not equivalence",
            },
            "exploratory_comparison_set": {
                "rule": "within each scope compare all 15 method pairs using two-sided paired seven-day block swaps of site-equal macro capacity-normalized seed-mean RMSE with Holm correction; retain the empirical winner and methods not shown worse than that winner at adjusted alpha 0.05",
                "interpretation": "not a formal model confidence set; retention means not eliminated by this comparison procedure, not equivalence or established superiority",
            },
        },
        "tasks": tasks,
    }


class Study:
    def __init__(self, output: Path, config: dict, *, write_global_provenance: bool = True):
        self.output, self.config = Path(output), config
        self.data = load_all_selection_data_v2()
        self.context_cache: dict = {}
        self.feature_cache: dict = {}
        self.heldout_unlocked = False
        if write_global_provenance:
            save_json(self.output / "data_provenance.json", {
                str(site): {
                    "feature_names": list(data.feature_names),
                    "provenance": data.provenance,
                    "preprocessing": data.preprocessing,
                    "statistics": data.statistics,
                }
                for site, data in self.data.items()
            })

    def context(self, task: dict, method: str) -> dict:
        target_only = method == "TargetSVQR"
        key = (task["name"], target_only)
        if key in self.context_cache:
            return self.context_cache[key]
        site = task["site"]
        target_data = self.data[site]
        folder = self.output / task["name"] / ("target_data" if target_only else "pooled_data")
        folder.mkdir(parents=True, exist_ok=True)

        budget = (self.config["within_target_fit_budget"] if task["mode"] == "within"
                  else self.config["cross_target_fit_budget"])
        target_ids, sampling = stratified_target_indices(target_data, budget)

        source_parts = []
        if not target_only:
            source_sites = [site] if task["mode"] == "within" else [s for s in self.data if s != site]
            cap = (self.config["within_site_source_cap"] if task["mode"] == "within"
                   else self.config["source_cap_per_other_site"])
            for source_site in source_sites:
                ids = np.flatnonzero(self.data[source_site].mask("train", "source"))
                if len(ids) > cap:
                    ids = ids[np.linspace(0, len(ids) - 1, cap, dtype=int)]
                source_parts.append((source_site, ids))

        source_n = sum(len(ids) for _, ids in source_parts)
        source_x = [self.data[source_site].X[ids] for source_site, ids in source_parts]
        source_y = [self.data[source_site].y_raw[ids] / CAPACITY[source_site]
                    for source_site, ids in source_parts]
        raw_x = np.concatenate(source_x + [target_data.X[target_ids]])
        raw_y = np.concatenate(source_y + [target_data.y_raw[target_ids] / CAPACITY[site]])
        x, y, transform, state = preprocess(raw_x, raw_y, target_data.feature_names,
                                             source_n=source_n)

        validation_ids = np.flatnonzero(target_data.mask("validation", "target"))
        source_site_values = (np.concatenate([np.full(len(ids), source_site, dtype=int)
                                              for source_site, ids in source_parts])
                              if source_n else np.empty(0, dtype=int))
        source_window_values = (np.concatenate([ids for _, ids in source_parts])
                                if source_n else np.empty(0, dtype=int))
        source_raw_rows = (np.concatenate([self.data[source_site].target_row_indices[ids]
                                           for source_site, ids in source_parts])
                           if source_n else np.empty(0, dtype=int))
        save_npz(
            folder / "fit_and_validation_ids.npz",
            source_site=source_site_values,
            source_window=source_window_values,
            source_raw_target_row=source_raw_rows,
            target_window=target_ids,
            target_raw_target_row=target_data.target_row_indices[target_ids],
            validation_window=validation_ids,
            validation_raw_target_row=target_data.target_row_indices[validation_ids],
        )
        save_json(folder / "sampling.json", sampling)
        save_json(folder / "preprocessing.json", state)
        counts = {
            "n_source_fit": source_n,
            "n_target_fit": len(target_ids),
            "n_validation": len(validation_ids),
            "fit_period": "2019",
            "validation_period": "2020 Q1",
        }
        save_json(folder / "selection_counts.json", counts)
        context = {
            "folder": folder, "x": x, "y": y, "transform": transform,
            "state": state, "source_n": source_n, "target_ids": target_ids,
            "validation_ids": validation_ids, "site": site,
            "data": target_data, "counts": counts,
        }
        self.context_cache[key] = context
        return context

    def _heldout_ids(self, context: dict, split: str) -> np.ndarray:
        if split not in ("calibration", "test"):
            raise ValueError("Held-out IDs are only defined for calibration and test.")
        if not self.heldout_unlocked:
            raise RuntimeError("Calibration/test access is forbidden before a valid selection lock.")
        return np.flatnonzero(context["data"].mask(split, "target"))

    def features(self, context: dict, seed: int, gamma: float, requested_rank: int) -> dict:
        key = (str(context["folder"]), int(seed), int(requested_rank), float(gamma))
        if key in self.feature_cache:
            return self.feature_cache[key]
        x, source_n = context["x"], context["source_n"]
        landmark_folder = context["folder"] / "landmarks" / f"seed{seed}" / f"rank{requested_rank}"
        landmark_folder.mkdir(parents=True, exist_ok=True)
        landmark_path = landmark_folder / "landmarks.npz"
        start = time.perf_counter()
        if landmark_path.exists():
            stored = np.load(landmark_path)
            centers = stored["centers"]
            if int(stored["requested_rank"]) != requested_rank:
                raise ValueError("Stored landmark rank does not match its path.")
        else:
            rng = np.random.default_rng(seed)
            source_positions = (rng.choice(source_n, min(source_n, self.config["landmark_pool_source_cap"]),
                                           replace=False) if source_n else np.empty(0, dtype=int))
            target_count = len(x) - source_n
            target_local = rng.choice(target_count, min(target_count, self.config["landmark_pool_target_cap"]),
                                      replace=False)
            target_positions = source_n + target_local
            pool_positions = np.r_[source_positions, target_positions]
            pool = x[pool_positions]
            if source_n:
                pool_weight = np.r_[np.full(len(source_positions), 0.5 / len(source_positions)),
                                    np.full(len(target_positions), 0.5 / len(target_positions))]
                rule = "domain_balanced_source_0.5_target_0.5"
            else:
                pool_weight = np.full(len(target_positions), 1.0 / len(target_positions))
                rule = "target_only_equal_weight"
            effective_rank = min(requested_rank, len(pool))
            if effective_rank < 1:
                raise ValueError("No fitting rows are available for landmark construction.")
            if effective_rank == len(pool):
                centers = pool.copy()
            else:
                centers = KMeans(n_clusters=effective_rank, n_init=10,
                                 random_state=seed).fit(pool, sample_weight=pool_weight).cluster_centers_
            save_npz(
                landmark_path, centers=centers, pool_positions=pool_positions,
                pool_weight=pool_weight, requested_rank=requested_rank,
                effective_rank=effective_rank, rule=np.asarray(rule),
            )

        map_folder = landmark_folder / ("gamma_" + format(float(gamma), ".17g"))
        map_folder.mkdir(parents=True, exist_ok=True)
        map_path = map_folder / "map.npz"
        if map_path.exists():
            stored_map = np.load(map_path)
            normalization = stored_map["normalization"]
            stored_centers = stored_map["centers"]
            if stored_centers.shape != centers.shape or not np.array_equal(stored_centers, centers):
                raise ValueError("Feature map and rank-specific landmarks disagree.")
        else:
            gram = rbf_kernel(centers, gamma=gamma)
            values, vectors = np.linalg.eigh((gram + gram.T) * 0.5 + 1e-6 * np.eye(len(gram)))
            if np.any(values <= 0):
                raise ValueError("Nonpositive stabilized landmark Gram eigenvalue.")
            normalization = (vectors / np.sqrt(values)) @ vectors.T
            save_npz(map_path, centers=centers, normalization=normalization,
                     gamma=gamma, requested_rank=requested_rank,
                     effective_rank=len(centers))
        z = rbf_kernel(x, centers, gamma=gamma) @ normalization
        result = {
            "z": z, "centers": centers, "normalization": normalization,
            "gamma": gamma, "requested_rank": requested_rank,
            "effective_rank": len(centers), "path": map_path,
            "construction_seconds": time.perf_counter() - start,
        }
        if len(self.feature_cache) >= 24:
            self.feature_cache.pop(next(iter(self.feature_cache)))
        self.feature_cache[key] = result
        return result

    @staticmethod
    def _validate_existing_fit(record: dict, task: dict, method: str, config: dict,
                               seed: int, tau: float, folder: Path) -> None:
        expected = (task["name"], method, config, seed, tau)
        observed = (record.get("task"), record.get("method"), record.get("config"),
                    record.get("seed"), record.get("tau"))
        if observed != expected:
            raise ValueError(f"Existing fit identity mismatch: {folder}")
        artifact = folder / ("model.joblib" if method == "HistGBR" else "model.npz")
        if not artifact.exists():
            raise ValueError(f"Completed fit record lacks model artifact: {artifact}")

    def fit(self, task: dict, method: str, config: dict, seed: int, tau: float,
            folder: Path, candidate_id: str) -> dict:
        folder.mkdir(parents=True, exist_ok=True)
        fit_path = folder / "fit.json"
        if fit_path.exists():
            record = json.loads(fit_path.read_text())
            self._validate_existing_fit(record, task, method, config, seed, tau, folder)
            return record

        context = self.context(task, method)
        source_n, target_n = context["source_n"], len(context["y"]) - context["source_n"]
        fraction = float(config.get("target_fraction", 1.0)) if source_n else 1.0
        if not 0 < fraction < 1 and source_n:
            raise ValueError("Pooled target loss fraction must lie strictly between zero and one.")
        start = time.perf_counter()
        if method == "HistGBR":
            options = {key: value for key, value in config.items()
                       if key not in ("target_fraction", "origin")}
            model = HistGradientBoostingRegressor(
                loss="quantile", quantile=tau, early_stopping=False,
                random_state=seed, **options)
            weights = np.r_[
                np.full(source_n, (1 - fraction) / source_n) if source_n else np.empty(0),
                np.full(target_n, fraction / target_n),
            ] * len(context["y"])
            model.fit(context["x"], context["y"], sample_weight=weights)
            joblib.dump(model, folder / "model.joblib")
            diagnostics = {
                "solver": "histogram gradient boosting", "converged": None,
                "iterations": model.n_iter_, "fit_seconds": time.perf_counter() - start,
            }
            feature_map = None
        else:
            feature = self.features(context, seed, config["gamma"], config["rank"])
            z, dimension = feature["z"], feature["effective_rank"]
            lambda1 = float(config["lambda1"])
            lambda2 = float(config["lambda2"])
            regularizer = np.eye(dimension)
            linear = np.zeros(dimension)
            component_audit = {
                "normalization": "not_applicable_target_only",
                "variance_trace_raw": 0.0, "discrepancy_trace_raw": 0.0,
            }
            if source_n:
                covariance, cross, discrepancy, component_audit = regularizer_components(
                    z, context["y"], source_n)
                regularizer += lambda1 * covariance + lambda2 * discrepancy
                linear += lambda1 * cross
            total = float(config["total_loss"])
            weights = np.r_[
                np.full(source_n, total * (1 - fraction) / source_n) if source_n else np.empty(0),
                np.full(target_n, total * fraction / target_n),
            ]
            attempts = []
            solution = None
            for attempt_index, options in enumerate(self.config["solver_attempts"]):
                warm = ({"w0": solution.w, "b0": solution.b}
                        if solution is not None else {})
                solution = refine_pinball(z, context["y"], regularizer, linear,
                                          weights, tau, **options, **warm)
                report = solution.report()
                attempts.append({key: value for key, value in report.items()
                                 if key != "history"})
                if solution.converged:
                    break
            save_npz(folder / "model.npz", w=solution.w, b=solution.b,
                     dual=solution.dual_variable, R=regularizer,
                     c=linear, C=weights, y=context["y"],
                     variance_trace_raw=component_audit["variance_trace_raw"],
                     discrepancy_trace_raw=component_audit["discrepancy_trace_raw"])
            diagnostics = solution.report()
            diagnostics.update({
                "solver": "ADMM", "fit_seconds_including_feature_access": time.perf_counter() - start,
                "requested_rank": config["rank"], "effective_rank": dimension,
                "attempt_count": len(attempts), "attempt_summaries": attempts,
                "regularizer_components": component_audit,
            })
            feature_map = str(feature["path"].relative_to(self.output))

        record = {
            "task": task["name"], "site": task["site"], "method": method,
            "candidate_id": candidate_id, "config": config,
            "seed": seed, "tau": tau, "feature_map": feature_map,
            "data_folder": str(context["folder"].relative_to(self.output)),
            "diagnostics": diagnostics, "selection_counts": context["counts"],
            "effective_target_fraction": fraction,
        }
        save_json(fit_path, record)  # completion marker, deliberately last
        return record

    def predict(self, context: dict, folder: Path, split: str) -> np.ndarray:
        record = json.loads((folder / "fit.json").read_text())
        if split == "validation":
            ids = context["validation_ids"]
        elif split in ("calibration", "test"):
            ids = self._heldout_ids(context, split)
        else:
            raise ValueError(f"Prediction split is not allowed: {split}")
        x = context["transform"](context["data"].X[ids])
        if record["method"] == "HistGBR":
            prediction = joblib.load(folder / "model.joblib").predict(x)
        else:
            feature_map = np.load(self.output / record["feature_map"])
            model = np.load(folder / "model.npz")
            z = (rbf_kernel(x, feature_map["centers"], gamma=float(feature_map["gamma"])) @
                 feature_map["normalization"])
            prediction = z @ model["w"] + float(model["b"])
        state = context["state"]
        return ((prediction * state["label_scale"] + state["label_mean"])
                * CAPACITY[context["site"]])

    def _base_configs(self, method: str) -> list[dict]:
        if method == "HistGBR":
            return [dict(candidate) for candidate in self.config["tree_candidates"]]
        anchor = self.config["base_regularizer_anchors"][method]
        key = "target_kernel_base_candidates" if method == "TargetSVQR" else "kernel_base_candidates"
        return [{**candidate, **anchor} for candidate in self.config[key]]

    def _tune_configs(self, task: dict, method: str, configs: list[dict],
                      stage: str, *, carried_incumbent: dict | None = None,
                      previous_configs: list[dict] = ()) -> dict:
        _assert_unique_configs(method, configs)
        prior_signatures = {effective_model_signature(method, candidate) for candidate in previous_configs}
        if any(effective_model_signature(method, candidate) in prior_signatures for candidate in configs):
            raise ValueError("A new stage query duplicates an earlier effective configuration.")
        if carried_incumbent is not None:
            if not carried_incumbent["eligible_for_selection"]:
                raise ValueError("The carried incumbent lacks a valid solver certificate.")
            if effective_model_signature(method, carried_incumbent["config"]) not in prior_signatures:
                raise ValueError("The carried incumbent was not queried in an earlier stage.")
        selection_path = self.output / task["name"] / method / f"{stage}_selection.json"
        if selection_path.exists():
            record = json.loads(selection_path.read_text())
            if (record.get("task"), record.get("method"), record.get("stage")) != (
                    task["name"], method, stage):
                raise ValueError(f"Existing selection identity mismatch: {selection_path}")
            stored_configs = [row["config"] for row in record["candidates"]]
            _assert_unique_configs(method, stored_configs)
            if record.get("carried_incumbent") != carried_incumbent:
                raise ValueError(f"Existing carried incumbent changed: {selection_path}")
            if stored_configs != configs:
                raise ValueError(f"Existing selection candidate list changed: {selection_path}")
            return record

        context = self.context(task, method)
        validation_ids = context["validation_ids"]
        y_validation = context["data"].y_raw[validation_ids]
        rows = []
        start = time.perf_counter()
        for index, candidate in enumerate(configs):
            candidate_id = f"{stage}_c{index:03d}"
            seed_rows = []
            fit_folders = {}
            for seed in self.config["selection_seeds_by_method"][method]:
                folder = (self.output / task["name"] / method / stage / "candidates" /
                          f"c{index:03d}" / f"seed{seed}")
                fit_record = self.fit(task, method, candidate, seed, 0.5, folder, candidate_id)
                prediction = self.predict(context, folder, "validation")
                metrics = point_metrics(y_validation, prediction)
                pd.DataFrame({
                    "window_index": validation_ids,
                    "raw_target_row": context["data"].target_row_indices[validation_ids],
                    "time": context["data"].timestamps[validation_ids].astype(str),
                    "y_true": y_validation, "prediction": prediction,
                }).to_csv(folder / "validation_predictions.csv", index=False)
                save_json(folder / "validation_metrics.json", metrics)
                seed_rows.append({"seed": seed, "metrics": metrics,
                                  "diagnostics": fit_record["diagnostics"]})
                fit_folders[str(seed)] = str(folder.relative_to(self.output))
            rmse = np.asarray([row["metrics"]["rmse"] for row in seed_rows])
            mae = np.asarray([row["metrics"]["mae"] for row in seed_rows])
            failed_seeds = ([row["seed"] for row in seed_rows
                             if method != "HistGBR" and
                             row["diagnostics"].get("converged") is not True])
            rows.append({
                "candidate_index": index, "candidate_id": candidate_id, "queried_stage": stage,
                "config": candidate, "seed_results": seed_rows,
                "mean_validation_rmse": float(rmse.mean()),
                "std_validation_rmse": float(rmse.std(ddof=0)),
                "mean_validation_mae": float(mae.mean()),
                "eligible_for_selection": not failed_seeds,
                "failed_certificate_seeds": failed_seeds,
                "fit_folders": fit_folders,
            })
        eligible = [row for row in rows if row["eligible_for_selection"]]
        if carried_incumbent is not None:
            eligible.append(carried_incumbent)
        if not eligible:
            raise RuntimeError(f"No {stage} candidate met the common solver certificate for "
                               f"{task['name']}/{method}.")
        best = min(eligible, key=lambda row: (row["mean_validation_rmse"],
                                              row is not carried_incumbent, row["candidate_index"]))
        selection = {
            "task": task["name"], "site": task["site"], "role": task["role"],
            "method": method, "stage": stage,
            "selection_rule": self.config["selection_rule"],
            "selected_candidate_index": best["candidate_index"],
            "selected_candidate_id": best["candidate_id"],
            "selected_source_stage": best["queried_stage"],
            "selected_candidate": best,
            "carried_incumbent": carried_incumbent,
            "selected_config": best["config"],
            "selected_mean_validation_rmse": best["mean_validation_rmse"],
            "selected_fit_folders": best["fit_folders"],
            "candidates": rows,
            "candidate_count": len(rows),
            "queried_effective_unique_count": len({effective_model_signature(method, row["config"]) for row in rows}),
            "selection_seconds": time.perf_counter() - start,
        }
        save_json(selection_path, selection)
        print(json.dumps({
            "phase": stage, "task": task["name"], "method": method,
            "selected": best["candidate_index"],
            "selected_source_stage": best["queried_stage"],
            "mean_validation_rmse": best["mean_validation_rmse"],
        }), flush=True)
        return selection

    def _jobs(self, task_names: set[str] | None = None,
              method_names: set[str] | None = None):
        for task in self.config["tasks"]:
            if task_names is not None and task["name"] not in task_names:
                continue
            for method in self.config["methods"]:
                if method_names is not None and method not in method_names:
                    continue
                yield task, method

    def tune_base(self, task_names: set[str] | None = None,
                  method_names: set[str] | None = None) -> list[dict]:
        selections = []
        previous_task = None
        for task, method in self._jobs(task_names, method_names):
            if previous_task is not None and task["name"] != previous_task:
                self.context_cache.clear()
                self.feature_cache.clear()
            selections.append(self._tune_configs(task, method,
                                                 self._base_configs(method), "base"))
            previous_task = task["name"]
        self.context_cache.clear()
        self.feature_cache.clear()
        return selections

    def tune_regularizers(self, task_names: set[str] | None = None,
                          method_names: set[str] | None = None) -> list[dict]:
        selections = []
        previous_task = None
        for task, method in self._jobs(task_names, method_names):
            if previous_task is not None and task["name"] != previous_task:
                self.context_cache.clear()
                self.feature_cache.clear()
            base_path = self.output / task["name"] / method / "base_selection.json"
            if not base_path.exists():
                raise RuntimeError(f"Base selection is incomplete: {base_path}")
            base = json.loads(base_path.read_text())
            previous_configs = [row["config"] for row in base["candidates"]]
            active = ACTIVE_REGULARIZERS.get(method, ())
            output_path = self.output / task["name"] / method / "regularized_selection.json"
            if active:
                configurations = [
                    {**base["selected_config"], **regularizer}
                    for regularizer in self.config["regularizer_candidates"][method]
                ]
                configurations = _complete_unique_candidates(
                    method, configurations, 20261240 + KERNEL_METHODS.index(method),
                    excluded=previous_configs, regularizer_center=base["selected_config"])
                middle_kind = "active_raw_regularizer_search"
            elif method in KERNEL_METHODS:
                configurations = _middle_kernel_candidates(method, base["selected_config"], excluded=previous_configs)
                middle_kind = "regularizer_equivalent_kernel_base_refinement"
            else:
                configurations = _middle_tree_candidates(base["selected_config"], excluded=previous_configs)
                middle_kind = "regularizer_equivalent_tree_refinement"
            selected = self._tune_configs(
                task, method, configurations, "middle",
                carried_incumbent=base["selected_candidate"], previous_configs=previous_configs)
            record = {**selected, "source_stage": "middle", "middle_kind": middle_kind,
                      "base_selection_sha256": digest(base_path)}
            if output_path.exists():
                if json.loads(output_path.read_text()) != record:
                    raise ValueError(f"Existing regularized selection changed: {output_path}")
            else:
                save_json(output_path, record)
            selections.append(record)
            previous_task = task["name"]
        self.context_cache.clear()
        self.feature_cache.clear()
        return selections

    def tune_joint(self, task_names: set[str] | None = None,
                   method_names: set[str] | None = None) -> list[dict]:
        final = []
        previous_task = None
        for task, method in self._jobs(task_names, method_names):
            if previous_task is not None and task["name"] != previous_task:
                self.context_cache.clear()
                self.feature_cache.clear()
            regularized_path = (self.output / task["name"] / method /
                                "regularized_selection.json")
            if not regularized_path.exists():
                raise RuntimeError(f"Regularizer stage is incomplete: {regularized_path}")
            regularized = json.loads(regularized_path.read_text())
            final_path = self.output / task["name"] / method / "final_selection.json"
            base = json.loads((self.output / task["name"] / method / "base_selection.json").read_text())
            previous_configs = [row["config"] for row in base["candidates"] + regularized["candidates"]]
            configurations = (_joint_kernel_candidates(method, regularized["selected_config"], excluded=previous_configs)
                              if method in KERNEL_METHODS else
                              _joint_tree_candidates(regularized["selected_config"], excluded=previous_configs))
            selected = self._tune_configs(
                task, method, configurations, "joint",
                carried_incumbent=regularized["selected_candidate"], previous_configs=previous_configs)
            final_record = {**selected, "source_stage": "joint",
                            "regularized_selection_sha256": digest(regularized_path)}
            if final_path.exists():
                if json.loads(final_path.read_text()) != final_record:
                    raise ValueError(f"Existing final selection changed: {final_path}")
            else:
                save_json(final_path, final_record)
            final.append(final_record)
            previous_task = task["name"]
        self.context_cache.clear()
        self.feature_cache.clear()
        return final

    def _expected_selection_files(self) -> set[str]:
        expected = set()
        for task in self.config["tasks"]:
            for method in self.config["methods"]:
                for name in ("base_selection.json", "middle_selection.json",
                             "regularized_selection.json", "joint_selection.json",
                             "final_selection.json"):
                    expected.add(f"{task['name']}/{method}/{name}")
        return expected

    def create_selection_lock(self, manifest: dict) -> dict:
        expected_keys = [(task["name"], method) for task in self.config["tasks"] for method in self.config["methods"]]
        if len(expected_keys) != 72 or len(set(expected_keys)) != 72:
            raise ValueError("A full selection lock requires 72 unique task-method keys.")
        expected = self._expected_selection_files()
        missing = [relative for relative in sorted(expected) if not (self.output / relative).exists()]
        if missing:
            raise RuntimeError(f"Cannot lock incomplete selection; first missing file: {missing[0]}")
        for task in self.config["tasks"]:
            for method in self.config["methods"]:
                all_queried = []
                prior_winner = None
                for stage in ("base", "middle", "joint"):
                    record = json.loads((self.output / task["name"] / method /
                                         f"{stage}_selection.json").read_text())
                    if record.get("candidate_count") != 18 or len(record.get("candidates", [])) != 18:
                        raise ValueError(f"Unequal validation-query budget in {task['name']}/{method}/{stage}.")
                    if (record.get("task"), record.get("method"), record.get("stage")) != (task["name"], method, stage):
                        raise ValueError("Selection record identity mismatch before locking.")
                    configs = [row["config"] for row in record["candidates"]]
                    _assert_unique_configs(method, configs)
                    if record.get("queried_effective_unique_count") != 18:
                        raise ValueError("Stage effective unique-query count is not 18.")
                    if record.get("carried_incumbent") != prior_winner:
                        raise ValueError("Carried incumbent does not match the preceding winner.")
                    eligible = [row for row in record["candidates"] if row["eligible_for_selection"]]
                    if prior_winner is not None:
                        eligible.append(prior_winner)
                    if not eligible:
                        raise ValueError("No eligible candidate or carried incumbent before locking.")
                    best = min(eligible, key=lambda row: (row["mean_validation_rmse"],
                               row is not prior_winner, row["candidate_index"]))
                    if (record["selected_candidate"] != best or record["selected_config"] != best["config"]
                            or record["selected_fit_folders"] != best["fit_folders"]
                            or record["selected_mean_validation_rmse"] != best["mean_validation_rmse"]):
                        raise ValueError("Stored winner is inconsistent with queried scores and carried incumbent.")
                    all_queried.extend(configs)
                    prior_winner = best
                _assert_unique_configs(method, all_queried, count=54)
                final = json.loads((self.output / task["name"] / method / "final_selection.json").read_text())
                if (final.get("selected_candidate") != prior_winner or
                        final.get("selected_config") != prior_winner["config"]):
                    raise ValueError("Final selection does not match the best staged incumbent.")
        files = {relative: digest(self.output / relative) for relative in sorted(expected)}
        final_records = [
            json.loads((self.output / task["name"] / method / "final_selection.json").read_text())
            for task in self.config["tasks"] for method in self.config["methods"]
        ]
        observed_keys = [(row.get("task"), row.get("method")) for row in final_records]
        if len(observed_keys) != 72 or set(observed_keys) != set(expected_keys):
            raise ValueError("Final selection records must contain 72 unique task-method identities.")
        lock = {
            "version": "selection_v2_lock_2",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest(self.output / "protocol.json"),
            "source_hashes": manifest["source_hashes"],
            "data_hashes": manifest["data_hashes"],
            "environment_versions": manifest["environment_versions"],
            "selection_files": files,
            "final_selections": final_records,
            "validation_query_budget": self.config["validation_query_budget"],
            "total_configuration_queries": (len(self.config["tasks"]) *
                                            len(self.config["methods"]) * 54),
            "data_provenance_sha256": digest(self.output / "data_provenance.json"),
            "heldout_policy": "no Q2 calibration or Q3-Q4 prediction was calculated before this lock",
        }
        lock_path = self.output / "selection_lock.json"
        if lock_path.exists():
            existing = json.loads(lock_path.read_text())
            comparable = {key: value for key, value in lock.items() if key != "created_utc"}
            previous = {key: value for key, value in existing.items() if key != "created_utc"}
            if previous != comparable:
                raise ValueError("Existing selection lock differs from current selections.")
            return existing
        save_json(lock_path, lock)
        return lock

    def unlock_heldout(self, manifest: dict) -> dict:
        lock_path = self.output / "selection_lock.json"
        if not lock_path.exists():
            raise RuntimeError("Held-out evaluation requires selection_lock.json.")
        lock = json.loads(lock_path.read_text())
        if lock["protocol_sha256"] != digest(self.output / "protocol.json"):
            raise ValueError("Selection-lock protocol hash mismatch.")
        if lock["source_hashes"] != manifest["source_hashes"] or lock["data_hashes"] != manifest["data_hashes"]:
            raise ValueError("Selection-lock source/data hashes mismatch.")
        if lock.get("environment_versions") != environment_versions():
            raise ValueError("Selection-lock environment versions mismatch.")
        if set(lock["selection_files"]) != self._expected_selection_files():
            raise ValueError("Selection lock has an incomplete or unexpected file list.")
        if any(digest(self.output / relative) != value
               for relative, value in lock["selection_files"].items()):
            raise ValueError("A locked selection file has changed.")
        if digest(self.output / "data_provenance.json") != lock["data_provenance_sha256"]:
            raise ValueError("Locked data provenance has changed.")
        expected_final = len(self.config["tasks"]) * len(self.config["methods"])
        if len(lock["final_selections"]) != expected_final:
            raise ValueError("Selection lock has missing or duplicate final selections.")
        self.heldout_unlocked = True
        return lock

    def evaluate(self, manifest: dict) -> list[dict]:
        lock = self.unlock_heldout(manifest)
        lock_sha256 = digest(self.output / "selection_lock.json")
        selected = {(row["task"], row["method"]): row for row in lock["final_selections"]}
        results = []
        for task in self.config["tasks"]:
            for method in self.config["methods"]:
                selection = selected[(task["name"], method)]
                config = selection["selected_config"]
                context = self.context(task, method)
                calibration_ids = self._heldout_ids(context, "calibration")
                test_ids = self._heldout_ids(context, "test")
                for seed in self.config["evaluation_seeds_by_method"][method]:
                    base = self.output / task["name"] / method / "evaluation" / f"seed{seed}"
                    result_path = base / "result.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        if result.get("selection_lock_sha256") != lock_sha256:
                            raise ValueError(f"Existing result used a different selection lock: {result_path}")
                        results.append(result)
                        continue
                    test_predictions, calibration_predictions, fits = [], [], []
                    for tau in self.config["taus"]:
                        folder = base / ("tau_" + format(tau, ".17g"))
                        fit_record = self.fit(task, method, config, seed, tau, folder,
                                              "locked_final")
                        if (method != "HistGBR" and
                                fit_record["diagnostics"].get("converged") is not True):
                            raise RuntimeError(
                                f"Locked evaluation fit missed the common solver certificate: {folder}")
                        fits.append({"path": str(folder.relative_to(self.output)), **fit_record})
                        test_predictions.append(self.predict(context, folder, "test"))
                        calibration_predictions.append(self.predict(context, folder, "calibration"))

                    data = context["data"]
                    y_test = data.y_raw[test_ids]
                    y_calibration = data.y_raw[calibration_ids]
                    raw_lower = np.minimum(test_predictions[1], test_predictions[2])
                    raw_upper = np.maximum(test_predictions[1], test_predictions[2])
                    cal_lower = np.minimum(calibration_predictions[1], calibration_predictions[2])
                    cal_upper = np.maximum(calibration_predictions[1], calibration_predictions[2])
                    calibration_scores = np.maximum.reduce([
                        cal_lower - y_calibration, y_calibration - cal_upper,
                        np.zeros(len(calibration_ids)),
                    ])
                    test_scores = np.maximum.reduce([
                        raw_lower - y_test, y_test - raw_upper, np.zeros(len(test_ids)),
                    ])
                    expansion, rank = conformal_quantile(calibration_scores, self.config["alpha"])
                    rolling = rolling_expansions(calibration_scores, test_scores,
                                                 self.config["rolling_window"], self.config["alpha"])
                    base.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        "window_index": test_ids,
                        "raw_target_row": data.target_row_indices[test_ids],
                        "time": data.timestamps[test_ids].astype(str),
                        "y_true": y_test, "prediction": test_predictions[0],
                        "raw_q025": test_predictions[1], "raw_q975": test_predictions[2],
                        "static_lower": raw_lower - expansion,
                        "static_upper": raw_upper + expansion,
                        "rolling_q": rolling,
                        "rolling_lower": raw_lower - rolling,
                        "rolling_upper": raw_upper + rolling,
                    }).to_csv(base / "test_predictions.csv", index=False)
                    pd.DataFrame({
                        "window_index": calibration_ids,
                        "raw_target_row": data.target_row_indices[calibration_ids],
                        "time": data.timestamps[calibration_ids].astype(str),
                        "y_true": y_calibration,
                        "q025": calibration_predictions[1],
                        "q975": calibration_predictions[2],
                        "score": calibration_scores,
                    }).to_csv(base / "calibration_predictions.csv", index=False)
                    result = {
                        "task": task["name"], "site": task["site"], "role": task["role"],
                        "method": method, "seed": seed,
                        "selected_config": config,
                        "selection_lock_sha256": lock_sha256,
                        "counts": {**context["counts"], "n_calibration": len(calibration_ids),
                                   "n_test": len(test_ids)},
                        "point": point_metrics(y_test, test_predictions[0]),
                        "raw_interval": interval_metrics(y_test, raw_lower, raw_upper,
                                                         self.config["alpha"]),
                        "static_interval": interval_metrics(y_test, raw_lower - expansion,
                                                            raw_upper + expansion,
                                                            self.config["alpha"]),
                        "rolling_interval": interval_metrics(y_test, raw_lower - rolling,
                                                             raw_upper + rolling,
                                                             self.config["alpha"]),
                        "calibration": {"n": len(calibration_ids), "rank": rank,
                                        "static_q": expansion,
                                        "rolling_window": self.config["rolling_window"]},
                        "fits": fits,
                    }
                    save_json(result_path, result)
                    results.append(result)
                    print(json.dumps({
                        "phase": "evaluation", "task": task["name"], "method": method,
                        "seed": seed, "rmse": result["point"]["rmse"],
                        "static_picp": result["static_interval"]["picp"],
                    }), flush=True)
            self.context_cache.clear()
            self.feature_cache.clear()
        save_json(self.output / "results.json", results)
        return results


def _experiment_hashes() -> tuple[dict, dict, str]:
    source_paths = [
        Path(__file__), ROOT / "review/selection_data_v2.py",
        ROOT / "review/selection_data.py", ROOT / "review/selection_sampling.py",
        ROOT / "review/temporal_data.py", ROOT / "review/convex_refinement.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    data_paths = sorted((ROOT / "data/wind_farm_data/data_original/wind_farms").glob("*.xlsx"))
    data_hashes = {str(path.relative_to(ROOT)): digest(path) for path in data_paths}
    reference = ROOT / "review/replied_comment_runs/chronological_v1/manifest.json"
    return source_hashes, data_hashes, digest(reference)


def _initialize(output: Path, config: dict, resume: bool) -> dict:
    source_hashes, data_hashes, reference_hash = _experiment_hashes()
    versions = environment_versions()
    if config.get("environment_versions") != versions:
        raise ValueError("Protocol environment versions differ from the current environment.")
    manifest_path = output / "manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError("Existing selection-v2 output requires explicit --resume.")
        if not manifest_path.exists():
            raise ValueError("Existing output directory lacks manifest.json.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("environment_versions") != versions:
            raise ValueError("Frozen selection-v2 environment versions changed.")
        if manifest.get("protocol") != config:
            raise ValueError("Frozen selection-v2 protocol changed.")
        if manifest.get("source_hashes") != source_hashes:
            raise ValueError("Frozen selection-v2 source hashes changed.")
        if manifest.get("data_hashes") != data_hashes:
            raise ValueError("Frozen source workbooks changed.")
        if manifest.get("reference_manifest_sha256") != reference_hash:
            raise ValueError("Frozen chronological reference manifest changed.")
        if digest(output / "protocol.json") != manifest["protocol_sha256"]:
            raise ValueError("protocol.json changed after freezing.")
        return manifest

    output.mkdir(parents=True)
    save_json(output / "protocol.json", config)
    manifest = {
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": config,
        "protocol_sha256": digest(output / "protocol.json"),
        "source_hashes": source_hashes,
        "data_hashes": data_hashes,
        "reference_manifest_sha256": reference_hash,
        "environment_versions": versions,
    }
    save_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "review/selection_runs/selection_v2")
    parser.add_argument("--phase", choices=("base", "regularizers", "joint", "lock",
                                             "select", "evaluate"), default="select")
    parser.add_argument("--task", action="append",
                        choices=tuple([f"within_site{site}" for site in range(1, 7)] +
                                      [f"cross_site{site}" for site in range(1, 7)]),
                        help="Run only this task; repeat as needed. Intended for disjoint parallel workers.")
    parser.add_argument("--method", action="append", choices=METHODS,
                        help="Run only this method; repeat as needed.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Alias for --phase evaluate; requires a pre-existing valid lock.")
    parser.add_argument("--freeze-only", action="store_true",
                        help="Write only protocol/manifest hashes; do not load data or fit a model.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        if args.phase != "select":
            parser.error("Use either --evaluate or an explicit --phase, not both.")
        args.phase = "evaluate"

    if args.phase == "evaluate" and (not args.resume or not args.output.exists()):
        parser.error("Evaluation requires a later --phase evaluate --resume call with an existing selection lock.")

    config = protocol()
    task_names = set(args.task) if args.task else None
    method_names = set(args.method) if args.method else None
    worker_mode = task_names is not None or method_names is not None
    if worker_mode and args.phase not in ("base", "regularizers", "joint"):
        parser.error("--task/--method filters are allowed only for base, regularizers, or joint; the full lock and evaluation are coordinator operations.")
    if worker_mode and not args.output.exists():
        parser.error("Initialize the shared output once with --freeze-only before starting filtered workers.")
    manifest = _initialize(args.output, config, args.resume)
    if args.freeze_only:
        return

    worker_record = None
    if worker_mode:
        identity = json.dumps({"phase": args.phase, "tasks": sorted(task_names or []),
                               "methods": sorted(method_names or [])}, sort_keys=True)
        worker_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
        worker_path = (args.output / "workers" /
                       f"{args.phase}_{worker_id}_pid{os.getpid()}.json")
        worker_record = {
            "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
            "phase": args.phase, "tasks": sorted(task_names or []),
            "methods": sorted(method_names or []), "pid": os.getpid(),
        }
        save_json(worker_path, worker_record)
    else:
        manifest["status"] = "running"
        manifest["active_phase"] = args.phase
        manifest.pop("error", None)
        save_json(args.output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=config["threads"]):
            study = Study(args.output, config, write_global_provenance=not worker_mode)
            if args.phase in ("base", "select"):
                study.tune_base(task_names, method_names)
                if not worker_mode:
                    manifest["last_stable_status"] = "base_complete"
                    save_json(args.output / "manifest.json", manifest)
            if args.phase in ("regularizers", "select"):
                study.tune_regularizers(task_names, method_names)
                if not worker_mode:
                    manifest["last_stable_status"] = "regularizers_complete"
                    save_json(args.output / "manifest.json", manifest)
            if args.phase in ("joint", "select"):
                study.tune_joint(task_names, method_names)
                if not worker_mode:
                    manifest["last_stable_status"] = "selection_complete_unlocked"
                    save_json(args.output / "manifest.json", manifest)
            if args.phase in ("lock", "select"):
                study.create_selection_lock(manifest)
                manifest["last_stable_status"] = "selection_locked"
                save_json(args.output / "manifest.json", manifest)
            if args.phase == "evaluate":
                study.evaluate(manifest)
                manifest["last_stable_status"] = "complete"
        if worker_mode:
            worker_record["status"] = "complete"
        else:
            manifest["status"] = manifest["last_stable_status"]
    except Exception as error:
        if worker_mode:
            worker_record["status"] = "failed"
            worker_record["error"] = repr(error)
        else:
            manifest["status"] = "failed"
            manifest["error"] = repr(error)
        raise
    finally:
        if worker_mode:
            worker_record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(worker_path, worker_record)
        else:
            manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
