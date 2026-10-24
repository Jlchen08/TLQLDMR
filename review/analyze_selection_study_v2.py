#!/usr/bin/env python3
"""Predeclared summaries and paired inference for selection-v2.

This analysis has a deliberately separate freeze and analysis step.  Run
``--freeze-only`` before held-out evaluation, then use ``--resume`` after the
selection-v2 runner has produced a valid selection lock and all locked
predictions.  The analysis never opens Q2 or Q3--Q4 output until it has
validated that lock against the frozen protocol, source files, data files,
and every locked selection record.

The six within-farm normal-to-extreme tasks form the primary scope.  The six
leave-one-farm-out tasks are reported in full as a secondary transfer-stress
scope.  No task, site, method, or evaluation seed is filtered.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "review/selection_runs/selection_v2"
DEFAULT_OUTPUT = ROOT / "review/selection_runs/selection_v2_analysis"
METHODS = ("Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR",
           "TargetSVQR", "HistGBR")
COMPARATORS = METHODS[1:]
KERNEL_METHODS = METHODS[:-1]
ACTIVE_REGULARIZERS = {
    "Full": ("lambda1", "lambda2"),
    "NoVariance": ("lambda2",),
    "NoDiscrepancy": ("lambda1",),
    "WeightedSVQR": (),
    "TargetSVQR": (),
}
CAPACITY = {1: 99.0, 2: 200.0, 3: 99.0, 4: 66.0, 5: 36.0, 6: 96.0}
SCOPES = {
    "within": tuple(f"within_site{site}" for site in range(1, 7)),
    "cross": tuple(f"cross_site{site}" for site in range(1, 7)),
}


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    """Atomically write strict JSON (NaN and Infinity are rejected)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def analysis_protocol() -> dict:
    return {
        "version": "selection_v2_analysis_2",
        "required_selection_lock_version": "selection_v2_lock_2",
        "reporting_rule": (
            "retain all 12 tasks x 6 methods and every predeclared method-specific "
            "evaluation seed; kernel methods use 0,1,2 and deterministic HistGBR uses 0"
        ),
        "primary_scope": {
            "name": "within",
            "tasks": list(SCOPES["within"]),
            "role": "six within-farm normal-to-extreme breadth tasks",
        },
        "secondary_scope": {
            "name": "cross",
            "tasks": list(SCOPES["cross"]),
            "role": "six leave-one-farm-out transfer-stress tasks",
        },
        "primary_endpoint": (
            "For each task and method, calculate RMSE separately for each locked "
            "evaluation seed, average the seed RMSEs, divide by the target-farm "
            "capacity, and then average the six task values with equal site weight."
        ),
        "rank_endpoint": (
            "Average of within-task ranks of seed-mean RMSE; lower is better and "
            "exact ties receive their average rank."
        ),
        "full_comparisons": {
            "comparators": list(COMPARATORS),
            "observation_loss": (
                "squared error in MW^2 averaged over each method's predeclared "
                "algorithmic evaluation seeds within each forecast timestamp"
            ),
            "effect": "Full minus comparator; negative values favor Full",
            "block": (
                "non-overlapping Monday--Sunday seven-calendar-day blocks, using a "
                "fixed 1970-01-05 Monday origin"
            ),
            "bootstrap": (
                "10,000 equal-probability block resamples within a task; each draw "
                "is a ratio of sampled block loss sums to sampled block counts; "
                "percentile 95% interval"
            ),
            "sign_flip": (
                "99,999 Monte Carlo two-sided paired block sign flips with the "
                "plus-one correction"
            ),
            "multiplicity": (
                "Holm adjustment in two separate 30-test families: 6 tasks x 5 "
                "comparators for within and for cross"
            ),
        },
        "first_tier": {
            "name": "conservative all-pairs non-rejection set",
            "procedure": (
                "Within each scope, compare all 15 method pairs by a two-sided "
                "seven-day block-swap randomization test of the site-equal macro "
                "capacity-normalized seed-mean RMSE difference. Apply Holm to all "
                "15 comparisons. The set contains the empirical endpoint winner "
                "and every method not shown worse than that winner at adjusted "
                "alpha 0.05."
            ),
            "interpretation": (
                "Membership means that inferiority to the empirical winner was not "
                "established under this block assumption; it is not evidence of "
                "equivalence and is not a formal Hansen model-confidence set."
            ),
        },
        "bootstrap_replicates": 10000,
        "sign_flip_replicates": 99999,
        "random_seed": 20261025,
        "alpha": 0.05,
        "heldout_guard": (
            "Do not read results.json, result.json, calibration predictions, or "
            "test predictions before selection_lock.json passes validation, including "
            "18 effective unique queries in each stage and 54 across stages for every "
            "one of the 72 task-method selections."
        ),
    }


def _stable_seed(base: int, *parts: object) -> int:
    text = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(values), dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("Holm adjustment requires a nonempty finite vector.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must lie in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _safe_locked_path(reference: Path, relative: str) -> Path:
    reference = reference.resolve()
    candidate = (reference / relative).resolve()
    if candidate != reference and reference not in candidate.parents:
        raise ValueError(f"Locked path escapes the experiment directory: {relative}")
    return candidate


def effective_model_signature(method: str, candidate: dict) -> tuple:
    """Independent representation of the parameters that affect a fitted model.

    This mirrors the frozen runner rule without importing executable runner
    code. Provenance and inactive coordinates do not create extra queries.
    """
    if method not in METHODS or not isinstance(candidate, dict):
        raise ValueError(f"Invalid method or candidate for signature: {method}")
    candidate = dict(candidate)
    if method == "HistGBR":
        keys = ("learning_rate", "max_leaf_nodes", "min_samples_leaf",
                "l2_regularization", "max_iter", "target_fraction")
    else:
        if method == "TargetSVQR":
            candidate.pop("target_fraction", None)
        defaults = {
            "Full": {"lambda1": 1.0, "lambda2": 1.0},
            "NoVariance": {"lambda1": 0.0, "lambda2": 1.0},
            "NoDiscrepancy": {"lambda1": 1.0, "lambda2": 0.0},
            "WeightedSVQR": {"lambda1": 0.0, "lambda2": 0.0},
            "TargetSVQR": {"lambda1": 0.0, "lambda2": 0.0},
        }[method]
        candidate = {**defaults, **candidate}
        for name in ("lambda1", "lambda2"):
            if name not in ACTIVE_REGULARIZERS[method]:
                candidate[name] = 0.0
        keys = ("gamma", "total_loss", "rank")
        if method != "TargetSVQR":
            keys += ("target_fraction",)
        keys += ACTIVE_REGULARIZERS[method]
    missing = [key for key in keys if key not in candidate]
    if missing:
        raise ValueError(f"Candidate for {method} lacks active parameter {missing[0]}.")
    return tuple((key, candidate[key]) for key in keys)


def validate_task_method_query_budget(reference: Path, task: str,
                                      method: str) -> set[tuple]:
    """Validate 18 new effective queries per stage and 54 across stages."""
    all_signatures: list[tuple] = []
    for stage in ("base", "middle", "joint"):
        path = Path(reference) / task / method / f"{stage}_selection.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("task"), record.get("method"), record.get("stage")) != (
                task, method, stage):
            raise ValueError(f"Selection-stage identity mismatch: {path}")
        candidates = record.get("candidates")
        if (record.get("candidate_count") != 18 or
                not isinstance(candidates, list) or len(candidates) != 18 or
                record.get("queried_effective_unique_count") != 18):
            raise ValueError(f"Selection stage does not declare 18 unique queries: {path}")
        try:
            signatures = [effective_model_signature(method, row["config"])
                          for row in candidates]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Malformed candidate inventory: {path}") from error
        if len(set(signatures)) != 18:
            raise ValueError(f"Selection stage has repeated effective models: {path}")
        all_signatures.extend(signatures)
    unique = set(all_signatures)
    if len(all_signatures) != 54 or len(unique) != 54:
        raise ValueError(
            f"Three stages do not contain 54 globally distinct effective queries: "
            f"{task}/{method}")
    return unique


def validate_selection_lock(reference: Path) -> tuple[dict, dict, dict, str]:
    """Validate selection metadata without opening any held-out result file."""
    reference = Path(reference)
    required = [reference / "selection_lock.json", reference / "protocol.json",
                reference / "manifest.json", reference / "data_provenance.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Locked selection metadata missing: {missing[0]}")

    # protocol.json and the lock contain selection information only.  In
    # particular, results.json is intentionally not opened in this function.
    lock = json.loads((reference / "selection_lock.json").read_text(encoding="utf-8"))
    protocol = json.loads((reference / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    if lock.get("version") != "selection_v2_lock_2":
        raise ValueError("Unexpected selection-lock version.")
    if lock.get("protocol_sha256") != digest(reference / "protocol.json"):
        raise ValueError("Selection-lock protocol hash mismatch.")
    if manifest.get("protocol_sha256") != lock["protocol_sha256"]:
        raise ValueError("Manifest and selection lock name different protocols.")
    if manifest.get("protocol") != protocol:
        raise ValueError("Manifest protocol differs from protocol.json.")
    if manifest.get("source_hashes") != lock.get("source_hashes"):
        raise ValueError("Manifest and selection-lock source hashes differ.")
    if manifest.get("data_hashes") != lock.get("data_hashes"):
        raise ValueError("Manifest and selection-lock data hashes differ.")
    if not lock.get("environment_versions"):
        raise ValueError("Selection lock lacks frozen environment versions.")
    if (manifest.get("environment_versions") != lock["environment_versions"] or
            protocol.get("environment_versions") != lock["environment_versions"]):
        raise ValueError("Protocol, manifest, and selection-lock environments differ.")
    if digest(reference / "data_provenance.json") != lock.get("data_provenance_sha256"):
        raise ValueError("Locked data provenance changed.")

    for collection_name in ("source_hashes", "data_hashes"):
        collection = lock.get(collection_name)
        if not isinstance(collection, dict) or not collection:
            raise ValueError(f"Selection lock lacks {collection_name}.")
        for relative, expected_hash in collection.items():
            path = _safe_locked_path(ROOT, relative)
            if not path.is_file() or digest(path) != expected_hash:
                raise ValueError(f"Frozen input changed: {relative}")

    methods = tuple(protocol.get("methods", []))
    tasks = tuple(row.get("name") for row in protocol.get("tasks", []))
    if methods != METHODS:
        raise ValueError("The locked protocol does not contain the six predeclared methods.")
    expected_tasks = SCOPES["within"] + SCOPES["cross"]
    if tasks != expected_tasks:
        raise ValueError("The locked protocol does not contain the 12 predeclared tasks.")
    expected_seed_map = {method: [0, 1, 2] for method in METHODS[:-1]}
    expected_seed_map["HistGBR"] = [0]
    if protocol.get("evaluation_seeds_by_method") != expected_seed_map:
        raise ValueError("The locked protocol has an unexpected method-specific seed map.")
    expected_queries = len(expected_tasks) * len(METHODS) * 54
    if lock.get("total_configuration_queries") != expected_queries:
        raise ValueError("Selection lock does not contain the predeclared 12 x 6 x 54 queries.")
    if (lock.get("validation_query_budget") != protocol.get("validation_query_budget") or
            protocol.get("validation_query_budget", {}).get("total_per_task_method") != 54):
        raise ValueError("Selection lock and protocol disagree on the 54-query budget.")

    selection_files = lock.get("selection_files")
    if not isinstance(selection_files, dict):
        raise ValueError("Selection lock lacks its selection-file inventory.")
    expected_selection_files = {
        f"{task}/{method}/{stage}_selection.json"
        for task in expected_tasks for method in METHODS
        for stage in ("base", "middle", "regularized", "joint", "final")
    }
    if set(selection_files) != expected_selection_files:
        raise ValueError("Selection lock has an incomplete or unexpected selection inventory.")
    for relative, expected_hash in selection_files.items():
        path = _safe_locked_path(reference, relative)
        if not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f"Locked selection record changed: {relative}")

    # A hash-consistent lock can still encode a nominal budget whose entries
    # collapse to the same fitted model. Recompute the effective signatures
    # independently before treating the lock as permission to read held-out
    # files.
    for task in expected_tasks:
        for method in METHODS:
            validate_task_method_query_budget(reference, task, method)

    final = lock.get("final_selections")
    if not isinstance(final, list) or len(final) != 72:
        raise ValueError("Selection lock must contain exactly 12 x 6 final selections.")
    keys = [(row.get("task"), row.get("method")) for row in final]
    expected_keys = [(task, method) for task in expected_tasks for method in METHODS]
    if sorted(keys) != sorted(expected_keys) or len(set(keys)) != len(keys):
        raise ValueError("Selection lock has missing or duplicate task-method records.")
    final_map = {(row["task"], row["method"]): row for row in final}
    for task, method in expected_keys:
        final_path = reference / task / method / "final_selection.json"
        if json.loads(final_path.read_text(encoding="utf-8")) != final_map[(task, method)]:
            raise ValueError(f"Locked final selection differs from its record: {final_path}")
    if lock.get("heldout_policy") != (
            "no Q2 calibration or Q3-Q4 prediction was calculated before this lock"):
        raise ValueError("Selection lock does not state the required held-out policy.")
    return lock, protocol, manifest, digest(reference / "selection_lock.json")


def load_locked_predictions(reference: Path) -> tuple[dict, dict, dict]:
    """Load all locked results and predictions after lock validation succeeds."""
    reference = Path(reference)
    lock, protocol, manifest, lock_hash = validate_selection_lock(reference)

    # This is the first held-out file opened by the analysis.
    results_path = reference / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError("Locked evaluation is incomplete: results.json is missing.")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(results, list):
        raise ValueError("results.json must be a list.")
    seed_map = protocol["evaluation_seeds_by_method"]
    expected_keys = [
        (task, method, seed)
        for task in SCOPES["within"] + SCOPES["cross"]
        for method in METHODS for seed in seed_map[method]
    ]
    result_map = {}
    for row in results:
        key = (row.get("task"), row.get("method"), row.get("seed"))
        if key in result_map:
            raise ValueError(f"Duplicate locked result: {key}")
        result_map[key] = row
    expected_result_count = 12 * sum(len(seed_map[method]) for method in METHODS)
    if set(result_map) != set(expected_keys) or len(result_map) != expected_result_count:
        raise ValueError("Locked results do not match all method-specific evaluation seeds.")

    frames: dict[tuple[str, str, int], pd.DataFrame] = {}
    input_hashes = {"selection_lock.json": lock_hash,
                    "results.json": digest(results_path)}
    locked_selection = {
        (row["task"], row["method"]): row["selected_config"]
        for row in lock["final_selections"]
    }
    canonical: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key in expected_keys:
        task, method, seed = key
        folder = reference / task / method / "evaluation" / f"seed{seed}"
        result_path = folder / "result.json"
        prediction_path = folder / "test_predictions.csv"
        if not result_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"Missing locked evaluation output under {folder}.")
        disk_result = json.loads(result_path.read_text(encoding="utf-8"))
        if disk_result != result_map[key]:
            raise ValueError(f"results.json differs from {result_path}.")
        if disk_result.get("selection_lock_sha256") != lock_hash:
            raise ValueError(f"Evaluation used a different selection lock: {result_path}")
        if (disk_result.get("task"), disk_result.get("method"), disk_result.get("seed")) != key:
            raise ValueError(f"Evaluation identity mismatch: {result_path}")
        if disk_result.get("selected_config") != locked_selection[(task, method)]:
            raise ValueError(f"Evaluation configuration differs from the lock: {result_path}")

        frame = pd.read_csv(prediction_path)
        required_columns = {"window_index", "time", "y_true", "prediction"}
        if not required_columns.issubset(frame.columns) or frame.empty:
            raise ValueError(f"Malformed test prediction file: {prediction_path}")
        if frame["window_index"].duplicated().any():
            raise ValueError(f"Duplicate test window identifiers: {prediction_path}")
        ids = frame["window_index"].to_numpy()
        times = pd.to_datetime(frame["time"], errors="raise").to_numpy()
        y = frame["y_true"].to_numpy(float)
        prediction = frame["prediction"].to_numpy(float)
        if np.any(~np.isfinite(y)) or np.any(~np.isfinite(prediction)):
            raise ValueError(f"Non-finite locked prediction: {prediction_path}")
        if task not in canonical:
            canonical[task] = (ids, times, y)
        else:
            expected_ids, expected_times, expected_y = canonical[task]
            if not (np.array_equal(ids, expected_ids) and
                    np.array_equal(times, expected_times) and
                    np.array_equal(y, expected_y)):
                raise ValueError(f"Prediction rows are not aligned within task {task}.")
        calculated_rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
        recorded_rmse = float(disk_result["point"]["rmse"])
        if int(disk_result["point"]["n"]) != len(frame):
            raise ValueError(f"Recorded test count does not match predictions: {prediction_path}")
        if not math.isclose(calculated_rmse, recorded_rmse, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"Recorded RMSE does not match predictions: {prediction_path}")
        frames[key] = frame.loc[:, ["window_index", "time", "y_true", "prediction"]].copy()
        input_hashes[str(result_path.relative_to(reference))] = digest(result_path)
        input_hashes[str(prediction_path.relative_to(reference))] = digest(prediction_path)

    return {
        "lock": lock, "protocol": protocol, "manifest": manifest,
        "lock_sha256": lock_hash, "input_hashes": input_hashes,
        "results": results,
    }, frames, result_map


def _site_from_task(task: str) -> int:
    try:
        site = int(task.rsplit("site", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Unrecognized task name: {task}") from error
    if site not in CAPACITY:
        raise ValueError(f"Unrecognized site in task: {task}")
    return site


def _calendar_blocks(times: Iterable[object]) -> np.ndarray:
    values = pd.DatetimeIndex(pd.to_datetime(list(times), errors="raise"))
    if values.hasnans:
        raise ValueError("A test timestamp is missing.")
    if values.tz is not None:
        values = values.tz_convert("UTC").tz_localize(None)
    day = values.to_numpy(dtype="datetime64[D]")
    origin = np.datetime64("1970-01-05", "D")  # Monday
    return ((day - origin).astype(np.int64) // 7).astype(np.int64)


def _error_matrix(frames: dict, task: str, method: str) -> np.ndarray:
    values = []
    seeds = sorted(seed for candidate_task, candidate_method, seed in frames
                   if candidate_task == task and candidate_method == method)
    if not seeds:
        raise ValueError(f"No evaluation seeds found for {task}/{method}.")
    for seed in seeds:
        frame = frames[(task, method, seed)]
        values.append((frame["prediction"].to_numpy(float) -
                       frame["y_true"].to_numpy(float)) ** 2)
    return np.asarray(values)


def summarize_point_results(frames: dict) -> tuple[list[dict], list[dict]]:
    rows = []
    for scope, tasks in SCOPES.items():
        for task in tasks:
            site = _site_from_task(task)
            capacity = CAPACITY[site]
            task_rows = []
            for method in METHODS:
                errors = _error_matrix(frames, task, method)
                seed_rmse = np.sqrt(errors.mean(axis=1))
                row = {
                    "scope": scope,
                    "task": task,
                    "site": site,
                    "method": method,
                    "n_test": int(errors.shape[1]),
                    "evaluation_seeds": int(errors.shape[0]),
                    "seed_rmse_mw": [float(value) for value in seed_rmse],
                    "seed_mean_rmse_mw": float(seed_rmse.mean()),
                    "capacity_mw": capacity,
                    "seed_mean_capacity_normalized_rmse": float(seed_rmse.mean() / capacity),
                    "seed_mean_capacity_normalized_rmse_percent": float(
                        100 * seed_rmse.mean() / capacity),
                }
                task_rows.append(row)
            ranks = pd.Series([row["seed_mean_rmse_mw"] for row in task_rows]).rank(
                method="average", ascending=True).to_numpy()
            for row, rank in zip(task_rows, ranks):
                row["task_rank"] = float(rank)
                rows.append(row)

    summary = []
    for scope in SCOPES:
        for method in METHODS:
            selected = [row for row in rows
                        if row["scope"] == scope and row["method"] == method]
            summary.append({
                "scope": scope,
                "role": ("primary" if scope == "within" else "secondary_stress"),
                "method": method,
                "tasks": len(selected),
                "site_equal_macro_capacity_normalized_rmse": float(np.mean([
                    row["seed_mean_capacity_normalized_rmse"] for row in selected])),
                "site_equal_macro_capacity_normalized_rmse_percent": float(np.mean([
                    row["seed_mean_capacity_normalized_rmse_percent"] for row in selected])),
                "average_task_rank": float(np.mean([row["task_rank"] for row in selected])),
                "task_wins_including_ties": int(sum(row["task_rank"] == 1 for row in selected)),
            })
    return rows, summary


def _block_sums(difference: np.ndarray, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    levels = np.unique(block)
    sums = np.asarray([difference[block == level].sum() for level in levels], dtype=float)
    counts = np.asarray([np.sum(block == level) for level in levels], dtype=int)
    if not len(sums) or np.any(counts <= 0):
        raise ValueError("Temporal blocking produced no usable blocks.")
    return sums, counts


def block_bootstrap_interval(block_sums: np.ndarray, block_counts: np.ndarray,
                             replicates: int, seed: int,
                             alpha: float = .05) -> tuple[float, float]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(block_sums), size=(replicates, len(block_sums)))
    values = block_sums[draws].sum(axis=1) / block_counts[draws].sum(axis=1)
    return (float(np.quantile(values, alpha / 2)),
            float(np.quantile(values, 1 - alpha / 2)))


def block_sign_flip_p(block_sums: np.ndarray, block_counts: np.ndarray,
                      replicates: int, seed: int, *, two_sided: bool = True) -> float:
    if replicates <= 0:
        raise ValueError("Sign-flip replicates must be positive.")
    observed = float(block_sums.sum() / block_counts.sum())
    rng = np.random.default_rng(seed)
    exceeded = 0
    remaining = replicates
    tolerance = 1e-14 * max(1., abs(observed))
    while remaining:
        count = min(10000, remaining)
        signs = rng.choice(np.array([-1., 1.]), size=(count, len(block_sums)))
        permuted = (signs @ block_sums) / block_counts.sum()
        if two_sided:
            exceeded += int(np.sum(np.abs(permuted) >= abs(observed) - tolerance))
        else:
            exceeded += int(np.sum(permuted >= observed - tolerance))
        remaining -= count
    return float((1 + exceeded) / (replicates + 1))


def full_pairwise_inference(frames: dict, *, bootstrap_replicates: int,
                            sign_flip_replicates: int, random_seed: int,
                            alpha: float = .05) -> list[dict]:
    reports = []
    for scope, tasks in SCOPES.items():
        for task in tasks:
            site = _site_from_task(task)
            full = _error_matrix(frames, task, "Full").mean(axis=0)
            reference_frame = frames[(task, "Full", 0)]
            blocks = _calendar_blocks(reference_frame["time"])
            for comparator in COMPARATORS:
                other = _error_matrix(frames, task, comparator).mean(axis=0)
                difference = full - other
                sums, counts = _block_sums(difference, blocks)
                key = (scope, task, comparator)
                low, high = block_bootstrap_interval(
                    sums, counts, bootstrap_replicates,
                    _stable_seed(random_seed, "bootstrap", *key), alpha)
                pvalue = block_sign_flip_p(
                    sums, counts, sign_flip_replicates,
                    _stable_seed(random_seed, "sign_flip", *key))
                reports.append({
                    "scope": scope,
                    "role": ("primary" if scope == "within" else "secondary_stress"),
                    "holm_family": f"{scope}_30_full_comparisons",
                    "task": task,
                    "site": site,
                    "comparator": comparator,
                    "n_test": int(len(difference)),
                    "full_evaluation_seeds_averaged_within_timestamp": int(
                        _error_matrix(frames, task, "Full").shape[0]),
                    "comparator_evaluation_seeds_averaged_within_timestamp": int(
                        _error_matrix(frames, task, comparator).shape[0]),
                    "calendar_blocks": int(len(sums)),
                    "block_size_min": int(counts.min()),
                    "block_size_max": int(counts.max()),
                    "full_minus_comparator_mse_mw2": float(difference.mean()),
                    "full_minus_comparator_normalized_mse": float(
                        difference.mean() / CAPACITY[site] ** 2),
                    "time_block_bootstrap_95_ci_mw2": [low, high],
                    "block_sign_flip_p_unadjusted": pvalue,
                })
    for scope in SCOPES:
        subset = [row for row in reports if row["scope"] == scope]
        if len(subset) != 30:
            raise AssertionError(f"Expected 30 Full comparisons in {scope}.")
        adjusted = holm_adjust(row["block_sign_flip_p_unadjusted"] for row in subset)
        for row, value in zip(subset, adjusted):
            row["block_sign_flip_p_holm_within_scope_30"] = float(value)
    return reports


def _scope_block_swap_p(frames: dict, tasks: tuple[str, ...], method_a: str,
                        method_b: str, replicates: int, seed: int) -> tuple[float, float]:
    """Randomization test for the exact macro seed-mean normalized-RMSE endpoint."""
    if replicates <= 0:
        raise ValueError("Block-swap replicates must be positive.")
    pieces = []
    observed_sites = []
    total_blocks = 0
    for task in tasks:
        site = _site_from_task(task)
        a = _error_matrix(frames, task, method_a)
        b = _error_matrix(frames, task, method_b)
        block = _calendar_blocks(frames[(task, method_a, 0)]["time"])
        levels = np.unique(block)
        midpoint_total = .5 * (a.sum(axis=1) + b.sum(axis=1))
        half_differences = np.asarray([
            .5 * (a[:, block == level].sum(axis=1) -
                  b[:, block == level].sum(axis=1))
            for level in levels
        ])
        pieces.append({
            "start": total_blocks, "stop": total_blocks + len(levels),
            "midpoint": midpoint_total, "half_difference": half_differences,
            "n": a.shape[1], "capacity": CAPACITY[site],
        })
        total_blocks += len(levels)
        observed_sites.append(float(
            np.sqrt(a.mean(axis=1)).mean() / CAPACITY[site] -
            np.sqrt(b.mean(axis=1)).mean() / CAPACITY[site]))
    observed = float(np.mean(observed_sites))

    rng = np.random.default_rng(seed)
    exceeded = 0
    remaining = replicates
    tolerance = 1e-14 * max(1., abs(observed))
    while remaining:
        count = min(5000, remaining)
        signs = rng.choice(np.array([-1., 1.]), size=(count, total_blocks))
        permuted = np.zeros(count)
        for piece in pieces:
            delta = (signs[:, piece["start"]:piece["stop"]] @
                     piece["half_difference"])
            a_total = np.maximum(piece["midpoint"][None, :] + delta, 0.)
            b_total = np.maximum(piece["midpoint"][None, :] - delta, 0.)
            site_difference = (
                np.sqrt(a_total / piece["n"]).mean(axis=1) -
                np.sqrt(b_total / piece["n"]).mean(axis=1)
            ) / piece["capacity"]
            permuted += site_difference / len(pieces)
        exceeded += int(np.sum(np.abs(permuted) >= abs(observed) - tolerance))
        remaining -= count
    return observed, float((1 + exceeded) / (replicates + 1))


def conservative_first_tier(frames: dict, scope_summary: list[dict], *,
                            sign_flip_replicates: int, random_seed: int,
                            alpha: float = .05) -> tuple[list[dict], list[dict]]:
    pairwise, membership = [], []
    for scope, tasks in SCOPES.items():
        endpoint = {
            row["method"]: row["site_equal_macro_capacity_normalized_rmse"]
            for row in scope_summary if row["scope"] == scope
        }
        winner = min(METHODS, key=lambda method: (endpoint[method], METHODS.index(method)))
        scope_pairs = []
        for method_a, method_b in itertools.combinations(METHODS, 2):
            difference, pvalue = _scope_block_swap_p(
                frames, tasks, method_a, method_b, sign_flip_replicates,
                _stable_seed(random_seed, "first_tier", scope, method_a, method_b))
            scope_pairs.append({
                "scope": scope,
                "holm_family": f"{scope}_15_all_method_pairs",
                "method_a": method_a,
                "method_b": method_b,
                "macro_nrmse_a_minus_b": difference,
                "block_swap_p_unadjusted_two_sided": pvalue,
            })
        adjusted = holm_adjust(
            row["block_swap_p_unadjusted_two_sided"] for row in scope_pairs)
        for row, value in zip(scope_pairs, adjusted):
            row["block_swap_p_holm_all_15"] = float(value)
        pairwise.extend(scope_pairs)

        for method in METHODS:
            if method == winner:
                # A self-comparison has no p-value.  Store the conservative
                # numeric identity value so strict CSV has no missing cell.
                adjusted_vs_winner = 1.0
                excluded = False
            else:
                match = next(row for row in scope_pairs
                             if {row["method_a"], row["method_b"]} == {method, winner})
                adjusted_vs_winner = match["block_swap_p_holm_all_15"]
                excluded = bool(endpoint[method] > endpoint[winner] and
                                adjusted_vs_winner < alpha)
            membership.append({
                "scope": scope,
                "method": method,
                "empirical_endpoint_winner": winner,
                "macro_capacity_normalized_rmse": endpoint[method],
                "difference_from_winner": float(endpoint[method] - endpoint[winner]),
                "holm_p_vs_winner": adjusted_vs_winner,
                "in_conservative_nonrejection_set": not excluded,
                "interpretation": (
                    "membership is non-rejection of inferiority, not equivalence"
                    if not excluded else
                    "worse than the empirical winner under the predeclared adjusted test"
                ),
            })
    return pairwise, membership


def _write_csv(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    if frame.isna().any().any():
        raise ValueError(f"Refusing to write missing values to {path}.")
    frame.to_csv(path, index=False, float_format="%.12g")


def _write_latex(output: Path, point_rows: list[dict], scope_rows: list[dict],
                 inference_rows: list[dict], membership_rows: list[dict]) -> None:
    point = pd.DataFrame(point_rows).pivot(index=["scope", "task", "site"],
                                           columns="method",
                                           values="seed_mean_rmse_mw").reset_index()
    point = point[["scope", "task", "site", *METHODS]]
    (output / "task_seed_mean_rmse.tex").write_text(point.to_latex(
        index=False, escape=True, float_format=lambda value: f"{value:.3f}",
        caption=("Locked Q3--Q4 RMSE (MW), averaged over the three algorithmic "
                 "evaluation seeds. All tasks and methods are retained."),
        label="tab:selection-v2-task-rmse"), encoding="utf-8")

    scope = pd.DataFrame(scope_rows)[[
        "scope", "method", "site_equal_macro_capacity_normalized_rmse_percent",
        "average_task_rank", "task_wins_including_ties"]].rename(columns={
            "site_equal_macro_capacity_normalized_rmse_percent": "macro NRMSE (percent)",
            "average_task_rank": "average rank", "task_wins_including_ties": "wins"})
    (output / "scope_summary.tex").write_text(scope.to_latex(
        index=False, escape=True, float_format=lambda value: f"{value:.3f}",
        caption=("Site-equal capacity-normalized RMSE and task ranks for the "
                 "primary within-farm and secondary cross-farm scopes."),
        label="tab:selection-v2-scope-summary"), encoding="utf-8")

    inference = pd.DataFrame([{
        "scope": row["scope"], "task": row["task"],
        "comparator": row["comparator"],
        "Full minus comparator (MW2)": row["full_minus_comparator_mse_mw2"],
        "CI low": row["time_block_bootstrap_95_ci_mw2"][0],
        "CI high": row["time_block_bootstrap_95_ci_mw2"][1],
        "Holm p": row["block_sign_flip_p_holm_within_scope_30"],
    } for row in inference_rows])
    (output / "full_pairwise_inference.tex").write_text(inference.to_latex(
        index=False, escape=True, longtable=True,
        float_format=lambda value: f"{value:.4g}",
        caption=("Paired seven-calendar-day block inference for Full minus each "
                 "comparator. Negative loss differences favor Full; Holm families "
                 "are separate for within and cross scopes."),
        label="tab:selection-v2-full-inference"), encoding="utf-8")

    tier = pd.DataFrame(membership_rows)[[
        "scope", "method", "empirical_endpoint_winner", "difference_from_winner",
        "holm_p_vs_winner", "in_conservative_nonrejection_set"]]
    (output / "conservative_first_tier.tex").write_text(tier.to_latex(
        index=False, escape=True, float_format=lambda value: f"{value:.4g}",
        caption=("Conservative all-pairs non-rejection set. Membership does not "
                 "establish equivalence or constitute a formal model-confidence set."),
        label="tab:selection-v2-first-tier"), encoding="utf-8")


def analyse(reference: Path, output: Path, protocol: dict) -> dict:
    metadata, frames, _ = load_locked_predictions(reference)
    point_rows, scope_rows = summarize_point_results(frames)
    inference_rows = full_pairwise_inference(
        frames,
        bootstrap_replicates=protocol["bootstrap_replicates"],
        sign_flip_replicates=protocol["sign_flip_replicates"],
        random_seed=protocol["random_seed"], alpha=protocol["alpha"])
    tier_pairs, membership_rows = conservative_first_tier(
        frames, scope_rows,
        sign_flip_replicates=protocol["sign_flip_replicates"],
        random_seed=protocol["random_seed"], alpha=protocol["alpha"])

    if len(point_rows) != 72 or len(scope_rows) != 12 or len(inference_rows) != 60:
        raise AssertionError("The analysis failed its complete-reporting invariant.")
    if len(tier_pairs) != 30 or len(membership_rows) != 12:
        raise AssertionError("The first-tier analysis failed its complete-reporting invariant.")

    _write_csv(output / "task_method_seed_mean_rmse.csv", point_rows)
    _write_csv(output / "scope_summary.csv", scope_rows)
    inference_csv = [{**row,
                      "time_block_bootstrap_95_ci_low_mw2":
                          row["time_block_bootstrap_95_ci_mw2"][0],
                      "time_block_bootstrap_95_ci_high_mw2":
                          row["time_block_bootstrap_95_ci_mw2"][1]}
                     for row in inference_rows]
    for row in inference_csv:
        del row["time_block_bootstrap_95_ci_mw2"]
    _write_csv(output / "full_pairwise_inference.csv", inference_csv)
    _write_csv(output / "first_tier_all_pair_tests.csv", tier_pairs)
    _write_csv(output / "conservative_first_tier.csv", membership_rows)
    _write_latex(output, point_rows, scope_rows, inference_rows, membership_rows)

    result = {
        "version": protocol["version"],
        "selection_lock_sha256": metadata["lock_sha256"],
        "reporting_counts": {
            "tasks": 12, "methods": 6,
            "evaluation_seeds_by_method": metadata["protocol"][
                "evaluation_seeds_by_method"],
            "locked_result_records": len(metadata["results"]),
            "task_method_rows": len(point_rows),
            "full_comparisons": len(inference_rows),
        },
        "primary_within_summary": [row for row in scope_rows
                                   if row["scope"] == "within"],
        "secondary_cross_summary": [row for row in scope_rows
                                    if row["scope"] == "cross"],
        "task_method_seed_mean_rmse": point_rows,
        "full_pairwise_inference": inference_rows,
        "first_tier_all_pair_tests": tier_pairs,
        "conservative_first_tier": membership_rows,
        "first_tier_caveat": protocol["first_tier"]["interpretation"],
    }
    save_json(output / "selection_v2_analysis.json", result)
    return {"result": result, "input_hashes": metadata["input_hashes"]}


def _initialize(output: Path, reference: Path, *, resume: bool,
                freeze_only: bool) -> tuple[dict, dict]:
    protocol = analysis_protocol()
    protocol_path = output / "analysis_protocol.json"
    manifest_path = output / "analysis_manifest.json"
    source_hash = digest(Path(__file__))
    if output.exists():
        if not resume:
            raise FileExistsError("Existing analysis output requires --resume.")
        if not protocol_path.is_file() or not manifest_path.is_file():
            raise ValueError("Existing analysis directory lacks frozen metadata.")
        frozen_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if frozen_protocol != protocol or manifest.get("protocol") != protocol:
            raise ValueError("Frozen analysis protocol changed.")
        if manifest.get("protocol_sha256") != digest(protocol_path):
            raise ValueError("Frozen analysis protocol hash changed.")
        if manifest.get("analysis_source_sha256") != source_hash:
            raise ValueError("Analysis source changed after protocol freeze.")
        if manifest.get("selection_root") != str(reference.resolve()):
            raise ValueError("Frozen analysis names a different selection root.")
        return protocol, manifest

    if not freeze_only:
        raise RuntimeError(
            "Freeze the analysis before held-out results are read: rerun with --freeze-only.")
    output.mkdir(parents=True)
    save_json(protocol_path, protocol)
    manifest = {
        "status": "frozen_before_analysis",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_root": str(reference.resolve()),
        "protocol": protocol,
        "protocol_sha256": digest(protocol_path),
        "analysis_source_sha256": source_hash,
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    save_json(manifest_path, manifest)
    return protocol, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze-only", action="store_true",
                        help="Freeze analysis rules without reading any selection output.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol, manifest = _initialize(args.output, args.selection_root,
                                     resume=args.resume, freeze_only=args.freeze_only)
    if args.freeze_only:
        return
    manifest["status"] = "running"
    manifest["analysis_started_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.pop("error", None)
    save_json(args.output / "analysis_manifest.json", manifest)
    try:
        report = analyse(args.selection_root, args.output, protocol)
        manifest["selection_input_hashes"] = report["input_hashes"]
        artifacts = [
            "task_method_seed_mean_rmse.csv", "scope_summary.csv",
            "full_pairwise_inference.csv", "first_tier_all_pair_tests.csv",
            "conservative_first_tier.csv", "task_seed_mean_rmse.tex",
            "scope_summary.tex", "full_pairwise_inference.tex",
            "conservative_first_tier.tex", "selection_v2_analysis.json",
        ]
        manifest["artifact_hashes"] = {
            name: digest(args.output / name) for name in artifacts
        }
        manifest["status"] = "complete"
        manifest["reporting_counts"] = report["result"]["reporting_counts"]
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(args.output / "analysis_manifest.json", manifest)


if __name__ == "__main__":
    main()
