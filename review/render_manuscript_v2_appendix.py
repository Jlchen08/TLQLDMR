#!/usr/bin/env python3
"""Render auditable selection-v2 appendix tables after completed analysis.

This presentation-only program reads the immutable selection metadata and the
hash-verified analysis products.  Held-out result and prediction files are read
only as bytes to recompute the 386 analysis-input digests; their contents are
never parsed.  The secondary RMSE table is derived only from the completed
analysis CSV after its row-level agreement with the analysis JSON is verified.
"""
from __future__ import annotations

import argparse
import ast
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "review/selection_runs/selection_v2"
DEFAULT_ANALYSIS = ROOT / "review/selection_runs/selection_v2_analysis"
DEFAULT_OUTPUT = ROOT / "review/selection_runs/selection_v2_manuscript_appendix"

METHODS = (
    "Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR",
    "TargetSVQR", "HistGBR",
)
KERNEL_METHODS = METHODS[:-1]
METHOD_LABELS = {
    "Full": "TL-QLDMR",
    "NoVariance": "No variance",
    "NoDiscrepancy": "No discrepancy",
    "WeightedSVQR": "Weighted SVQR",
    "TargetSVQR": "Target SVQR",
    "HistGBR": "HistGBR",
}
SCOPES = ("within", "cross")
TASKS = {
    scope: tuple(f"{scope}_site{site}" for site in range(1, 7))
    for scope in SCOPES
}
STAGES = ("base", "middle", "joint")
ACTIVE_REGULARIZERS = {
    "Full": ("lambda1", "lambda2"),
    "NoVariance": ("lambda2",),
    "NoDiscrepancy": ("lambda1",),
    "WeightedSVQR": (),
    "TargetSVQR": (),
}
OUTPUT_FILES = (
    "selection_v2_selected_configs.tex",
    "selection_v2_secondary_cross_rmse.tex",
    "selection_v2_optimization_summary.tex",
    "selection_v2_optimization_summary.csv",
)
SELECTION_SOURCE_FILES = (
    "review/run_selection_study_v2.py",
    "review/selection_data_v2.py",
    "review/selection_data.py",
    "review/selection_sampling.py",
    "review/temporal_data.py",
    "review/convex_refinement.py",
)
SELECTION_DATA_FILES = tuple(
    "data/wind_farm_data/data_original/wind_farms/"
    f"Wind farm site {site} (Nominal capacity-{capacity}MW).xlsx"
    for site, capacity in ((1, 99), (2, 200), (3, 99), (4, 66), (5, 36), (6, 96))
)
ANALYSIS_ARTIFACT_FILES = (
    "task_method_seed_mean_rmse.csv",
    "scope_summary.csv",
    "full_pairwise_inference.csv",
    "first_tier_all_pair_tests.csv",
    "conservative_first_tier.csv",
    "task_seed_mean_rmse.tex",
    "scope_summary.tex",
    "full_pairwise_inference.tex",
    "conservative_first_tier.tex",
    "selection_v2_analysis.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _strict_int(value: object, label: str, *, minimum: int | None = None) -> int:
    """Accept JSON integers only; never silently truncate floats or booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def _strict_csv_int(value: object, label: str, *, minimum: int | None = None) -> int:
    text = str(value)
    if not text or (text[0] == "-" and not text[1:].isdigit()) or \
            (text[0] != "-" and not text.isdigit()):
        raise ValueError(f"{label} must be an integer.")
    result = int(text)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return result


def _effective_signature(method: str, config: object) -> tuple:
    """Canonical fitted-model signature, independent of provenance fields."""
    if method not in METHODS or not isinstance(config, dict):
        raise ValueError(f"Malformed candidate configuration for {method}.")
    candidate = dict(config)
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
    missing = [name for name in keys if name not in candidate]
    if missing:
        raise ValueError(f"Candidate for {method} lacks {missing[0]}.")
    if method == "HistGBR":
        for name in ("max_leaf_nodes", "min_samples_leaf", "max_iter"):
            _strict_int(candidate[name], f"{method} {name}", minimum=1)
    else:
        _strict_int(candidate["rank"], f"{method} rank", minimum=1)
    integer_names = {"rank", "max_leaf_nodes", "min_samples_leaf", "max_iter"}
    signature = []
    for name in keys:
        value = candidate[name]
        if name in integer_names:
            value = _strict_int(value, f"{method} {name}", minimum=1)
        else:
            value = _finite_number(value, f"{method} {name}")
        # Canonical numeric values prevent JSON strings such as "0.1" from
        # manufacturing an apparently distinct effective query.
        signature.append((name, value))
    values = dict(signature)
    if method == "HistGBR":
        if not (.01 <= values["learning_rate"] <= .3):
            raise ValueError("HistGBR learning rate lies outside the frozen search space.")
        if values["max_leaf_nodes"] not in {7, 15, 31, 63} or \
                values["min_samples_leaf"] not in {5, 10, 20, 40, 80} or \
                values["max_iter"] not in {100, 200, 400, 600} or \
                values["l2_regularization"] not in {0.0, 0.1, 1.0, 10.0, 100.0} or \
                not (.05 <= values["target_fraction"] <= .995):
            raise ValueError("HistGBR configuration lies outside the frozen search space.")
    else:
        expected_ranks = {32, 64, 128} if method == "TargetSVQR" else {128, 256, 512}
        if (values["rank"] not in expected_ranks or
                not (1e-4 <= values["gamma"] <= .1) or
                not (100.0 <= values["total_loss"] <= 2.4e6)):
            raise ValueError(f"{method} configuration lies outside the frozen search space.")
        if method != "TargetSVQR" and not (
                .05 <= values["target_fraction"] <= .995):
            raise ValueError(f"{method} target fraction lies outside the frozen search space.")
        if any(not (1e-5 <= values[name] <= 1000.0)
               for name in ACTIVE_REGULARIZERS[method]):
            raise ValueError(f"{method} regularizer lies outside the frozen search space.")
    return tuple(signature)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} is not a SHA-256 digest.")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not a SHA-256 digest.") from error
    return value


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("A hashed input path is empty or malformed.")
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Hashed input escapes its root: {relative}")
    return candidate


def _verify_hashes(root: Path, hashes: object, required: Iterable[str], label: str) -> dict:
    if not isinstance(hashes, dict):
        raise ValueError(f"{label} lacks a hash inventory.")
    for name in required:
        path = _safe_path(root, name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} input: {path}")
        expected = _sha256(hashes.get(name), f"{label} hash for {name}")
        if digest(path) != expected:
            raise ValueError(f"{label} hash mismatch: {name}")
    return hashes


def _expected_keys() -> list[tuple[str, str]]:
    return [(task, method) for scope in SCOPES for task in TASKS[scope]
            for method in METHODS]


def _expected_analysis_protocol() -> dict:
    """Independent copy of the frozen formal analysis contract."""
    return {
        "version": "selection_v2_analysis_2",
        "required_selection_lock_version": "selection_v2_lock_2",
        "reporting_rule": (
            "retain all 12 tasks x 6 methods and every predeclared method-specific "
            "evaluation seed; kernel methods use 0,1,2 and deterministic HistGBR uses 0"
        ),
        "primary_scope": {
            "name": "within", "tasks": list(TASKS["within"]),
            "role": "six within-farm normal-to-extreme breadth tasks",
        },
        "secondary_scope": {
            "name": "cross", "tasks": list(TASKS["cross"]),
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
            "comparators": list(METHODS[1:]),
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


def _validate_selection(selection: Path, input_root: Path) -> tuple[dict, dict, dict, str]:
    selection = Path(selection)
    names = ("selection_lock.json", "protocol.json", "manifest.json",
             "data_provenance.json")
    for name in names:
        if not (selection / name).is_file():
            raise FileNotFoundError(f"Selection lock is incomplete: {selection / name}")
    lock = _read_json(selection / "selection_lock.json")
    protocol = _read_json(selection / "protocol.json")
    manifest = _read_json(selection / "manifest.json")
    if not all(isinstance(value, dict) for value in (lock, protocol, manifest)):
        raise ValueError("Selection metadata must contain JSON objects.")
    if lock.get("version") != "selection_v2_lock_2":
        raise ValueError("Unexpected or missing selection-v2 lock.")
    if protocol.get("version") != "selection_v2":
        raise ValueError("Unexpected selection protocol version.")
    if manifest.get("status") != "complete" or manifest.get("last_stable_status") != "complete":
        raise ValueError("Selection evaluation is not complete.")
    protocol_hash = digest(selection / "protocol.json")
    if lock.get("protocol_sha256") != protocol_hash or manifest.get("protocol_sha256") != protocol_hash:
        raise ValueError("Selection protocol hash chain is incomplete or inconsistent.")
    if manifest.get("protocol") != protocol:
        raise ValueError("Selection manifest and protocol differ.")
    if tuple(protocol.get("methods", ())) != METHODS:
        raise ValueError("Selection protocol does not contain the six expected methods.")
    expected_tasks = TASKS["within"] + TASKS["cross"]
    if tuple(row.get("name") for row in protocol.get("tasks", ())) != expected_tasks:
        raise ValueError("Selection protocol does not contain the 12 expected tasks.")
    if (lock.get("validation_query_budget") != protocol.get("validation_query_budget") or
            protocol.get("validation_query_budget", {}).get("total_per_task_method") != 54 or
            lock.get("total_configuration_queries") != 12 * 6 * 54):
        raise ValueError("Selection lock does not certify the complete 12 x 6 x 54 budget.")
    for stage in STAGES:
        _strict_int(protocol["validation_query_budget"].get(stage),
                    f"{stage} query budget", minimum=1)
    _strict_int(protocol["validation_query_budget"].get("total_per_task_method"),
                "total query budget", minimum=1)
    _strict_int(lock.get("total_configuration_queries"),
                "total configuration queries", minimum=1)
    if manifest.get("source_hashes") != lock.get("source_hashes") or \
            manifest.get("data_hashes") != lock.get("data_hashes"):
        raise ValueError("Selection source/data hash chain is inconsistent.")
    expected_frozen = {
        "source_hashes": set(SELECTION_SOURCE_FILES),
        "data_hashes": set(SELECTION_DATA_FILES),
    }
    for name, expected_names in expected_frozen.items():
        hashes = lock.get(name)
        if not isinstance(hashes, dict) or set(hashes) != expected_names:
            raise ValueError(f"Selection lock has an incomplete or unexpected {name} inventory.")
        _verify_hashes(input_root, hashes, expected_names, name)
    chronological = _safe_path(
        input_root, "review/replied_comment_runs/chronological_v1/manifest.json")
    if (not chronological.is_file() or
            manifest.get("reference_manifest_sha256") != digest(chronological)):
        raise ValueError("Selection chronological-reference hash is missing or inconsistent.")
    if (not lock.get("environment_versions") or
            lock.get("environment_versions") != manifest.get("environment_versions") or
            lock.get("environment_versions") != protocol.get("environment_versions")):
        raise ValueError("Selection environment hash chain is inconsistent.")
    if digest(selection / "data_provenance.json") != lock.get("data_provenance_sha256"):
        raise ValueError("Selection data-provenance hash mismatch.")
    if lock.get("heldout_policy") != (
            "no Q2 calibration or Q3-Q4 prediction was calculated before this lock"):
        raise ValueError("Selection lock lacks the formal held-out policy.")

    expected_files = {
        f"{task}/{method}/{stage}_selection.json"
        for task, method in _expected_keys()
        for stage in ("base", "middle", "regularized", "joint", "final")
    }
    file_hashes = lock.get("selection_files")
    if not isinstance(file_hashes, dict) or set(file_hashes) != expected_files:
        raise ValueError("Selection lock has an incomplete selection-record inventory.")
    _verify_hashes(selection, file_hashes, expected_files, "locked selection")
    final = lock.get("final_selections")
    if not isinstance(final, list) or len(final) != 72:
        raise ValueError("Selection lock must contain 72 final configurations.")
    final_map = {}
    for row in final:
        if not isinstance(row, dict):
            raise ValueError("A locked final selection is malformed.")
        key = (row.get("task"), row.get("method"))
        if key in final_map:
            raise ValueError(f"Duplicate locked final selection: {key}")
        final_map[key] = row
    if set(final_map) != set(_expected_keys()):
        raise ValueError("Locked final selections do not cover the 12 x 6 design.")
    for key, row in final_map.items():
        task, method = key
        disk = _read_json(selection / task / method / "final_selection.json")
        if disk != row:
            raise ValueError(f"Locked final selection differs from disk: {task}/{method}")
        config = row.get("selected_config")
        candidate = row.get("selected_candidate")
        if not isinstance(config, dict) or not isinstance(candidate, dict) or \
                candidate.get("config") != config:
            raise ValueError(f"Malformed selected configuration: {task}/{method}")
        if row.get("selected_source_stage") not in STAGES:
            raise ValueError(f"Invalid selected source stage: {task}/{method}")
    lock_hash = digest(selection / "selection_lock.json")
    return lock, protocol, final_map, lock_hash


def _expected_analysis_inputs(protocol: dict) -> set[str]:
    seed_map = protocol.get("evaluation_seeds_by_method")
    expected_seed_map = {method: [0, 1, 2] for method in KERNEL_METHODS}
    expected_seed_map["HistGBR"] = [0]
    if seed_map != expected_seed_map:
        raise ValueError("Analysis chain has an unexpected evaluation-seed design.")
    names = {"selection_lock.json", "results.json"}
    for task, method in _expected_keys():
        for seed in seed_map[method]:
            base = f"{task}/{method}/evaluation/seed{seed}"
            names.add(f"{base}/result.json")
            names.add(f"{base}/test_predictions.csv")
    return names


def _csv_value_matches(cell: str, value: object) -> bool:
    if value is None:
        return cell == ""
    if isinstance(value, bool):
        return cell == str(value)
    if isinstance(value, int):
        return cell == str(value)
    if isinstance(value, float):
        try:
            parsed = float(cell)
        except ValueError:
            return False
        # The frozen analyzer serializes CSV floats with ``%.12g`` while the
        # companion JSON retains the binary float.  Accept either an exact
        # round trip or exactly that declared 12-significant-digit rendering;
        # do not use a broad tolerance that could hide a changed value.
        return (math.isfinite(parsed) and
                (parsed == value or cell == format(value, ".12g")))
    if isinstance(value, (list, tuple)):
        try:
            parsed = ast.literal_eval(cell)
        except (SyntaxError, ValueError):
            return False
        return parsed == list(value)
    return cell == str(value)


def _validate_analysis(analysis: Path, selection: Path, protocol: dict,
                       lock_hash: str, input_root: Path) -> list[dict[str, str]]:
    analysis = Path(analysis)
    required = ("analysis_manifest.json", "analysis_protocol.json",
                "selection_v2_analysis.json", "task_method_seed_mean_rmse.csv")
    for name in required:
        if not (analysis / name).is_file():
            raise FileNotFoundError(f"Completed analysis is missing: {analysis / name}")
    manifest = _read_json(analysis / "analysis_manifest.json")
    report = _read_json(analysis / "selection_v2_analysis.json")
    analysis_protocol = _read_json(analysis / "analysis_protocol.json")
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError("Selection-v2 analysis is not complete.")
    if not isinstance(report, dict) or not isinstance(analysis_protocol, dict):
        raise ValueError("Analysis metadata is malformed.")
    if analysis_protocol != _expected_analysis_protocol():
        raise ValueError("Unexpected formal analysis protocol/version.")
    if report.get("version") != analysis_protocol["version"]:
        raise ValueError("Analysis report and formal protocol versions differ.")
    if manifest.get("protocol") != analysis_protocol or \
            manifest.get("protocol_sha256") != digest(analysis / "analysis_protocol.json"):
        raise ValueError("Analysis protocol hash chain is inconsistent.")
    if manifest.get("selection_root") != str(Path(selection).resolve()):
        raise ValueError("Analysis manifest names a different selection root.")
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(
            ANALYSIS_ARTIFACT_FILES):
        raise ValueError("Analysis artifact hash inventory is incomplete or unexpected.")
    hashes = _verify_hashes(
        analysis, artifact_hashes, ANALYSIS_ARTIFACT_FILES, "analysis artifact")
    del hashes
    if report.get("selection_lock_sha256") != lock_hash:
        raise ValueError("Analysis report does not descend from the current selection lock.")
    counts = report.get("reporting_counts", {})
    if counts.get("tasks") != 12 or counts.get("methods") != 6 or \
            counts.get("task_method_rows") != 72 or \
            counts.get("locked_result_records") != 192 or \
            counts.get("full_comparisons") != 60 or \
            counts.get("evaluation_seeds_by_method") != \
            protocol.get("evaluation_seeds_by_method"):
        raise ValueError("Analysis report does not certify the full 12 x 6 design.")
    if manifest.get("reporting_counts") != counts:
        raise ValueError("Analysis manifest and report disagree on reporting counts.")
    source_path = _safe_path(input_root, "review/analyze_selection_study_v2.py")
    if not source_path.is_file() or manifest.get("analysis_source_sha256") != digest(source_path):
        raise ValueError("Analysis source hash does not match the actual analysis script.")
    input_hashes = manifest.get("selection_input_hashes")
    expected_inputs = _expected_analysis_inputs(protocol)
    if not isinstance(input_hashes, dict) or set(input_hashes) != expected_inputs:
        raise ValueError("Analysis input hash chain is incomplete or unexpected.")
    for name, value in input_hashes.items():
        expected = _sha256(value, f"analysis input hash for {name}")
        path = _safe_path(selection, name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Analysis input byte hash mismatch: {name}")
    if input_hashes.get("selection_lock.json") != lock_hash:
        raise ValueError("Analysis manifest names a different selection lock.")

    with (analysis / "task_method_seed_mean_rmse.csv").open(
            encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        csv_fields = reader.fieldnames
    required_columns = {"scope", "task", "site", "method", "seed_mean_rmse_mw"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError("Analysis RMSE CSV lacks required columns.")
    json_rows = report.get("task_method_seed_mean_rmse")
    if not isinstance(json_rows, list) or len(json_rows) != 72:
        raise ValueError("Analysis JSON lacks the complete task-method RMSE rows.")
    if (not isinstance(csv_fields, list) or
            len(csv_fields) != len(set(csv_fields))):
        raise ValueError("Analysis RMSE CSV lacks a header.")
    observed = []
    for row_index, (row, json_row) in enumerate(zip(rows, json_rows)):
        if not isinstance(json_row, dict) or set(csv_fields) != set(json_row):
            raise ValueError(f"CSV/analysis JSON fields differ at row {row_index}.")
        for name in csv_fields:
            if not _csv_value_matches(row[name], json_row[name]):
                raise ValueError(
                    f"CSV/analysis JSON value differs at row {row_index}, field {name}.")
        key = (row["task"], row["method"])
        scope = row["scope"]
        if scope not in SCOPES or row["task"] not in TASKS[scope]:
            raise ValueError(f"Unexpected analysis task/scope: {scope}/{row['task']}")
        expected_site = int(row["task"].rsplit("site", 1)[1])
        if _strict_csv_int(row["site"], "analysis site", minimum=1) != expected_site:
            raise ValueError(f"Analysis site mismatch: {row['task']}")
        rmse = float(row["seed_mean_rmse_mw"])
        if not math.isfinite(rmse) or rmse < 0:
            raise ValueError(f"Invalid analysis RMSE: {row['task']}/{row['method']}")
        observed.append(key)
    if observed != _expected_keys() or len(set(observed)) != 72:
        raise ValueError("Analysis RMSE CSV is incomplete or contains duplicate rows.")
    return rows


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite.")
    return number


def _selection_statistics(selection: Path) -> list[dict[str, object]]:
    totals: dict[tuple[str, str], dict[str, object]] = {}
    for scope in SCOPES:
        for method in METHODS:
            totals[(scope, method)] = {
                "scope": scope, "method": method,
                "queried": {stage: 0 for stage in STAGES},
                "eligible_candidates": 0, "failed_seed_fits": 0,
                "solver_attempts": 0, "solver_iterations": 0,
                "fit_seconds": 0.0,
            }
    for scope in SCOPES:
        for task in TASKS[scope]:
            for method in METHODS:
                target = totals[(scope, method)]
                expected_seeds = [0] if method == "HistGBR" else [0, 1, 2]
                prior_winner = None
                all_signatures = []
                for stage in STAGES:
                    record = _read_json(selection / task / method / f"{stage}_selection.json")
                    if not isinstance(record, dict) or \
                            (record.get("task"), record.get("method"), record.get("stage")) != \
                            (task, method, stage):
                        raise ValueError(f"Selection-stage identity mismatch: {task}/{method}/{stage}")
                    candidates = record.get("candidates")
                    if _strict_int(record.get("candidate_count"),
                                   "candidate count", minimum=1) != 18 or \
                            _strict_int(record.get("queried_effective_unique_count"),
                                        "effective query count", minimum=1) != 18 or \
                            not isinstance(candidates, list) or len(candidates) != 18:
                        raise ValueError(f"Incomplete 18-query stage: {task}/{method}/{stage}")
                    if record.get("carried_incumbent") != prior_winner:
                        raise ValueError(f"Broken carried-incumbent chain: {task}/{method}/{stage}")
                    signatures = []
                    target["queried"][stage] += len(candidates)
                    for index, candidate in enumerate(candidates):
                        if not isinstance(candidate, dict):
                            raise ValueError("A queried candidate is malformed.")
                        if (_strict_int(candidate.get("candidate_index"),
                                        "candidate index", minimum=0) != index or
                                candidate.get("candidate_id") != f"{stage}_c{index:03d}" or
                                candidate.get("queried_stage") != stage):
                            raise ValueError(
                                f"Candidate ID/index/stage mismatch: {task}/{method}/{stage}")
                        signatures.append(_effective_signature(method, candidate.get("config")))
                        seeds = candidate.get("seed_results")
                        if not isinstance(seeds, list) or len(seeds) != len(expected_seeds):
                            raise ValueError(f"Selection seed inventory mismatch: {task}/{method}/{stage}")
                        observed_seeds = [
                            _strict_int(row.get("seed"), "selection seed", minimum=0)
                            if isinstance(row, dict) else None for row in seeds]
                        if observed_seeds != expected_seeds:
                            raise ValueError(f"Selection seed inventory mismatch: {task}/{method}/{stage}")
                        failed = candidate.get("failed_certificate_seeds")
                        if not isinstance(failed, list):
                            raise ValueError(f"Missing certificate inventory: {task}/{method}/{stage}")
                        failed = [_strict_int(seed, "failed-certificate seed", minimum=0)
                                  for seed in failed]
                        derived_failed = []
                        validation_rmses = []
                        for seed_row in seeds:
                            diagnostics = seed_row.get("diagnostics")
                            if not isinstance(diagnostics, dict):
                                raise ValueError("A selection fit lacks solver diagnostics.")
                            metrics = seed_row.get("metrics")
                            if not isinstance(metrics, dict):
                                raise ValueError("A selection fit lacks validation metrics.")
                            rmse = _finite_number(
                                metrics.get("rmse"), "seed validation RMSE")
                            if rmse < 0:
                                raise ValueError("A selection fit contains a negative validation RMSE.")
                            validation_rmses.append(rmse)
                            if method != "HistGBR" and diagnostics.get("converged") is not True:
                                derived_failed.append(seed_row["seed"])
                            if method == "HistGBR":
                                attempts = 1
                                iterations = _strict_int(
                                    diagnostics.get("iterations"), "HistGBR iterations", minimum=0)
                                seconds = _finite_number(
                                    diagnostics.get("fit_seconds"), "HistGBR fit seconds")
                            else:
                                summaries = diagnostics.get("attempt_summaries")
                                if not isinstance(summaries, list) or not summaries:
                                    raise ValueError("A kernel fit lacks its solver-attempt summaries.")
                                if _strict_int(diagnostics.get("attempt_count"),
                                               "kernel attempt count", minimum=1) != len(summaries):
                                    raise ValueError("Kernel attempt count disagrees with its summaries.")
                                attempts = len(summaries)
                                iterations = sum(_strict_int(
                                    item.get("iterations"), "ADMM attempt iterations", minimum=0)
                                    for item in summaries)
                                seconds = _finite_number(
                                    diagnostics.get("fit_seconds_including_feature_access"),
                                    "kernel fit seconds")
                            if seconds < 0:
                                raise ValueError("A solver diagnostic contains a negative count or duration.")
                            target["solver_attempts"] += attempts
                            target["solver_iterations"] += int(iterations)
                            target["fit_seconds"] += seconds
                        stored_mean = _finite_number(
                            candidate.get("mean_validation_rmse"),
                            "mean validation RMSE")
                        derived_mean = math.fsum(validation_rmses) / len(validation_rmses)
                        if not math.isclose(stored_mean, derived_mean,
                                            rel_tol=1e-13, abs_tol=1e-13):
                            raise ValueError(
                                "Candidate mean validation RMSE disagrees with its seed metrics.")
                        if failed != derived_failed:
                            raise ValueError("Failed-certificate seeds disagree with fit diagnostics.")
                        eligible = candidate.get("eligible_for_selection")
                        if not isinstance(eligible, bool) or eligible != (not failed):
                            raise ValueError("Candidate eligibility disagrees with certificate failures.")
                        target["eligible_candidates"] += int(eligible)
                        target["failed_seed_fits"] += len(failed)
                        fit_folders = candidate.get("fit_folders")
                        expected_folders = {
                            str(seed):
                                f"{task}/{method}/{stage}/candidates/c{index:03d}/seed{seed}"
                            for seed in expected_seeds
                        }
                        if fit_folders != expected_folders:
                            raise ValueError(
                                f"Candidate fit-folder chain is inconsistent: "
                                f"{task}/{method}/{stage}")
                    if len(set(signatures)) != 18:
                        raise ValueError(f"Stage effective signatures are not unique: {task}/{method}/{stage}")
                    if set(signatures) & set(all_signatures):
                        raise ValueError(f"Effective candidate repeated across stages: {task}/{method}/{stage}")
                    all_signatures.extend(signatures)

                    eligible_rows = [(candidate, True) for candidate in candidates
                                     if candidate["eligible_for_selection"]]
                    if prior_winner is not None:
                        eligible_rows.append((prior_winner, False))
                    if not eligible_rows:
                        raise ValueError(f"No eligible staged candidate: {task}/{method}/{stage}")
                    winner, _ = min(
                        eligible_rows,
                        key=lambda item: (
                            _finite_number(item[0].get("mean_validation_rmse"),
                                           "mean validation RMSE"),
                            item[1],
                            _strict_int(item[0].get("candidate_index"),
                                        "winner candidate index", minimum=0),
                        ),
                    )
                    selected_index = _strict_int(
                        record.get("selected_candidate_index"),
                        "selected candidate index", minimum=0)
                    if (record.get("selected_candidate") != winner or
                            record.get("selected_config") != winner.get("config") or
                            selected_index != winner.get("candidate_index") or
                            record.get("selected_candidate_id") != winner.get("candidate_id") or
                            record.get("selected_source_stage") != winner.get("queried_stage") or
                            record.get("selected_mean_validation_rmse") !=
                            winner.get("mean_validation_rmse") or
                            record.get("selected_fit_folders") != winner.get("fit_folders")):
                        raise ValueError(f"Stored stage winner is inconsistent: {task}/{method}/{stage}")
                    prior_winner = winner
                if len(all_signatures) != 54 or len(set(all_signatures)) != 54:
                    raise ValueError(f"Task-method design lacks 54 unique signatures: {task}/{method}")
                regularized = _read_json(
                    selection / task / method / "regularized_selection.json")
                middle = _read_json(selection / task / method / "middle_selection.json")
                middle_kind = (
                    "active_raw_regularizer_search"
                    if ACTIVE_REGULARIZERS.get(method) else
                    ("regularizer_equivalent_kernel_base_refinement"
                     if method in KERNEL_METHODS else
                     "regularizer_equivalent_tree_refinement")
                )
                expected_regularized = {
                    **middle,
                    "source_stage": "middle",
                    "middle_kind": middle_kind,
                    "base_selection_sha256": digest(
                        selection / task / method / "base_selection.json"),
                }
                if regularized != expected_regularized:
                    raise ValueError(f"Regularized wrapper differs from middle selection: {task}/{method}")
                final = _read_json(selection / task / method / "final_selection.json")
                joint = _read_json(selection / task / method / "joint_selection.json")
                expected_final = {
                    **joint,
                    "source_stage": "joint",
                    "regularized_selection_sha256": digest(
                        selection / task / method / "regularized_selection.json"),
                }
                if final != expected_final:
                    raise ValueError(f"Final wrapper differs from joint selection: {task}/{method}")
    rows = [totals[(scope, method)] for scope in SCOPES for method in METHODS]
    for row in rows:
        if row["queried"] != {stage: 108 for stage in STAGES}:
            raise ValueError("Optimization summary does not contain six complete tasks per scope.")
    return rows


def _escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _fmt_config(value: object) -> str:
    number = _finite_number(value, "configuration value")
    if number == 0:
        return "0"
    absolute = abs(number)
    if absolute < .001 or absolute >= 10000:
        return f"{number:.3g}"
    return f"{number:.4g}"


def _table_open(caption: str, label: str, columns: str) -> list[str]:
    return [
        r"\begin{table*}[pos=t,width=\FullWidth]", r"\centering", r"\color{blue}", f"\\caption{{{caption}}}",
        f"\\label{{{label}}}", r"\small", r"\setlength{\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{columns}}}", r"\toprule",
    ]


def _selected_config_tables(final_map: dict[tuple[str, str], dict]) -> str:
    lines: list[str] = []
    scope_label = {"within": "primary within-farm", "cross": "secondary cross-farm"}
    for scope in SCOPES:
        lines.extend(_table_open(
            "Locked kernel configurations for the " + scope_label[scope] +
            " tasks. The source stage identifies where the retained incumbent was first queried.",
            f"tab:selection-v2-{scope}-kernel-configs", "lllrlllll"))
        lines.append(r"Farm & Method & Stage & $r$ & $\gamma$ & $A$ & $p$ & $\lambda_1$ & $\lambda_2$ \\")
        lines.append(r"\midrule")
        for site, task in enumerate(TASKS[scope], start=1):
            for method in KERNEL_METHODS:
                row = final_map[(task, method)]
                config = row["selected_config"]
                required = ("rank", "gamma", "total_loss", "lambda1", "lambda2")
                if any(name not in config for name in required):
                    raise ValueError(f"Locked kernel configuration lacks a displayed field: {task}/{method}")
                p = 1.0 if method == "TargetSVQR" else config.get("target_fraction")
                if p is None:
                    raise ValueError(f"Locked pooled kernel configuration lacks p: {task}/{method}")
                values = [
                    f"Farm {site}", METHOD_LABELS[method], row["selected_source_stage"],
                    str(_strict_int(config["rank"], "rank", minimum=1)),
                    _fmt_config(config["gamma"]), _fmt_config(config["total_loss"]),
                    _fmt_config(p), _fmt_config(config["lambda1"]),
                    _fmt_config(config["lambda2"]),
                ]
                lines.append(" & ".join(_escape(value) for value in values) + r" \\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])

    lines.extend(_table_open(
        "Locked HistGBR configurations for the primary and secondary tasks.",
        "tab:selection-v2-histgbr-configs", "lllrrrrrr"))
    lines.append(r"Scope & Farm & Stage & LR & Leaves & Min. leaf & $L_2$ & Iter. & $p$ \\")
    lines.append(r"\midrule")
    for scope in SCOPES:
        for site, task in enumerate(TASKS[scope], start=1):
            row = final_map[(task, "HistGBR")]
            config = row["selected_config"]
            fields = ("learning_rate", "max_leaf_nodes", "min_samples_leaf",
                      "l2_regularization", "max_iter", "target_fraction")
            if any(name not in config for name in fields):
                raise ValueError(f"Locked HistGBR configuration lacks a displayed field: {task}")
            values = [scope.title(), f"Farm {site}", row["selected_source_stage"],
                      _fmt_config(config["learning_rate"]),
                      str(_strict_int(config["max_leaf_nodes"], "maximum leaves", minimum=1)),
                      str(_strict_int(config["min_samples_leaf"], "minimum leaf size", minimum=1)),
                      _fmt_config(config["l2_regularization"]),
                      str(_strict_int(config["max_iter"], "HistGBR iterations", minimum=1)),
                      _fmt_config(config["target_fraction"])]
            lines.append(" & ".join(_escape(value) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _cross_rmse_table(rows: list[dict[str, str]]) -> str:
    values = {(row["task"], row["method"]): float(row["seed_mean_rmse_mw"])
              for row in rows if row["scope"] == "cross"}
    if set(values) != {(task, method) for task in TASKS["cross"] for method in METHODS}:
        raise ValueError("Secondary cross-farm analysis rows are incomplete.")
    lines = _table_open(
        "Q3--Q4 RMSE (MW) for all six secondary leave-one-farm-out transfer-stress tasks. "
        "Kernel entries are means over the three predeclared evaluation seeds; HistGBR is deterministic.",
        "tab:selection-v2-secondary-cross-rmse", "lrrrrrr")
    lines.append("Farm & " + " & ".join(_escape(METHOD_LABELS[m]) for m in METHODS) + r" \\")
    lines.append(r"\midrule")
    for site, task in enumerate(TASKS["cross"], start=1):
        lines.append(f"Farm {site} & " + " & ".join(
            f"{values[(task, method)]:.3f}" for method in METHODS) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _optimization_table(rows: list[dict[str, object]]) -> str:
    lines = _table_open(
        "Audited Q1 optimization totals by scope and method. B, M, and J are newly queried "
        "candidates in the base, middle, and joint stages. Eligibility and certificate failures "
        "refer only to newly queried candidates. Iterations sum the reported iterations over all "
        "solver attempts; fit time sums recorded per-fit wall time and is descriptive because tasks ran in parallel.",
        "tab:selection-v2-optimization-summary", "llrrrrrrrr")
    lines.append(r"Scope & Method & B & M & J & Eligible & Failed seed fits & Attempts & Iterations & Fit h \\")
    lines.append(r"\midrule")
    for row in rows:
        queried = row["queried"]
        values = [row["scope"].title(), METHOD_LABELS[row["method"]],
                  queried["base"], queried["middle"], queried["joint"],
                  row["eligible_candidates"], row["failed_seed_fits"],
                  row["solver_attempts"], row["solver_iterations"],
                  f"{float(row['fit_seconds']) / 3600:.2f}"]
        lines.append(" & ".join(_escape(value) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _optimization_csv(rows: list[dict[str, object]]) -> str:
    """Return the audited solver accounting in a machine-readable form."""
    columns = (
        "scope", "method", "base_queries", "middle_queries", "joint_queries",
        "total_queries", "eligible_candidates", "failed_seed_fits",
        "solver_attempts", "solver_iterations", "fit_seconds", "fit_hours",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        queried = row["queried"]
        seconds = float(row["fit_seconds"])
        writer.writerow({
            "scope": row["scope"],
            "method": row["method"],
            "base_queries": queried["base"],
            "middle_queries": queried["middle"],
            "joint_queries": queried["joint"],
            "total_queries": sum(int(queried[stage]) for stage in STAGES),
            "eligible_candidates": row["eligible_candidates"],
            "failed_seed_fits": row["failed_seed_fits"],
            "solver_attempts": row["solver_attempts"],
            "solver_iterations": row["solver_iterations"],
            "fit_seconds": format(seconds, ".17g"),
            "fit_hours": format(seconds / 3600.0, ".17g"),
        })
    return buffer.getvalue()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _input_snapshot(selection: Path, analysis: Path, input_root: Path,
                    lock: dict, protocol: dict) -> dict[str, str]:
    """Hash every byte source on which validation or rendered content depends."""
    paths: dict[str, Path] = {
        "selection/selection_lock.json": selection / "selection_lock.json",
        "selection/protocol.json": selection / "protocol.json",
        "selection/manifest.json": selection / "manifest.json",
        "selection/data_provenance.json": selection / "data_provenance.json",
        "analysis/analysis_manifest.json": analysis / "analysis_manifest.json",
        "analysis/analysis_protocol.json": analysis / "analysis_protocol.json",
        "analysis/selection_v2_analysis.json": analysis / "selection_v2_analysis.json",
        "analysis/task_method_seed_mean_rmse.csv":
            analysis / "task_method_seed_mean_rmse.csv",
        "frozen/review/replied_comment_runs/chronological_v1/manifest.json":
            _safe_path(
                input_root,
                "review/replied_comment_runs/chronological_v1/manifest.json"),
        "source/review/analyze_selection_study_v2.py":
            _safe_path(input_root, "review/analyze_selection_study_v2.py"),
        "source/review/render_manuscript_v2_appendix.py": Path(__file__).resolve(),
    }
    for relative in ANALYSIS_ARTIFACT_FILES:
        paths[f"analysis/{relative}"] = _safe_path(analysis, relative)
    for collection in ("source_hashes", "data_hashes"):
        for relative in lock[collection]:
            paths[f"frozen/{relative}"] = _safe_path(input_root, relative)
    for relative in lock["selection_files"]:
        paths[f"selection/{relative}"] = _safe_path(selection, relative)
    for relative in _expected_analysis_inputs(protocol):
        paths[f"analysis-input/{relative}"] = _safe_path(selection, relative)
    snapshot = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot input disappeared: {path}")
        snapshot[label] = digest(path)
    return dict(sorted(snapshot.items()))


def _load_validated(selection: Path, analysis: Path,
                    input_root: Path) -> dict[str, object]:
    lock, protocol, final_map, lock_hash = _validate_selection(selection, input_root)
    rmse_rows = _validate_analysis(
        analysis, selection, protocol, lock_hash, input_root)
    optimization_rows = _selection_statistics(selection)
    return {
        "lock": lock, "protocol": protocol, "final_map": final_map,
        "lock_hash": lock_hash, "rmse_rows": rmse_rows,
        "optimization_rows": optimization_rows,
        "snapshot": _input_snapshot(selection, analysis, input_root, lock, protocol),
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing target."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(source), -100,
                           os.fsencode(destination), 1)  # RENAME_NOREPLACE
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"Appendix output already exists: {destination}")
        if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise OSError(error, os.strerror(error), str(destination))
    # A check followed by os.rename() can replace a directory created by an
    # uncooperative process in the intervening instant.  Refuse publication on
    # platforms lacking an atomic no-replace directory rename.
    raise OSError(
        errno.ENOTSUP,
        "Atomic no-replace directory publication is unavailable on this platform",
        str(destination),
    )


def render(selection: Path, analysis: Path, output: Path, *,
           input_root: Path = ROOT) -> dict:
    """Validate the complete chain, stage outputs, and atomically publish once."""
    selection, analysis, output = Path(selection), Path(analysis), Path(output)
    input_root = Path(input_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    publisher_lock = output.parent / f".{output.name}.publish.lock"
    try:
        lock_fd = os.open(publisher_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            f"Another renderer is publishing this appendix target: {output}") from error
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        if output.exists():
            raise FileExistsError(f"Refusing to replace an existing appendix render: {output}")
        first = _load_validated(selection, analysis, input_root)
        _atomic_text(staging / OUTPUT_FILES[0],
                     _selected_config_tables(first["final_map"]))
        _atomic_text(staging / OUTPUT_FILES[1],
                     _cross_rmse_table(first["rmse_rows"]))
        _atomic_text(staging / OUTPUT_FILES[2],
                     _optimization_table(first["optimization_rows"]))
        _atomic_text(staging / OUTPUT_FILES[3],
                     _optimization_csv(first["optimization_rows"]))

        # Parse and hash the entire dependency graph again immediately before
        # publication. This detects concurrent mutation and guarantees that the
        # TeX corresponds to the byte snapshot recorded below.
        second = _load_validated(selection, analysis, input_root)
        for name in ("lock", "protocol", "final_map", "lock_hash", "rmse_rows",
                     "optimization_rows", "snapshot"):
            if second[name] != first[name]:
                raise RuntimeError(f"Input changed during appendix rendering: {name}")
        manifest = {
            "version": "selection_v2_manuscript_appendix_render_2",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "selection_root": str(selection.resolve()),
            "analysis_root": str(analysis.resolve()),
            "selection_lock_sha256": second["lock_hash"],
            "heldout_access_policy": (
                "Held-out results and predictions were read only as opaque bytes for SHA-256 "
                "verification and were never parsed; the cross-task table reads only the "
                "hash-verified completed-analysis CSV."
            ),
            "validated_counts": {
                "locked_configs": len(second["final_map"]),
                "secondary_cross_rmse_rows": 6 * 6,
                "optimization_scope_method_rows": len(second["optimization_rows"]),
                "queried_candidates": 12 * 6 * 54,
            },
            "inputs": second["snapshot"],
            "outputs": {name: digest(staging / name) for name in OUTPUT_FILES},
        }
        _atomic_text(staging / "manifest.json",
                     json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        final_snapshot = _input_snapshot(
            selection, analysis, input_root, second["lock"], second["protocol"])
        if final_snapshot != second["snapshot"]:
            raise RuntimeError("Input bytes changed immediately before appendix publication.")
        _rename_noreplace(staging, output)
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        os.close(lock_fd)
        try:
            publisher_lock.unlink()
        except FileNotFoundError:
            pass
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.selection_root, args.analysis_root, args.output)


if __name__ == "__main__":
    main()
