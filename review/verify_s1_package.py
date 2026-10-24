#!/usr/bin/env python3
"""Verify S1's outer manifest, exclusions, and nested frozen-v2 hash chains."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


LINE = re.compile(r"^([0-9a-f]{64})\s+(\d+)\s+(.+)$")
HEX64 = re.compile(r"[0-9a-f]{64}")
SELECTION = Path("review/selection_runs/selection_v2")
SELECTION_ANALYSIS = Path("review/selection_runs/selection_v2_analysis")
MATCHED = Path("review/selection_runs/matched_ablation_v2")
MATCHED_ANALYSIS = Path("review/selection_runs/matched_ablation_v2_analysis")
RENDER = Path("review/selection_runs/selection_v2_render")
AMENDMENT_LOCK = Path("review/selection_runs/solver_certificate_amendment_v2.json")
AMENDMENT_RUNNER = Path("review/resume_solver_certificate_v2.py")
AMENDMENT_VERSION = "solver_certificate_amendment_v2_2"
CHRONOLOGICAL = Path("review/replied_comment_runs/chronological_v1")
PROBABILITY = Path("review/probabilistic_runs/probabilistic_selection_v2")
PROBABILITY_ANALYSIS = Path("review/probabilistic_runs/probabilistic_analysis_v2")
PROBABILITY_PRESENTATION = Path("review/probabilistic_runs/probabilistic_presentation_v1")
EVIDENCE_INVENTORY = Path("PACKAGE_EVIDENCE_INVENTORY.json")
FIT_RECORD_INVENTORY = Path("fit_record_inventory.json")
FIT_RECORD_INVENTORY_VERSION = "selection_v2_fit_record_inventory_2"
FIT_RECORD_HASH_AUTHORITY = (
    "package-time source-record checksum inventory; not an original experiment lock "
    "or preregistration"
)
FIT_RECORD_SOURCE_VALIDATION = (
    "Before packaging, every listed bottom-level record was read and matched exactly "
    "to its selection or locked main-result embedding; selection-candidate certificate "
    "failures were retained only with consistent ineligibility, and every locked main "
    "kernel fit carried diagnostics.converged=true."
)
LEGACY_PROBABILITY_LIMITATION = (
    "The probability study has post-completion integrity hashes, but its original "
    "manifest did not freeze every transitive loader/sampling dependency or bind the "
    "chronological reference manifest; those historical hashes cannot be reconstructed."
)
PROBABILITY_CURRENT_CHECK = (
    "At packaging time, run_selection_study.protocol() exactly matched the archived "
    "selection_v1/protocol.json object. This current-source consistency check does not "
    "retroactively create missing pre-run hashes."
)
PACKAGE_TIME_SNAPSHOT = (
    "package-time snapshot; not an original experiment lock or preregistration"
)
README_POINT_WORKFLOW = (
    "python review/run_selection_study_v2.py --output review/selection_runs/"
    "reproduced_selection_v2 --freeze-only",
    "python review/run_selection_study_v2.py --output review/selection_runs/"
    "reproduced_selection_v2 --phase base --resume",
    "python review/run_selection_study_v2.py --output review/selection_runs/"
    "reproduced_selection_v2 --phase regularizers --resume",
    "python review/run_selection_study_v2.py --output review/selection_runs/"
    "reproduced_selection_v2 --phase joint --resume",
    "python review/run_selection_study_v2.py --output review/selection_runs/"
    "reproduced_selection_v2 --phase lock --resume",
    "python review/analyze_selection_study_v2.py --selection-root review/selection_runs/"
    "reproduced_selection_v2 --output review/selection_runs/"
    "reproduced_selection_v2_analysis --freeze-only",
    "python review/run_matched_ablation_v2.py --selection-output review/selection_runs/"
    "reproduced_selection_v2 --output review/selection_runs/"
    "reproduced_matched_ablation_v2 --phase freeze",
    "python review/analyze_matched_ablation_v2.py --matched-output review/selection_runs/"
    "reproduced_matched_ablation_v2 --output review/selection_runs/"
    "reproduced_matched_ablation_v2_analysis --freeze-only",
    "python review/run_selection_study_v2.py --output review/selection_runs/"
    "reproduced_selection_v2 --phase evaluate --resume",
    "python review/run_matched_ablation_v2.py --selection-output review/selection_runs/"
    "reproduced_selection_v2 --output review/selection_runs/"
    "reproduced_matched_ablation_v2 --phase evaluate --resume",
    "python review/analyze_selection_study_v2.py --selection-root review/selection_runs/"
    "reproduced_selection_v2 --output review/selection_runs/"
    "reproduced_selection_v2_analysis --resume",
    "python review/analyze_matched_ablation_v2.py --matched-output review/selection_runs/"
    "reproduced_matched_ablation_v2 --output review/selection_runs/"
    "reproduced_matched_ablation_v2_analysis --resume",
    "python review/render_selection_study_v2.py --selection-analysis review/selection_runs/"
    "reproduced_selection_v2_analysis --matched-analysis review/selection_runs/"
    "reproduced_matched_ablation_v2_analysis --output review/selection_runs/"
    "reproduced_selection_v2_render",
    "python review/render_manuscript_v2_appendix.py --selection-root review/selection_runs/"
    "reproduced_selection_v2 --analysis-root review/selection_runs/"
    "reproduced_selection_v2_analysis --output review/selection_runs/"
    "reproduced_selection_v2_manuscript_appendix",
    "python review/summarize_v2_revision_evidence.py --selection-analysis review/selection_runs/"
    "reproduced_selection_v2_analysis --matched-analysis review/selection_runs/"
    "reproduced_matched_ablation_v2_analysis --output review/selection_runs/"
    "reproduced_revision_evidence_v2",
)
REQUIRED_PUBLICATION_SCRIPTS = (
    Path("review/render_manuscript_v2_appendix.py"),
    Path("review/summarize_v2_revision_evidence.py"),
)
METHODS = ("Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR", "TargetSVQR", "HistGBR")
TASKS = tuple(f"within_site{s}" for s in range(1, 7)) + tuple(f"cross_site{s}" for s in range(1, 7))
STAGES = ("base", "middle", "regularized", "joint", "final")
VARIANTS = ("Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR")
MATCHED_SEEDS = (0, 1, 2)
MATCHED_TAU = 0.5
MATCHED_CONFIG_KEYS = {
    "gamma", "total_loss", "target_fraction", "rank", "lambda1", "lambda2",
}
ACTIVE_REGULARIZERS = {
    "Full": ("lambda1", "lambda2"), "NoVariance": ("lambda2",),
    "NoDiscrepancy": ("lambda1",), "WeightedSVQR": (), "TargetSVQR": (),
}
CALIBRATION_HEADER = (
    "window_index", "raw_target_row", "time", "y_true", "q025", "q975", "score",
)
MATCHED_PREDICTION_HEADER = (
    "window_index", "raw_target_row", "time", "y_true", "prediction",
)
SELECTION_ANALYSIS_FILES = {
    "task_method_seed_mean_rmse.csv", "scope_summary.csv", "full_pairwise_inference.csv",
    "first_tier_all_pair_tests.csv", "conservative_first_tier.csv",
    "task_seed_mean_rmse.tex", "scope_summary.tex", "full_pairwise_inference.tex",
    "conservative_first_tier.tex", "selection_v2_analysis.json",
}
RENDER_FILES = {
    "primary_within_rmse.tex", "scope_macro_summary.tex", "full_paired_inference.tex",
    "matched_ablation.tex", "within_site_normalized_rmse.pdf",
    "within_site_normalized_rmse.png", "scope_average_rank.pdf",
    "scope_average_rank.png", "render_protocol.json",
}
RENDER_SELECTION_INPUT_FILES = {
    "selection_v2_analysis.json", "task_method_seed_mean_rmse.csv",
    "scope_summary.csv", "full_pairwise_inference.csv",
}
RENDER_MATCHED_INPUT_FILES = {
    "summary.json", "task_metrics.csv", "aggregate_metrics.csv", "paired_inference.csv",
}
PROBABILITY_METHOD_SEEDS = {
    method: ((0,) if method == "HistGBR" else (0, 1, 2)) for method in METHODS
}
SELECTION_METHOD_SEEDS = {
    method: ((0,) if method == "HistGBR" else (0, 1, 2)) for method in METHODS
}
PROBABILITY_ANALYSIS_FILES = {
    "protocol.json", "manifest.json", "aggregate_results.json",
    "paired_probability_inference.json", "posthoc_integrity_inventory.json",
    "probabilistic_scores.tex", "probabilistic_coverage.tex",
    "probabilistic_inference.tex", "probabilistic_reliability.pdf",
    "probabilistic_reliability.svg", "probabilistic_coverage_sharpness.pdf",
    "probabilistic_coverage_sharpness.svg",
}
PROBABILITY_PRESENTATION_FILES = {
    "manifest.json", "probabilistic_reliability.pdf", "probabilistic_reliability.svg",
    "probabilistic_coverage_sharpness.pdf", "probabilistic_coverage_sharpness.svg",
}

# Construct these strings so the verifier does not trigger its own content scan.
FORBIDDEN_TEXT = (
    "--enforce-tl-" + "dominance",
    "post_" + "adjustment",
    "Best TL-QLDMR config on " + "test",
    "test_" + "selected",
)
FORBIDDEN_SEGMENTS = {
    "paired_point_v2", "paired_point_v3", "residual_diagnostics",
    "corrected_validation_runs", "solver_refinement_diagnostics", "probabilistic_v1",
}
TEXT_SUFFIXES = {".py", ".json", ".csv", ".tex", ".txt", ".md"}
SPLIT_ASSIGNMENT_HEADER = (
    "window_index", "target_row_index", "target_timestamp", "split", "domain",
    "power_mw", "wind_rule", "cold_rule", "hot_rule", "above_25ms_rule",
)
SPLIT_BOUNDARIES = (
    ("train", datetime(2019, 1, 1), datetime(2020, 1, 1)),
    ("validation", datetime(2020, 1, 1), datetime(2020, 4, 1)),
    ("calibration", datetime(2020, 4, 1), datetime(2020, 7, 1)),
    ("test", datetime(2020, 7, 1), datetime(2021, 1, 1)),
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


_NANOSECOND_TIMESTAMP = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?$"
)


def _parse_naive_timestamp(value: str) -> tuple[datetime, int]:
    """Parse pandas ISO timestamps without discarding nanosecond ordering."""
    match = _NANOSECOND_TIMESTAMP.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid naive ISO timestamp: {value!r}")
    return datetime.fromisoformat(match.group(1)), int(
        (match.group(2) or "0").ljust(9, "0"))


def _object_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe(value: str | Path, errors: list[str], label: str) -> Path | None:
    if not isinstance(value, (str, Path)):
        errors.append(f"Unsafe {label} path: {value}")
        return None
    raw = str(value)
    path = Path(value)
    if (path.is_absolute() or not path.parts or ".." in path.parts or
            "\\" in raw or raw != path.as_posix()):
        errors.append(f"Unsafe {label} path: {value}")
        return None
    return path


def _json(path: Path, errors: list[str], label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Missing {label}: {path}")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"Malformed {label}: {path} ({error})")
    return None


def _hash_mapping(base: Path, mapping: object, errors: list[str], label: str) -> set[Path]:
    if not isinstance(mapping, dict) or not mapping:
        errors.append(f"Missing or empty {label} hash inventory")
        return set()
    paths: set[Path] = set()
    for name, expected in mapping.items():
        relative = _safe(name, errors, label)
        if relative is None:
            continue
        if relative in paths:
            errors.append(f"Duplicate canonical {label} path: {name}")
            continue
        path = base / relative
        if not path.is_file():
            errors.append(f"Missing {label} artifact: {relative}")
        elif not isinstance(expected, str) or digest(path) != expected:
            errors.append(f"{label} SHA-256 mismatch: {relative}")
        paths.add(relative)
    return paths


def _exact_subtree_files(root: Path, subtree: Path, expected: set[Path],
                         errors: list[str], label: str) -> None:
    base = root / subtree
    actual = {path.relative_to(base) for path in base.rglob("*") if path.is_file()} \
        if base.is_dir() else set()
    if actual != expected:
        extra = sorted(actual - expected, key=str)
        missing = sorted(expected - actual, key=str)
        detail = extra[0] if extra else (missing[0] if missing else "unknown")
        errors.append(f"{label} subtree inventory changed: {detail}")


def _effective_signature(method: str, candidate: object) -> tuple:
    if method not in METHODS or not isinstance(candidate, dict):
        raise ValueError(f"Invalid candidate for {method}")
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
    return tuple((key, candidate[key]) for key in keys)


def _derive_matched_variants(full_config: object) -> dict[str, dict]:
    if not isinstance(full_config, dict):
        raise ValueError("Full configuration is not a mapping")
    missing = MATCHED_CONFIG_KEYS - set(full_config)
    if missing:
        raise ValueError(f"Full configuration lacks {sorted(missing)}")
    try:
        if not all(math.isfinite(float(full_config[name])) for name in MATCHED_CONFIG_KEYS):
            raise ValueError("Full configuration contains a nonfinite coordinate")
        rank = int(full_config["rank"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Full configuration contains an invalid coordinate") from error
    if rank <= 0 or float(full_config["rank"]) != rank:
        raise ValueError("Full requested rank is not a positive integer")
    if float(full_config["lambda1"]) <= 0 or float(full_config["lambda2"]) <= 0:
        raise ValueError("Full does not activate both components")
    variants = {name: dict(full_config) for name in VARIANTS}
    variants["NoVariance"]["lambda1"] = 0.0
    variants["NoDiscrepancy"]["lambda2"] = 0.0
    variants["WeightedSVQR"]["lambda1"] = 0.0
    variants["WeightedSVQR"]["lambda2"] = 0.0
    return variants


def _expected_shared_artifacts(task: str, full_config: dict) -> dict[str, str]:
    base = Path(task) / "pooled_data"
    result = {
        "fit_and_validation_ids": str(base / "fit_and_validation_ids.npz"),
        "sampling": str(base / "sampling.json"),
        "preprocessing": str(base / "preprocessing.json"),
        "selection_counts": str(base / "selection_counts.json"),
    }
    gamma = format(float(full_config["gamma"]), ".17g")
    rank = int(full_config["rank"])
    for seed in MATCHED_SEEDS:
        landmark = base / "landmarks" / f"seed{seed}" / f"rank{rank}" / "landmarks.npz"
        feature_map = landmark.parent / f"gamma_{gamma}" / "map.npz"
        result[f"seed{seed}_landmarks"] = str(landmark)
        result[f"seed{seed}_feature_map"] = str(feature_map)
    return result


def _matched_configuration_index(configurations: dict[str, Any], selection: dict[str, Any],
                                 errors: list[str]) -> dict[str, dict] | None:
    try:
        if configurations.get("base_selection_lock_sha256") != digest(
                selection["base"] / "selection_lock.json"):
            raise ValueError("configurations name a different base selection lock")
        rows = configurations.get("tasks")
        if not isinstance(rows, list) or [row.get("task") for row in rows] != list(TASKS):
            raise ValueError("configurations do not contain the canonical 12-task order")
        protocol_tasks = {row.get("name"): row for row in selection["protocol"].get("tasks", [])}
        final = {(row.get("task"), row.get("method")): row
                 for row in selection["lock"].get("final_selections", [])}
        by_task: dict[str, dict] = {}
        for row in rows:
            task = row["task"]
            protocol_task, base_row = protocol_tasks.get(task), final.get((task, "Full"))
            if protocol_task is None or base_row is None:
                raise ValueError(f"task is absent from base lock: {task}")
            for field in ("site", "mode", "role"):
                if row.get(field) != protocol_task.get(field):
                    raise ValueError(f"task {field} changed: {task}")
            selection_file = Path(task) / "Full" / "final_selection.json"
            disk = _json(selection["base"] / selection_file, errors, "base Full selection")
            if (row.get("base_full_selection_file") != str(selection_file) or
                    row.get("base_full_selection_sha256") != digest(
                        selection["base"] / selection_file) or disk != base_row):
                raise ValueError(f"task does not bind the locked Full selection: {task}")
            full_config = row.get("full_config")
            if full_config != base_row.get("selected_config"):
                raise ValueError(f"Full configuration differs from base lock: {task}")
            if row.get("variants") != _derive_matched_variants(full_config):
                raise ValueError(f"variants change more than the two lambdas: {task}")
            if row.get("shared_artifacts") != _expected_shared_artifacts(task, full_config):
                raise ValueError(f"shared-artifact paths changed: {task}")
            by_task[task] = row
        return by_task
    except (KeyError, TypeError, ValueError, OSError) as error:
        errors.append(f"matched configuration identity invalid: {error}")
        return None


def _verify_render_input_files(mapping: object, selection_root: Path, matched_root: Path,
                               errors: list[str]) -> None:
    expected = {
        **{name: digest(selection_root / name) for name in RENDER_SELECTION_INPUT_FILES},
        **{name: digest(matched_root / name) for name in RENDER_MATCHED_INPUT_FILES},
    }
    if not isinstance(mapping, dict) or len(mapping) != len(expected):
        errors.append("selection-v2 render input-file inventory is incomplete")
        return
    observed: dict[str, str] = {}
    for raw_path, recorded_hash in mapping.items():
        name = Path(raw_path).name
        if name in observed or name not in expected:
            errors.append(f"selection-v2 render has an unexpected input: {raw_path}")
            return
        observed[name] = recorded_hash
    if observed != expected:
        errors.append("selection-v2 render input-file hashes changed")


def _verify_query_records(base: Path, errors: list[str]) -> tuple[int, list[dict[str, Any]]]:
    fit_count = 0
    fit_inventory: list[dict[str, Any]] = []
    try:
        for task in TASKS:
            for method in METHODS:
                signatures = []
                for stage in ("base", "middle", "joint"):
                    path = base / task / method / f"{stage}_selection.json"
                    record = json.loads(path.read_text(encoding="utf-8"))
                    candidates = record.get("candidates")
                    if ((record.get("task"), record.get("method"), record.get("stage")) !=
                            (task, method, stage) or record.get("candidate_count") != 18 or
                            record.get("queried_effective_unique_count") != 18 or
                            not isinstance(candidates, list) or len(candidates) != 18):
                        raise ValueError(f"malformed stage {task}/{method}/{stage}")
                    current = [_effective_signature(method, row.get("config"))
                               for row in candidates]
                    if len(set(current)) != 18:
                        raise ValueError(f"repeated stage query {task}/{method}/{stage}")
                    expected_seeds = list(SELECTION_METHOD_SEEDS[method])
                    for index, candidate in enumerate(candidates):
                        seed_results = candidate.get("seed_results")
                        fit_folders = candidate.get("fit_folders")
                        if (candidate.get("candidate_index") != index or
                                candidate.get("candidate_id") != f"{stage}_c{index:03d}" or
                                candidate.get("queried_stage") != stage or
                                not isinstance(seed_results, list) or
                                [item.get("seed") for item in seed_results] !=
                                expected_seeds or
                                not isinstance(fit_folders, dict) or
                                fit_folders != {str(seed):
                                    f"{task}/{method}/{stage}/candidates/c{index:03d}/seed{seed}"
                                    for seed in expected_seeds}):
                            raise ValueError(
                                f"malformed fit identity {task}/{method}/{stage}/c{index:03d}")
                        failed = []
                        rmse, mae = [], []
                        for seed_result in seed_results:
                            seed = seed_result.get("seed")
                            metrics = seed_result.get("metrics")
                            diagnostics = seed_result.get("diagnostics")
                            if (set(seed_result) != {"seed", "metrics", "diagnostics"} or
                                    not isinstance(metrics, dict) or
                                    not isinstance(diagnostics, dict) or
                                    type(metrics.get("n")) is not int or metrics["n"] <= 0 or
                                    any(not math.isfinite(float(metrics.get(name, math.nan)))
                                        for name in ("rmse", "mae", "r2"))):
                                raise ValueError(
                                    f"malformed fit evidence {task}/{method}/{stage}/c{index:03d}")
                            converged = diagnostics.get("converged")
                            if ((method == "HistGBR" and converged is not None) or
                                    (method != "HistGBR" and type(converged) is not bool)):
                                raise ValueError(
                                    "selection fit has malformed certificate status: "
                                    f"{task}/{method}/{stage}/c{index:03d}/seed{seed}")
                            fit_folder = Path(fit_folders[str(seed)])
                            fit_inventory.append({
                                "key": f"{task}|{method}|{stage}|c{index:03d}|seed{seed}",
                                "fit_record": str(fit_folder / "fit.json"),
                                "validation_metrics": str(
                                    fit_folder / "validation_metrics.json"),
                                "config_sha256": _object_digest(candidate["config"]),
                                "diagnostics_sha256": _object_digest(diagnostics),
                                "metrics_sha256": _object_digest(metrics),
                                "certificate_status": (
                                    "not_applicable_tree" if method == "HistGBR" else
                                    ("passed" if converged else
                                     "failed_ineligible_candidate")
                                ),
                            })
                            rmse.append(float(metrics["rmse"]))
                            mae.append(float(metrics["mae"]))
                            if method != "HistGBR" and not converged:
                                failed.append(seed_result["seed"])
                        mean_rmse = sum(rmse) / len(rmse)
                        mean_mae = sum(mae) / len(mae)
                        std_rmse = math.sqrt(sum((value - mean_rmse) ** 2 for value in rmse) /
                                             len(rmse))
                        if (candidate.get("failed_certificate_seeds") != failed or
                                candidate.get("eligible_for_selection") is not (not failed) or
                                not math.isclose(float(candidate.get("mean_validation_rmse", math.nan)),
                                                 mean_rmse, rel_tol=1e-12, abs_tol=1e-12) or
                                not math.isclose(float(candidate.get("std_validation_rmse", math.nan)),
                                                 std_rmse, rel_tol=1e-12, abs_tol=1e-12) or
                                not math.isclose(float(candidate.get("mean_validation_mae", math.nan)),
                                                 mean_mae, rel_tol=1e-12, abs_tol=1e-12)):
                            raise ValueError(
                                f"inconsistent fit summary {task}/{method}/{stage}/c{index:03d}")
                        fit_count += len(expected_seeds)
                    signatures.extend(current)
                if len(set(signatures)) != 54:
                    raise ValueError(f"not 54 distinct queries for {task}/{method}")
        if fit_count != 10368:
            raise ValueError("selection-fit count is not 10368")
    except (AttributeError, OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError,
            TypeError, ValueError) as error:
        errors.append(f"selection-v2 query structure invalid: {error}")
    return fit_count, sorted(fit_inventory, key=lambda row: row["key"])


def _verify_fit_record_inventory(base: Path,
                                 expected_selection: list[dict[str, Any]],
                                 expected_evaluation: list[dict[str, Any]],
                                 errors: list[str]) -> Path:
    relative = FIT_RECORD_INVENTORY
    inventory = _json(base / relative, errors, "selection fit-record inventory")
    if not isinstance(inventory, dict):
        return relative
    if set(inventory) != {
            "version", "hash_authority", "source_validation", "counts",
            "selection_fit_records", "evaluation_fit_records"}:
        errors.append("selection fit-record inventory schema changed")
    if (inventory.get("version") != FIT_RECORD_INVENTORY_VERSION or
            inventory.get("hash_authority") != FIT_RECORD_HASH_AUTHORITY or
            inventory.get("source_validation") != FIT_RECORD_SOURCE_VALIDATION or
            inventory.get("counts") != {
                "selection_fit_records": 10368,
                "selection_validation_metric_records": 10368,
                "evaluation_fit_records": 576,
            }):
        errors.append("selection fit-record inventory identity/counts changed")

    def check_rows(label: str, observed: object,
                   expected: list[dict[str, Any]], raw_hash_fields: set[str]) -> None:
        if not isinstance(observed, list) or len(observed) != len(expected):
            errors.append(f"{label} fit-record inventory length changed")
            return
        expected_keys = set(expected[0]) | raw_hash_fields if expected else raw_hash_fields
        observed_order = []
        for index, (row, reference) in enumerate(zip(observed, expected)):
            if not isinstance(row, dict) or set(row) != expected_keys:
                errors.append(f"{label} fit-record inventory schema changed at row {index}")
                return
            observed_order.append(row.get("key"))
            if any(row.get(field) != value for field, value in reference.items()):
                errors.append(f"{label} fit-record inventory embedding changed at row {index}")
                return
            if any(not isinstance(row.get(field), str) or
                   not HEX64.fullmatch(row[field]) for field in raw_hash_fields):
                errors.append(f"{label} fit-record inventory SHA-256 changed at row {index}")
                return
        expected_order = [row["key"] for row in expected]
        if observed_order != expected_order or len(set(observed_order)) != len(observed_order):
            errors.append(f"{label} fit-record inventory order/identity changed")

    check_rows(
        "selection", inventory.get("selection_fit_records"), expected_selection,
        {"fit_record_sha256", "validation_metrics_sha256"},
    )
    check_rows(
        "evaluation", inventory.get("evaluation_fit_records"), expected_evaluation,
        {"fit_record_sha256"},
    )
    return relative


def _verify_calibration_csv(path: Path, expected_rows: object,
                            errors: list[str]) -> None:
    if type(expected_rows) is not int or expected_rows <= 0:
        errors.append(f"Invalid selection calibration-row count: {path}")
        return
    count, previous_index, previous_target = 0, -1, -1
    previous_time: tuple[datetime, int] | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CALIBRATION_HEADER:
                raise ValueError("header changed")
            for count, row in enumerate(reader, start=1):
                index = int(row["window_index"])
                target = int(row["raw_target_row"])
                timestamp = _parse_naive_timestamp(row["time"])
                y_true, q025, q975, score = (float(row[name]) for name in
                                              ("y_true", "q025", "q975", "score"))
                expected_score = max(min(q025, q975) - y_true,
                                     y_true - max(q025, q975), 0.0)
                if (index <= previous_index or target <= previous_target or
                        (previous_time is not None and timestamp <= previous_time) or
                        not all(math.isfinite(value) for value in
                                (y_true, q025, q975, score)) or score < 0 or
                        not math.isclose(score, expected_score, rel_tol=1e-10,
                                         abs_tol=1e-10)):
                    raise ValueError("row identity, order, or score changed")
                previous_index, previous_target, previous_time = index, target, timestamp
    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as error:
        errors.append(f"Malformed selection calibration CSV {path}: {error}")
        return
    if count != expected_rows:
        errors.append(f"Selection calibration row count mismatch: {path}")


def _verify_matched_prediction_csv(path: Path, expected_rows: object,
                                   expected_metrics: object,
                                   errors: list[str]) -> None:
    if (type(expected_rows) is not int or expected_rows <= 0 or
            not isinstance(expected_metrics, dict)):
        errors.append(f"Invalid matched prediction summary: {path}")
        return
    count, previous_index, previous_target = 0, -1, -1
    previous_time: tuple[datetime, int] | None = None
    targets, predictions = [], []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != MATCHED_PREDICTION_HEADER:
                raise ValueError("header changed")
            for count, row in enumerate(reader, start=1):
                index, target = int(row["window_index"]), int(row["raw_target_row"])
                timestamp = _parse_naive_timestamp(row["time"])
                y_true, prediction = float(row["y_true"]), float(row["prediction"])
                if (index <= previous_index or target <= previous_target or
                        (previous_time is not None and timestamp <= previous_time) or
                        not math.isfinite(y_true) or not math.isfinite(prediction)):
                    raise ValueError("row identity, order, or prediction changed")
                previous_index, previous_target, previous_time = index, target, timestamp
                targets.append(y_true)
                predictions.append(prediction)
    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as error:
        errors.append(f"Malformed matched prediction CSV {path}: {error}")
        return
    if count != expected_rows:
        errors.append(f"Matched prediction row count mismatch: {path}")
        return
    mean_y = sum(targets) / len(targets)
    squared = sum((truth - prediction) ** 2
                  for truth, prediction in zip(targets, predictions))
    denominator = sum((truth - mean_y) ** 2 for truth in targets)
    observed = {
        "n": len(targets), "rmse": math.sqrt(squared / len(targets)),
        "mae": sum(abs(truth - prediction)
                   for truth, prediction in zip(targets, predictions)) / len(targets),
        "r2": 1 - squared / denominator if denominator else None,
    }
    for name, value in observed.items():
        expected = expected_metrics.get(name)
        if (value is None and expected is not None) or (value is not None and
                (not isinstance(expected, (int, float)) or
                 not math.isclose(value, float(expected), rel_tol=1e-10, abs_tol=1e-10))):
            errors.append(f"Matched prediction metric mismatch ({name}): {path}")


def verify_manifest(root: Path, errors: list[str]) -> tuple[dict[str, tuple[str, int]], set[str]]:
    manifest = root / "MANIFEST.sha256"
    declared: dict[str, tuple[str, int]] = {}
    if not manifest.is_file():
        errors.append("Missing MANIFEST.sha256")
        return declared, set()
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        errors.append(f"Cannot read MANIFEST.sha256: {error}")
        return declared, set()
    for number, line in enumerate(lines, start=1):
        match = LINE.fullmatch(line)
        if not match:
            errors.append(f"Malformed manifest line {number}")
            continue
        expected_hash, expected_bytes, relative = match.groups()
        if _safe(relative, errors, "manifest") is None:
            continue
        if relative in declared:
            errors.append(f"Duplicate manifest path: {relative}")
            continue
        declared[relative] = (expected_hash, int(expected_bytes))
    tree = list(root.rglob("*"))
    for path in tree:
        if path.is_symlink():
            errors.append(f"Symbolic links are not allowed in S1: {path.relative_to(root)}")
    actual = {str(path.relative_to(root)) for path in tree
              if path.is_file() and path != manifest}
    errors.extend(f"Missing payload: {name}" for name in sorted(set(declared) - actual))
    errors.extend(f"Unlisted payload: {name}" for name in sorted(actual - set(declared)))
    for relative in sorted(set(declared) & actual):
        path = root / relative
        if path.is_symlink():
            continue
        expected_hash, expected_bytes = declared[relative]
        if path.stat().st_size != expected_bytes:
            errors.append(f"Byte-count mismatch: {relative}")
        if digest(path) != expected_hash:
            errors.append(f"SHA-256 mismatch: {relative}")
    return declared, actual


def verify_exclusions(root: Path, actual: set[str], errors: list[str]) -> None:
    allowed_v1 = "review/selection_runs/selection_v1/protocol.json"
    selection_v1: list[str] = []
    legacy_paths: list[str] = []
    historical_paths: list[str] = []
    outcome_chosen_paths: list[str] = []
    for relative in sorted(actual):
        path = Path(relative)
        if (any(part == "selection_v1" or part.startswith(("selection_v1_", "selection_v1-"))
                for part in path.parts) and relative != allowed_v1):
            selection_v1.append(relative)
        if any(part == forbidden or part.startswith((forbidden + "_", forbidden + "-"))
               for part in (*path.parts, path.stem) for forbidden in FORBIDDEN_SEGMENTS):
            legacy_paths.append(relative)
        if any(re.fullmatch(r"experiment[1-5](?:[_-].*)?", part, re.IGNORECASE)
               for part in path.parts):
            historical_paths.append(relative)
        if ("test_" + "selected") in relative.lower():
            outcome_chosen_paths.append(relative)
        full = root / relative
        if full.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            payload = full.read_bytes()
        except OSError as error:
            errors.append(f"Cannot scan payload: {relative} ({error})")
            continue
        for forbidden in FORBIDDEN_TEXT:
            if forbidden.encode() in payload:
                errors.append(f"Forbidden legacy content in {relative}: {forbidden}")
    for label, paths in (
        ("Forbidden selection-v1 point artifact", selection_v1),
        ("Forbidden legacy artifact path", legacy_paths),
        ("Forbidden historical experiment path", historical_paths),
        ("Forbidden test-selected path", outcome_chosen_paths),
    ):
        if paths:
            errors.append(f"{label}: {paths[0]} ({len(paths)} file(s))")


def _verify_selection(root: Path, errors: list[str]) -> dict[str, Any] | None:
    base = root / SELECTION
    manifest = _json(base / "manifest.json", errors, "selection-v2 manifest")
    protocol = _json(base / "protocol.json", errors, "selection-v2 protocol")
    lock = _json(base / "selection_lock.json", errors, "selection-v2 lock")
    results = _json(base / "results.json", errors, "selection-v2 results")
    if not (isinstance(manifest, dict) and isinstance(protocol, dict) and
            isinstance(lock, dict) and isinstance(results, list)):
        return None
    if manifest.get("status") != "complete" or manifest.get("last_stable_status") != "complete":
        errors.append("selection-v2 manifest is not complete")
    if protocol.get("version") != "selection_v2" or lock.get("version") != "selection_v2_lock_2":
        errors.append("Unexpected selection-v2 protocol or lock version")
    if lock.get("protocol_sha256") != digest(base / "protocol.json"):
        errors.append("selection-v2 protocol hash mismatch")
    if manifest.get("protocol") != protocol or manifest.get("protocol_sha256") != lock.get("protocol_sha256"):
        errors.append("selection-v2 protocol/manifest/lock mismatch")
    for key in ("source_hashes", "data_hashes", "environment_versions"):
        if manifest.get(key) != lock.get(key):
            errors.append(f"selection-v2 manifest/lock {key} mismatch")
    _hash_mapping(root, lock.get("source_hashes"), errors, "selection source")
    provenance = base / "data_provenance.json"
    if not provenance.is_file() or digest(provenance) != lock.get("data_provenance_sha256"):
        errors.append("selection-v2 data-provenance hash mismatch")
    if lock.get("total_configuration_queries") != 3888:
        errors.append("selection-v2 query count is not 3888")
    if lock.get("validation_query_budget") != protocol.get("validation_query_budget"):
        errors.append("selection-v2 query-budget mismatch")
    if protocol.get("methods") != list(METHODS) or tuple(
            row.get("name") for row in protocol.get("tasks", [])) != TASKS:
        errors.append("selection-v2 task/method design changed")
    if protocol.get("taus") != [0.5, 0.025, 0.975]:
        errors.append("selection-v2 locked evaluation quantiles changed")
    if protocol.get("selection_seeds_by_method") != {
            method: list(seeds) for method, seeds in SELECTION_METHOD_SEEDS.items()}:
        errors.append("selection-v2 selection seed design changed")

    expected_files = {f"{task}/{method}/{stage}_selection.json" for task in TASKS
                      for method in METHODS for stage in STAGES}
    if set(lock.get("selection_files", {})) != expected_files:
        errors.append("selection-v2 selection-file inventory is incomplete or unexpected")
    _hash_mapping(base, lock.get("selection_files"), errors, "selection locked record")
    selection_fit_count, selection_fit_inventory = _verify_query_records(base, errors)
    final = lock.get("final_selections", [])
    final_keys = [(row.get("task"), row.get("method")) for row in final]
    expected_final = {(task, method) for task in TASKS for method in METHODS}
    if len(final_keys) != 72 or len(set(final_keys)) != 72 or set(final_keys) != expected_final:
        errors.append("selection-v2 final-selection inventory is not 12 x 6")

    seed_map = protocol.get("evaluation_seeds_by_method", {})
    expected_results = {(task, method, seed) for task in TASKS for method in METHODS
                        for seed in seed_map.get(method, [])}
    observed = [(row.get("task"), row.get("method"), row.get("seed")) for row in results]
    if len(expected_results) != 192 or len(observed) != 192 or len(set(observed)) != 192 \
            or set(observed) != expected_results:
        errors.append("selection-v2 result inventory is not the locked 192-record design")
    by_key = {(row.get("task"), row.get("method"), row.get("seed")): row for row in results}
    selected = {(row.get("task"), row.get("method")): row.get("selected_config")
                for row in final if isinstance(row, dict)}
    lock_hash = digest(base / "selection_lock.json")
    calibration_paths: set[Path] = set()
    amended_fit_paths: set[Path] = set()
    evaluation_fit_inventory: list[dict[str, Any]] = []
    fit_count = 0
    for task, method, seed in expected_results:
        folder = base / task / method / "evaluation" / f"seed{seed}"
        disk = _json(folder / "result.json", errors, "selection per-seed result")
        if disk != by_key.get((task, method, seed)):
            errors.append(f"selection aggregate/per-seed result mismatch: {folder}")
        if isinstance(disk, dict) and disk.get("selection_lock_sha256") != lock_hash:
            errors.append(f"selection result lock mismatch: {folder}")
        if not (folder / "test_predictions.csv").is_file():
            errors.append(f"Missing selection prediction: {folder}")
        calibration = folder / "calibration_predictions.csv"
        if not calibration.is_file():
            errors.append(f"Missing selection calibration prediction: {folder}")
        else:
            calibration_paths.add(calibration.relative_to(root))
        if isinstance(disk, dict):
            if calibration.is_file():
                _verify_calibration_csv(
                    calibration, disk.get("counts", {}).get("n_calibration"), errors)
            if disk.get("selected_config") != selected.get((task, method)):
                errors.append(f"selection result configuration mismatch: {folder}")
            fits = disk.get("fits")
            if not isinstance(fits, list) or len(fits) != 3:
                errors.append(f"selection result does not contain three tau fits: {folder}")
            elif any(not isinstance(fit, dict) for fit in fits):
                errors.append(f"selection result contains a non-mapping tau fit: {folder}")
            else:
                fit_count += len(fits)
                if [fit.get("tau") for fit in fits] != protocol.get("taus"):
                    errors.append(f"selection result tau-fit order changed: {folder}")
                for fit, tau in zip(fits, protocol.get("taus", [])):
                    expected_path = str((Path(task) / method / "evaluation" / f"seed{seed}" /
                                         ("tau_" + format(float(tau), ".17g"))))
                    fit_relative = Path(expected_path) / "fit.json"
                    expected_fit = {
                        "task": task, "method": method,
                        "config": selected.get((task, method)), "seed": seed,
                        "tau": tau, "candidate_id": "locked_final", "path": expected_path,
                    }
                    if any(fit.get(field) != value for field, value in expected_fit.items()):
                        errors.append(f"selection fit identity changed: {folder}")
                        break
                    embedded = dict(fit)
                    embedded.pop("path", None)
                    diagnostics = embedded.get("diagnostics")
                    if not isinstance(diagnostics, dict):
                        errors.append(f"selection fit diagnostics are malformed: {folder}")
                        break
                    if method != "HistGBR" and diagnostics.get("converged") is not True:
                        errors.append(
                            f"selection kernel fit lacks solver certificate: {fit_relative}")
                    if "solver_certificate_amendment" in diagnostics:
                        amended_fit_paths.add(fit_relative)
                    evaluation_fit_inventory.append({
                        "key": (f"{task}|{method}|seed{seed}|tau"
                                f"{format(float(tau), '.17g')}"),
                        "fit_record": str(fit_relative),
                        "embedded_fit_sha256": _object_digest(embedded),
                        "certificate_status": (
                            "not_applicable_tree" if method == "HistGBR" else "passed"
                        ),
                    })
    if fit_count != 576:
        errors.append("selection-v2 evaluation fit count is not 576")
    fit_record_inventory_path = _verify_fit_record_inventory(
        base, selection_fit_inventory,
        sorted(evaluation_fit_inventory, key=lambda row: row["key"]), errors,
    )
    return {"base": base, "manifest": manifest, "protocol": protocol,
            "lock": lock, "results": results,
            "calibration_paths": calibration_paths,
            "selection_fit_inventory": selection_fit_inventory,
            "evaluation_fit_inventory": sorted(
                evaluation_fit_inventory, key=lambda row: row["key"]),
            "amended_fit_paths": amended_fit_paths,
            "fit_record_inventory_path": fit_record_inventory_path,
            "selection_fit_count": selection_fit_count,
            "evaluation_fit_count": fit_count}


def _verify_selection_analysis(root: Path, selection: dict[str, Any],
                               errors: list[str]) -> dict[str, Any] | None:
    base = root / SELECTION_ANALYSIS
    manifest = _json(base / "analysis_manifest.json", errors, "selection analysis manifest")
    protocol = _json(base / "analysis_protocol.json", errors, "selection analysis protocol")
    if not isinstance(manifest, dict) or not isinstance(protocol, dict):
        return None
    if manifest.get("status") != "complete":
        errors.append("selection-v2 analysis is not complete")
    if (protocol.get("version") != "selection_v2_analysis_2" or
            manifest.get("protocol") != protocol or
            manifest.get("protocol_sha256") != digest(base / "analysis_protocol.json")):
        errors.append("selection-v2 analysis protocol/manifest mismatch")
    source = root / "review/analyze_selection_study_v2.py"
    if not source.is_file() or manifest.get("analysis_source_sha256") != digest(source):
        errors.append("selection-v2 analysis source hash mismatch")
    counts = manifest.get("reporting_counts", {})
    for key, expected in {"tasks": 12, "methods": 6, "locked_result_records": 192,
                          "task_method_rows": 72, "full_comparisons": 60}.items():
        if counts.get(key) != expected:
            errors.append(f"selection-v2 analysis count mismatch: {key}")
    artifacts = _hash_mapping(base, manifest.get("artifact_hashes"), errors,
                              "selection analysis output")
    if {str(path) for path in artifacts} != SELECTION_ANALYSIS_FILES:
        errors.append("selection-v2 analysis artifact inventory changed")
    inputs = _hash_mapping(selection["base"], manifest.get("selection_input_hashes"), errors,
                           "selection analysis input")
    if not {Path("selection_lock.json"), Path("results.json")}.issubset(inputs):
        errors.append("selection-v2 analysis input inventory omits lock/results")
    expected_inputs = {Path("selection_lock.json"), Path("results.json")}
    for task in TASKS:
        for method in METHODS:
            for seed in selection["protocol"]["evaluation_seeds_by_method"][method]:
                folder = Path(task) / method / "evaluation" / f"seed{seed}"
                expected_inputs |= {folder / "result.json", folder / "test_predictions.csv"}
    if inputs != expected_inputs:
        errors.append("selection-v2 analysis input inventory is not the canonical 386 files")
    return {"base": base, "manifest": manifest, "protocol": protocol,
            "artifact_paths": artifacts}


def _verify_matched(root: Path, selection: dict[str, Any], errors: list[str],
                    *, models_declared_omitted: bool) -> dict[str, Any] | None:
    base = root / MATCHED
    manifest = _json(base / "manifest.json", errors, "matched manifest")
    protocol = _json(base / "protocol.json", errors, "matched protocol")
    configurations = _json(base / "configurations.json", errors, "matched configurations")
    lock = _json(base / "matched_lock.json", errors, "matched lock")
    results = _json(base / "results.json", errors, "matched results")
    if not (isinstance(manifest, dict) and isinstance(protocol, dict) and
            isinstance(configurations, dict) and isinstance(lock, dict) and
            isinstance(results, list)):
        return None
    if (manifest.get("status"), manifest.get("heldout_unlocked"),
            manifest.get("completed_evaluation_keys")) != ("complete", True, 144):
        errors.append("matched-ablation-v2 manifest is not complete for 144 keys")
    if (protocol.get("version") != "matched_ablation_v2" or
            configurations.get("version") != "matched_ablation_v2_configurations_1" or
            lock.get("version") != "matched_ablation_v2_lock_1"):
        errors.append("Unexpected matched-ablation-v2 frozen version")
    if (manifest.get("version") != "matched_ablation_v2_manifest_1" or
            protocol.get("base_experiment") != "selection_v2" or
            protocol.get("task_order") != selection["protocol"].get("tasks") or
            protocol.get("variants") != list(VARIANTS) or
            protocol.get("evaluation_seeds") != list(MATCHED_SEEDS) or
            protocol.get("tau") != MATCHED_TAU):
        errors.append("matched-ablation-v2 protocol design changed")
    if lock.get("protocol_sha256") != digest(base / "protocol.json"):
        errors.append("matched protocol hash mismatch")
    if lock.get("configurations_sha256") != digest(base / "configurations.json"):
        errors.append("matched configurations hash mismatch")
    _hash_mapping(root, lock.get("source_hashes"), errors, "matched source")
    if (lock.get("environment_versions") != protocol.get("environment_versions") or
            manifest.get("protocol_sha256") != lock.get("protocol_sha256") or
            manifest.get("configurations_sha256") != lock.get("configurations_sha256")):
        errors.append("matched-ablation-v2 frozen identities disagree")
    frozen_base = lock.get("base_selection", {})
    expected_base = {
        "selection_lock_sha256": digest(selection["base"] / "selection_lock.json"),
        "protocol_sha256": selection["lock"].get("protocol_sha256"),
        "source_hashes": selection["lock"].get("source_hashes"),
        "data_hashes": selection["lock"].get("data_hashes"),
        "environment_versions": selection["lock"].get("environment_versions"),
        "data_provenance_sha256": selection["lock"].get("data_provenance_sha256"),
    }
    for key, expected in expected_base.items():
        if frozen_base.get(key) != expected:
            errors.append(f"matched/base selection identity mismatch: {key}")
    fit_only = _hash_mapping(selection["base"], lock.get("fit_only_artifact_hashes"), errors,
                             "matched fit-only")
    configuration_tasks = configurations.get("tasks")
    task_records = _matched_configuration_index(configurations, selection, errors)
    declared_shared: set[Path] = set()
    if not isinstance(configuration_tasks, list) or len(configuration_tasks) != 12:
        errors.append("matched configurations do not contain 12 tasks")
    else:
        for task in configuration_tasks:
            shared = task.get("shared_artifacts")
            if not isinstance(shared, dict) or len(shared) != 10:
                errors.append("a matched task does not name 10 shared fit-only artifacts")
                continue
            for value in shared.values():
                relative = _safe(value, errors, "matched shared artifact")
                if relative is not None:
                    declared_shared.add(relative)
    if fit_only != declared_shared or len(fit_only) != 120:
        errors.append("matched fit-only inventory is not the locked 12 x 10 design")
    expected_keys = [f"{task}|{variant}|seed{seed}" for task in TASKS
                     for variant in VARIANTS for seed in range(3)]
    if lock.get("expected_evaluation_keys") != expected_keys:
        errors.append("matched lock does not declare the canonical 144 keys")
    if manifest.get("matched_lock_sha256") != digest(base / "matched_lock.json"):
        errors.append("matched manifest lock hash mismatch")
    if manifest.get("base_selection_lock_sha256") != digest(
            selection["base"] / "selection_lock.json"):
        errors.append("matched manifest base-selection lock hash mismatch")
    if manifest.get("results_sha256") != digest(base / "results.json"):
        errors.append("matched results hash mismatch")
    keys = [row.get("evaluation_key") for row in results]
    if len(results) != 144 or len(set(keys)) != 144 or keys != expected_keys:
        errors.append("matched results do not match the locked 144-key order")
    if task_records is None:
        return None
    by_key = {row.get("evaluation_key"): row for row in results}
    result_paths: set[Path] = set()
    amended_fit_paths: set[Path] = set()
    retained_amendment_models: set[Path] = set()
    amendment_lock_path = root / AMENDMENT_LOCK
    if amendment_lock_path.is_file():
        amendment = _json(amendment_lock_path, errors, "solver amendment lock")
        if isinstance(amendment, dict):
            for raw in amendment.get("preexisting_completion_artifacts", {}):
                relative = _safe(raw, errors, "pre-amendment artifact")
                if relative is not None and relative.is_relative_to(MATCHED):
                    within = relative.relative_to(MATCHED)
                    if within.name == "model.npz":
                        retained_amendment_models.add(within)
    for key in expected_keys:
        task, variant, seed_name = key.split("|")
        seed = int(seed_name.removeprefix("seed"))
        task_record = task_records[task]
        folder = base / "evaluation" / task / variant / seed_name
        row = by_key.get(key, {})
        disk = _json(folder / "result.json", errors, "matched per-key result")
        if disk != row:
            errors.append(f"matched aggregate/per-key result mismatch: {key}")
        expected_identity = {
            "task": task, "site": task_record["site"], "mode": task_record["mode"],
            "role": task_record["role"], "variant": variant, "seed": seed,
            "tau": MATCHED_TAU, "config": task_record["variants"][variant],
            "matched_lock_sha256": digest(base / "matched_lock.json"),
            "base_selection_lock_sha256": digest(selection["base"] / "selection_lock.json"),
            "shared_artifacts": task_record["shared_artifacts"],
            "feature_map": task_record["shared_artifacts"][f"seed{seed}_feature_map"],
            "data_folder": str(Path(task_record["shared_artifacts"]["preprocessing"]).parent),
        }
        for field, expected in expected_identity.items():
            if row.get(field) != expected:
                errors.append(f"matched result {field} mismatch: {key}")
        expected_files = {
            "fit_record": str((folder / "fit.json").relative_to(base)),
            "prediction_file": str((folder / "test_predictions.csv").relative_to(base)),
            "model_file": str((folder / "model.npz").relative_to(base)),
        }
        result_paths.add((folder / "result.json").relative_to(base))
        for field, expected in expected_files.items():
            if row.get(field) != expected:
                errors.append(f"matched noncanonical {field}: {key}")
        for field, hash_field in (("fit_record", "fit_record_sha256"),
                                  ("prediction_file", "prediction_sha256")):
            relative = _safe(row.get(field, ""), errors, f"matched {field}")
            if relative is None:
                continue
            path = base / relative
            if not path.is_file() or digest(path) != row.get(hash_field):
                errors.append(f"matched {field} hash mismatch: {key}")
            result_paths.add(relative)
        prediction_relative = _safe(row.get("prediction_file", ""), errors,
                                    "matched prediction")
        if prediction_relative is not None and (base / prediction_relative).is_file():
            _verify_matched_prediction_csv(
                base / prediction_relative, row.get("counts", {}).get("n_test"),
                row.get("point"), errors)
        # Model binaries are an explicitly indexed omission. If supplied, they
        # still have to match the result record.
        model = _safe(row.get("model_file", ""), errors, "matched model")
        if model is not None:
            model_path = base / model
            if (model_path.exists() and models_declared_omitted and
                    model not in retained_amendment_models):
                errors.append(f"matched model present despite indexed omission: {key}")
            elif model_path.exists() and digest(model_path) != row.get("model_sha256"):
                errors.append(f"matched model hash mismatch: {key}")
            elif not model_path.exists() and not models_declared_omitted:
                errors.append(f"matched model missing without an indexed omission: {key}")
        fit = _json(folder / "fit.json", errors, "matched fit record")
        if isinstance(fit, dict):
            expected_fit = {
                "task": task, "method": variant, "config": task_record["variants"][variant],
                "seed": seed, "tau": MATCHED_TAU,
                "candidate_id": "matched_ablation_v2_locked",
                "feature_map": expected_identity["feature_map"],
                "data_folder": expected_identity["data_folder"],
            }
            for field, expected in expected_fit.items():
                if fit.get(field) != expected:
                    errors.append(f"matched fit {field} mismatch: {key}")
            if fit.get("diagnostics", {}).get("converged") is not True:
                errors.append(f"matched fit lacks convergence certificate: {key}")
            if row.get("diagnostics") != fit.get("diagnostics"):
                errors.append(f"matched result/fit diagnostics mismatch: {key}")
            if "solver_certificate_amendment" in fit.get("diagnostics", {}):
                amended_fit_paths.add(Path(expected_files["fit_record"]))
    return {"base": base, "manifest": manifest, "protocol": protocol,
            "configurations": configurations, "lock": lock, "results": results,
            "fit_only_paths": fit_only, "result_paths": result_paths,
            "amended_fit_paths": amended_fit_paths}


def _verify_matched_analysis(root: Path, matched: dict[str, Any],
                             errors: list[str]) -> dict[str, Any] | None:
    base = root / MATCHED_ANALYSIS
    protocol = _json(base / "analysis_protocol.json", errors, "matched analysis protocol")
    lock = _json(base / "analysis_lock.json", errors, "matched analysis lock")
    manifest = _json(base / "manifest.json", errors, "matched analysis manifest")
    summary = _json(base / "summary.json", errors, "matched analysis summary")
    if not all(isinstance(value, dict) for value in (protocol, lock, manifest, summary)):
        return None
    if manifest.get("status") != "complete" or manifest.get("heldout_results_read") is not True:
        errors.append("matched-ablation-v2 analysis is not complete")
    if (protocol.get("version") != "matched_ablation_v2_analysis_1" or
            lock.get("version") != "matched_ablation_v2_analysis_lock_1"):
        errors.append("Unexpected matched-ablation-v2 analysis version")
    if manifest.get("version") != "matched_ablation_v2_analysis_manifest_1":
        errors.append("Unexpected matched-ablation-v2 analysis manifest version")
    if lock.get("analysis_protocol_sha256") != digest(base / "analysis_protocol.json"):
        errors.append("matched analysis protocol hash mismatch")
    _hash_mapping(root, lock.get("source_hashes"), errors, "matched analysis source")
    if (lock.get("environment_versions") != protocol.get("environment_versions") or
            lock.get("matched_protocol_sha256") != matched["lock"].get("protocol_sha256") or
            lock.get("matched_configurations_sha256") != matched["lock"].get("configurations_sha256") or
            lock.get("expected_evaluation_keys") != matched["lock"].get("expected_evaluation_keys")):
        errors.append("matched analysis lock disagrees with its experiment or protocol")
    if lock.get("matched_lock_sha256") != digest(matched["base"] / "matched_lock.json"):
        errors.append("matched analysis lock points to a different experiment")
    if manifest.get("analysis_lock_sha256") != digest(base / "analysis_lock.json"):
        errors.append("matched analysis manifest lock hash mismatch")
    outputs = _hash_mapping(base, manifest.get("output_hashes"), errors,
                            "matched analysis output")
    if set(protocol.get("output_files", [])) != {str(path) for path in outputs} or len(outputs) != 7:
        errors.append("matched analysis output inventory is not the declared seven files")
    if (summary.get("version") != "matched_ablation_v2_analysis_results_1" or
            summary.get("analysis_lock_sha256") != digest(base / "analysis_lock.json") or
            summary.get("matched_lock_sha256") != digest(matched["base"] / "matched_lock.json") or
            summary.get("matched_results_sha256") != digest(matched["base"] / "results.json") or
            summary.get("complete_evaluation_keys") != 144 or
            summary.get("task_metric_rows") != 48 or
            summary.get("paired_tests") != {"within": 18, "cross": 18}):
        errors.append("matched analysis identity or design counts changed")
    return {"base": base, "manifest": manifest, "output_paths": outputs}


def _verify_solver_amendment(root: Path, selection: dict[str, Any],
                             matched: dict[str, Any],
                             index: dict[str, Any] | None,
                             errors: list[str]) -> dict[str, set[Path]]:
    """Verify the one performance-blind certificate amendment and its audit chain."""
    empty = {"selection_paths": set(), "matched_paths": set()}
    main_amended = set(selection.get("amended_fit_paths", set()))
    matched_amended = set(matched.get("amended_fit_paths", set()))
    lock_path = root / AMENDMENT_LOCK
    declared = index.get("solver_certificate_amendment") \
        if isinstance(index, dict) else None

    if not main_amended and not matched_amended:
        if lock_path.exists() or declared is not None:
            errors.append(
                "A solver amendment is packaged or declared, but no fit embeds its provenance")
        return empty
    if len(main_amended) != 1 or matched_amended:
        errors.append(
            "The archived study must contain exactly one amended main fit and no amended "
            "matched fit")
        return empty

    runner_path = root / AMENDMENT_RUNNER
    if not runner_path.is_file() or not lock_path.is_file():
        errors.append("The solver-certificate amendment runner or lock is missing")
        return empty

    amendment = _json(lock_path, errors, "solver amendment lock")
    if not isinstance(amendment, dict):
        return empty

    if amendment.get("version") != AMENDMENT_VERSION:
        errors.append("Unexpected solver-certificate amendment version")
    expected_rule = [{
        "max_iter": 96000, "gap_rtol": 0.001,
        "gap_atol": 1e-7, "check_every": 25,
    }]
    if amendment.get("rule", {}).get("conditional_warm_restart") != expected_rule:
        errors.append("The solver-amendment warm-restart rule changed")
    if (amendment.get("amendment_runner") != str(AMENDMENT_RUNNER) or
            amendment.get("amendment_runner_sha256") != digest(runner_path)):
        errors.append("The frozen solver-amendment runner changed")
    if (amendment.get("selection", {}).get("output") != str(SELECTION) or
            amendment.get("matched", {}).get("output") != str(MATCHED) or
            amendment.get("selection", {}).get("selection_lock_sha256") !=
            digest(selection["base"] / "selection_lock.json") or
            amendment.get("selection", {}).get("protocol_sha256") !=
            digest(selection["base"] / "protocol.json") or
            amendment.get("selection", {}).get("source_hashes") !=
            selection["lock"].get("source_hashes") or
            amendment.get("selection", {}).get("data_hashes") !=
            selection["lock"].get("data_hashes") or
            amendment.get("matched", {}).get("matched_lock_sha256") !=
            digest(matched["base"] / "matched_lock.json") or
            amendment.get("matched", {}).get("source_hashes") !=
            matched["lock"].get("source_hashes")):
        errors.append("The solver amendment no longer binds the frozen studies")

    completion_commitments = amendment.get("preexisting_completion_artifacts", {})
    if (not isinstance(completion_commitments, dict) or
            len(completion_commitments) != 110 or
            amendment.get("partial_evaluation_disclosure", {}).get(
                "main_result_records_already_present") != 10 or
            amendment.get("partial_evaluation_disclosure", {}).get(
                "matched_result_records_already_present") != 5):
        errors.append("The pre-amendment completion commitment inventory changed")
        completion_commitments = {}
    for raw_relative, expected_hash in completion_commitments.items():
        relative = _safe(raw_relative, errors, "pre-amendment artifact")
        if relative is None:
            continue
        path = root / relative
        if not path.is_file() or digest(path) != expected_hash:
            errors.append(f"A pre-amendment completion artifact changed: {relative}")

    allowed_manifest_transitions = {
        SELECTION_ANALYSIS / "analysis_manifest.json",
        MATCHED_ANALYSIS / "manifest.json",
    }
    expected_analysis_commitments = {
        SELECTION_ANALYSIS / "analysis_protocol.json",
        SELECTION_ANALYSIS / "analysis_manifest.json",
        MATCHED_ANALYSIS / "analysis_protocol.json",
        MATCHED_ANALYSIS / "analysis_lock.json",
        MATCHED_ANALYSIS / "manifest.json",
    }
    analysis_commitments = amendment.get("preexisting_analysis_freezes", {})
    observed_analysis_paths = {
        relative for raw in analysis_commitments
        if (relative := _safe(raw, errors, "pre-amendment analysis file")) is not None
    } if isinstance(analysis_commitments, dict) else set()
    if observed_analysis_paths != expected_analysis_commitments:
        errors.append("The pre-amendment analysis commitment inventory changed")
        analysis_commitments = {}
    observed_manifest_transitions: set[Path] = set()
    for raw_relative, expected_hash in analysis_commitments.items():
        relative = _safe(raw_relative, errors, "pre-amendment analysis file")
        if relative is None:
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"A committed analysis file is missing: {relative}")
        elif digest(path) != expected_hash:
            if relative not in allowed_manifest_transitions:
                errors.append(
                    f"An immutable pre-amendment analysis file changed: {relative}")
            else:
                observed_manifest_transitions.add(relative)
    if observed_manifest_transitions != allowed_manifest_transitions:
        errors.append("The pre-amendment analysis-manifest transition inventory changed")

    trigger_relative = _safe(
        amendment.get("trigger", {}).get("fit_record", ""), errors,
        "solver amendment trigger")
    if trigger_relative is None or not trigger_relative.is_relative_to(SELECTION):
        errors.append("The solver-amendment trigger is not a main-study fit")
        return empty
    trigger_within_selection = trigger_relative.relative_to(SELECTION)
    if main_amended != {trigger_within_selection}:
        errors.append(
            "Embedded amended-fit provenance disagrees with the amendment trigger")
    trigger_folder = root / trigger_relative.parent
    try:
        fit_path = trigger_folder / "fit.json"
        model_path = trigger_folder / "model.npz"
        fit_archive = trigger_folder / "fit.pre_amendment.json"
        model_archive = trigger_folder / "model.pre_amendment.npz"
        intent_path = trigger_folder / "solver_amendment_intent.json"
        completion_path = trigger_folder / "solver_amendment.json"
        if not all(path.is_file() for path in (
                fit_path, model_path, fit_archive, model_archive,
                intent_path, completion_path)):
            raise ValueError("the amended fit lacks its six-file audit chain")
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        original_fit = json.loads(fit_archive.read_text(encoding="utf-8"))
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        amendment_hash = digest(lock_path)
        relative_folder = str(trigger_relative.parent)
        for sidecar, label in ((intent, "intent"), (completion, "completion")):
            if (sidecar.get("version") != AMENDMENT_VERSION or
                    sidecar.get("fit_folder") != relative_folder or
                    sidecar.get("amendment_lock_sha256") != amendment_hash):
                raise ValueError(f"the solver-amendment {label} identity changed")
        for key in ("original_fit_sha256", "original_model_sha256"):
            if intent.get(key) != completion.get(key):
                raise ValueError(f"original hash accounting changed: {key}")
        if (digest(fit_archive) != intent.get("original_fit_sha256") or
                digest(model_archive) != intent.get("original_model_sha256") or
                digest(fit_path) != completion.get("amended_fit_sha256") or
                digest(model_path) != completion.get("amended_model_sha256")):
            raise ValueError("an original or amended solver artifact changed")
        trigger = amendment.get("trigger", {})
        if (trigger.get("fit_record_sha256") != digest(fit_archive) or
                trigger.get("model_sha256") != digest(model_archive) or
                trigger.get("identity") != {
                    key: original_fit.get(key)
                    for key in ("task", "method", "seed", "tau")
                }):
            raise ValueError("the archived optimizer-only trigger changed")
        original_diagnostics = original_fit.get("diagnostics", {})
        if any(original_diagnostics.get(key) != value for key, value in
               trigger.get("optimizer_diagnostics", {}).items()):
            raise ValueError("the solver-amendment trigger diagnostics changed")
        for field in (
            "task", "site", "method", "config", "seed", "tau", "candidate_id",
            "feature_map", "data_folder", "selection_counts",
            "effective_target_fraction",
        ):
            if original_fit.get(field) != fit.get(field):
                raise ValueError(f"the amendment changed locked fit identity: {field}")
        diagnostics = fit.get("diagnostics", {})
        expected_provenance = {
            "version": AMENDMENT_VERSION,
            "amendment_lock": str(AMENDMENT_LOCK),
            "amendment_lock_sha256": amendment_hash,
            "original_fit_sha256": intent["original_fit_sha256"],
            "original_model_sha256": intent["original_model_sha256"],
            "extension_attempts_used": 1,
            "gap_tolerances_changed": False,
        }
        if diagnostics.get("solver_certificate_amendment") != expected_provenance:
            raise ValueError("the amended-fit diagnostics provenance changed")
        reports = completion.get("extension_attempts")
        if (completion.get("converged") is not True or
                diagnostics.get("converged") is not True or
                not isinstance(reports, list) or len(reports) != 1):
            raise ValueError("the amended fit lacks its completed certificate")
        report = reports[0]
        for key, expected in expected_rule[0].items():
            if report.get(key) != expected:
                raise ValueError(f"the restart rule changed: {key}")
        if (type(report.get("iterations")) is not int or
                not 0 <= report["iterations"] <= expected_rule[0]["max_iter"] or
                report.get("converged") is not True or
                diagnostics.get("attempt_count") != 3 or
                len(diagnostics.get("attempt_summaries", [])) != 3 or
                diagnostics["attempt_summaries"][-1] != report or
                any(diagnostics.get(key) != value for key, value in report.items()
                    if key not in ("max_iter", "check_every"))):
            raise ValueError("the amended-fit attempt accounting changed")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"Completed solver-amendment chain validation failed: {error}")

    expected_sidecar = trigger_folder / "solver_amendment.json"
    main_sidecars = sorted(selection["base"].rglob("solver_amendment.json"))
    matched_sidecars = sorted(matched["base"].rglob("solver_amendment.json"))
    if main_sidecars != [expected_sidecar] or matched_sidecars:
        errors.append("The completed solver-amendment sidecar inventory changed")

    selection_paths: set[Path] = set()
    matched_paths: set[Path] = set()
    for raw_relative in completion_commitments:
        relative = _safe(raw_relative, errors, "pre-amendment artifact")
        if relative is None:
            continue
        if relative.is_relative_to(SELECTION):
            selection_paths.add(relative.relative_to(SELECTION))
        elif relative.is_relative_to(MATCHED):
            matched_paths.add(relative.relative_to(MATCHED))
        else:
            errors.append(f"Unexpected pre-amendment artifact root: {relative}")

    for name in (
        "fit.json", "model.npz", "fit.pre_amendment.json",
        "model.pre_amendment.npz", "solver_amendment_intent.json",
        "solver_amendment.json",
    ):
        path = trigger_folder / name
        if not path.is_file():
            errors.append(f"Amended-fit audit artifact is missing: {path}")
        else:
            selection_paths.add(path.relative_to(selection["base"]))

    supersession_declaration = amendment.get("superseded_preexecution_draft", {})
    supersession_relative = _safe(
        supersession_declaration.get("record", ""), errors,
        "solver amendment supersession")
    if supersession_relative is not None:
        supersession_path = root / supersession_relative
        supersession = _json(
            supersession_path, errors, "solver amendment supersession")
        if (not supersession_path.is_file() or
                digest(supersession_path) !=
                supersession_declaration.get("record_sha256") or
                not isinstance(supersession, dict) or
                supersession.get("executed_fit_count") != 0):
            errors.append("The unexecuted amendment-draft supersession record changed")
        elif (draft_relative := _safe(
                supersession.get("superseded_lock", ""), errors,
                "superseded solver amendment draft")) is not None:
            draft_path = root / draft_relative
            if (not draft_path.is_file() or
                    digest(draft_path) != supersession.get("superseded_lock_sha256")):
                errors.append("The unexecuted superseded amendment draft changed")

    expected_declaration = {
        "version": AMENDMENT_VERSION,
        "lock": str(AMENDMENT_LOCK),
        "lock_sha256": digest(lock_path),
        "amended_main_fits": 1,
        "amended_matched_fits": 0,
        "advanced_analysis_manifests": sorted(
            str(path) for path in allowed_manifest_transitions),
    }
    if declared != expected_declaration:
        errors.append("PACKAGE_INDEX solver-amendment declaration changed")
    return {"selection_paths": selection_paths, "matched_paths": matched_paths}


def _verify_render(root: Path, selection_analysis: dict[str, Any],
                   matched_analysis: dict[str, Any],
                   errors: list[str]) -> dict[str, Any] | None:
    base = root / RENDER
    manifest = _json(base / "manifest.json", errors, "selection-v2 render manifest")
    if not isinstance(manifest, dict):
        return
    if manifest.get("version") != "selection_v2_render_manifest_1" or manifest.get("status") != "complete":
        errors.append("selection-v2 render is not a complete supported version")
    source = root / "review/render_selection_study_v2.py"
    if not source.is_file() or manifest.get("source_sha256") != digest(source):
        errors.append("selection-v2 render source hash mismatch")
    inputs = manifest.get("inputs", {})
    if inputs.get("selection_analysis_manifest_sha256") != digest(
            selection_analysis["base"] / "analysis_manifest.json"):
        errors.append("render selection-analysis manifest hash mismatch")
    if inputs.get("matched_analysis_manifest_sha256") != digest(
            matched_analysis["base"] / "manifest.json"):
        errors.append("render matched-analysis manifest hash mismatch")
    _verify_render_input_files(inputs.get("files"), selection_analysis["base"],
                               matched_analysis["base"], errors)
    expected = {"selection_task_method_rows": 72, "selection_scope_rows": 12,
                "selection_full_inference_rows": 60, "matched_task_variant_rows": 48,
                "matched_aggregate_rows": 8, "matched_inference_rows": 36}
    if manifest.get("validated_counts") != expected:
        errors.append("selection-v2 render completeness counts changed")
    outputs = _hash_mapping(base, manifest.get("outputs"), errors,
                            "selection-v2 render output")
    if {str(path) for path in outputs} != RENDER_FILES:
        errors.append("selection-v2 render output inventory changed")
    return {"base": base, "output_paths": outputs}


def _planned_chronological(protocol: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    jobs: list[str] = []
    aliases: dict[str, str] = {}
    for seed in protocol["seeds"]:
        for variant in protocol["components"]:
            jobs.extend(f"components_{variant}_{task}_s{seed}"
                        for task in ("point", "interval"))
        jobs.extend(f"scarcity_n{budget}_s{seed}" for budget in protocol["label_budgets"]
                    if budget != "all")
    for scheme in protocol["weight_schemes"]:
        if scheme != "domain_balanced":
            jobs.extend(f"weights_{scheme}_{task}" for task in ("point", "interval"))
    for quantile in protocol["threshold_quantiles"]:
        for task in ("point", "interval"):
            name = f"threshold_q{quantile}_{task}"
            if quantile == 98:
                aliases[name] = f"components_Full_{task}_s0"
            else:
                jobs.append(name)
    for seed in protocol["seeds"]:
        for landmarks in protocol["landmark_grid"]:
            name = f"landmarks_m{landmarks}_s{seed}"
            if landmarks == protocol["model"]["landmarks"]:
                aliases[name] = f"components_Full_point_s{seed}"
            else:
                jobs.append(name)
    for fraction in protocol["source_fractions"]:
        name = f"scale_fraction{fraction:g}"
        if fraction == 1:
            aliases[name] = "components_Full_point_s0"
        else:
            jobs.append(name)
    for site in protocol["cross_farm_sites"]:
        for variant in ("Full", "TargetOnly"):
            jobs.extend(f"crossfarm_site{site}_{variant}_{task}"
                        for task in ("point", "interval"))
    return jobs, aliases


def _declared_subtree(root: Path, subtree: Path, mapping: object,
                      expected: set[Path], errors: list[str], label: str) -> None:
    declared = _hash_mapping(root, mapping, errors, label)
    expected_full = {subtree / path for path in expected}
    if declared != expected_full:
        errors.append(f"{label} is not the exact canonical inventory")
    base = root / subtree
    actual = {path.relative_to(root) for path in base.rglob("*") if path.is_file()} \
        if base.is_dir() else set()
    if actual != expected_full:
        errors.append(f"{label} subtree contains missing or extra files")


def _verify_chronological(root: Path, evidence: object,
                          errors: list[str]) -> dict[str, Any] | None:
    base = root / CHRONOLOGICAL
    manifest = _json(base / "manifest.json", errors, "chronological manifest")
    protocol = _json(root / "review/replied_study_protocol.json", errors,
                     "chronological protocol")
    definitions = _json(base / "threshold_definitions.json", errors,
                        "chronological threshold definitions")
    verification = _json(base / "verification.json", errors, "chronological verification")
    if not all(isinstance(value, dict) for value in (manifest, protocol, verification)) \
            or not isinstance(definitions, list):
        return None
    if manifest.get("status") != "complete" or manifest.get("protocol") != protocol:
        errors.append("chronological-v1 is incomplete or its protocol changed")
    _hash_mapping(root, manifest.get("source_hashes"), errors, "chronological source")
    data_hashes = manifest.get("data_hashes")
    if (not isinstance(data_hashes, dict) or len(data_hashes) != 6 or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in data_hashes.values())):
        errors.append("chronological raw-data hash inventory is not six SHA-256 values")
    try:
        jobs, aliases = _planned_chronological(protocol)
        for row in definitions:
            if not isinstance(row, dict):
                raise TypeError("threshold-definition row is not an object")
            if row.get("identical_domain_mask_to_25") is False:
                jobs.extend(f"threshold_q{row['q']}_cutout{row['cutout_ms']}_{task}"
                            for task in ("point", "interval"))
    except (AttributeError, KeyError, TypeError) as error:
        errors.append(f"chronological protocol is malformed: {error}")
        return None
    if (len(jobs), len(set(jobs)), len(aliases), manifest.get("completed_fits")) != \
            (94, 94, 6, 94):
        errors.append("chronological job/alias/completion counts changed")
    verification_reports: set[Path] = set()
    branches = verification.get("branches")
    verification_aliases = verification.get("aliases")
    global_checks = verification.get("global_checks")
    if all(isinstance(rows, list) for rows in
           (branches, verification_aliases, global_checks)):
        if any(not isinstance(row, dict)
               for rows in (branches, verification_aliases, global_checks)
               for row in rows):
            errors.append("chronological verification rows are malformed")
            return None
        branch_names = {
            row.get("name") for row in branches if row.get("status") == "passed"
        }
        alias_names = {
            row.get("name") for row in verification_aliases
            if row.get("status") == "passed"
        }
        if (verification.get("status") != "passed" or verification.get("pending") or
                branch_names != set(jobs) or alias_names != set(aliases) or
                any(row.get("status") != "passed" for row in global_checks)):
            errors.append(
                "chronological independent verification is not complete and passed")
    else:
        report_names = {
            "branch_verification_report": "verification_stage1.json",
            "completion_verification_report": "verification_crossfarm_final.json",
            "independent_prediction_summary": "study_summary.json",
        }
        hashes = verification.get("source_reports_sha256")
        if (verification.get("status") != "passed" or
                verification.get("completed_experiment_branches") != len(jobs) or
                verification.get("completed_quantile_model_fits") != 123 or
                verification.get("aliases") != len(aliases) or
                any(verification.get(key) != value
                    for key, value in report_names.items()) or
                not isinstance(hashes, dict) or
                set(hashes) != set(report_names.values())):
            errors.append("chronological verification summary is incomplete")
        reports: dict[str, dict[str, Any]] = {}
        for name in sorted(set(report_names.values())):
            path = base / name
            report = _json(path, errors, f"chronological {name}")
            if (not isinstance(report, dict) or not isinstance(hashes, dict) or
                    not isinstance(hashes.get(name), str) or
                    not path.is_file() or digest(path) != hashes.get(name)):
                errors.append(f"chronological verification report hash mismatch: {name}")
            else:
                reports[name] = report
                verification_reports.add(Path(name))

        stage1 = reports.get("verification_stage1.json")
        if isinstance(stage1, dict):
            rows = (stage1.get("branches"), stage1.get("aliases"),
                    stage1.get("global_checks"))
            if (not all(isinstance(value, list) for value in rows) or
                    any(not isinstance(row, dict)
                        for value in rows if isinstance(value, list) for row in value)):
                errors.append("chronological stage-1 verification rows are malformed")
            else:
                stage_branches, stage_aliases, stage_globals = rows
                if (stage1.get("status") != "passed_partial" or stage1.get("pending") or
                        stage1.get("passed_branches") != len(jobs) or
                        {row.get("name") for row in stage_branches
                         if row.get("status") == "passed"} != set(jobs) or
                        {row.get("name") for row in stage_aliases
                         if row.get("status") == "passed"} != set(aliases) or
                        any(row.get("status") != "passed" for row in stage_globals) or
                        any(aliases.get(row.get("name")) != row.get("reference")
                            for row in stage_aliases)):
                    errors.append("chronological stage-1 verification is incomplete")

        completion = reports.get("verification_crossfarm_final.json")
        if isinstance(completion, dict):
            rows = (completion.get("branches"), completion.get("aliases"),
                    completion.get("global_checks"))
            if (not all(isinstance(value, list) for value in rows) or
                    any(not isinstance(row, dict)
                        for value in rows if isinstance(value, list) for row in value)):
                errors.append("chronological completion verification rows are malformed")
            else:
                completion_branches, completion_aliases, completion_globals = rows
                cross_jobs = {name for name in jobs if name.startswith("crossfarm_")}
                if (completion.get("status") != "passed" or
                        completion.get("pending") or
                        completion.get("manifest_status_at_start") != "complete" or
                        completion.get("passed_branches") != len(cross_jobs) or
                        completion_aliases or
                        {row.get("name") for row in completion_branches
                         if row.get("status") == "passed"} != cross_jobs or
                        any(row.get("status") != "passed"
                            for row in completion_globals)):
                    errors.append("chronological completion verification is incomplete")

        summary = reports.get("study_summary.json")
        if isinstance(summary, dict):
            prediction = summary.get("prediction_verification")
            available = summary.get("available_branches_at_start")
            if (summary.get("manifest_status") != "complete" or
                    summary.get("completed_unique_branches") != len(jobs) or
                    summary.get("completed_unique_quantile_fits") != 123 or
                    summary.get("missing_expected_branches") or summary.get("errors") or
                    summary.get("aliases") != aliases or
                    not isinstance(available, list) or
                    set(available) != set(jobs) | set(aliases) or
                    not isinstance(prediction, dict) or set(prediction) != set(jobs)):
                errors.append("chronological independent prediction summary is incomplete")
            elif isinstance(prediction, dict):
                for name, row in prediction.items():
                    metric_checks = row.get("metrics_match_predictions") \
                        if isinstance(row, dict) else None
                    result_path = base / name / "result.json"
                    prediction_path = base / name / "predictions.csv"
                    if (not isinstance(row, dict) or not result_path.is_file() or
                            not prediction_path.is_file() or
                            row.get("result_sha256") != digest(result_path) or
                            row.get("predictions_sha256") !=
                            digest(prediction_path) or
                            not isinstance(metric_checks, dict) or not metric_checks or
                            not all(value is True for value in metric_checks.values())):
                        errors.append(
                            f"chronological independent prediction check changed: {name}")
                        break
    expected = {
        Path("manifest.json"), Path("summary.csv"), Path("threshold_definitions.json"),
        Path("verification.json"),
        *(Path("data") / f"farm{site}_{suffix}"
          for site in range(1, 7) for suffix in ("audit.json", "ids.npz")),
    }
    expected |= verification_reports
    for name in jobs:
        folder = Path(name)
        expected |= {folder / "result.json", folder / "predictions.csv",
                     folder / "preprocessing.json", folder / "fit_ids.npz"}
        if "interval" in name:
            expected.add(folder / "calibration.csv")
    expected |= {Path(name) / "alias.json" for name in aliases}
    if not isinstance(evidence, dict):
        errors.append("Missing chronological evidence inventory")
    else:
        if (evidence.get("artifact_hash_authority") != PACKAGE_TIME_SNAPSHOT or
                evidence.get("jobs") != 94 or evidence.get("aliases") != 6):
            errors.append("chronological evidence counts changed")
        _declared_subtree(root, CHRONOLOGICAL, evidence.get("files"), expected,
                          errors, "chronological evidence")
    return {"base": base, "manifest": manifest, "jobs": jobs, "aliases": aliases}


def _probability_csv_paths() -> tuple[set[Path], set[Path], set[Path]]:
    validation = {Path("within_site6") / method / "candidates" / f"c{index:02d}" /
                  "validation_quantiles.csv"
                  for method in METHODS for index in range(16)}
    calibration = {Path("within_site6") / method / f"final_seed{seed}" /
                   "calibration_quantiles.csv"
                   for method in METHODS for seed in PROBABILITY_METHOD_SEEDS[method]}
    test = {path.with_name("test_quantiles.csv") for path in calibration}
    return validation, calibration, test


def _expected_probability_current_check(root: Path) -> dict[str, Any]:
    runner = root / "review/run_selection_study.py"
    archived = root / "review/selection_runs/selection_v1/protocol.json"
    return {
        "status": "passed", "statement": PROBABILITY_CURRENT_CHECK,
        "runner_sha256": digest(runner) if runner.is_file() else None,
        "archived_protocol_sha256": digest(archived) if archived.is_file() else None,
        "methods": 6, "tasks": 7, "candidate_count_per_method": 16,
    }


def _verify_probability(root: Path, evidence: object, errors: list[str]) -> None:
    base = root / PROBABILITY
    manifest = _json(base / "manifest.json", errors, "probability selection manifest")
    protocol = _json(base / "protocol.json", errors, "probability selection protocol")
    lock = _json(base / "selection_lock.json", errors, "probability selection lock")
    results = _json(base / "results.json", errors, "probability selection results")
    if not (isinstance(manifest, dict) and isinstance(protocol, dict) and
            isinstance(lock, dict) and isinstance(results, list)):
        return
    if (manifest.get("status") != "complete" or manifest.get("protocol") != protocol or
            protocol.get("version") != "probabilistic_selection_v2" or
            protocol.get("candidate_budget") != 16 or
            protocol.get("kernel_seeds") != [0, 1, 2] or
            protocol.get("tree_seeds") != [0] or
            manifest.get("selection_records") != 6 or
            manifest.get("probabilistic_records") != 16):
        errors.append("probability selection-v2 design is incomplete or changed")
    dependency = root / "review/selection_runs/selection_v1/protocol.json"
    if (not dependency.is_file() or
            manifest.get("base_candidate_protocol_sha256") != digest(dependency)):
        errors.append("probability selection-v1 protocol dependency hash mismatch")
    sources = _hash_mapping(root, manifest.get("source_hashes"), errors,
                            "probability source")
    if Path("review/run_selection_study.py") not in sources:
        errors.append("probability source inventory omits its v1 runner dependency")
    if lock.get("protocol_sha256") != digest(base / "protocol.json"):
        errors.append("probability selection-lock protocol hash mismatch")
    expected_selection_names = {f"within_site6/{method}/selection.json" for method in METHODS}
    if set(lock.get("selection_files", {})) != expected_selection_names:
        errors.append("probability selection-file inventory changed")
    selection_paths = _hash_mapping(base, lock.get("selection_files"), errors,
                                    "probability locked selection")
    selections = lock.get("selections")
    if (not isinstance(selections, list) or len(selections) != 6 or
            [row.get("method") for row in selections] != list(METHODS)):
        errors.append("probability lock does not contain six ordered selections")
    else:
        for row in selections:
            disk = _json(base / f"within_site6/{row['method']}/selection.json", errors,
                         "probability selected record")
            if disk != row:
                errors.append(f"probability embedded selection changed: {row['method']}")
    expected_keys = [(method, seed) for method in METHODS
                     for seed in PROBABILITY_METHOD_SEEDS[method]]
    keys = [(row.get("method"), row.get("seed")) for row in results]
    if keys != expected_keys:
        errors.append("probability results do not contain the canonical 16 records")
    result_paths: set[Path] = set()
    for row in results:
        try:
            path = Path("within_site6") / row["method"] / f"final_seed{row['seed']}" / "result.json"
        except KeyError:
            errors.append("probability result identity is malformed")
            continue
        if _json(base / path, errors, "probability per-seed result") != row:
            errors.append(f"probability aggregate/per-seed result mismatch: {path}")
        result_paths.add(path)

    analysis_base = root / PROBABILITY_ANALYSIS
    analysis_manifest = _json(analysis_base / "manifest.json", errors,
                              "probability analysis manifest")
    analysis_protocol = _json(analysis_base / "protocol.json", errors,
                              "probability analysis protocol")
    if not isinstance(analysis_manifest, dict) or not isinstance(analysis_protocol, dict):
        return
    if (analysis_manifest.get("status") != "complete" or
            analysis_manifest.get("protocol") != analysis_protocol or
            analysis_protocol.get("version") != "probabilistic_analysis_v2" or
            analysis_manifest.get("experiment_protocol_sha256") != digest(base / "protocol.json") or
            analysis_manifest.get("experiment_runner_sha256") != digest(
                root / "review/run_probabilistic_selection_study.py") or
            analysis_manifest.get("methods") != 6 or
            analysis_manifest.get("paired_comparisons") != 5):
        errors.append("probability analysis identity or completion counts changed")
    _hash_mapping(root, analysis_manifest.get("source_hashes"), errors,
                  "probability analysis source")
    expected_inputs = {
        str(PROBABILITY / name): digest(base / name)
        for name in ("manifest.json", "selection_lock.json", "results.json")
    }
    for method, seed in expected_keys:
        relative = (PROBABILITY / "within_site6" / method / f"final_seed{seed}" /
                    "test_quantiles.csv")
        expected_inputs[str(relative)] = digest(root / relative)
    if analysis_manifest.get("input_hashes") != expected_inputs:
        errors.append("probability analysis input hash inventory changed")

    inventory = _json(analysis_base / "posthoc_integrity_inventory.json", errors,
                      "probability posthoc integrity inventory")
    validation, calibration, test = _probability_csv_paths()
    if isinstance(inventory, dict):
        if (inventory.get("role") !=
                "post-completion integrity inventory; not preregistration" or
                inventory.get("counts") != {"candidate_validation_csv": 96,
                                            "selected_calibration_csv": 16,
                                            "selected_test_csv": 16}):
            errors.append("probability post-completion inventory metadata changed")
        controls = {
            PROBABILITY / name for name in
            ("protocol.json", "manifest.json", "selection_lock.json", "results.json")
        } | {
            PROBABILITY_ANALYSIS / name for name in
            ("protocol.json", "manifest.json", "aggregate_results.json",
             "paired_probability_inference.json")
        } | {
            Path(name) for name in (
                "review/run_probabilistic_selection_study.py",
                "review/run_probabilistic_analysis_v2.py",
                "review/render_probabilistic_figures.py",
                "review/create_probabilistic_integrity_inventory.py",
            )
        }
        expected_posthoc = controls | {
            PROBABILITY / path for path in validation | calibration | test
        }
        posthoc_paths = _hash_mapping(root, inventory.get("sha256"), errors,
                                      "probability posthoc")
        if posthoc_paths != expected_posthoc:
            errors.append("probability post-completion inventory is not the canonical 140 files")

    presentation_base = root / PROBABILITY_PRESENTATION
    presentation = _json(presentation_base / "manifest.json", errors,
                         "probability presentation manifest")
    if isinstance(presentation, dict):
        expected_outputs = PROBABILITY_PRESENTATION_FILES - {"manifest.json"}
        if (presentation.get("input") != str(
                PROBABILITY_ANALYSIS / "aggregate_results.json") or
                presentation.get("input_sha256") != digest(
                    analysis_base / "aggregate_results.json") or
                presentation.get("source") != "review/render_probabilistic_figures.py" or
                presentation.get("source_sha256") != digest(
                    root / "review/render_probabilistic_figures.py") or
                set(presentation.get("outputs", {})) != expected_outputs):
            errors.append("probability presentation identity or output inventory changed")
        else:
            _hash_mapping(presentation_base, presentation["outputs"], errors,
                          "probability presentation output")

    expected_selection = {
        Path("protocol.json"), Path("manifest.json"), Path("selection_lock.json"),
        Path("results.json"), *selection_paths, *result_paths,
        *validation, *calibration, *test,
    }
    if not isinstance(evidence, dict):
        errors.append("Missing probability evidence inventory")
        return
    if (evidence.get("provenance_class") != "legacy/posthoc" or
            evidence.get("legacy_provenance_limitation") != LEGACY_PROBABILITY_LIMITATION or
            evidence.get("current_candidate_protocol_check") !=
            _expected_probability_current_check(root)):
        errors.append("probability legacy provenance declaration or current check changed")
    _declared_subtree(root, PROBABILITY, evidence.get("selection_files"),
                      expected_selection, errors, "probability selection evidence")
    _declared_subtree(root, PROBABILITY_ANALYSIS, evidence.get("analysis_files"),
                      {Path(name) for name in PROBABILITY_ANALYSIS_FILES}, errors,
                      "probability analysis evidence")
    _declared_subtree(root, PROBABILITY_PRESENTATION,
                      evidence.get("presentation_files"),
                      {Path(name) for name in PROBABILITY_PRESENTATION_FILES}, errors,
                      "probability presentation evidence")


def _verify_evidence_inventory(root: Path, selection: dict[str, Any],
                               errors: list[str]) -> dict[str, Any] | None:
    inventory = _json(root / EVIDENCE_INVENTORY, errors, "package evidence inventory")
    if not isinstance(inventory, dict):
        return None
    if inventory.get("version") != "s1_evidence_inventory_1":
        errors.append("Unexpected package evidence inventory version")
    if set(inventory) != {"version", "selection_v2", "chronological_v1",
                         "probability_v2"}:
        errors.append("Package evidence inventory top-level schema changed")
    selection_evidence = inventory.get("selection_v2")
    if not isinstance(selection_evidence, dict):
        errors.append("Missing selection-v2 evidence inventory")
    else:
        if set(selection_evidence) != {
                "calibration_hash_authority", "selection_fits",
                "evaluation_records", "evaluation_tau_fits",
                "calibration_predictions", "fit_record_hash_authority",
                "fit_record_inventory", "fit_record_inventory_sha256"}:
            errors.append("Selection-v2 evidence schema changed")
        if (selection_evidence.get("calibration_hash_authority") !=
                PACKAGE_TIME_SNAPSHOT or
                selection_evidence.get("fit_record_hash_authority") !=
                FIT_RECORD_HASH_AUTHORITY or
                selection_evidence.get("fit_record_inventory") !=
                str(SELECTION / FIT_RECORD_INVENTORY) or
                selection_evidence.get("selection_fits") != 10368 or
                selection_evidence.get("evaluation_records") != 192 or
                selection_evidence.get("evaluation_tau_fits") != 576):
            errors.append("selection-v2 evidence counts changed")
        fit_inventory = root / SELECTION / FIT_RECORD_INVENTORY
        if (not fit_inventory.is_file() or
                selection_evidence.get("fit_record_inventory_sha256") !=
                digest(fit_inventory)):
            errors.append("selection-v2 fit-record inventory evidence hash changed")
        declared = _hash_mapping(
            root, selection_evidence.get("calibration_predictions"), errors,
            "selection calibration evidence")
        if declared != selection["calibration_paths"] or len(declared) != 192:
            errors.append("selection calibration evidence is not the canonical 192 files")
    return inventory


def _verify_package_metadata(root: Path, selection: dict[str, Any],
                             index: dict[str, Any] | None,
                             evidence: dict[str, Any] | None,
                             errors: list[str]) -> None:
    environment = _json(root / "environment.json", errors, "package environment")
    provenance = _json(root / "PACKAGING_PROVENANCE.json", errors,
                       "packaging provenance")
    try:
        readme = (root / "README.md").read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as error:
        errors.append(f"Missing or malformed package README: {error}")
        readme = ""
    expected_check = _expected_probability_current_check(root)
    expected_probability = {
        "class": "legacy/posthoc",
        "limitation": LEGACY_PROBABILITY_LIMITATION,
        "current_candidate_protocol_check": expected_check,
    }
    if isinstance(index, dict) and index.get("probability_provenance") != expected_probability:
        errors.append("PACKAGE_INDEX probability provenance declaration changed")
    if not isinstance(provenance, dict):
        pass
    elif (provenance.get("version") != "s1_v2_exact_bytes_2" or
          provenance.get("probability_provenance") != expected_probability or
          provenance.get("fit_record_policy") != FIT_RECORD_HASH_AUTHORITY or
          "packaging runtime is not experimental provenance" not in
          provenance.get("environment_policy", "")):
        errors.append("PACKAGING_PROVENANCE legacy or environment declaration changed")
    has_amendment = bool(selection.get("amended_fit_paths"))
    amendment_lock = root / AMENDMENT_LOCK
    index_amendment = index.get("solver_certificate_amendment") \
        if isinstance(index, dict) else None
    provenance_amendment = provenance.get("solver_certificate_amendment") \
        if isinstance(provenance, dict) else None
    if has_amendment:
        expected_amendment = {
            "version": AMENDMENT_VERSION,
            "lock": str(AMENDMENT_LOCK),
            "lock_sha256": digest(amendment_lock) if amendment_lock.is_file() else None,
            "amended_main_fits": 1,
            "amended_matched_fits": 0,
            "advanced_analysis_manifests": sorted((
                str(SELECTION_ANALYSIS / "analysis_manifest.json"),
                str(MATCHED_ANALYSIS / "manifest.json"),
            )),
        }
        if index_amendment != expected_amendment:
            errors.append("PACKAGE_INDEX solver-amendment declaration changed")
        expected_provenance_amendment = {
            **expected_amendment,
            "interpretation": (
                "Performance-blind numerical-certificate amendment; objective, data, feature "
                "map, configuration, tolerance, prediction, and analysis rules were unchanged. "
                "The two named state manifests subsequently advanced to their separately "
                "validated completed states; the immutable analysis protocols and lock did "
                "not change."
            ),
        }
        if provenance_amendment != expected_provenance_amendment:
            errors.append("PACKAGING_PROVENANCE solver-amendment declaration changed")
        amendment_phrases = (
            "performance-blind solver amendment",
            "relative gap 0.0010217",
            "96,000-iteration warm restart",
            "no matched fit used the amendment",
            "three immutable analysis protocol/lock files",
            "two state manifests",
        )
        if any(phrase not in readme for phrase in amendment_phrases):
            errors.append("README omits a required solver-amendment disclosure")
    elif (index_amendment is not None or provenance_amendment is not None or
          amendment_lock.exists()):
        errors.append("Package declares a solver amendment without amended-fit provenance")
    probability_evidence = evidence.get("probability_v2", {}) \
        if isinstance(evidence, dict) else {}
    chronological_evidence = evidence.get("chronological_v1", {}) \
        if isinstance(evidence, dict) else {}
    if isinstance(chronological_evidence, dict) and set(chronological_evidence) != {
            "artifact_hash_authority", "jobs", "aliases", "files"}:
        errors.append("Chronological evidence schema changed")
    if isinstance(probability_evidence, dict) and set(probability_evidence) != {
            "provenance_class", "legacy_provenance_limitation",
            "current_candidate_protocol_check", "selection_files",
            "analysis_files", "presentation_files"}:
        errors.append("Probability evidence schema changed")
    if (probability_evidence.get("provenance_class") != "legacy/posthoc" or
            probability_evidence.get("legacy_provenance_limitation") !=
            LEGACY_PROBABILITY_LIMITATION or
            probability_evidence.get("current_candidate_protocol_check") != expected_check):
        errors.append("Evidence inventory probability provenance declaration changed")
    if (LEGACY_PROBABILITY_LIMITATION not in readme or
            PROBABILITY_CURRENT_CHECK not in readme or
            PACKAGE_TIME_SNAPSHOT not in readme):
        errors.append("README omits a required provenance limitation or current check")
    if ("main analysis protocol was frozen while selection was still in progress" not in
            readme or "archived run froze that analysis protocol earlier" not in readme or
            "main analysis protocol was then frozen" in readme):
        errors.append("README misstates the archived main-analysis freeze timing")
    readme_lines = readme.splitlines()
    for relative in REQUIRED_PUBLICATION_SCRIPTS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"Missing required publication/revision script: {relative}")
    workflow_positions = []
    for command in README_POINT_WORKFLOW:
        if readme_lines.count(command) != 1:
            errors.append("README point-workflow command inventory changed")
            break
        workflow_positions.append(readme_lines.index(command))
    if (len(workflow_positions) == len(README_POINT_WORKFLOW) and
            workflow_positions != sorted(workflow_positions)):
        errors.append("README point-workflow freeze/evaluation order changed")
    if not isinstance(environment, dict):
        return
    packaging_runtime = environment.get("packaging_runtime")
    if (set(environment) != {"version", "frozen_selection_v2_environment",
                             "packaging_runtime", "interpretation"} or
            environment.get("version") != "s1_environment_2" or
            environment.get("frozen_selection_v2_environment") !=
            selection["lock"].get("environment_versions") or
            not isinstance(packaging_runtime, dict) or
            set(packaging_runtime) != {"python", "implementation", "platform",
                                       "machine", "processor", "packages"} or
            "archive-construction runtime" not in environment.get("interpretation", "")):
        errors.append("Package environment conflates frozen experiment and packaging runtimes")
        return
    frozen = environment["frozen_selection_v2_environment"]
    lines = ["# Versions frozen by the selection-v2 lock."]
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "joblib",
                 "threadpoolctl"):
        if frozen.get(name):
            lines.append(f"{name}=={frozen[name]}")
    lines.append(
        "# Presentation/solver packages recorded at packaging time; not selection-v2 lock fields.")
    for name in ("matplotlib", "cvxopt"):
        if packaging_runtime["packages"].get(name):
            lines.append(f"{name}=={packaging_runtime['packages'][name]}")
    expected_requirements = "\n".join(lines) + "\n"
    try:
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeDecodeError) as error:
        errors.append(f"Missing or malformed package requirements: {error}")
    else:
        if requirements != expected_requirements:
            errors.append("requirements.txt conflates frozen and packaging-only versions")


def _verify_package_index(root: Path, errors: list[str]) -> tuple[bool, dict[str, Any] | None]:
    index = _json(root / "PACKAGE_INDEX.json", errors, "package index")
    if not isinstance(index, dict):
        return False, None
    if index.get("version") != "selection_v2_matched_2":
        errors.append("Unexpected PACKAGE_INDEX version")
    expected_order = [
        "selection-v2 freeze", "base selection started", "selection-analysis freeze",
        "base selection completed", "regularizer selection", "joint selection", "selection lock",
        "matched-ablation freeze", "matched-analysis freeze", "selection evaluation",
        "matched evaluation", "selection analysis", "matched analysis",
        "publication rendering",
    ]
    if index.get("workflow_order") != expected_order:
        errors.append("PACKAGE_INDEX workflow order changed")
    design = index.get("v2_design", {})
    expected_design = {
        "selection_lock_version": "selection_v2_lock_2", "tasks": 12, "methods": 6,
        "queries_per_task_method": 54, "total_configuration_queries": 3888,
        "selection_fits": 10368, "evaluation_records": 192,
        "evaluation_tau_fits": 576,
        "matched_lock_version": "matched_ablation_v2_lock_1",
        "matched_evaluation_records": 144,
    }
    if design != expected_design:
        errors.append("PACKAGE_INDEX v2 design changed")
    fit_inventory_path = root / SELECTION / FIT_RECORD_INVENTORY
    expected_fit_inventory = {
        "path": str(SELECTION / FIT_RECORD_INVENTORY),
        "sha256": digest(fit_inventory_path) if fit_inventory_path.is_file() else None,
        "version": FIT_RECORD_INVENTORY_VERSION,
        "hash_authority": FIT_RECORD_HASH_AUTHORITY,
        "selection_fit_records": 10368,
        "selection_validation_metric_records": 10368,
        "evaluation_fit_records": 576,
    }
    if index.get("fit_record_inventory") != expected_fit_inventory:
        errors.append("PACKAGE_INDEX fit-record inventory declaration changed")
    if index.get("selection_v1_allowlist") != [
            "review/selection_runs/selection_v1/protocol.json"]:
        errors.append("PACKAGE_INDEX selection-v1 allowlist changed")
    if index.get("evidence_inventory") != str(EVIDENCE_INVENTORY):
        errors.append("PACKAGE_INDEX evidence-inventory path changed")
    omitted = index.get("omitted_artifact_classes", [])
    models_declared = isinstance(omitted, list) and "final fitted-model binaries" in omitted
    if not models_declared:
        errors.append("PACKAGE_INDEX does not declare omitted final fitted-model binaries")
    if not isinstance(omitted, list) or "full selection and main fit histories" not in omitted:
        errors.append("PACKAGE_INDEX does not declare omitted full fit histories")
    return models_declared, index


def _verify_split_assignments(root: Path, index: dict[str, Any] | None,
                              errors: list[str]) -> None:
    """Validate the exact six-file inventory and every published assignment row."""
    records = index.get("split_assignments") if isinstance(index, dict) else None
    if not isinstance(records, list) or len(records) != 6:
        errors.append("PACKAGE_INDEX split-assignment inventory is not six farms")
        return
    expected_files = {Path(f"farm{site}.csv.gz") for site in range(1, 7)}
    base = root / "split_assignments"
    actual_files = ({path.relative_to(base) for path in base.rglob("*") if path.is_file()}
                    if base.is_dir() else set())
    if actual_files != expected_files:
        errors.append("split-assignment subtree inventory changed")
    expected_record_keys = {"site", "rows", "file", "sha256"}
    previous_sites: list[int] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            errors.append("split-assignment index record schema changed")
            continue
        site = record.get("site")
        if type(site) is not int or site not in range(1, 7):
            errors.append("split-assignment site is invalid")
            continue
        previous_sites.append(site)
        expected_relative = Path("split_assignments") / f"farm{site}.csv.gz"
        relative = _safe(record.get("file"), errors, "split assignment")
        if relative != expected_relative:
            errors.append(f"split-assignment path is noncanonical for farm {site}")
            continue
        path = root / relative
        if (not path.is_file() or not isinstance(record.get("sha256"), str) or
                digest(path) != record.get("sha256")):
            errors.append(f"split-assignment SHA-256 mismatch for farm {site}")
            continue
        declared_rows = record.get("rows")
        if type(declared_rows) is not int or not (1 <= declared_rows <= 100_000):
            errors.append(f"split-assignment row count is invalid for farm {site}")
            continue
        row_count = 0
        previous_target_row = -1
        previous_time: datetime | None = None
        seen_splits: set[str] = set()
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader, None)
                if header != list(SPLIT_ASSIGNMENT_HEADER):
                    raise ValueError("header changed")
                for row_count, row in enumerate(reader, start=1):
                    if row_count > declared_rows or len(row) != len(SPLIT_ASSIGNMENT_HEADER):
                        raise ValueError("row count or width changed")
                    window_index = int(row[0])
                    target_row = int(row[1])
                    timestamp = datetime.fromisoformat(row[2])
                    split, domain = row[3], row[4]
                    power = float(row[5])
                    flags = tuple(int(value) for value in row[6:10])
                    if (window_index != row_count - 1 or target_row <= previous_target_row or
                            timestamp.tzinfo is not None or
                            (previous_time is not None and timestamp <= previous_time) or
                            not math.isfinite(power) or any(flag not in (0, 1) for flag in flags)):
                        raise ValueError("row identity, order, or numeric value changed")
                    expected_split = next((name for name, start, end in SPLIT_BOUNDARIES
                                           if start <= timestamp < end), None)
                    if split != expected_split:
                        raise ValueError("timestamp/split assignment changed")
                    expected_domain = "target" if any(flags) else "source"
                    if domain != expected_domain:
                        raise ValueError("meteorological flags/domain assignment changed")
                    seen_splits.add(split)
                    previous_target_row, previous_time = target_row, timestamp
        except (OSError, EOFError, UnicodeDecodeError, csv.Error, ValueError) as error:
            errors.append(f"Malformed split assignment for farm {site}: {error}")
            continue
        if row_count != declared_rows:
            errors.append(f"split-assignment row count mismatch for farm {site}")
        if seen_splits != {name for name, _, _ in SPLIT_BOUNDARIES}:
            errors.append(f"split-assignment temporal coverage changed for farm {site}")
    if previous_sites != list(range(1, 7)):
        errors.append("split-assignment farm order or uniqueness changed")


def verify_v2(root: Path, errors: list[str]) -> None:
    models_declared_omitted, index = _verify_package_index(root, errors)
    _verify_split_assignments(root, index, errors)
    selection = _verify_selection(root, errors)
    if selection is None:
        return
    evidence = _verify_evidence_inventory(root, selection, errors)
    _verify_package_metadata(root, selection, index, evidence, errors)
    chronological_evidence = evidence.get("chronological_v1") \
        if isinstance(evidence, dict) else None
    chronological = _verify_chronological(root, chronological_evidence, errors)
    if (chronological is not None and
            selection["manifest"].get("reference_manifest_sha256") !=
            digest(chronological["base"] / "manifest.json")):
        errors.append("selection-v2 does not bind the packaged chronological manifest")
    selection_analysis = _verify_selection_analysis(root, selection, errors)
    matched = _verify_matched(root, selection, errors,
                              models_declared_omitted=models_declared_omitted)
    if selection_analysis is None or matched is None:
        return
    amendment_paths = _verify_solver_amendment(
        root, selection, matched, index, errors)
    matched_analysis = _verify_matched_analysis(root, matched, errors)
    render = None
    if matched_analysis is not None:
        render = _verify_render(root, selection_analysis, matched_analysis, errors)

    selection_expected = {
        Path("protocol.json"), Path("manifest.json"), Path("data_provenance.json"),
        Path("selection_lock.json"), Path("results.json"),
        selection["fit_record_inventory_path"],
        *(Path(name) for name in selection["lock"].get("selection_files", {})),
        *matched["fit_only_paths"],
        *amendment_paths["selection_paths"],
        *(Path(task) / "target_data" / "selection_counts.json" for task in TASKS),
    }
    for task in TASKS:
        for method in METHODS:
            for seed in selection["protocol"].get("evaluation_seeds_by_method", {}).get(
                    method, []):
                folder = Path(task) / method / "evaluation" / f"seed{seed}"
                selection_expected |= {
                    folder / "result.json", folder / "test_predictions.csv",
                    folder / "calibration_predictions.csv",
                }
    _exact_subtree_files(root, SELECTION, selection_expected, errors,
                         "selection-v2")
    _exact_subtree_files(
        root, SELECTION_ANALYSIS,
        {Path("analysis_protocol.json"), Path("analysis_manifest.json"),
         *selection_analysis["artifact_paths"]},
        errors, "selection-v2 analysis")
    _exact_subtree_files(
        root, MATCHED,
        {Path("protocol.json"), Path("configurations.json"), Path("matched_lock.json"),
         Path("manifest.json"), Path("results.json"), *matched["result_paths"],
         *amendment_paths["matched_paths"]},
        errors, "matched-ablation-v2")
    if matched_analysis is not None:
        _exact_subtree_files(
            root, MATCHED_ANALYSIS,
            {Path("analysis_protocol.json"), Path("analysis_lock.json"),
             Path("manifest.json"), *matched_analysis["output_paths"]},
            errors, "matched-ablation-v2 analysis")
    if render is not None:
        _exact_subtree_files(
            root, RENDER, {Path("manifest.json"), *render["output_paths"]},
            errors, "selection-v2 render")
    if isinstance(index, dict) and isinstance(evidence, dict):
        probability_inventory = evidence.get("probability_v2", {})
        chronological_inventory = evidence.get("chronological_v1", {})
        expected_counts = {
            "common_setting": len(chronological_inventory.get("files", {})),
            "probabilistic_selection_v2": len(
                probability_inventory.get("selection_files", {})),
            "probabilistic_analysis_v2": len(
                probability_inventory.get("analysis_files", {})),
            "probabilistic_presentation_v1": len(
                probability_inventory.get("presentation_files", {})),
            "selection_v2": len(selection_expected),
            "selection_v2_analysis": 2 + len(selection_analysis["artifact_paths"]),
            "matched_ablation_v2": len({
                Path("protocol.json"), Path("configurations.json"),
                Path("matched_lock.json"), Path("manifest.json"), Path("results.json"),
                *matched["result_paths"], *amendment_paths["matched_paths"],
            }),
            "matched_ablation_v2_analysis": (
                3 + len(matched_analysis["output_paths"])
                if matched_analysis is not None else 0),
            "selection_v2_render": 1 + (len(render["output_paths"])
                                         if render is not None else 0),
        }
        if index.get("artifact_file_counts") != expected_counts:
            errors.append("PACKAGE_INDEX artifact-file counts changed")
    probability_evidence = evidence.get("probability_v2") \
        if isinstance(evidence, dict) else None
    _verify_probability(root, probability_evidence, errors)


def verify(root: Path, *, require_v2: bool = True) -> dict[str, Any]:
    root = Path(root).resolve()
    errors: list[str] = []
    declared, actual = verify_manifest(root, errors)
    verify_exclusions(root, actual, errors)
    if require_v2:
        try:
            verify_v2(root, errors)
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            errors.append(f"Nested S1 verification could not complete: {error}")
    return {"status": "passed" if not errors else "failed", "package_root": str(root),
            "manifest_entries": len(declared), "payload_files": len(actual), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = verify(args.root)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.json_output:
        args.json_output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
