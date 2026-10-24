#!/usr/bin/env python3
"""Build S1 only after the complete, frozen selection-v2 workflow exists."""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "TLQLDMR_R1" / "Supplementary_Reproducibility_S1"
ZIP = ROOT / "TLQLDMR_R1" / "Supplementary_Reproducibility_S1.zip"

SELECTION_REL = Path("review/selection_runs/selection_v2")
SELECTION_ANALYSIS_REL = Path("review/selection_runs/selection_v2_analysis")
MATCHED_REL = Path("review/selection_runs/matched_ablation_v2")
MATCHED_ANALYSIS_REL = Path("review/selection_runs/matched_ablation_v2_analysis")
RENDER_REL = Path("review/selection_runs/selection_v2_render")
AMENDMENT_LOCK_REL = Path(
    "review/selection_runs/solver_certificate_amendment_v2.json"
)
AMENDMENT_VERSION = "solver_certificate_amendment_v2_2"
CHRONO_REL = Path("review/replied_comment_runs/chronological_v1")
CHRONO_SOURCE_SNAPSHOTS = {
    Path("Train_TL_QLDMR.py"):
        CHRONO_REL / "frozen_sources" / "Train_TL_QLDMR.py",
}
PROBABILITY_REL = Path("review/probabilistic_runs/probabilistic_selection_v2")
PROBABILITY_ANALYSIS_REL = Path("review/probabilistic_runs/probabilistic_analysis_v2")
PROBABILITY_PRESENTATION_REL = Path("review/probabilistic_runs/probabilistic_presentation_v1")
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

CODE_FILES = (
    "Train_TL_QLDMR.py", "Kernel_Function.py", "review/temporal_data.py",
    "review/selection_data.py", "review/selection_data_v2.py",
    "review/selection_sampling.py", "review/convex_refinement.py",
    "review/replied_study_protocol.json", "review/run_replied_comment_study.py",
    "review/summarize_replied_comment_study.py", "review/verify_replied_comment_study.py",
    "review/plot_replied_comment_study.py",
    # Kept only because the frozen probability protocol binds this source.
    "review/run_selection_study.py", "review/run_selection_study_v2.py",
    "review/resume_solver_certificate_v2.py",
    "review/analyze_selection_study_v2.py", "review/run_matched_ablation_v2.py",
    "review/analyze_matched_ablation_v2.py", "review/render_selection_study_v2.py",
    "review/render_manuscript_v2_appendix.py",
    "review/summarize_v2_revision_evidence.py",
    "review/run_probabilistic_selection_study.py", "review/run_probabilistic_analysis_v2.py",
    "review/run_probabilistic_analysis_portable.py", "review/render_probabilistic_figures.py",
    "review/create_probabilistic_integrity_inventory.py", "review/summarize_extreme_events.py",
    "review/build_reproducibility_supplement.py", "review/verify_s1_package.py",
)


class PackageNotReady(RuntimeError):
    """The formal workflow has not reached a complete immutable state."""


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


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PackageNotReady(f"Required frozen file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise PackageNotReady(f"Required frozen JSON is malformed: {path}") from error


def _safe_relative(value: str | Path, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"Unsafe {label} path: {value}")
    raw = str(value)
    path = Path(value)
    if (path.is_absolute() or not path.parts or ".." in path.parts or
            "\\" in raw or raw != path.as_posix()):
        raise ValueError(f"Unsafe {label} path: {value}")
    return path


def _require_hashes(base: Path, mapping: object, *, label: str) -> set[Path]:
    if not isinstance(mapping, dict) or not mapping:
        raise PackageNotReady(f"{label} hash inventory is missing or empty.")
    paths: set[Path] = set()
    for value, expected in mapping.items():
        relative = _safe_relative(value, label=label)
        if relative in paths:
            raise ValueError(f"Duplicate canonical {label} path: {value}")
        path = base / relative
        if not path.is_file():
            raise PackageNotReady(f"{label} artifact is missing: {path}")
        if not isinstance(expected, str) or digest(path) != expected:
            raise ValueError(f"{label} artifact hash mismatch: {path}")
        paths.add(relative)
    return paths


def _require_sources(source_root: Path, mapping: object, *, label: str) -> set[Path]:
    paths = _require_hashes(source_root, mapping, label=label)
    if any(path.suffix not in {".py", ".json"} for path in paths):
        raise ValueError(f"Unexpected file type in {label} inventory.")
    return paths


def _require_chronological_sources(
        source_root: Path, mapping: object) -> tuple[set[Path], dict[Path, Path]]:
    """Validate historical source hashes, using explicit byte-exact snapshots.

    The workspace training module advanced after the chronological study.  The
    historical manifest still binds the exact source used by that study, so S1
    packages that manifest-bound snapshot at the module's canonical root path.
    """
    if not isinstance(mapping, dict) or not mapping:
        raise PackageNotReady("chronological source hash inventory is missing or empty.")
    paths: set[Path] = set()
    overrides: dict[Path, Path] = {}
    for value, expected in mapping.items():
        relative = _safe_relative(value, label="chronological source")
        if relative in paths:
            raise ValueError(f"Duplicate chronological source path: {value}")
        if relative.suffix not in {".py", ".json"}:
            raise ValueError("Unexpected file type in chronological source inventory.")
        if not isinstance(expected, str):
            raise ValueError(
                f"chronological source hash is malformed: {relative}")
        current_path = source_root / relative
        if current_path.is_file() and digest(current_path) == expected:
            source_relative = relative
            source_path = current_path
        else:
            source_relative = CHRONO_SOURCE_SNAPSHOTS.get(relative, relative)
            source_path = source_root / source_relative
            if not source_path.is_file():
                raise PackageNotReady(
                    f"chronological source artifact is missing: {source_path}")
            if digest(source_path) != expected:
                raise ValueError(
                    f"chronological source artifact hash mismatch: {source_path}")
        if source_relative != relative:
            overrides[relative] = source_relative
        paths.add(relative)
    return paths, overrides


def _complete(manifest: Mapping[str, Any], *, label: str) -> None:
    if manifest.get("status") != "complete":
        raise PackageNotReady(f"{label} is not complete (status={manifest.get('status')!r}).")


def _expected_selection_files() -> set[str]:
    return {f"{task}/{method}/{stage}_selection.json"
            for task in TASKS for method in METHODS for stage in STAGES}


def _effective_signature(method: str, candidate: object) -> tuple:
    if method not in METHODS or not isinstance(candidate, dict):
        raise ValueError(f"Invalid candidate for {method}.")
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
    try:
        return tuple((key, candidate[key]) for key in keys)
    except KeyError as error:
        raise ValueError(f"Candidate for {method} lacks {error.args[0]}.") from error


def _derive_matched_variants(full_config: object) -> dict[str, dict]:
    if not isinstance(full_config, dict):
        raise ValueError("Matched Full configuration is not a mapping.")
    missing = MATCHED_CONFIG_KEYS - set(full_config)
    if missing:
        raise ValueError(f"Matched Full configuration lacks keys: {sorted(missing)}")
    numeric = [full_config[name] for name in MATCHED_CONFIG_KEYS]
    try:
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("Matched Full configuration contains a nonfinite coordinate.")
        rank = int(full_config["rank"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Matched Full configuration contains an invalid coordinate.") from error
    if rank <= 0 or float(full_config["rank"]) != rank:
        raise ValueError("Matched Full requested rank must be a positive integer.")
    if float(full_config["lambda1"]) <= 0 or float(full_config["lambda2"]) <= 0:
        raise ValueError("Matched Full configuration must activate both components.")
    variants = {name: dict(full_config) for name in VARIANTS}
    variants["NoVariance"]["lambda1"] = 0.0
    variants["NoDiscrepancy"]["lambda2"] = 0.0
    variants["WeightedSVQR"]["lambda1"] = 0.0
    variants["WeightedSVQR"]["lambda2"] = 0.0
    return variants


def _expected_shared_artifacts(task: str, full_config: Mapping[str, Any]) -> dict[str, str]:
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


def _validate_matched_configurations(selection: dict[str, Any],
                                     configurations: Mapping[str, Any]) -> dict[str, dict]:
    if configurations.get("base_selection_lock_sha256") != digest(
            selection["root"] / "selection_lock.json"):
        raise ValueError("Matched configurations name a different base selection lock.")
    frozen_tasks = configurations.get("tasks")
    if not isinstance(frozen_tasks, list) or [row.get("task") for row in frozen_tasks] != list(TASKS):
        raise ValueError("Matched configurations do not contain the canonical 12-task order.")
    protocol_tasks = {row.get("name"): row for row in selection["protocol"].get("tasks", [])}
    final = {(row.get("task"), row.get("method")): row
             for row in selection["lock"].get("final_selections", [])}
    by_task: dict[str, dict] = {}
    for row in frozen_tasks:
        task = row["task"]
        protocol_task = protocol_tasks.get(task)
        base_row = final.get((task, "Full"))
        if protocol_task is None or base_row is None:
            raise ValueError(f"Matched task is absent from the base lock: {task}")
        for field in ("site", "mode", "role"):
            if row.get(field) != protocol_task.get(field):
                raise ValueError(f"Matched task {field} changed: {task}")
        selection_file = Path(task) / "Full" / "final_selection.json"
        if (row.get("base_full_selection_file") != str(selection_file) or
                row.get("base_full_selection_sha256") != digest(selection["root"] / selection_file) or
                _read_json(selection["root"] / selection_file) != base_row):
            raise ValueError(f"Matched task does not bind the locked Full selection: {task}")
        full_config = row.get("full_config")
        if full_config != base_row.get("selected_config"):
            raise ValueError(f"Matched Full configuration differs from the base lock: {task}")
        if row.get("variants") != _derive_matched_variants(full_config):
            raise ValueError(f"Matched variants change more than the two lambdas: {task}")
        expected_shared = _expected_shared_artifacts(task, full_config)
        if row.get("shared_artifacts") != expected_shared:
            raise ValueError(f"Matched shared-artifact paths changed: {task}")
        by_task[task] = dict(row)
    return by_task


def _validate_render_inputs(mapping: object, selection_root: Path,
                            matched_root: Path) -> None:
    expected = {
        **{name: digest(selection_root / name) for name in RENDER_SELECTION_INPUT_FILES},
        **{name: digest(matched_root / name) for name in RENDER_MATCHED_INPUT_FILES},
    }
    if not isinstance(mapping, dict) or len(mapping) != len(expected):
        raise ValueError("selection-v2 render input-file inventory is incomplete.")
    observed: dict[str, str] = {}
    for raw_path, recorded_hash in mapping.items():
        name = Path(raw_path).name
        if name in observed or name not in expected:
            raise ValueError(f"selection-v2 render has an unexpected input: {raw_path}")
        observed[name] = recorded_hash
    if observed != expected:
        raise ValueError("selection-v2 render input-file hashes changed.")


def _planned_chronological(protocol: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
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


def _validate_chronological_verification(
        root: Path, verification: Mapping[str, Any], jobs: list[str],
        aliases: Mapping[str, str]) -> set[Path]:
    """Validate either the legacy inline report or its final hash-bound summary."""
    branches = verification.get("branches")
    verification_aliases = verification.get("aliases")
    global_checks = verification.get("global_checks")
    if all(isinstance(rows, list)
           for rows in (branches, verification_aliases, global_checks)):
        if any(not isinstance(row, dict)
               for rows in (branches, verification_aliases, global_checks)
               for row in rows):
            raise ValueError("chronological verification rows are malformed.")
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
            raise ValueError(
                "chronological independent verification is not complete and passed.")
        return set()

    report_names = {
        "branch_verification_report": "verification_stage1.json",
        "completion_verification_report": "verification_crossfarm_final.json",
        "independent_prediction_summary": "study_summary.json",
    }
    if (verification.get("status") != "passed" or
            verification.get("completed_experiment_branches") != len(jobs) or
            verification.get("completed_quantile_model_fits") != 123 or
            verification.get("aliases") != len(aliases) or
            any(verification.get(key) != value for key, value in report_names.items())):
        raise ValueError("chronological verification summary is incomplete.")
    hashes = verification.get("source_reports_sha256")
    expected_names = set(report_names.values())
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        raise ValueError("chronological verification report hashes are incomplete.")
    reports: dict[str, Mapping[str, Any]] = {}
    for name in sorted(expected_names):
        path = root / name
        if (not path.is_file() or not isinstance(hashes.get(name), str) or
                digest(path) != hashes[name]):
            raise ValueError(f"chronological verification report hash mismatch: {name}")
        report = _read_json(path)
        if not isinstance(report, dict):
            raise ValueError(f"chronological verification report is malformed: {name}")
        reports[name] = report

    stage1 = reports["verification_stage1.json"]
    stage1_branches = stage1.get("branches")
    stage1_aliases = stage1.get("aliases")
    stage1_globals = stage1.get("global_checks")
    if (not all(isinstance(rows, list)
                for rows in (stage1_branches, stage1_aliases, stage1_globals)) or
            any(not isinstance(row, dict)
                for rows in (stage1_branches, stage1_aliases, stage1_globals)
                for row in rows) or
            stage1.get("status") != "passed_partial" or stage1.get("pending") or
            stage1.get("passed_branches") != len(jobs) or
            {row.get("name") for row in stage1_branches
             if row.get("status") == "passed"} != set(jobs) or
            {row.get("name") for row in stage1_aliases
             if row.get("status") == "passed"} != set(aliases) or
            any(row.get("status") != "passed" for row in stage1_globals)):
        raise ValueError("chronological stage-1 verification is incomplete.")
    for row in stage1_aliases:
        if aliases.get(row.get("name")) != row.get("reference"):
            raise ValueError("chronological stage-1 alias mapping changed.")

    completion = reports["verification_crossfarm_final.json"]
    completion_branches = completion.get("branches")
    completion_aliases = completion.get("aliases")
    completion_globals = completion.get("global_checks")
    cross_jobs = {name for name in jobs if name.startswith("crossfarm_")}
    if (not all(isinstance(rows, list)
                for rows in (completion_branches, completion_aliases,
                             completion_globals)) or
            any(not isinstance(row, dict)
                for rows in (completion_branches, completion_aliases,
                             completion_globals) for row in rows) or
            completion.get("status") != "passed" or completion.get("pending") or
            completion.get("manifest_status_at_start") != "complete" or
            completion.get("passed_branches") != len(cross_jobs) or
            completion_aliases or
            {row.get("name") for row in completion_branches
             if row.get("status") == "passed"} != cross_jobs or
            any(row.get("status") != "passed" for row in completion_globals)):
        raise ValueError("chronological completion verification is incomplete.")

    summary = reports["study_summary.json"]
    prediction_verification = summary.get("prediction_verification")
    available = summary.get("available_branches_at_start")
    if (summary.get("manifest_status") != "complete" or
            summary.get("completed_unique_branches") != len(jobs) or
            summary.get("completed_unique_quantile_fits") != 123 or
            summary.get("missing_expected_branches") or summary.get("errors") or
            summary.get("aliases") != dict(aliases) or
            not isinstance(available, list) or set(available) != set(jobs) | set(aliases) or
            not isinstance(prediction_verification, dict) or
            set(prediction_verification) != set(jobs)):
        raise ValueError("chronological independent prediction summary is incomplete.")
    for name, row in prediction_verification.items():
        if (not isinstance(row, dict) or
                row.get("result_sha256") != digest(root / name / "result.json") or
                row.get("predictions_sha256") !=
                digest(root / name / "predictions.csv") or
                not isinstance(row.get("metrics_match_predictions"), dict) or
                not row["metrics_match_predictions"] or
                not all(value is True
                        for value in row["metrics_match_predictions"].values())):
            raise ValueError(
                f"chronological independent prediction check changed: {name}")
    return {Path(name) for name in expected_names}


def _validate_chronological(source_root: Path) -> dict[str, Any]:
    root = source_root / CHRONO_REL
    manifest = _read_json(root / "manifest.json")
    protocol = _read_json(source_root / "review/replied_study_protocol.json")
    if manifest.get("status") != "complete" or manifest.get("protocol") != protocol:
        raise PackageNotReady("chronological-v1 is incomplete or its protocol changed.")
    _, source_overrides = _require_chronological_sources(
        source_root, manifest.get("source_hashes"))
    _require_hashes(source_root, manifest.get("data_hashes"), label="chronological data")
    jobs, aliases = _planned_chronological(protocol)
    definitions = _read_json(root / "threshold_definitions.json")
    if not isinstance(definitions, list):
        raise ValueError("chronological threshold definitions are malformed.")
    for row in definitions:
        if row.get("identical_domain_mask_to_25") is False:
            jobs.extend(f"threshold_q{row['q']}_cutout{row['cutout_ms']}_{task}"
                        for task in ("point", "interval"))
    if (len(jobs) != 94 or len(aliases) != 6 or len(jobs) != len(set(jobs)) or
            manifest.get("completed_fits") != len(jobs)):
        raise ValueError("chronological job inventory differs from its completed-fit count.")

    verification = _read_json(root / "verification.json")
    if not isinstance(verification, dict):
        raise ValueError("chronological verification summary is malformed.")
    verification_reports = _validate_chronological_verification(
        root, verification, jobs, aliases)

    included = {
        Path("manifest.json"), Path("summary.csv"), Path("threshold_definitions.json"),
        Path("verification.json"),
        *(Path("data") / f"farm{site}_{suffix}"
          for site in range(1, 7) for suffix in ("audit.json", "ids.npz")),
    }
    included |= verification_reports
    omitted = {
        Path(name) for name in (
            "environment.json", "study_tables.tex",
            "study_summary.json", "verification_stage1.json",
            "verification_crossfarm_final.json",
        ) if (root / name).is_file()
    } - verification_reports
    omitted |= {
        source_relative.relative_to(CHRONO_REL)
        for source_relative in source_overrides.values()
    }
    point_tau = protocol["point_tau"]
    interval_taus = protocol["interval_taus"]
    for name in jobs:
        task = "interval" if "interval" in name else "point"
        folder = Path(name)
        included |= {folder / "result.json", folder / "predictions.csv",
                     folder / "preprocessing.json", folder / "fit_ids.npz"}
        if task == "interval":
            included.add(folder / "calibration.csv")
        omitted.add(folder / "training.log")
        taus = interval_taus if task == "interval" else [point_tau]
        omitted |= {folder / f"model_tau_{tau}.npz" for tau in taus}
    included |= {Path(name) / "alias.json" for name in aliases}
    for relative in included | omitted:
        if not (root / relative).is_file():
            raise PackageNotReady(f"chronological canonical artifact is missing: {relative}")
    actual = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    if actual != included | omitted:
        unexpected = sorted(actual - included - omitted, key=str)
        missing = sorted((included | omitted) - actual, key=str)
        detail = unexpected[0] if unexpected else missing[0]
        raise ValueError(f"chronological artifact inventory has an unexpected difference: {detail}")
    hashes = {str(path): digest(root / path) for path in sorted(included, key=str)}
    return {"root": root, "manifest": manifest, "paths": included, "hashes": hashes,
            "omitted_paths": omitted, "jobs": jobs, "aliases": aliases,
            "source_overrides": source_overrides}


def _probability_csv_paths() -> tuple[set[Path], set[Path], set[Path]]:
    validation = {Path("within_site6") / method / "candidates" / f"c{index:02d}" /
                  "validation_quantiles.csv"
                  for method in METHODS for index in range(16)}
    calibration = {Path("within_site6") / method / f"final_seed{seed}" /
                   "calibration_quantiles.csv"
                   for method in METHODS for seed in PROBABILITY_METHOD_SEEDS[method]}
    test = {path.with_name("test_quantiles.csv") for path in calibration}
    return validation, calibration, test


def _current_probability_protocol_check(source_root: Path) -> dict[str, Any]:
    """Execute only the candidate-protocol constructor and compare its object exactly.

    This is a current-source consistency check.  It deliberately does not claim that
    the missing transitive hashes existed before the legacy probability run.
    """
    runner = source_root / "review/run_selection_study.py"
    archived = source_root / "review/selection_runs/selection_v1/protocol.json"
    expected = _read_json(archived)
    try:
        source = runner.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(runner))
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "protocol"]
        assignments = [
            node for node in tree.body if isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == "METHODS"
                for target in node.targets)
        ]
        if len(functions) != 1 or len(assignments) > 1:
            raise ValueError("expected one protocol() definition and at most one METHODS assignment")
        uses_methods = any(
            isinstance(node, ast.Name) and node.id == "METHODS" and
            isinstance(node.ctx, ast.Load) for node in ast.walk(functions[0]))
        if assignments:
            methods = ast.literal_eval(assignments[0].value)
        elif uses_methods:
            raise ValueError("protocol() references a missing METHODS assignment")
        else:
            methods = list(METHODS)
        from scipy.stats import qmc
        namespace: dict[str, Any] = {"np": np, "qmc": qmc, "METHODS": methods}
        protocol_module = ast.fix_missing_locations(
            ast.Module(body=[functions[0]], type_ignores=[]))
        exec(compile(protocol_module, str(runner), "exec"), namespace)
        current = namespace["protocol"]()
    except Exception as error:
        raise PackageNotReady(
            f"Cannot execute the current probability candidate protocol: {error}"
        ) from error
    if current != expected:
        raise ValueError(
            "Current run_selection_study.protocol() differs from the archived "
            "selection-v1 candidate protocol."
        )
    if (current.get("methods") != list(METHODS) or
            len(current.get("tasks", [])) != 7 or
            current.get("candidate_count_per_method") != 16):
        raise ValueError("Current probability candidate protocol has an unexpected design.")
    return {
        "status": "passed",
        "statement": PROBABILITY_CURRENT_CHECK,
        "runner_sha256": digest(runner),
        "archived_protocol_sha256": digest(archived),
        "methods": 6,
        "tasks": 7,
        "candidate_count_per_method": 16,
    }


def _validate_probability(source_root: Path) -> dict[str, Any]:
    root = source_root / PROBABILITY_REL
    manifest = _read_json(root / "manifest.json")
    protocol = _read_json(root / "protocol.json")
    lock = _read_json(root / "selection_lock.json")
    if (manifest.get("status") != "complete" or manifest.get("protocol") != protocol or
            protocol.get("version") != "probabilistic_selection_v2" or
            protocol.get("candidate_budget") != 16 or protocol.get("kernel_seeds") != [0, 1, 2] or
            protocol.get("tree_seeds") != [0] or manifest.get("selection_records") != 6 or
            manifest.get("probabilistic_records") != 16):
        raise PackageNotReady("probability selection-v2 design is incomplete or changed.")
    dependency = source_root / "review/selection_runs/selection_v1/protocol.json"
    if manifest.get("base_candidate_protocol_sha256") != digest(dependency):
        raise ValueError("probability v1 candidate-protocol dependency changed.")
    _require_sources(source_root, manifest.get("source_hashes"), label="probability source")
    if lock.get("protocol_sha256") != digest(root / "protocol.json"):
        raise ValueError("probability selection-lock protocol hash mismatch.")
    expected_selection = {f"within_site6/{method}/selection.json" for method in METHODS}
    if set(lock.get("selection_files", {})) != expected_selection:
        raise ValueError("probability selection-file inventory changed.")
    selection_paths = _require_hashes(root, lock["selection_files"],
                                      label="probability locked selection")
    selections = lock.get("selections")
    if (not isinstance(selections, list) or len(selections) != 6 or
            [row.get("method") for row in selections] != list(METHODS)):
        raise ValueError("probability lock does not contain the six ordered selections.")
    for row in selections:
        if _read_json(root / f"within_site6/{row['method']}/selection.json") != row:
            raise ValueError(f"probability embedded selection changed: {row['method']}")

    results = _read_json(root / "results.json")
    expected_keys = [(method, seed) for method in METHODS
                     for seed in PROBABILITY_METHOD_SEEDS[method]]
    keys = [(row.get("method"), row.get("seed")) for row in results or []]
    if len(keys) != 16 or keys != expected_keys:
        raise ValueError("probability results do not contain the canonical 16 records.")
    result_paths: set[Path] = set()
    for row in results:
        path = Path("within_site6") / row["method"] / f"final_seed{row['seed']}" / "result.json"
        if _read_json(root / path) != row:
            raise ValueError(f"probability aggregate/per-seed result mismatch: {path}")
        result_paths.add(path)

    analysis_root = source_root / PROBABILITY_ANALYSIS_REL
    analysis_manifest = _read_json(analysis_root / "manifest.json")
    analysis_protocol = _read_json(analysis_root / "protocol.json")
    if (analysis_manifest.get("status") != "complete" or
            analysis_manifest.get("protocol") != analysis_protocol or
            analysis_protocol.get("version") != "probabilistic_analysis_v2" or
            analysis_manifest.get("experiment_protocol_sha256") != digest(root / "protocol.json") or
            analysis_manifest.get("experiment_runner_sha256") != digest(
                source_root / "review/run_probabilistic_selection_study.py") or
            analysis_manifest.get("methods") != 6 or
            analysis_manifest.get("paired_comparisons") != 5):
        raise ValueError("probability analysis identity or completion counts changed.")
    _require_sources(source_root, analysis_manifest.get("source_hashes"),
                     label="probability analysis source")
    expected_inputs = {
        str((root / name).relative_to(source_root)): digest(root / name)
        for name in ("manifest.json", "selection_lock.json", "results.json")
    }
    for method, seed in expected_keys:
        path = root / "within_site6" / method / f"final_seed{seed}" / "test_quantiles.csv"
        expected_inputs[str(path.relative_to(source_root))] = digest(path)
    if analysis_manifest.get("input_hashes") != expected_inputs:
        raise ValueError("probability analysis input hash inventory changed.")

    inventory = _read_json(analysis_root / "posthoc_integrity_inventory.json")
    validation, calibration, test = _probability_csv_paths()
    if (inventory.get("role") != "post-completion integrity inventory; not preregistration" or
            inventory.get("counts") != {"candidate_validation_csv": 96,
                                        "selected_calibration_csv": 16,
                                        "selected_test_csv": 16}):
        raise ValueError("probability post-completion inventory metadata changed.")
    controls = {
        PROBABILITY_REL / name for name in
        ("protocol.json", "manifest.json", "selection_lock.json", "results.json")
    } | {
        PROBABILITY_ANALYSIS_REL / name for name in
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
    expected_inventory = controls | {
        PROBABILITY_REL / path for path in validation | calibration | test
    }
    if set(inventory.get("sha256", {})) != {str(path) for path in expected_inventory}:
        raise ValueError("probability post-completion hash inventory is not the canonical 140 files.")
    _require_hashes(source_root, inventory["sha256"], label="probability posthoc")

    for name in PROBABILITY_ANALYSIS_FILES:
        if not (analysis_root / name).is_file():
            raise PackageNotReady(f"probability analysis artifact is missing: {name}")
    presentation_root = source_root / PROBABILITY_PRESENTATION_REL
    presentation = _read_json(presentation_root / "manifest.json")
    expected_outputs = PROBABILITY_PRESENTATION_FILES - {"manifest.json"}
    if (presentation.get("input") != str((analysis_root / "aggregate_results.json").relative_to(source_root)) or
            presentation.get("input_sha256") != digest(analysis_root / "aggregate_results.json") or
            presentation.get("source") != "review/render_probabilistic_figures.py" or
            presentation.get("source_sha256") != digest(
                source_root / "review/render_probabilistic_figures.py") or
            set(presentation.get("outputs", {})) != expected_outputs):
        raise ValueError("probability presentation identity or output inventory changed.")
    for name, expected_hash in presentation["outputs"].items():
        if digest(presentation_root / name) != expected_hash:
            raise ValueError(f"probability presentation output hash mismatch: {name}")

    selection_paths |= {Path("protocol.json"), Path("manifest.json"), Path("selection_lock.json"),
                        Path("results.json"), *result_paths, *validation, *calibration, *test}
    selection_hashes = {str(path): digest(root / path) for path in sorted(selection_paths, key=str)}
    analysis_paths = {Path(name) for name in PROBABILITY_ANALYSIS_FILES}
    analysis_hashes = {str(path): digest(analysis_root / path)
                       for path in sorted(analysis_paths, key=str)}
    presentation_paths = {Path(name) for name in PROBABILITY_PRESENTATION_FILES}
    presentation_hashes = {str(path): digest(presentation_root / path)
                           for path in sorted(presentation_paths, key=str)}
    return {
        "selection": {"root": root, "paths": selection_paths, "hashes": selection_hashes},
        "analysis": {"root": analysis_root, "paths": analysis_paths, "hashes": analysis_hashes},
        "presentation": {"root": presentation_root, "paths": presentation_paths,
                         "hashes": presentation_hashes},
        "legacy_limitation": LEGACY_PROBABILITY_LIMITATION,
        "current_candidate_protocol_check": _current_probability_protocol_check(source_root),
    }


def _validate_query_records(root: Path) -> tuple[int, list[dict[str, Any]]]:
    fit_count = 0
    fit_inventory: list[dict[str, Any]] = []
    fit_record_keys = {
        "task", "site", "method", "candidate_id", "config", "seed", "tau",
        "feature_map", "data_folder", "diagnostics", "selection_counts",
        "effective_target_fraction",
    }
    for task in TASKS:
        for method in METHODS:
            data_subfolder = "target_data" if method == "TargetSVQR" else "pooled_data"
            expected_counts = _read_json(
                root / task / data_subfolder / "selection_counts.json")
            signatures = []
            for stage in ("base", "middle", "joint"):
                record = _read_json(root / task / method / f"{stage}_selection.json")
                candidates = record.get("candidates")
                if ((record.get("task"), record.get("method"), record.get("stage")) !=
                        (task, method, stage) or record.get("candidate_count") != 18 or
                        record.get("queried_effective_unique_count") != 18 or
                        not isinstance(candidates, list) or len(candidates) != 18):
                    raise ValueError(f"Malformed 18-query selection stage: {task}/{method}/{stage}")
                stage_signatures = [_effective_signature(method, row.get("config"))
                                    for row in candidates]
                if len(set(stage_signatures)) != 18:
                    raise ValueError(f"Repeated effective query: {task}/{method}/{stage}")
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
                            f"Malformed selection-fit identity: {task}/{method}/{stage}/c{index:03d}")
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
                                f"Malformed selection-fit evidence: {task}/{method}/{stage}/c{index:03d}")
                        fit_folder = Path(fit_folders[str(seed)])
                        fit_path = root / fit_folder / "fit.json"
                        metrics_path = root / fit_folder / "validation_metrics.json"
                        fit_record = _read_json(fit_path)
                        validation_metrics = _read_json(metrics_path)
                        expected_fit = {
                            "task": task, "site": int(task.rsplit("site", 1)[1]),
                            "method": method, "config": candidate["config"],
                            "seed": seed, "tau": 0.5,
                            "candidate_id": f"{stage}_c{index:03d}",
                            "feature_map": (
                                None if method == "HistGBR" else
                                f"{task}/{data_subfolder}/landmarks/seed{seed}/"
                                f"rank{int(candidate['config']['rank'])}/gamma_"
                                f"{format(float(candidate['config']['gamma']), '.17g')}/map.npz"
                            ),
                            "data_folder": f"{task}/{data_subfolder}",
                            "selection_counts": expected_counts,
                            "effective_target_fraction": (
                                1.0 if method == "TargetSVQR" else
                                float(candidate["config"].get("target_fraction", 1.0))
                            ),
                        }
                        if (set(fit_record) != fit_record_keys or
                                any(fit_record.get(field) != value
                                    for field, value in expected_fit.items())):
                            raise ValueError(
                                f"Selection fit-record identity changed: {fit_path}")
                        if fit_record.get("diagnostics") != diagnostics:
                            raise ValueError(
                                f"Selection fit diagnostics disagree with its record: {fit_path}")
                        if validation_metrics != metrics:
                            raise ValueError(
                                f"Selection validation metrics disagree with their record: "
                                f"{metrics_path}")
                        converged = diagnostics.get("converged")
                        if ((method == "HistGBR" and converged is not None) or
                                (method != "HistGBR" and type(converged) is not bool)):
                            raise ValueError(
                                f"Selection fit has malformed certificate status: {fit_path}")
                        fit_inventory.append({
                            "key": f"{task}|{method}|{stage}|c{index:03d}|seed{seed}",
                            "fit_record": str(fit_path.relative_to(root)),
                            "fit_record_sha256": digest(fit_path),
                            "validation_metrics": str(metrics_path.relative_to(root)),
                            "validation_metrics_sha256": digest(metrics_path),
                            "config_sha256": _object_digest(candidate["config"]),
                            "diagnostics_sha256": _object_digest(diagnostics),
                            "metrics_sha256": _object_digest(metrics),
                            "certificate_status": (
                                "not_applicable_tree" if method == "HistGBR" else
                                ("passed" if converged else "failed_ineligible_candidate")
                            ),
                        })
                        rmse.append(float(metrics["rmse"]))
                        mae.append(float(metrics["mae"]))
                        if method != "HistGBR" and not converged:
                            failed.append(seed_result["seed"])
                    expected_eligible = not failed
                    if (candidate.get("failed_certificate_seeds") != failed or
                            candidate.get("eligible_for_selection") is not expected_eligible or
                            not math.isclose(float(candidate.get("mean_validation_rmse", math.nan)),
                                             float(np.mean(rmse)), rel_tol=1e-12, abs_tol=1e-12) or
                            not math.isclose(float(candidate.get("std_validation_rmse", math.nan)),
                                             float(np.std(rmse, ddof=0)), rel_tol=1e-12,
                                             abs_tol=1e-12) or
                            not math.isclose(float(candidate.get("mean_validation_mae", math.nan)),
                                             float(np.mean(mae)), rel_tol=1e-12, abs_tol=1e-12)):
                        raise ValueError(
                            f"Inconsistent selection-fit summary: {task}/{method}/{stage}/c{index:03d}")
                    fit_count += len(expected_seeds)
                signatures.extend(stage_signatures)
            if len(set(signatures)) != 54:
                raise ValueError(f"Selection stages are not 54 distinct queries: {task}/{method}")
    if fit_count != 10368:
        raise ValueError("Selection records do not certify the canonical 10,368 fits.")
    if len(fit_inventory) != fit_count or len({row["key"] for row in fit_inventory}) != fit_count:
        raise ValueError("Selection fit-record inventory is incomplete or duplicated.")
    return fit_count, sorted(fit_inventory, key=lambda row: row["key"])


def _validate_calibration_csv(path: Path, expected_rows: object) -> None:
    if type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError(f"Invalid calibration-row count for {path}.")
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
        raise ValueError(f"Malformed selection calibration CSV {path}: {error}") from error
    if count != expected_rows:
        raise ValueError(f"Selection calibration row count mismatch: {path}")


def _validate_matched_prediction_csv(path: Path, expected_rows: object,
                                     expected_metrics: object) -> None:
    if (type(expected_rows) is not int or expected_rows <= 0 or
            not isinstance(expected_metrics, dict)):
        raise ValueError(f"Invalid matched prediction summary for {path}.")
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
        raise ValueError(f"Malformed matched prediction CSV {path}: {error}") from error
    if count != expected_rows:
        raise ValueError(f"Matched prediction row count mismatch: {path}")
    y, prediction = np.asarray(targets), np.asarray(predictions)
    residual = y - prediction
    denominator = float(np.sum((y - y.mean()) ** 2))
    observed = {
        "n": len(y), "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual ** 2) / denominator) if denominator else None,
    }
    for name, value in observed.items():
        expected = expected_metrics.get(name)
        if (value is None and expected is not None) or (value is not None and
                (not isinstance(expected, (int, float)) or
                 not math.isclose(value, float(expected), rel_tol=1e-10, abs_tol=1e-10))):
            raise ValueError(f"Matched prediction metric mismatch ({name}): {path}")


def _validate_selection(source_root: Path) -> dict[str, Any]:
    root = source_root / SELECTION_REL
    # Deliberately stop here on an unfinished run; no held-out file is opened.
    manifest = _read_json(root / "manifest.json")
    _complete(manifest, label="selection-v2 evaluation")
    if manifest.get("last_stable_status") != "complete":
        raise PackageNotReady("selection-v2 lacks its complete stable-state marker.")

    protocol = _read_json(root / "protocol.json")
    lock = _read_json(root / "selection_lock.json")
    if protocol.get("version") != "selection_v2" or lock.get("version") != "selection_v2_lock_2":
        raise ValueError("Unexpected selection-v2 protocol or lock version.")
    if lock.get("protocol_sha256") != digest(root / "protocol.json"):
        raise ValueError("selection-v2 protocol hash mismatch.")
    if manifest.get("protocol") != protocol or manifest.get("protocol_sha256") != lock["protocol_sha256"]:
        raise ValueError("selection-v2 protocol, manifest, and lock disagree.")
    for key in ("source_hashes", "data_hashes", "environment_versions"):
        if manifest.get(key) != lock.get(key):
            raise ValueError(f"selection-v2 manifest and lock disagree on {key}.")
    _require_sources(source_root, lock.get("source_hashes"), label="selection source")
    _require_hashes(source_root, lock.get("data_hashes"), label="selection data")
    provenance = root / "data_provenance.json"
    if not provenance.is_file() or digest(provenance) != lock.get("data_provenance_sha256"):
        raise ValueError("selection-v2 data-provenance hash mismatch.")
    if lock.get("total_configuration_queries") != 3888:
        raise ValueError("selection-v2 lock does not certify 12 x 6 x 54 queries.")
    if lock.get("validation_query_budget") != protocol.get("validation_query_budget"):
        raise ValueError("selection-v2 query budgets disagree.")
    if protocol.get("methods") != list(METHODS) or tuple(
            row.get("name") for row in protocol.get("tasks", [])) != TASKS:
        raise ValueError("selection-v2 task or method order changed.")
    if protocol.get("taus") != [0.5, 0.025, 0.975]:
        raise ValueError("selection-v2 locked evaluation quantiles changed.")
    if protocol.get("selection_seeds_by_method") != {
            method: list(seeds) for method, seeds in SELECTION_METHOD_SEEDS.items()}:
        raise ValueError("selection-v2 selection seed design changed.")

    if set(lock.get("selection_files", {})) != _expected_selection_files():
        raise ValueError("selection-v2 selection-file inventory is incomplete or unexpected.")
    selection_paths = _require_hashes(root, lock["selection_files"],
                                      label="selection-v2 locked selection")
    selection_fit_count, selection_fit_inventory = _validate_query_records(root)
    final = lock.get("final_selections")
    expected_final = {(task, method) for task in TASKS for method in METHODS}
    final_keys = [(row.get("task"), row.get("method")) for row in final or []]
    if len(final_keys) != 72 or len(set(final_keys)) != 72 or set(final_keys) != expected_final:
        raise ValueError("selection-v2 lock lacks the 72 unique final selections.")

    results = _read_json(root / "results.json")
    seed_map = protocol.get("evaluation_seeds_by_method", {})
    expected_keys = {(task, method, seed) for task in TASKS for method in METHODS
                     for seed in seed_map.get(method, [])}
    observed = [(row.get("task"), row.get("method"), row.get("seed")) for row in results]
    if len(expected_keys) != 192 or len(observed) != 192 or len(set(observed)) != 192 \
            or set(observed) != expected_keys:
        raise ValueError("selection-v2 results do not contain the complete 192-record design.")
    lock_sha = digest(root / "selection_lock.json")
    by_key = {(row.get("task"), row.get("method"), row.get("seed")): row for row in results}
    selected = {(row.get("task"), row.get("method")): row.get("selected_config")
                for row in final}
    calibration_paths: set[Path] = set()
    amended_fit_paths: set[Path] = set()
    evaluation_fit_inventory: list[dict[str, Any]] = []
    evaluation_fit_count = 0
    for task, method, seed in expected_keys:
        folder = root / task / method / "evaluation" / f"seed{seed}"
        result_path, prediction = folder / "result.json", folder / "test_predictions.csv"
        calibration = folder / "calibration_predictions.csv"
        if not result_path.is_file() or not prediction.is_file() or not calibration.is_file():
            raise PackageNotReady(f"selection-v2 evaluation artifact is missing: {folder}")
        disk = _read_json(result_path)
        if disk != by_key[(task, method, seed)]:
            raise ValueError(f"selection-v2 aggregate/per-seed mismatch: {result_path}")
        if disk.get("selection_lock_sha256") != lock_sha:
            raise ValueError(f"selection-v2 result used a different lock: {result_path}")
        if disk.get("selected_config") != selected[(task, method)]:
            raise ValueError(f"selection-v2 result used a different configuration: {result_path}")
        _validate_calibration_csv(calibration, disk.get("counts", {}).get("n_calibration"))
        fits = disk.get("fits")
        if not isinstance(fits, list) or len(fits) != 3:
            raise ValueError(f"selection-v2 result does not contain three tau fits: {result_path}")
        if any(not isinstance(fit, dict) for fit in fits):
            raise ValueError(f"selection-v2 fit is not a mapping: {result_path}")
        if [fit.get("tau") for fit in fits] != protocol["taus"]:
            raise ValueError(f"selection-v2 result tau-fit order changed: {result_path}")
        for fit, tau in zip(fits, protocol["taus"]):
            expected_path = str((Path(task) / method / "evaluation" / f"seed{seed}" /
                                 ("tau_" + format(float(tau), ".17g"))))
            expected_fit = {"task": task, "method": method, "config": selected[(task, method)],
                            "seed": seed, "tau": tau, "candidate_id": "locked_final",
                            "path": expected_path}
            if any(fit.get(field) != value for field, value in expected_fit.items()):
                raise ValueError(f"selection-v2 fit identity changed: {result_path}")
            fit_path = root / expected_path / "fit.json"
            fit_record = _read_json(fit_path)
            embedded = dict(fit)
            embedded.pop("path", None)
            if "path" in fit_record or embedded != fit_record:
                raise ValueError(
                    f"selection-v2 embedded fit disagrees with fit.json: {fit_path}")
            if (method != "HistGBR" and
                    fit_record.get("diagnostics", {}).get("converged") is not True):
                raise ValueError(
                    f"selection-v2 kernel fit lacks the solver certificate: {fit_path}")
            if "solver_certificate_amendment" in fit_record.get("diagnostics", {}):
                amended_fit_paths.add(fit_path.relative_to(root))
            evaluation_fit_inventory.append({
                "key": (f"{task}|{method}|seed{seed}|tau"
                        f"{format(float(tau), '.17g')}"),
                "fit_record": str(fit_path.relative_to(root)),
                "fit_record_sha256": digest(fit_path),
                "embedded_fit_sha256": _object_digest(embedded),
                "certificate_status": (
                    "not_applicable_tree" if method == "HistGBR" else "passed"
                ),
            })
        evaluation_fit_count += len(fits)
        calibration_paths.add(calibration.relative_to(root))
    if evaluation_fit_count != 576:
        raise ValueError("selection-v2 evaluation does not contain the canonical 576 tau fits.")
    if (len(evaluation_fit_inventory) != evaluation_fit_count or
            len({row["key"] for row in evaluation_fit_inventory}) != evaluation_fit_count):
        raise ValueError("selection-v2 evaluation fit-record inventory is incomplete or duplicated.")
    fit_record_inventory = {
        "version": FIT_RECORD_INVENTORY_VERSION,
        "hash_authority": FIT_RECORD_HASH_AUTHORITY,
        "source_validation": FIT_RECORD_SOURCE_VALIDATION,
        "counts": {
            "selection_fit_records": selection_fit_count,
            "selection_validation_metric_records": selection_fit_count,
            "evaluation_fit_records": evaluation_fit_count,
        },
        "selection_fit_records": selection_fit_inventory,
        "evaluation_fit_records": sorted(
            evaluation_fit_inventory, key=lambda row: row["key"]),
    }
    return {"root": root, "manifest": manifest, "protocol": protocol, "lock": lock,
            "results": results, "selection_paths": selection_paths,
            "selection_fit_count": selection_fit_count,
            "evaluation_fit_count": evaluation_fit_count,
            "fit_record_inventory": fit_record_inventory,
            "amended_fit_paths": amended_fit_paths,
            "calibration_paths": calibration_paths,
            "calibration_hashes": {str(path): digest(root / path)
                                   for path in sorted(calibration_paths, key=str)}}


def _validate_selection_analysis(source_root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    root = source_root / SELECTION_ANALYSIS_REL
    manifest = _read_json(root / "analysis_manifest.json")
    _complete(manifest, label="selection-v2 analysis")
    protocol = _read_json(root / "analysis_protocol.json")
    if (protocol.get("version") != "selection_v2_analysis_2" or
            manifest.get("protocol") != protocol or
            manifest.get("protocol_sha256") != digest(root / "analysis_protocol.json")):
        raise ValueError("selection-v2 analysis protocol and manifest disagree.")
    if manifest.get("analysis_source_sha256") != digest(source_root / "review/analyze_selection_study_v2.py"):
        raise ValueError("selection-v2 analysis source hash mismatch.")
    counts = manifest.get("reporting_counts", {})
    if any(counts.get(key) != value for key, value in {
        "tasks": 12, "methods": 6, "locked_result_records": 192,
        "task_method_rows": 72, "full_comparisons": 60,
    }.items()):
        raise ValueError("selection-v2 analysis reporting counts changed.")
    artifacts = _require_hashes(root, manifest.get("artifact_hashes"),
                                label="selection-v2 analysis")
    if {str(path) for path in artifacts} != SELECTION_ANALYSIS_FILES:
        raise ValueError("selection-v2 analysis artifact inventory changed.")
    inputs = _require_hashes(selection["root"], manifest.get("selection_input_hashes"),
                             label="selection-v2 analysis input")
    if not {Path("selection_lock.json"), Path("results.json")}.issubset(inputs):
        raise ValueError("selection-v2 analysis input inventory omits its lock or results.")
    expected_inputs = {Path("selection_lock.json"), Path("results.json")}
    for task in TASKS:
        for method in METHODS:
            for seed in selection["protocol"]["evaluation_seeds_by_method"][method]:
                folder = Path(task) / method / "evaluation" / f"seed{seed}"
                expected_inputs |= {folder / "result.json", folder / "test_predictions.csv"}
    if inputs != expected_inputs:
        raise ValueError("selection-v2 analysis input inventory is not the canonical 386 files.")
    return {"root": root, "manifest": manifest, "protocol": protocol,
            "artifact_paths": artifacts,
            "input_paths": inputs}


def _validate_matched(source_root: Path, selection: dict[str, Any]) -> dict[str, Any]:
    root = source_root / MATCHED_REL
    manifest = _read_json(root / "manifest.json")
    _complete(manifest, label="matched-ablation-v2 evaluation")
    if manifest.get("heldout_unlocked") is not True or manifest.get("completed_evaluation_keys") != 144:
        raise PackageNotReady("matched-ablation-v2 does not certify all 144 evaluations.")
    protocol = _read_json(root / "protocol.json")
    configurations = _read_json(root / "configurations.json")
    lock = _read_json(root / "matched_lock.json")
    if (protocol.get("version") != "matched_ablation_v2" or
            configurations.get("version") != "matched_ablation_v2_configurations_1" or
            lock.get("version") != "matched_ablation_v2_lock_1"):
        raise ValueError("Unexpected matched-ablation-v2 frozen version.")
    if (manifest.get("version") != "matched_ablation_v2_manifest_1" or
            protocol.get("base_experiment") != "selection_v2" or
            protocol.get("task_order") != selection["protocol"].get("tasks") or
            protocol.get("variants") != list(VARIANTS) or
            protocol.get("evaluation_seeds") != list(MATCHED_SEEDS) or
            protocol.get("tau") != MATCHED_TAU):
        raise ValueError("Matched-ablation-v2 protocol design changed.")
    if (lock.get("protocol_sha256") != digest(root / "protocol.json") or
            lock.get("configurations_sha256") != digest(root / "configurations.json")):
        raise ValueError("matched-ablation-v2 protocol/configuration hash mismatch.")
    _require_sources(source_root, lock.get("source_hashes"), label="matched source")
    if (lock.get("environment_versions") != protocol.get("environment_versions") or
            manifest.get("protocol_sha256") != lock.get("protocol_sha256") or
            manifest.get("configurations_sha256") != lock.get("configurations_sha256")):
        raise ValueError("matched-ablation-v2 frozen identities disagree.")
    base = lock.get("base_selection", {})
    expected_base = {
        "selection_lock_sha256": digest(selection["root"] / "selection_lock.json"),
        "protocol_sha256": selection["lock"]["protocol_sha256"],
        "source_hashes": selection["lock"]["source_hashes"],
        "data_hashes": selection["lock"]["data_hashes"],
        "environment_versions": selection["lock"]["environment_versions"],
        "data_provenance_sha256": selection["lock"]["data_provenance_sha256"],
    }
    if any(base.get(key) != value for key, value in expected_base.items()):
        raise ValueError("matched-ablation-v2 names a different base selection lock.")
    fit_only = _require_hashes(selection["root"], lock.get("fit_only_artifact_hashes"),
                               label="matched fit-only")
    configuration_tasks = configurations.get("tasks")
    task_records = _validate_matched_configurations(selection, configurations)
    declared_shared: set[Path] = set()
    for task in configuration_tasks:
        shared = task.get("shared_artifacts")
        if not isinstance(shared, dict) or len(shared) != 10:
            raise ValueError("Each matched task must name 10 shared fit-only artifacts.")
        declared_shared |= {_safe_relative(value, label="matched shared artifact")
                            for value in shared.values()}
    if fit_only != declared_shared or len(fit_only) != 120:
        raise ValueError("matched fit-only inventory is not the locked 12 x 10 design.")
    expected_keys = [f"{task}|{variant}|seed{seed}" for task in TASKS
                     for variant in VARIANTS for seed in range(3)]
    if lock.get("expected_evaluation_keys") != expected_keys:
        raise ValueError("matched-ablation-v2 lock does not declare the canonical 144 keys.")
    if manifest.get("matched_lock_sha256") != digest(root / "matched_lock.json"):
        raise ValueError("matched-ablation-v2 manifest lock hash mismatch.")
    if manifest.get("base_selection_lock_sha256") != digest(
            selection["root"] / "selection_lock.json"):
        raise ValueError("matched-ablation-v2 manifest base-lock hash mismatch.")
    results_path = root / "results.json"
    if not results_path.is_file() or manifest.get("results_sha256") != digest(results_path):
        raise ValueError("matched-ablation-v2 results hash mismatch.")
    results = _read_json(results_path)
    keys = [row.get("evaluation_key") for row in results]
    if len(results) != 144 or len(set(keys)) != 144 or keys != expected_keys:
        raise ValueError("matched-ablation-v2 results do not match the locked key order.")
    result_paths: set[Path] = {Path("results.json")}
    amended_fit_paths: set[Path] = set()
    for row in results:
        task, variant, seed_name = row["evaluation_key"].split("|")
        seed = int(seed_name.removeprefix("seed"))
        task_record = task_records[task]
        result_rel = Path("evaluation") / task / variant / seed_name / "result.json"
        if _read_json(root / result_rel) != row:
            raise ValueError(f"matched aggregate/per-key result mismatch: {result_rel}")
        result_paths.add(result_rel)
        expected_identity = {
            "task": task, "site": task_record["site"], "mode": task_record["mode"],
            "role": task_record["role"], "variant": variant, "seed": seed,
            "tau": MATCHED_TAU, "config": task_record["variants"][variant],
            "matched_lock_sha256": digest(root / "matched_lock.json"),
            "base_selection_lock_sha256": digest(selection["root"] / "selection_lock.json"),
            "shared_artifacts": task_record["shared_artifacts"],
            "feature_map": task_record["shared_artifacts"][f"seed{seed}_feature_map"],
            "data_folder": str(Path(task_record["shared_artifacts"]["preprocessing"]).parent),
        }
        for field, expected in expected_identity.items():
            if row.get(field) != expected:
                raise ValueError(f"matched-ablation-v2 result {field} mismatch: {row['evaluation_key']}")
        folder = result_rel.parent
        expected_files = {
            "fit_record": str(folder / "fit.json"),
            "prediction_file": str(folder / "test_predictions.csv"),
            "model_file": str(folder / "model.npz"),
        }
        for field, expected in expected_files.items():
            if row.get(field) != expected:
                raise ValueError(f"matched-ablation-v2 noncanonical {field}: {row['evaluation_key']}")
        for field, hash_field in (("fit_record", "fit_record_sha256"),
                                  ("prediction_file", "prediction_sha256")):
            relative = _safe_relative(row.get(field, ""), label=f"matched {field}")
            artifact = root / relative
            if not artifact.is_file() or digest(artifact) != row.get(hash_field):
                raise ValueError(f"matched-ablation-v2 {field} hash mismatch: {relative}")
            result_paths.add(relative)
        _validate_matched_prediction_csv(
            root / expected_files["prediction_file"],
            row.get("counts", {}).get("n_test"), row.get("point"))
        # Models are verified in the source run, then omitted from S1 for size.
        model = _safe_relative(row.get("model_file", ""), label="matched model")
        if not (root / model).is_file() or digest(root / model) != row.get("model_sha256"):
            raise ValueError(f"matched-ablation-v2 model hash mismatch: {model}")
        fit = _read_json(root / expected_files["fit_record"])
        expected_fit = {
            "task": task, "method": variant, "config": task_record["variants"][variant],
            "seed": seed, "tau": MATCHED_TAU,
            "candidate_id": "matched_ablation_v2_locked",
            "feature_map": expected_identity["feature_map"],
            "data_folder": expected_identity["data_folder"],
        }
        for field, expected in expected_fit.items():
            if fit.get(field) != expected:
                raise ValueError(f"matched-ablation-v2 fit {field} mismatch: {row['evaluation_key']}")
        if fit.get("diagnostics", {}).get("converged") is not True:
            raise ValueError(f"matched-ablation-v2 fit lacks convergence certificate: {row['evaluation_key']}")
        if row.get("diagnostics") != fit.get("diagnostics"):
            raise ValueError(f"matched-ablation-v2 result/fit diagnostics mismatch: {row['evaluation_key']}")
        if "solver_certificate_amendment" in fit.get("diagnostics", {}):
            amended_fit_paths.add(Path(expected_files["fit_record"]))
    return {"root": root, "manifest": manifest, "protocol": protocol,
            "configurations": configurations, "lock": lock, "results": results,
            "fit_only_paths": fit_only, "result_paths": result_paths,
            "amended_fit_paths": amended_fit_paths}


def _validate_matched_analysis(source_root: Path, matched: dict[str, Any]) -> dict[str, Any]:
    root = source_root / MATCHED_ANALYSIS_REL
    manifest = _read_json(root / "manifest.json")
    _complete(manifest, label="matched-ablation-v2 analysis")
    if manifest.get("heldout_results_read") is not True:
        raise PackageNotReady("matched-ablation-v2 analysis has not read the complete results.")
    protocol = _read_json(root / "analysis_protocol.json")
    lock = _read_json(root / "analysis_lock.json")
    if (protocol.get("version") != "matched_ablation_v2_analysis_1" or
            lock.get("version") != "matched_ablation_v2_analysis_lock_1"):
        raise ValueError("Unexpected matched-ablation-v2 analysis version.")
    if manifest.get("version") != "matched_ablation_v2_analysis_manifest_1":
        raise ValueError("Unexpected matched-ablation-v2 analysis manifest version.")
    if lock.get("analysis_protocol_sha256") != digest(root / "analysis_protocol.json"):
        raise ValueError("matched-ablation-v2 analysis protocol hash mismatch.")
    _require_sources(source_root, lock.get("source_hashes"), label="matched analysis source")
    if (lock.get("environment_versions") != protocol.get("environment_versions") or
            lock.get("matched_protocol_sha256") != matched["lock"].get("protocol_sha256") or
            lock.get("matched_configurations_sha256") != matched["lock"].get("configurations_sha256") or
            lock.get("expected_evaluation_keys") != matched["lock"].get("expected_evaluation_keys")):
        raise ValueError("matched analysis lock disagrees with its experiment or protocol.")
    if lock.get("matched_lock_sha256") != digest(matched["root"] / "matched_lock.json"):
        raise ValueError("matched analysis names a different matched lock.")
    if manifest.get("analysis_lock_sha256") != digest(root / "analysis_lock.json"):
        raise ValueError("matched analysis manifest lock hash mismatch.")
    outputs = _require_hashes(root, manifest.get("output_hashes"),
                              label="matched-ablation-v2 analysis")
    if set(protocol.get("output_files", [])) != {str(path) for path in outputs} or len(outputs) != 7:
        raise ValueError("matched analysis output inventory is not the declared seven files.")
    summary = _read_json(root / "summary.json")
    if (summary.get("version") != "matched_ablation_v2_analysis_results_1" or
            summary.get("analysis_lock_sha256") != digest(root / "analysis_lock.json") or
            summary.get("matched_lock_sha256") != digest(matched["root"] / "matched_lock.json") or
            summary.get("matched_results_sha256") != digest(matched["root"] / "results.json") or
            summary.get("complete_evaluation_keys") != 144 or summary.get("task_metric_rows") != 48 or
            summary.get("paired_tests") != {"within": 18, "cross": 18}):
        raise ValueError("matched-ablation-v2 analysis identity or reporting counts changed.")
    return {"root": root, "manifest": manifest, "lock": lock, "output_paths": outputs}


def _validate_solver_amendment(source_root: Path, selection: dict[str, Any],
                               matched: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and inventory the performance-blind certificate amendment.

    The amendment is required exactly when a completed fit declares amendment
    provenance.  Synthetic or future runs with no amended fit retain the
    original package contract without inventing an amendment record.
    """
    main_amended = set(selection.get("amended_fit_paths", set()))
    matched_amended = set(matched.get("amended_fit_paths", set()))
    if not main_amended and not matched_amended:
        return None
    if len(main_amended) != 1 or matched_amended:
        raise ValueError(
            "The archived study must contain exactly one amended main fit and no "
            "amended matched fit."
        )

    runner = source_root / "review/resume_solver_certificate_v2.py"
    lock_path = source_root / AMENDMENT_LOCK_REL
    if not runner.is_file() or not lock_path.is_file():
        raise PackageNotReady("The solver-certificate amendment source or lock is missing.")
    amendment = _read_json(lock_path)
    if amendment.get("version") != AMENDMENT_VERSION:
        raise ValueError("Unexpected solver-certificate amendment version.")
    if (amendment.get("amendment_runner") !=
            str(Path("review/resume_solver_certificate_v2.py")) or
            amendment.get("amendment_runner_sha256") != digest(runner)):
        raise ValueError("The frozen solver-amendment runner changed.")
    expected_extension_rule = [{
        "max_iter": 96000, "gap_rtol": 0.001,
        "gap_atol": 1e-7, "check_every": 25,
    }]
    if amendment.get("rule", {}).get("conditional_warm_restart") != expected_extension_rule:
        raise ValueError("The frozen solver-amendment restart rule changed.")
    if (amendment.get("selection", {}).get("output") != str(SELECTION_REL) or
            amendment.get("matched", {}).get("output") != str(MATCHED_REL) or
            amendment["selection"].get("selection_lock_sha256") !=
            digest(selection["root"] / "selection_lock.json") or
            amendment["selection"].get("protocol_sha256") !=
            digest(selection["root"] / "protocol.json") or
            amendment["selection"].get("source_hashes") !=
            selection["lock"].get("source_hashes") or
            amendment["selection"].get("data_hashes") !=
            selection["lock"].get("data_hashes") or
            amendment["matched"].get("matched_lock_sha256") !=
            digest(matched["root"] / "matched_lock.json") or
            amendment["matched"].get("source_hashes") !=
            matched["lock"].get("source_hashes")):
        raise ValueError("The solver amendment no longer binds the frozen studies.")

    completion_commitments = amendment.get("preexisting_completion_artifacts", {})
    if (not isinstance(completion_commitments, dict) or
            len(completion_commitments) != 110 or
            amendment.get("partial_evaluation_disclosure", {}).get(
                "main_result_records_already_present") != 10 or
            amendment.get("partial_evaluation_disclosure", {}).get(
                "matched_result_records_already_present") != 5):
        raise ValueError("The pre-amendment completion commitment inventory changed.")
    for raw_relative, expected_hash in completion_commitments.items():
        relative = _safe_relative(raw_relative, label="pre-amendment artifact")
        path = source_root / relative
        if not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f"A pre-amendment completion artifact changed: {relative}")

    # The two manifests were state records, not analysis specifications: after
    # this lock they advanced from frozen/not-started to their separately
    # validated complete state.  The protocol and matched analysis lock remain
    # byte-identical to the pre-amendment commitments.
    allowed_manifest_transitions = {
        SELECTION_ANALYSIS_REL / "analysis_manifest.json",
        MATCHED_ANALYSIS_REL / "manifest.json",
    }
    expected_analysis_commitments = {
        SELECTION_ANALYSIS_REL / "analysis_protocol.json",
        SELECTION_ANALYSIS_REL / "analysis_manifest.json",
        MATCHED_ANALYSIS_REL / "analysis_protocol.json",
        MATCHED_ANALYSIS_REL / "analysis_lock.json",
        MATCHED_ANALYSIS_REL / "manifest.json",
    }
    analysis_commitments = amendment.get("preexisting_analysis_freezes", {})
    if (not isinstance(analysis_commitments, dict) or
            {_safe_relative(path, label="pre-amendment analysis file")
             for path in analysis_commitments} != expected_analysis_commitments):
        raise ValueError("The pre-amendment analysis commitment inventory changed.")
    observed_manifest_transitions: set[Path] = set()
    for raw_relative, expected_hash in analysis_commitments.items():
        relative = _safe_relative(raw_relative, label="pre-amendment analysis file")
        path = source_root / relative
        if not path.is_file():
            raise PackageNotReady(f"A committed analysis file is missing: {relative}")
        if digest(path) == expected_hash:
            continue
        if relative not in allowed_manifest_transitions:
            raise ValueError(f"An immutable pre-amendment analysis file changed: {relative}")
        observed_manifest_transitions.add(relative)
    if observed_manifest_transitions != allowed_manifest_transitions:
        raise ValueError("The pre-amendment analysis-manifest transition inventory changed.")

    trigger_fit_rel = _safe_relative(
        amendment.get("trigger", {}).get("fit_record", ""),
        label="solver amendment trigger",
    )
    selection_prefix = SELECTION_REL
    matched_prefix = MATCHED_REL
    if not trigger_fit_rel.is_relative_to(selection_prefix):
        raise ValueError("The amendment trigger is not a main-study fit.")
    trigger_within_selection = trigger_fit_rel.relative_to(selection_prefix)
    if main_amended != {trigger_within_selection}:
        raise ValueError("Embedded amended-fit provenance disagrees with the amendment trigger.")
    trigger_folder = source_root / trigger_fit_rel.parent
    fit_path = trigger_folder / "fit.json"
    model_path = trigger_folder / "model.npz"
    fit_archive = trigger_folder / "fit.pre_amendment.json"
    model_archive = trigger_folder / "model.pre_amendment.npz"
    intent_path = trigger_folder / "solver_amendment_intent.json"
    completion_path = trigger_folder / "solver_amendment.json"
    required_chain = (
        fit_path, model_path, fit_archive, model_archive, intent_path, completion_path)
    if not all(path.is_file() for path in required_chain):
        raise PackageNotReady("The amended fit lacks its six-file audit chain.")
    fit = _read_json(fit_path)
    original_fit = _read_json(fit_archive)
    intent = _read_json(intent_path)
    completion = _read_json(completion_path)
    amendment_hash = digest(lock_path)
    relative_folder = str(trigger_fit_rel.parent)
    for sidecar, label in ((intent, "intent"), (completion, "completion")):
        if (sidecar.get("version") != AMENDMENT_VERSION or
                sidecar.get("fit_folder") != relative_folder or
                sidecar.get("amendment_lock_sha256") != amendment_hash):
            raise ValueError(f"The solver-amendment {label} identity changed.")
    for key in ("original_fit_sha256", "original_model_sha256"):
        if intent.get(key) != completion.get(key):
            raise ValueError(f"Original solver-amendment hash accounting changed: {key}")
    if (digest(fit_archive) != intent.get("original_fit_sha256") or
            digest(model_archive) != intent.get("original_model_sha256") or
            digest(fit_path) != completion.get("amended_fit_sha256") or
            digest(model_path) != completion.get("amended_model_sha256")):
        raise ValueError("An original or amended solver artifact changed.")
    if (amendment.get("trigger", {}).get("fit_record_sha256") != digest(fit_archive) or
            amendment.get("trigger", {}).get("model_sha256") != digest(model_archive)):
        raise ValueError("The archived optimizer-only trigger changed.")
    identity = {key: original_fit.get(key) for key in ("task", "method", "seed", "tau")}
    if identity != amendment.get("trigger", {}).get("identity"):
        raise ValueError("The solver-amendment trigger identity changed.")
    trigger_diagnostics = amendment.get("trigger", {}).get("optimizer_diagnostics", {})
    original_diagnostics = original_fit.get("diagnostics", {})
    if any(original_diagnostics.get(key) != value
           for key, value in trigger_diagnostics.items()):
        raise ValueError("The solver-amendment trigger diagnostics changed.")
    for field in (
        "task", "site", "method", "config", "seed", "tau", "candidate_id",
        "feature_map", "data_folder", "selection_counts", "effective_target_fraction",
    ):
        if original_fit.get(field) != fit.get(field):
            raise ValueError(f"The solver amendment changed locked fit identity: {field}")
    diagnostics = fit.get("diagnostics", {})
    expected_provenance = {
        "version": AMENDMENT_VERSION,
        "amendment_lock": str(AMENDMENT_LOCK_REL),
        "amendment_lock_sha256": amendment_hash,
        "original_fit_sha256": intent["original_fit_sha256"],
        "original_model_sha256": intent["original_model_sha256"],
        "extension_attempts_used": 1,
        "gap_tolerances_changed": False,
    }
    if diagnostics.get("solver_certificate_amendment") != expected_provenance:
        raise ValueError("The amended-fit diagnostics provenance changed.")
    reports = completion.get("extension_attempts")
    if (completion.get("converged") is not True or
            diagnostics.get("converged") is not True or
            not isinstance(reports, list) or len(reports) != 1):
        raise ValueError("The amended fit lacks its completed numerical certificate.")
    report = reports[0]
    for key, expected in expected_extension_rule[0].items():
        if report.get(key) != expected:
            raise ValueError(f"The solver-amendment restart rule changed: {key}")
    if (type(report.get("iterations")) is not int or
            not 0 <= report["iterations"] <= expected_extension_rule[0]["max_iter"] or
            report.get("converged") is not True or
            diagnostics.get("attempt_count") != 3 or
            len(diagnostics.get("attempt_summaries", [])) != 3 or
            diagnostics["attempt_summaries"][-1] != report or
            any(diagnostics.get(key) != value for key, value in report.items()
                if key not in ("max_iter", "check_every"))):
        raise ValueError("The amended-fit attempt accounting changed.")

    main_sidecars = sorted(selection["root"].rglob("solver_amendment.json"))
    matched_sidecars = sorted(matched["root"].rglob("solver_amendment.json"))
    if main_sidecars != [trigger_folder / "solver_amendment.json"] or matched_sidecars:
        raise ValueError("The completed solver-amendment sidecar inventory changed.")

    selection_paths: set[Path] = set()
    matched_paths: set[Path] = set()
    for raw_relative in amendment.get("preexisting_completion_artifacts", {}):
        relative = _safe_relative(raw_relative, label="pre-amendment artifact")
        if relative.is_relative_to(selection_prefix):
            selection_paths.add(relative.relative_to(selection_prefix))
        elif relative.is_relative_to(matched_prefix):
            matched_paths.add(relative.relative_to(matched_prefix))
        else:
            raise ValueError(f"Unexpected pre-amendment artifact root: {relative}")

    for name in (
        "fit.json", "model.npz", "fit.pre_amendment.json",
        "model.pre_amendment.npz", "solver_amendment_intent.json",
        "solver_amendment.json",
    ):
        path = trigger_folder / name
        if not path.is_file():
            raise PackageNotReady(f"Amended-fit audit artifact is missing: {path}")
        selection_paths.add(path.relative_to(selection["root"]))

    supersession_rel = _safe_relative(
        amendment.get("superseded_preexecution_draft", {}).get("record", ""),
        label="solver amendment supersession",
    )
    supersession = _read_json(source_root / supersession_rel)
    if (digest(source_root / supersession_rel) !=
            amendment.get("superseded_preexecution_draft", {}).get("record_sha256") or
            supersession.get("executed_fit_count") != 0):
        raise ValueError("The unexecuted amendment-draft supersession record changed.")
    draft_rel = _safe_relative(
        supersession.get("superseded_lock", ""),
        label="superseded solver amendment draft",
    )
    static_paths = {AMENDMENT_LOCK_REL, supersession_rel, draft_rel}
    for relative in static_paths:
        if not (source_root / relative).is_file():
            raise PackageNotReady(f"Solver-amendment provenance is missing: {relative}")
    if digest(source_root / draft_rel) != supersession.get("superseded_lock_sha256"):
        raise ValueError("The unexecuted superseded amendment draft changed.")

    return {
        "version": AMENDMENT_VERSION,
        "lock_path": AMENDMENT_LOCK_REL,
        "lock_sha256": digest(lock_path),
        "amended_main_fits": len(main_amended),
        "amended_matched_fits": len(matched_amended),
        "advanced_analysis_manifests": sorted(
            (str(path) for path in observed_manifest_transitions)),
        "selection_paths": selection_paths,
        "matched_paths": matched_paths,
        "static_paths": static_paths,
    }


def _validate_render(source_root: Path, selection_analysis: dict[str, Any],
                     matched_analysis: dict[str, Any]) -> dict[str, Any]:
    root = source_root / RENDER_REL
    manifest = _read_json(root / "manifest.json")
    _complete(manifest, label="selection-v2 rendering")
    if manifest.get("version") != "selection_v2_render_manifest_1":
        raise ValueError("Unexpected selection-v2 render manifest version.")
    if manifest.get("source_sha256") != digest(source_root / "review/render_selection_study_v2.py"):
        raise ValueError("selection-v2 render source hash mismatch.")
    inputs = manifest.get("inputs", {})
    if (inputs.get("selection_analysis_manifest_sha256") !=
            digest(selection_analysis["root"] / "analysis_manifest.json") or
            inputs.get("matched_analysis_manifest_sha256") !=
            digest(matched_analysis["root"] / "manifest.json")):
        raise ValueError("selection-v2 render manifest names different analysis inputs.")
    _validate_render_inputs(inputs.get("files"), selection_analysis["root"],
                            matched_analysis["root"])
    expected_counts = {
        "selection_task_method_rows": 72, "selection_scope_rows": 12,
        "selection_full_inference_rows": 60, "matched_task_variant_rows": 48,
        "matched_aggregate_rows": 8, "matched_inference_rows": 36,
    }
    if manifest.get("validated_counts") != expected_counts:
        raise ValueError("selection-v2 render completeness counts changed.")
    outputs = _require_hashes(root, manifest.get("outputs"), label="selection-v2 render")
    if {str(path) for path in outputs} != RENDER_FILES:
        raise ValueError("selection-v2 render output inventory changed.")
    return {"root": root, "manifest": manifest, "output_paths": outputs}


def _validate_probability_dependency(source_root: Path) -> None:
    root = source_root / "review/probabilistic_runs/probabilistic_selection_v2"
    manifest = _read_json(root / "manifest.json")
    protocol = _read_json(root / "protocol.json")
    lock = _read_json(root / "selection_lock.json")
    _complete(manifest, label="probability selection-v2")
    if manifest.get("protocol") != protocol:
        raise ValueError("Probability manifest and protocol.json disagree.")
    dependency = source_root / "review/selection_runs/selection_v1/protocol.json"
    if not dependency.is_file() or manifest.get("base_candidate_protocol_sha256") != digest(dependency):
        raise ValueError("Probability study's sole selection-v1 protocol dependency changed.")
    sources = _require_sources(source_root, manifest.get("source_hashes"),
                               label="probability source")
    if Path("review/run_selection_study.py") not in sources:
        raise ValueError("Probability source inventory omits its v1 runner dependency.")
    if lock.get("protocol_sha256") != digest(root / "protocol.json"):
        raise ValueError("Probability selection-lock protocol hash mismatch.")


def validate_completed_v2(source_root: Path = ROOT) -> dict[str, Any]:
    """Validate every completion gate before any staging or replacement."""
    source_root = Path(source_root).resolve()
    selection = _validate_selection(source_root)
    selection_analysis = _validate_selection_analysis(source_root, selection)
    matched = _validate_matched(source_root, selection)
    matched_analysis = _validate_matched_analysis(source_root, matched)
    amendment = _validate_solver_amendment(source_root, selection, matched)
    render = _validate_render(source_root, selection_analysis, matched_analysis)
    chronological = _validate_chronological(source_root)
    if selection["manifest"].get("reference_manifest_sha256") != digest(
            chronological["root"] / "manifest.json"):
        raise ValueError("selection-v2 does not bind the packaged chronological manifest.")
    probability = _validate_probability(source_root)
    state = {"selection": selection, "selection_analysis": selection_analysis,
             "matched": matched, "matched_analysis": matched_analysis, "render": render,
             "amendment": amendment, "chronological": chronological,
             "probability": probability}
    copy_groups = {
        "selection": (selection["root"], _selection_package_paths(state)),
        "selection_analysis": (
            selection_analysis["root"],
            {Path("analysis_protocol.json"), Path("analysis_manifest.json"),
             *selection_analysis["artifact_paths"]}),
        "matched": (matched["root"], _matched_package_paths(state)),
        "matched_analysis": (
            matched_analysis["root"],
            {Path("analysis_protocol.json"), Path("analysis_lock.json"), Path("manifest.json"),
             *matched_analysis["output_paths"]}),
        "render": (render["root"],
                   {Path("manifest.json"), Path("render_protocol.json"),
                    *render["output_paths"]}),
        "chronological": (chronological["root"], chronological["paths"]),
        "probability_selection": (probability["selection"]["root"],
                                  probability["selection"]["paths"]),
        "probability_analysis": (probability["analysis"]["root"],
                                 probability["analysis"]["paths"]),
        "probability_presentation": (probability["presentation"]["root"],
                                     probability["presentation"]["paths"]),
    }
    state["copy_hashes"] = {
        name: {str(path): digest(root / path) for path in sorted(paths, key=str)}
        for name, (root, paths) in copy_groups.items()
    }
    static_relatives = {
        *(Path(value) for value in CODE_FILES),
        Path("TLQLDMR_R1/event_composition.tex"),
        Path("review/selection_runs/selection_v1/protocol.json"),
        Path("review/extreme_event_summary.json"),
    }
    if amendment is not None:
        static_relatives.update(amendment["static_paths"])
    state["copy_hashes"]["static"] = {}
    state["static_sources"] = {}
    for relative in sorted(static_relatives, key=str):
        source_relative = chronological["source_overrides"].get(relative, relative)
        path = source_root / source_relative
        if not path.is_file():
            raise PackageNotReady(f"Required package source is missing: {relative}")
        state["copy_hashes"]["static"][str(relative)] = digest(path)
        state["static_sources"][str(relative)] = source_relative
    # Guard the raw-data inputs used later to derive split_assignments as well.
    state["copy_hashes"]["selection_data"] = dict(selection["lock"]["data_hashes"])
    return state


def copy_exact(source: Path, destination: Path, *, expected_hash: str | None = None) -> None:
    """Copy immutable material without rewriting workstation-path strings."""
    if not Path(source).is_file():
        raise FileNotFoundError(source)
    if expected_hash is not None and digest(source) != expected_hash:
        raise ValueError(f"Source changed after validation: {source}")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    copied_hash = digest(destination)
    if (expected_hash is not None and copied_hash != expected_hash) or digest(source) != copied_hash:
        raise OSError(f"Byte-exact copy verification failed: {source}")


def copy_relative(source_root: Path, staging: Path, relative: str | Path,
                  *, expected_hash: str | None = None) -> None:
    relative = _safe_relative(relative, label="package")
    copy_exact(source_root / relative, staging / relative, expected_hash=expected_hash)


def copy_relative_set(source: Path, destination: Path, relatives: Iterable[Path],
                      *, expected_hashes: Mapping[str, str] | None = None) -> int:
    count = 0
    for relative in sorted(set(relatives), key=str):
        relative = _safe_relative(relative, label="artifact")
        expected = None if expected_hashes is None else expected_hashes.get(str(relative))
        if expected_hashes is not None and expected is None:
            raise ValueError(f"Artifact lacks its validation snapshot: {relative}")
        copy_exact(source / relative, destination / relative, expected_hash=expected)
        count += 1
    return count


def copy_record_tree(source: Path, destination: Path, *, presentation: bool = False) -> int:
    """Copy records and row outputs, while omitting fitted-model binaries/logs."""
    if not source.is_dir():
        raise FileNotFoundError(source)
    allowed = {".json", ".csv", ".tex", ".txt"}
    if presentation:
        allowed |= {".pdf", ".svg", ".png"}
    count = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        keep = path.suffix.lower() in allowed
        keep |= path.suffix.lower() == ".npz" and path.name.endswith("ids.npz")
        if keep:
            copy_exact(path, destination / path.relative_to(source))
            count += 1
    return count


def _selection_package_paths(state: dict[str, Any]) -> set[Path]:
    selection = state["selection"]
    paths = {Path("protocol.json"), Path("manifest.json"), Path("data_provenance.json"),
             Path("selection_lock.json"), Path("results.json"),
             *selection["selection_paths"], *state["selection_analysis"]["input_paths"],
             *state["matched"]["fit_only_paths"], *selection["calibration_paths"],
             *(Path(task) / "target_data" / "selection_counts.json"
               for task in TASKS)}
    if state.get("amendment") is not None:
        paths.update(state["amendment"]["selection_paths"])
    return paths


def _matched_package_paths(state: dict[str, Any]) -> set[Path]:
    paths = {Path("protocol.json"), Path("configurations.json"), Path("matched_lock.json"),
             Path("manifest.json"), Path("results.json"), *state["matched"]["result_paths"]}
    if state.get("amendment") is not None:
        paths.update(state["amendment"]["matched_paths"])
    return paths


def write_split_assignments(source_root: Path, staging: Path,
                            *, expected_data_hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    _require_hashes(source_root, expected_data_hashes, label="split-assignment source data")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from review.selection_data_v2 import load_all_selection_data_v2

    records = []
    for site, farm in sorted(load_all_selection_data_v2(mmap_mode="r").items()):
        thresholds = farm.preprocessing["domain_thresholds"]
        wind = np.asarray(farm.target_weather["wind_hub"])
        temperature = np.asarray(farm.target_weather["temperature"])
        rules = {"wind": wind > thresholds["wind_ms"],
                 "cold": temperature < thresholds["temperature_low_c"],
                 "hot": temperature > thresholds["temperature_high_c"],
                 "cutout": wind > thresholds["cutout_ms"]}
        target = rules["wind"] | rules["cold"] | rules["hot"] | rules["cutout"]
        if not np.array_equal(target, np.asarray(farm.is_extreme)):
            raise AssertionError(f"Farm {site} domain reconstruction mismatch")
        path = staging / "split_assignments" / f"farm{site}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                import io
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.writer(text)
                    writer.writerow(["window_index", "target_row_index", "target_timestamp", "split",
                                     "domain", "power_mw", "wind_rule", "cold_rule", "hot_rule",
                                     "above_25ms_rule"])
                    for index in range(len(farm.timestamps)):
                        writer.writerow([
                            index, int(farm.target_row_indices[index]),
                            np.datetime_as_string(farm.timestamps[index], unit="s"),
                            str(farm.split[index]), "target" if target[index] else "source",
                            format(float(farm.y_raw[index]), ".12g"), int(rules["wind"][index]),
                            int(rules["cold"][index]), int(rules["hot"][index]),
                            int(rules["cutout"][index]),
                        ])
        records.append({"site": site, "rows": len(farm.timestamps),
                        "file": str(path.relative_to(staging)),
                        "sha256": digest(path)})
    _require_hashes(source_root, expected_data_hashes, label="split-assignment source data")
    return records


def package_environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "cvxopt",
                 "openpyxl", "joblib", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or None, "packages": packages}


def write_requirements(staging: Path, frozen: Mapping[str, Any],
                       packaging: Mapping[str, Any]) -> None:
    lines = ["# Versions frozen by the selection-v2 lock."]
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "joblib",
                 "threadpoolctl"):
        if frozen.get(name):
            lines.append(f"{name}=={frozen[name]}")
    lines.append("# Presentation/solver packages recorded at packaging time; not selection-v2 lock fields.")
    for name in ("matplotlib", "cvxopt"):
        if packaging["packages"].get(name):
            lines.append(f"{name}=={packaging['packages'][name]}")
    (staging / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prefixed_hashes(prefix: Path, hashes: Mapping[str, str]) -> dict[str, str]:
    return {str(prefix / _safe_relative(relative, label="evidence")): value
            for relative, value in sorted(hashes.items())}


def write_evidence_inventory(staging: Path, state: Mapping[str, Any]) -> None:
    calibration = _prefixed_hashes(
        SELECTION_REL, state["selection"]["calibration_hashes"])
    chronological = _prefixed_hashes(
        CHRONO_REL, state["copy_hashes"]["chronological"])
    probability_selection = _prefixed_hashes(
        PROBABILITY_REL, state["copy_hashes"]["probability_selection"])
    probability_analysis = _prefixed_hashes(
        PROBABILITY_ANALYSIS_REL, state["copy_hashes"]["probability_analysis"])
    probability_presentation = _prefixed_hashes(
        PROBABILITY_PRESENTATION_REL,
        state["copy_hashes"]["probability_presentation"])
    payload = {
        "version": "s1_evidence_inventory_1",
        "selection_v2": {
            "calibration_hash_authority": PACKAGE_TIME_SNAPSHOT,
            "fit_record_hash_authority": FIT_RECORD_HASH_AUTHORITY,
            "fit_record_inventory": str(SELECTION_REL / FIT_RECORD_INVENTORY),
            "fit_record_inventory_sha256": digest(
                staging / SELECTION_REL / FIT_RECORD_INVENTORY),
            "selection_fits": state["selection"]["selection_fit_count"],
            "evaluation_records": len(state["selection"]["results"]),
            "evaluation_tau_fits": state["selection"]["evaluation_fit_count"],
            "calibration_predictions": calibration,
        },
        "chronological_v1": {
            "artifact_hash_authority": PACKAGE_TIME_SNAPSHOT,
            "jobs": len(state["chronological"]["jobs"]),
            "aliases": len(state["chronological"]["aliases"]),
            "files": chronological,
        },
        "probability_v2": {
            "provenance_class": "legacy/posthoc",
            "legacy_provenance_limitation": LEGACY_PROBABILITY_LIMITATION,
            "current_candidate_protocol_check":
                state["probability"]["current_candidate_protocol_check"],
            "selection_files": probability_selection,
            "analysis_files": probability_analysis,
            "presentation_files": probability_presentation,
        },
    }
    (staging / EVIDENCE_INVENTORY).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_packaging_provenance(staging: Path, state: Mapping[str, Any]) -> None:
    payload = {
        "version": "s1_v2_exact_bytes_2",
        "copy_policy": "Frozen source, JSON, CSV, and lock-required NPZ files are copied byte for byte.",
        "path_interpretation": (
            "Absolute paths in frozen records describe the execution host. Verification resolves "
            "archived artifacts from the package layout without changing those immutable bytes."
        ),
        "model_policy": (
            "Candidate and final fitted-model binaries plus full selection/main fit histories are "
            "omitted except for the small solver-amendment audit chain, whose pre-amendment and "
            "amended model files are retained. For selection-v2, the builder checks every source "
            "fit/metric record against "
            "its embedding and retains a compact package-time checksum/digest inventory. Locks, "
            "selected configurations, embedded or matched fit records, row-level predictions, IDs, "
            "preprocessing, lock-required landmarks, and maps are retained. Legacy chronological "
            "and probability contents are exactly enumerated in the evidence inventory."
        ),
        "fit_record_policy": FIT_RECORD_HASH_AUTHORITY,
        "package_byte_authority": "MANIFEST.sha256 covers every released payload file.",
        "environment_policy": (
            "environment.json reports the frozen selection-v2 environment separately from "
            "the later packaging runtime; the packaging runtime is not experimental provenance."
        ),
        "probability_provenance": {
            "class": "legacy/posthoc",
            "limitation": LEGACY_PROBABILITY_LIMITATION,
            "current_candidate_protocol_check":
                state["probability"]["current_candidate_protocol_check"],
        },
    }
    if state.get("amendment") is not None:
        amendment = state["amendment"]
        payload["solver_certificate_amendment"] = {
            "version": amendment["version"],
            "lock": str(amendment["lock_path"]),
            "lock_sha256": amendment["lock_sha256"],
            "amended_main_fits": amendment["amended_main_fits"],
            "amended_matched_fits": amendment["amended_matched_fits"],
            "advanced_analysis_manifests": amendment["advanced_analysis_manifests"],
            "interpretation": (
                "Performance-blind numerical-certificate amendment; objective, data, feature "
                "map, configuration, tolerance, prediction, and analysis rules were unchanged. "
                "The two named state manifests subsequently advanced to their separately "
                "validated completed states; the immutable analysis protocols and lock did "
                "not change."
            ),
        }
    (staging / "PACKAGING_PROVENANCE.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(staging: Path) -> None:
    (staging / "README.md").write_text(f"""# Supplementary Reproducibility Package S1

This package accompanies the revised TL-QLDMR manuscript. It contains exact frozen records for the predeclared selection-v2 study, the independently locked matched component ablation, their analyses and publication rendering, the chronological common-setting study, and the probability study.

The six raw workbooks are the public State Grid Renewable Energy Generation dataset (Chen and Xu, 2022, *Scientific Data* 9, 577; data DOI `10.6084/m9.figshare.17304221.v4`, CC0). Place the original files under `data/wind_farm_data/data_original/wind_farms/`. `split_assignments/` maps every eligible window to its workbook row, timestamp, split, label, domain, and meteorological rule flags.

`review/selection_runs/selection_v2/` contains the 12-task, six-method lock, 3,888 validation queries backed by 10,368 seed-level selection fits, 192 locked evaluation records, row predictions, and fit-only artifacts required by the matched lock. Its compact `fit_record_inventory.json` records source-file SHA-256 values and identity/metric/diagnostic digests after the builder matched all 10,368 selection fit/validation-metric pairs and all 576 main fit records to their embedded records. Selection-stage certificate failures are retained as evidence only when the candidate's failed-seed list and ineligibility flag agree; every locked main kernel fit must be certified. These checksums are a package-time source-record inventory, not an original experiment lock or preregistration. `review/selection_runs/matched_ablation_v2/` contains the independent 144-key matched lock and evaluation records. `review/selection_runs/selection_v1/protocol.json` and `review/run_selection_study.py` are retained only because the probability protocol binds them; no selection-v1 point predictions are included.

The main analysis protocol was frozen while selection was still in progress, before the selection lock and before any held-out fit. The selection-v2 lock was then created before its 576 three-quantile held-out fits, followed by the matched experiment lock and matched analysis freeze. Only after all four locks existed were the main 192-record and matched 144-record evaluations opened.

One locked main-study fit exhausted the original 3,000- and 12,000-iteration attempts with relative gap 0.0010217 against the unchanged 0.001 tolerance. Before its predictive scores were inspected, a performance-blind solver amendment committed every then-existing completion artifact and analysis file, and allowed one additional 96,000-iteration warm restart only for an otherwise exact locked fit with no completed result. The objective, data, feature map, configuration, seed, quantile, tolerance, prediction procedure, and analysis rules were unchanged. That one main fit met the certificate; no matched fit used the amendment. The three immutable analysis protocol/lock files still match their pre-amendment hashes. The two state manifests named in `solver_certificate_amendment.advanced_analysis_manifests` subsequently advanced from their committed pre-analysis states to the separately validated completed states distributed here; this expected transition did not change either analysis specification. The amendment runner, lock, unexecuted superseded draft, original and amended trigger records/models, completion sidecars, and every pre-amendment completion artifact bound by the lock are retained here.

Probability provenance is legacy/posthoc. {LEGACY_PROBABILITY_LIMITATION} {PROBABILITY_CURRENT_CHECK}

The calibration-CSV and chronological artifact hashes in `PACKAGE_EVIDENCE_INVENTORY.json` are a {PACKAGE_TIME_SNAPSHOT}; they preserve the bytes distributed in S1 but are not presented as pre-run locks.

The root `Train_TL_QLDMR.py` in this package is the byte-exact source snapshot bound by the chronological-study manifest. The live workstation module advanced after that study, so the builder validates and packages the historical snapshot rather than substituting the later workspace file.

`environment.json` separates versions frozen by the selection-v2 lock from the runtime used only to assemble this archive. Packaging-time versions must not be interpreted as the environment that produced frozen experiments.

Run from this package root and first verify every outer and nested hash:

```text
PYTHONDONTWRITEBYTECODE=1 python review/verify_s1_package.py
```

For a clean reproduction, use the order below. It freezes the main analysis immediately after the selection lock; the archived run froze that analysis protocol earlier, while selection was still in progress. In both cases the analysis rules and matched experiment are frozen before either held-out evaluation is opened.

```text
python review/run_selection_study_v2.py --output review/selection_runs/reproduced_selection_v2 --freeze-only
python review/run_selection_study_v2.py --output review/selection_runs/reproduced_selection_v2 --phase base --resume
python review/run_selection_study_v2.py --output review/selection_runs/reproduced_selection_v2 --phase regularizers --resume
python review/run_selection_study_v2.py --output review/selection_runs/reproduced_selection_v2 --phase joint --resume
python review/run_selection_study_v2.py --output review/selection_runs/reproduced_selection_v2 --phase lock --resume
python review/analyze_selection_study_v2.py --selection-root review/selection_runs/reproduced_selection_v2 --output review/selection_runs/reproduced_selection_v2_analysis --freeze-only
python review/run_matched_ablation_v2.py --selection-output review/selection_runs/reproduced_selection_v2 --output review/selection_runs/reproduced_matched_ablation_v2 --phase freeze
python review/analyze_matched_ablation_v2.py --matched-output review/selection_runs/reproduced_matched_ablation_v2 --output review/selection_runs/reproduced_matched_ablation_v2_analysis --freeze-only
python review/run_selection_study_v2.py --output review/selection_runs/reproduced_selection_v2 --phase evaluate --resume
python review/run_matched_ablation_v2.py --selection-output review/selection_runs/reproduced_selection_v2 --output review/selection_runs/reproduced_matched_ablation_v2 --phase evaluate --resume
python review/analyze_selection_study_v2.py --selection-root review/selection_runs/reproduced_selection_v2 --output review/selection_runs/reproduced_selection_v2_analysis --resume
python review/analyze_matched_ablation_v2.py --matched-output review/selection_runs/reproduced_matched_ablation_v2 --output review/selection_runs/reproduced_matched_ablation_v2_analysis --resume
python review/render_selection_study_v2.py --selection-analysis review/selection_runs/reproduced_selection_v2_analysis --matched-analysis review/selection_runs/reproduced_matched_ablation_v2_analysis --output review/selection_runs/reproduced_selection_v2_render
python review/render_manuscript_v2_appendix.py --selection-root review/selection_runs/reproduced_selection_v2 --analysis-root review/selection_runs/reproduced_selection_v2_analysis --output review/selection_runs/reproduced_selection_v2_manuscript_appendix
python review/summarize_v2_revision_evidence.py --selection-analysis review/selection_runs/reproduced_selection_v2_analysis --matched-analysis review/selection_runs/reproduced_matched_ablation_v2_analysis --output review/selection_runs/reproduced_revision_evidence_v2
```

The three selection phases may be partitioned across disjoint task workers, but only an unfiltered coordinator may create the lock. The chronological and probability workflows use the supplied runners:

```text
python review/run_replied_comment_study.py --protocol review/replied_study_protocol.json --output review/replied_comment_runs/reproduced_chronological_v1
python review/verify_replied_comment_study.py --run-dir review/replied_comment_runs/reproduced_chronological_v1 --require-complete --output analysis/common_setting_verification_recomputed.json
python review/run_probabilistic_selection_study.py --output review/probabilistic_runs/reproduced_probabilistic_selection_v2
python -m review.run_probabilistic_analysis_portable --experiment review/probabilistic_runs/reproduced_probabilistic_selection_v2 --output review/probabilistic_runs/reproduced_probabilistic_analysis_v2
python review/render_probabilistic_figures.py --input review/probabilistic_runs/reproduced_probabilistic_analysis_v2/aggregate_results.json --output review/probabilistic_runs/reproduced_probabilistic_presentation_v1
```

Frozen metadata retain execution-host paths because changing those bytes would invalidate their hashes. Candidate validation-prediction CSV files, full selection/main fit histories, and candidate/final model binaries are omitted for size except for the solver-amendment audit chain described above. For selection-v2 and its matched ablation, exact locks, a compact checksum/digest inventory for all source-run fit records, matched fit records, row predictions, and lock-required fit-only arrays are included. The package verifier matches every compact inventory digest to its locked selection or main-result embedding and checks the inventory's exact counts and provenance label. `PACKAGE_EVIDENCE_INVENTORY.json` gives the exact included chronological and probability inventories, binds the compact fit-record inventory, and separately snapshots all 192 main calibration CSV files at packaging time.
""", encoding="utf-8")


def _write_manifest(staging: Path) -> int:
    lines = [f"{digest(path)}  {path.stat().st_size:12d}  {path.relative_to(staging)}"
             for path in sorted(staging.rglob("*"))
             if path.is_file() and path.name != "MANIFEST.sha256"]
    (staging / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _build_staging(source_root: Path, staging: Path, state: dict[str, Any]) -> int:
    write_readme(staging)
    static_hashes = state["copy_hashes"]["static"]
    for relative in CODE_FILES:
        source_relative = state["static_sources"][relative]
        copy_exact(source_root / source_relative, staging / relative,
                   expected_hash=static_hashes[relative])
    copy_relative(source_root, staging, "TLQLDMR_R1/event_composition.tex",
                  expected_hash=static_hashes["TLQLDMR_R1/event_composition.tex"])
    if state.get("amendment") is not None:
        for relative in sorted(state["amendment"]["static_paths"], key=str):
            copy_relative(source_root, staging, relative,
                          expected_hash=static_hashes[str(relative)])

    counts: dict[str, int] = {}
    counts["common_setting"] = copy_relative_set(
        state["chronological"]["root"], staging / CHRONO_REL,
        state["chronological"]["paths"],
        expected_hashes=state["copy_hashes"]["chronological"])
    # This is the only allowed selection-v1 artifact.
    copy_relative(source_root, staging, "review/selection_runs/selection_v1/protocol.json",
                  expected_hash=static_hashes[
                      "review/selection_runs/selection_v1/protocol.json"])
    for name, relative, state_name in (
        ("probabilistic_selection_v2", PROBABILITY_REL, "selection"),
        ("probabilistic_analysis_v2", PROBABILITY_ANALYSIS_REL, "analysis"),
        ("probabilistic_presentation_v1", PROBABILITY_PRESENTATION_REL, "presentation"),
    ):
        counts[name] = copy_relative_set(
            state["probability"][state_name]["root"], staging / relative,
            state["probability"][state_name]["paths"],
            expected_hashes=state["copy_hashes"][f"probability_{state_name}"])

    counts["selection_v2"] = copy_relative_set(
        state["selection"]["root"], staging / SELECTION_REL,
        _selection_package_paths(state),
        expected_hashes=state["copy_hashes"]["selection"])
    fit_inventory_path = staging / SELECTION_REL / FIT_RECORD_INVENTORY
    fit_inventory_path.write_text(
        json.dumps(state["selection"]["fit_record_inventory"], indent=2,
                   ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    counts["selection_v2"] += 1
    counts["selection_v2_analysis"] = copy_relative_set(
        state["selection_analysis"]["root"], staging / SELECTION_ANALYSIS_REL,
        {Path("analysis_protocol.json"), Path("analysis_manifest.json"),
         *state["selection_analysis"]["artifact_paths"]},
        expected_hashes=state["copy_hashes"]["selection_analysis"])
    counts["matched_ablation_v2"] = copy_relative_set(
        state["matched"]["root"], staging / MATCHED_REL,
        _matched_package_paths(state),
        expected_hashes=state["copy_hashes"]["matched"])
    counts["matched_ablation_v2_analysis"] = copy_relative_set(
        state["matched_analysis"]["root"], staging / MATCHED_ANALYSIS_REL,
        {Path("analysis_protocol.json"), Path("analysis_lock.json"), Path("manifest.json"),
         *state["matched_analysis"]["output_paths"]},
        expected_hashes=state["copy_hashes"]["matched_analysis"])
    counts["selection_v2_render"] = copy_relative_set(
        state["render"]["root"], staging / RENDER_REL,
        {Path("manifest.json"), Path("render_protocol.json"),
         *state["render"]["output_paths"]},
        expected_hashes=state["copy_hashes"]["render"])

    analysis = staging / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    copy_exact(source_root / "review/extreme_event_summary.json",
               analysis / "extreme_event_summary.json",
               expected_hash=static_hashes["review/extreme_event_summary.json"])
    split_records = write_split_assignments(
        source_root, staging,
        expected_data_hashes=state["copy_hashes"]["selection_data"])
    packaging_environment = package_environment()
    frozen_environment = state["selection"]["lock"]["environment_versions"]
    environment = {
        "version": "s1_environment_2",
        "frozen_selection_v2_environment": frozen_environment,
        "packaging_runtime": packaging_environment,
        "interpretation": (
            "The frozen block records the experiment environment bound by the selection-v2 "
            "lock. The packaging block records only the later archive-construction runtime."
        ),
    }
    (staging / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_requirements(staging, frozen_environment, packaging_environment)
    write_evidence_inventory(staging, state)
    index = {
        "package": "Supplementary Reproducibility Package S1",
        "version": "selection_v2_matched_2",
        "raw_data_doi": "10.6084/m9.figshare.17304221.v4",
        "data_article_doi": "10.1038/s41597-022-01696-6",
        "workflow_order": [
            "selection-v2 freeze", "base selection started", "selection-analysis freeze",
            "base selection completed", "regularizer selection", "joint selection", "selection lock",
            "matched-ablation freeze", "matched-analysis freeze", "selection evaluation",
            "matched evaluation", "selection analysis", "matched analysis",
            "publication rendering",
        ],
        "v2_design": {
            "selection_lock_version": "selection_v2_lock_2", "tasks": 12, "methods": 6,
            "queries_per_task_method": 54, "total_configuration_queries": 3888,
            "selection_fits": 10368, "evaluation_records": 192,
            "evaluation_tau_fits": 576,
            "matched_lock_version": "matched_ablation_v2_lock_1",
            "matched_evaluation_records": 144,
        },
        "split_assignments": split_records, "artifact_file_counts": counts,
        "evidence_inventory": str(EVIDENCE_INVENTORY),
        "fit_record_inventory": {
            "path": str(SELECTION_REL / FIT_RECORD_INVENTORY),
            "sha256": digest(fit_inventory_path),
            "version": FIT_RECORD_INVENTORY_VERSION,
            "hash_authority": FIT_RECORD_HASH_AUTHORITY,
            "selection_fit_records": state["selection"]["selection_fit_count"],
            "selection_validation_metric_records":
                state["selection"]["selection_fit_count"],
            "evaluation_fit_records": state["selection"]["evaluation_fit_count"],
        },
        "probability_provenance": {
            "class": "legacy/posthoc",
            "limitation": LEGACY_PROBABILITY_LIMITATION,
            "current_candidate_protocol_check":
                state["probability"]["current_candidate_protocol_check"],
        },
        "selection_v1_allowlist": ["review/selection_runs/selection_v1/protocol.json"],
        "omitted_artifact_classes": [
            "public raw workbooks", "candidate validation-prediction CSV files",
            "full selection and main fit histories",
            "candidate fitted-model binaries", "final fitted-model binaries",
            "solver-amendment model binaries retained as audit evidence",
            "worker logs and non-JSON internal diagnostics",
        ],
    }
    if state.get("amendment") is not None:
        amendment = state["amendment"]
        index["solver_certificate_amendment"] = {
            "version": amendment["version"],
            "lock": str(amendment["lock_path"]),
            "lock_sha256": amendment["lock_sha256"],
            "amended_main_fits": amendment["amended_main_fits"],
            "amended_matched_fits": amendment["amended_matched_fits"],
            "advanced_analysis_manifests": amendment["advanced_analysis_manifests"],
        }
    (staging / "PACKAGE_INDEX.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_packaging_provenance(staging, state)
    return _write_manifest(staging)


def _write_zip(staging: Path, destination_name: str, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                archive.write(path, Path(destination_name) / path.relative_to(staging))


def _verify_zip_matches_staging(staging: Path, destination_name: str,
                                archive_path: Path) -> None:
    """Require a one-to-one, byte-exact regular-file view of staging in the ZIP."""
    if (not destination_name or destination_name in {".", ".."} or
            Path(destination_name).name != destination_name or
            "/" in destination_name or "\\" in destination_name):
        raise ValueError("Unsafe S1 ZIP top-level directory name.")
    expected = {
        str(Path(destination_name) / path.relative_to(staging)): path
        for path in staging.rglob("*") if path.is_file() and not path.is_symlink()
    }
    if any(path.is_symlink() for path in staging.rglob("*")):
        raise ValueError("Symbolic links are not allowed in staged S1 content.")
    observed: dict[str, zipfile.ZipInfo] = {}
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            bad_crc = archive.testzip()
            if bad_crc is not None:
                raise ValueError(f"S1 ZIP CRC failure: {bad_crc}")
            for info in archive.infolist():
                name = info.filename
                relative = _safe_relative(name, label="ZIP member")
                if (info.is_dir() or relative.as_posix() != name or
                        not relative.parts or relative.parts[0] != destination_name):
                    raise ValueError(f"Unsafe or unexpected S1 ZIP member: {name}")
                mode = (info.external_attr >> 16) & 0o170000
                if mode not in (0, stat.S_IFREG) or info.flag_bits & 0x1:
                    raise ValueError(f"Unsupported S1 ZIP member type: {name}")
                if name in observed:
                    raise ValueError(f"Duplicate S1 ZIP member: {name}")
                observed[name] = info
            if set(observed) != set(expected):
                raise ValueError("S1 ZIP member inventory differs from staged content.")
            for name, source in expected.items():
                info = observed[name]
                if info.file_size != source.stat().st_size:
                    raise ValueError(f"S1 ZIP byte-count mismatch: {name}")
                payload = archive.read(info)
                if hashlib.sha256(payload).hexdigest() != digest(source):
                    raise ValueError(f"S1 ZIP content mismatch: {name}")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Cannot validate staged S1 ZIP: {error}") from error


def _publish(staging: Path, temporary_zip: Path, destination: Path, zip_path: Path) -> None:
    directory_backup = destination.with_name(destination.name + ".previous")
    zip_backup = zip_path.with_name(zip_path.name + ".previous")
    lexists = lambda path: os.path.lexists(path)
    if (not staging.is_dir() or staging.is_symlink() or
            not temporary_zip.is_file() or temporary_zip.is_symlink()):
        raise ValueError("Publication inputs must be a regular staging directory and ZIP file.")
    if lexists(directory_backup) or lexists(zip_backup):
        raise FileExistsError("Refusing to overwrite a stale S1 backup.")
    if lexists(destination) and (destination.is_symlink() or not destination.is_dir()):
        raise ValueError("Existing S1 destination is not a regular directory.")
    if lexists(zip_path) and (zip_path.is_symlink() or not zip_path.is_file()):
        raise ValueError("Existing S1 ZIP destination is not a regular file.")
    destination_existed = lexists(destination)
    zip_existed = lexists(zip_path)
    moved_directory = moved_zip = False
    published_directory = published_zip = False
    try:
        if lexists(destination):
            destination.rename(directory_backup)
            moved_directory = True
        if lexists(zip_path):
            zip_path.rename(zip_backup)
            moved_zip = True
        staging.rename(destination)
        published_directory = True
        temporary_zip.replace(zip_path)
        published_zip = True
    except BaseException as error:
        # A filesystem operation can complete and still raise (for example, an
        # injected failure after rename). Infer that state from source/target
        # existence before restoring the previous pair.
        directory_replaced = published_directory or (
            not lexists(staging) and lexists(destination) and
            (moved_directory or not destination_existed))
        zip_replaced = published_zip or (
            not lexists(temporary_zip) and lexists(zip_path) and
            (moved_zip or not zip_existed))
        rollback_errors: list[str] = []
        for description, operation in (
            ("remove new directory",
             lambda: shutil.rmtree(destination) if directory_replaced and
             lexists(destination) else None),
            ("remove new ZIP",
             lambda: zip_path.unlink() if zip_replaced and lexists(zip_path) else None),
            ("restore old directory",
             lambda: directory_backup.rename(destination) if moved_directory and
             lexists(directory_backup) else None),
            ("restore old ZIP",
             lambda: zip_backup.rename(zip_path) if moved_zip and
             lexists(zip_backup) else None),
        ):
            try:
                operation()
            except BaseException as rollback_error:
                rollback_errors.append(f"{description}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "S1 publication failed and rollback was incomplete: " +
                "; ".join(rollback_errors)
            ) from error
        raise
    else:
        if directory_backup.exists():
            shutil.rmtree(directory_backup)
        if zip_backup.exists():
            zip_backup.unlink()


def build(source_root: Path = ROOT, destination: Path = DEST,
          zip_path: Path = ZIP) -> dict[str, Any]:
    source_root, destination, zip_path = map(Path, (source_root, destination, zip_path))
    # No mkdir, temporary directory, rename, or deletion occurs before this gate.
    state = validate_completed_v2(source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tlqldmr-s1-", dir=destination.parent) as temporary:
        staging = Path(temporary) / destination.name
        staging.mkdir()
        manifest_entries = _build_staging(source_root, staging, state)
        from review.verify_s1_package import verify as verify_s1
        verification = verify_s1(staging)
        if verification["status"] != "passed":
            detail = verification["errors"][0] if verification["errors"] else "unknown error"
            raise ValueError(f"Staged S1 failed self-verification: {detail}")
        with tempfile.NamedTemporaryFile(prefix="tlqldmr-s1-", suffix=".zip",
                                         dir=zip_path.parent, delete=False) as stream:
            temporary_zip = Path(stream.name)
        try:
            _write_zip(staging, destination.name, temporary_zip)
            _verify_zip_matches_staging(staging, destination.name, temporary_zip)
            _publish(staging, temporary_zip, destination, zip_path)
        finally:
            if temporary_zip.exists():
                temporary_zip.unlink()
    report = {"directory": str(destination), "zip": str(zip_path),
              "zip_bytes": zip_path.stat().st_size, "zip_sha256": digest(zip_path),
              "manifest_entries": manifest_entries}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-ready", action="store_true",
                        help="validate every completion gate without writing S1")
    args = parser.parse_args()
    try:
        if args.check_ready:
            state = validate_completed_v2(ROOT)
            print(json.dumps({"status": "ready",
                              "selection_results": len(state["selection"]["results"]),
                              "matched_results": len(state["matched"]["results"])}, indent=2))
        else:
            build()
        return 0
    except PackageNotReady as error:
        print(json.dumps({"status": "not_ready", "reason": str(error)},
                         indent=2, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
