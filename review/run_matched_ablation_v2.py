#!/usr/bin/env python3
"""Frozen, matched component ablation for the selection-v2 TL-QLDMR model.

This experiment is deliberately split into two invocations.  ``freeze`` reads
only the already locked selection-v2 metadata and fit-only artifacts.  It
derives four configurations from each task's locked Full configuration and
then creates an immutable matched-ablation lock.  It neither constructs a
``Study`` nor requests calibration/test indices or outcomes.

Only a later explicit ``--phase evaluate --resume`` invocation unlocks the
selection-v2 study and evaluates the frozen configurations on 2020 Q3-Q4.
The four variants share the Full configuration's target/source fitting rows,
preprocessing, Nystr\"om landmarks, feature map, and seeds; the only permitted
changes are setting the two component coefficients to zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from review.selection_data import digest
import review.run_selection_study_v2 as selection

VARIANTS = ("Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR")
SEEDS = (0, 1, 2)
TAU = 0.5
PREDICTION_COLUMNS = ("window_index", "raw_target_row", "time", "y_true", "prediction")
REQUIRED_CONFIG_KEYS = {
    "gamma", "total_loss", "target_fraction", "rank", "lambda1", "lambda2",
}


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _save_csv(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write a CSV completion artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def protocol() -> dict:
    tasks = selection.protocol()["tasks"]
    return {
        "version": "matched_ablation_v2",
        "base_experiment": "selection_v2",
        "task_order": tasks,
        "variants": list(VARIANTS),
        "evaluation_seeds": list(SEEDS),
        "tau": TAU,
        "configuration_rule": {
            "Full": "the locked selection-v2 Full configuration without alteration",
            "NoVariance": "the same configuration with lambda1=0 only",
            "NoDiscrepancy": "the same configuration with lambda2=0 only",
            "WeightedSVQR": "the same configuration with lambda1=lambda2=0 only",
        },
        "matching_rule": (
            "Within each task and seed, all four variants use the same Full-selected "
            "gamma, total loss mass, target fraction, requested rank, source/target "
            "fit rows, fit-only preprocessing, Nyström landmarks, and feature map."
        ),
        "fit_period": "2019; identical locked selection-v2 fitting rows",
        "selection_use": "no additional Q1 selection queries",
        "calibration_use": "none",
        "evaluation_period": "2020 Q3-Q4 reused test period",
        "evaluation_target": "point prediction only at tau=0.5",
        "heldout_gate": (
            "freeze must precede every selection-v2 or matched-ablation held-out "
            "evaluation; only a separate evaluate --resume invocation may unlock Q3-Q4"
        ),
        "solver_certificate": selection.protocol()["nonconvergence_rule"],
        "environment_versions": selection.environment_versions(),
        "analysis_plan": {
            "status": "declared before matched-ablation Q3-Q4 evaluation",
            "primary_scope": "all six within-site normal-to-extreme tasks",
            "secondary_scope": "all six leave-one-farm-out transfer-stress tasks",
            "primary_aggregate_estimands": [
                "six-within-site equal-weight macro normalized RMSE: at each site average the three separate-seed RMSE values, divide by site capacity, then average the six sites",
                "six-within-site equal-weight mean rank: rank the four variants by each site's three-seed mean RMSE using average ranks for exact ties, then average ranks across six sites",
            ],
            "secondary_aggregate_estimands": [
                "the corresponding six-cross-site equal-weight macro normalized RMSE",
                "the corresponding six-cross-site equal-weight mean rank",
            ],
            "task_comparisons": [
                "Full minus NoVariance", "Full minus NoDiscrepancy",
                "Full minus WeightedSVQR",
            ],
            "paired_loss": (
                "at every task timestamp, average squared error across seeds 0,1,2 "
                "within each variant, then subtract removal loss from Full loss"
            ),
            "blocks": (
                "non-overlapping Monday-Sunday seven-calendar-day blocks with fixed "
                "1970-01-05 Monday origin; retain all observations in each sampled block"
            ),
            "bootstrap": {
                "replicates": 10000,
                "confidence_level": 0.95,
                "rule": (
                    "sample whole blocks with equal probability and replacement; each "
                    "replicate divides sampled loss-difference sums by sampled counts"
                ),
            },
            "sign_flip": {
                "replicates": 99999,
                "sidedness": "two-sided",
                "rule": "independent whole-block sign flips with plus-one correction",
            },
            "multiplicity": (
                "Holm correction in two separate families: 6 sites x 3 Full-versus-removal "
                "tests = 18 within tests and 18 cross tests"
            ),
            "random_seed": 20261026,
            "complete_reporting": (
                "report 12 tasks x 4 variants x 3 seeds; no task, seed, or variant may "
                "be removed using held-out performance"
            ),
            "interpretation": (
                "component effects are conditional on each task's locked Full "
                "configuration and are not independently retuned procedure comparisons; "
                "failure to reject is not evidence of equivalence"
            ),
        },
    }


def derive_variants(full_config: dict) -> dict[str, dict]:
    """Return four matched configs, changing only the two lambda coordinates."""
    missing = REQUIRED_CONFIG_KEYS - set(full_config)
    if missing:
        raise ValueError(f"Locked Full configuration lacks keys: {sorted(missing)}")
    numeric = {name: float(full_config[name]) for name in REQUIRED_CONFIG_KEYS - {"rank"}}
    if not all(math.isfinite(value) for value in numeric.values()):
        raise ValueError("Locked Full configuration contains a nonfinite coordinate.")
    if int(full_config["rank"]) <= 0:
        raise ValueError("Locked Full requested rank must be positive.")
    if float(full_config["lambda1"]) <= 0 or float(full_config["lambda2"]) <= 0:
        raise ValueError("Matched removal requires a locked Full model with both components active.")
    variants = {name: dict(full_config) for name in VARIANTS}
    variants["NoVariance"]["lambda1"] = 0.0
    variants["NoDiscrepancy"]["lambda2"] = 0.0
    variants["WeightedSVQR"]["lambda1"] = 0.0
    variants["WeightedSVQR"]["lambda2"] = 0.0
    return variants


def _assert_only_lambda_changes(full: dict, variants: dict[str, dict]) -> None:
    if tuple(variants) != VARIANTS:
        raise ValueError("Matched variant order or membership changed.")
    expected = derive_variants(full)
    if variants != expected:
        raise ValueError("A matched configuration changes more than the declared lambdas.")
    fixed = set(full) - {"lambda1", "lambda2"}
    for variant in VARIANTS:
        if set(variants[variant]) != set(full):
            raise ValueError(f"{variant} changed the Full configuration schema.")
        if any(variants[variant][key] != full[key] for key in fixed):
            raise ValueError(f"{variant} changed a non-lambda Full coordinate.")


def _expected_selection_files(config: dict) -> set[str]:
    names = ("base_selection.json", "middle_selection.json",
             "regularized_selection.json", "joint_selection.json",
             "final_selection.json")
    return {
        f"{task['name']}/{method}/{name}"
        for task in config["tasks"] for method in config["methods"] for name in names
    }


def _heldout_artifacts(selection_output: Path) -> list[str]:
    """Find evidence that the base study's held-out gate was already crossed."""
    found = []
    if (selection_output / "results.json").exists():
        found.append("results.json")
    for task_dir in selection_output.iterdir() if selection_output.exists() else ():
        if not task_dir.is_dir() or not (task_dir.name.startswith("within_site") or
                                         task_dir.name.startswith("cross_site")):
            continue
        for evaluation in task_dir.glob("*/evaluation"):
            if evaluation.is_dir() and any(path.is_file() for path in evaluation.rglob("*")):
                found.append(str(evaluation.relative_to(selection_output)))
    return sorted(set(found))


def _artifact_paths(selection_output: Path, task: dict, config: dict) -> dict[str, str]:
    """Resolve the fit-only artifacts shared by all matched variants."""
    base = selection_output / task["name"] / "pooled_data"
    result = {
        "fit_and_validation_ids": str((base / "fit_and_validation_ids.npz").relative_to(selection_output)),
        "sampling": str((base / "sampling.json").relative_to(selection_output)),
        "preprocessing": str((base / "preprocessing.json").relative_to(selection_output)),
        "selection_counts": str((base / "selection_counts.json").relative_to(selection_output)),
    }
    gamma = format(float(config["gamma"]), ".17g")
    rank = int(config["rank"])
    for seed in SEEDS:
        landmark = base / "landmarks" / f"seed{seed}" / f"rank{rank}" / "landmarks.npz"
        feature_map = landmark.parent / f"gamma_{gamma}" / "map.npz"
        result[f"seed{seed}_landmarks"] = str(landmark.relative_to(selection_output))
        result[f"seed{seed}_feature_map"] = str(feature_map.relative_to(selection_output))
    return result


def _validate_map_identity(selection_output: Path, paths: dict[str, str],
                           config: dict) -> None:
    for seed in SEEDS:
        landmark_path = selection_output / paths[f"seed{seed}_landmarks"]
        map_path = selection_output / paths[f"seed{seed}_feature_map"]
        with np.load(landmark_path) as landmarks, np.load(map_path) as feature_map:
            if int(landmarks["requested_rank"]) != int(config["rank"]):
                raise ValueError(f"Locked landmark requested rank changed: {landmark_path}")
            if int(feature_map["requested_rank"]) != int(config["rank"]):
                raise ValueError(f"Locked feature-map requested rank changed: {map_path}")
            if float(feature_map["gamma"]) != float(config["gamma"]):
                raise ValueError(f"Locked feature-map gamma changed: {map_path}")
            if not np.array_equal(landmarks["centers"], feature_map["centers"]):
                raise ValueError(f"Locked landmark and feature-map centers disagree: {map_path}")


def _verify_base_selection(selection_output: Path, *, require_preregistered: bool) -> dict:
    """Validate the complete selection-v2 lock without constructing a Study."""
    selection_output = Path(selection_output).resolve()
    current_protocol = selection.protocol()
    manifest = selection._initialize(selection_output, current_protocol, True)
    protocol_path = selection_output / "protocol.json"
    lock_path = selection_output / "selection_lock.json"
    provenance_path = selection_output / "data_provenance.json"
    if not lock_path.exists():
        raise RuntimeError("Matched ablation requires a completed selection-v2 lock.")
    lock = _read_json(lock_path)
    if lock.get("version") != "selection_v2_lock_2":
        raise ValueError("Matched ablation accepts only the audited selection_v2_lock_2.")
    if lock.get("protocol_sha256") != digest(protocol_path):
        raise ValueError("Base selection-lock protocol hash mismatch.")
    for key in ("source_hashes", "data_hashes", "environment_versions"):
        if lock.get(key) != manifest.get(key):
            raise ValueError(f"Base selection-lock {key} mismatch.")
    if lock.get("environment_versions") != selection.environment_versions():
        raise ValueError("Base selection-lock environment versions changed.")
    expected_files = _expected_selection_files(current_protocol)
    if set(lock.get("selection_files", {})) != expected_files:
        raise ValueError("Base selection lock has an incomplete or unexpected file list.")
    for relative, expected_hash in lock["selection_files"].items():
        if digest(selection_output / relative) != expected_hash:
            raise ValueError(f"A base locked selection file changed: {relative}")
    if digest(provenance_path) != lock.get("data_provenance_sha256"):
        raise ValueError("Base locked data provenance changed.")
    if require_preregistered:
        exposed = _heldout_artifacts(selection_output)
        if exposed:
            raise RuntimeError(
                "Matched-ablation freeze must precede held-out evaluation; found " + exposed[0])
        if (manifest.get("active_phase") == "evaluate" or
                manifest.get("last_stable_status") == "complete" or
                manifest.get("status") in ("evaluating", "complete")):
            raise RuntimeError(
                "Matched-ablation freeze cannot follow a base held-out evaluation marker.")

    expected_keys = {(task["name"], method)
                     for task in current_protocol["tasks"]
                     for method in current_protocol["methods"]}
    if len(expected_keys) != 72:
        raise ValueError("The audited base design must contain 72 task-method identities.")
    expected_queries = 12 * 6 * 54
    if lock.get("total_configuration_queries") != expected_queries:
        raise ValueError("Base selection lock does not certify all 12 x 6 x 54 queries.")
    budget = lock.get("validation_query_budget", {})
    if budget.get("total_per_task_method") != 54:
        raise ValueError("Base selection lock does not certify 54 queries per task-method.")
    verified_query_count = 0
    for task in current_protocol["tasks"]:
        for method in current_protocol["methods"]:
            queried = []
            for stage in ("base", "middle", "joint"):
                record = _read_json(selection_output / task["name"] / method /
                                    f"{stage}_selection.json")
                if (record.get("task"), record.get("method"), record.get("stage")) != (
                        task["name"], method, stage):
                    raise ValueError("Base selection stage identity changed.")
                candidates = record.get("candidates", [])
                configs = [row.get("config") for row in candidates]
                selection._assert_unique_configs(method, configs, count=18)
                if (record.get("candidate_count") != 18 or
                        record.get("queried_effective_unique_count") != 18):
                    raise ValueError("Base selection stage does not certify 18 unique queries.")
                queried.extend(configs)
            selection._assert_unique_configs(method, queried, count=54)
            verified_query_count += len(queried)
    if verified_query_count != expected_queries:
        raise ValueError("Independently verified base query count is not 12 x 6 x 54.")
    final_rows = lock.get("final_selections", [])
    observed_keys = [(row.get("task"), row.get("method")) for row in final_rows]
    if (len(observed_keys) != 72 or len(set(observed_keys)) != 72 or
            set(observed_keys) != expected_keys):
        raise ValueError("Base selection lock has missing or duplicate final selections.")
    by_key = {(row["task"], row["method"]): row for row in final_rows}

    tasks = []
    artifact_hashes: dict[str, str] = {}
    for task in current_protocol["tasks"]:
        row = by_key[(task["name"], "Full")]
        selection_path = selection_output / task["name"] / "Full" / "final_selection.json"
        if _read_json(selection_path) != row:
            raise ValueError(f"Base lock and Full final selection disagree: {selection_path}")
        full_config = row.get("selected_config")
        variants = derive_variants(full_config)
        _assert_only_lambda_changes(full_config, variants)
        paths = _artifact_paths(selection_output, task, full_config)
        for label, relative in paths.items():
            path = selection_output / relative
            if not path.is_file():
                raise RuntimeError(f"Missing locked Full fit-only artifact: {relative}")
            artifact_hashes[relative] = digest(path)
        _validate_map_identity(selection_output, paths, full_config)
        tasks.append({
            "task": task["name"], "site": task["site"], "mode": task["mode"],
            "role": task["role"], "base_full_selection_file": str(
                selection_path.relative_to(selection_output)),
            "base_full_selection_sha256": digest(selection_path),
            "full_config": full_config, "variants": variants,
            "shared_artifacts": paths,
        })
    return {
        "selection_output": str(selection_output),
        "manifest": manifest,
        "protocol": current_protocol,
        "lock": lock,
        "selection_lock_sha256": digest(lock_path),
        "tasks": tasks,
        "fit_only_artifact_hashes": dict(sorted(artifact_hashes.items())),
    }


def _source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__), ROOT / "review/run_selection_study_v2.py",
        ROOT / "review/selection_data_v2.py", ROOT / "review/selection_data.py",
        ROOT / "review/selection_sampling.py", ROOT / "review/temporal_data.py",
        ROOT / "review/convex_refinement.py",
    ]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def _configuration_document(base: dict) -> dict:
    return {
        "version": "matched_ablation_v2_configurations_1",
        "base_selection_lock_sha256": base["selection_lock_sha256"],
        "tasks": base["tasks"],
    }


def expected_evaluation_keys(configurations: dict) -> list[str]:
    return [f"{task['task']}|{variant}|seed{seed}"
            for task in configurations["tasks"]
            for variant in VARIANTS for seed in SEEDS]


def freeze(output: Path, selection_output: Path, *, resume: bool = False) -> dict:
    """Create the independent lock without opening a Study or held-out split."""
    output = Path(output).resolve()
    if output.exists():
        if not resume:
            raise FileExistsError("Existing matched-ablation output requires explicit --resume.")
        return verify_matched_lock(output, selection_output)

    base = _verify_base_selection(selection_output, require_preregistered=True)
    config = protocol()
    configurations = _configuration_document(base)
    source_hashes = _source_hashes()

    staging = output.with_name(output.name + f".tmp.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        selection.save_json(staging / "protocol.json", config)
        selection.save_json(staging / "configurations.json", configurations)
        lock = {
            "version": "matched_ablation_v2_lock_1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": digest(staging / "protocol.json"),
            "configurations_sha256": digest(staging / "configurations.json"),
            "source_hashes": source_hashes,
            "environment_versions": selection.environment_versions(),
            "base_selection": {
                "output": base["selection_output"],
                "selection_lock_sha256": base["selection_lock_sha256"],
                "protocol_sha256": base["lock"]["protocol_sha256"],
                "source_hashes": base["lock"]["source_hashes"],
                "data_hashes": base["lock"]["data_hashes"],
                "environment_versions": base["lock"]["environment_versions"],
                "data_provenance_sha256": base["lock"]["data_provenance_sha256"],
            },
            "fit_only_artifact_hashes": base["fit_only_artifact_hashes"],
            "expected_evaluation_keys": expected_evaluation_keys(configurations),
            "heldout_policy": (
                "This lock was created before any base or matched Q2/Q3-Q4 evaluation. "
                "Freeze did not instantiate Study. Evaluation requires a later explicit "
                "evaluate --resume call and uses Q3-Q4 only."
            ),
        }
        selection.save_json(staging / "matched_lock.json", lock)
        manifest = {
            "version": "matched_ablation_v2_manifest_1",
            "status": "matched_locked",
            "created_utc": lock["created_utc"],
            "protocol_sha256": lock["protocol_sha256"],
            "configurations_sha256": lock["configurations_sha256"],
            "matched_lock_sha256": digest(staging / "matched_lock.json"),
            "base_selection_lock_sha256": base["selection_lock_sha256"],
            "heldout_unlocked": False,
        }
        selection.save_json(staging / "manifest.json", manifest)
        staging.replace(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    # Repeat the preregistration check after the atomic publish.  This keeps a
    # concurrent base-evaluation start from silently landing between the first
    # check and publication of the matched lock.
    try:
        return verify_matched_lock(output, selection_output,
                                   require_base_preregistered=True)
    except BaseException:
        # Never leave a postcheck failure looking like a usable frozen lock.
        if (output / "manifest.json").is_file():
            failed = _read_json(output / "manifest.json")
            failed.update({"status": "invalid_preregistration_postcheck",
                           "preregistration_valid": False})
            selection.save_json(output / "manifest.json", failed)
        raise


def verify_matched_lock(output: Path, selection_output: Path, *,
                        require_base_preregistered: bool = False) -> dict:
    output = Path(output).resolve()
    selection_output = Path(selection_output).resolve()
    required = [output / name for name in
                ("protocol.json", "configurations.json", "matched_lock.json", "manifest.json")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Matched-ablation output is not fully locked.")
    config = _read_json(output / "protocol.json")
    configurations = _read_json(output / "configurations.json")
    lock = _read_json(output / "matched_lock.json")
    manifest = _read_json(output / "manifest.json")
    if config != protocol():
        raise ValueError("Frozen matched-ablation protocol changed.")
    if lock.get("version") != "matched_ablation_v2_lock_1":
        raise ValueError("Unsupported matched-ablation lock version.")
    if configurations.get("version") != "matched_ablation_v2_configurations_1":
        raise ValueError("Unsupported matched-ablation configuration version.")
    if manifest.get("version") != "matched_ablation_v2_manifest_1":
        raise ValueError("Unsupported matched-ablation manifest version.")
    state = (manifest.get("status"), manifest.get("heldout_unlocked"))
    if state not in (("matched_locked", False), ("evaluating", True),
                     ("complete", True)):
        raise ValueError("Matched manifest has an invalid held-out gate state.")
    if lock.get("protocol_sha256") != digest(output / "protocol.json"):
        raise ValueError("Matched lock protocol hash mismatch.")
    if lock.get("configurations_sha256") != digest(output / "configurations.json"):
        raise ValueError("Matched lock configuration hash mismatch.")
    if lock.get("source_hashes") != _source_hashes():
        raise ValueError("Frozen matched-ablation source hashes changed.")
    if lock.get("environment_versions") != selection.environment_versions():
        raise ValueError("Frozen matched-ablation environment versions changed.")
    if Path(lock["base_selection"]["output"]).resolve() != selection_output:
        raise ValueError("Matched lock points to a different base selection output.")
    base = _verify_base_selection(
        selection_output, require_preregistered=require_base_preregistered)
    frozen_base = lock["base_selection"]
    expected_base = {
        "output": base["selection_output"],
        "selection_lock_sha256": base["selection_lock_sha256"],
        "protocol_sha256": base["lock"]["protocol_sha256"],
        "source_hashes": base["lock"]["source_hashes"],
        "data_hashes": base["lock"]["data_hashes"],
        "environment_versions": base["lock"]["environment_versions"],
        "data_provenance_sha256": base["lock"]["data_provenance_sha256"],
    }
    if frozen_base != expected_base:
        raise ValueError("Base selection identity changed after matched freeze.")
    if lock.get("fit_only_artifact_hashes") != base["fit_only_artifact_hashes"]:
        raise ValueError("A frozen fit-only ID, preprocessing, landmark, or map file changed.")
    expected_configurations = _configuration_document(base)
    if configurations != expected_configurations:
        raise ValueError("Frozen matched configurations no longer derive from the base lock.")
    for task in configurations["tasks"]:
        _assert_only_lambda_changes(task["full_config"], task["variants"])
    expected_keys = expected_evaluation_keys(configurations)
    if lock.get("expected_evaluation_keys") != expected_keys or len(expected_keys) != 12 * 4 * 3:
        raise ValueError("Matched lock does not declare all 12 x 4 x 3 evaluations.")
    if manifest.get("matched_lock_sha256") != digest(output / "matched_lock.json"):
        raise ValueError("Matched manifest lock hash mismatch.")
    if manifest.get("protocol_sha256") != lock["protocol_sha256"] or \
            manifest.get("configurations_sha256") != lock["configurations_sha256"]:
        raise ValueError("Matched manifest identity differs from the lock.")
    return {
        "protocol": config, "configurations": configurations, "lock": lock,
        "manifest": manifest, "base": base,
        "matched_lock_sha256": digest(output / "matched_lock.json"),
    }


def _validate_existing_result(result: dict, *, key: str, matched_lock_sha256: str,
                              config: dict, prediction_path: Path,
                              output: Path | None = None,
                              task_record: dict | None = None,
                              base_selection_lock_sha256: str | None = None) -> None:
    """Validate a completion marker and every artifact it identifies.

    The optional frozen-task arguments are supplied by the real resume path.
    Keeping the elementary identity checks first also gives a focused failure
    when a result from a different matched lock is encountered.
    """
    if result.get("evaluation_key") != key:
        raise ValueError(f"Existing matched result key changed: {key}")
    if result.get("matched_lock_sha256") != matched_lock_sha256:
        raise ValueError(f"Existing matched result used a different lock: {key}")
    if result.get("config") != config:
        raise ValueError(f"Existing matched result configuration changed: {key}")
    if not prediction_path.is_file() or result.get("prediction_sha256") != digest(prediction_path):
        raise ValueError(f"Existing matched prediction artifact changed: {key}")
    fit_path = prediction_path.parent / "fit.json"
    model_path = prediction_path.parent / "model.npz"
    if not fit_path.is_file() or result.get("fit_record_sha256") != digest(fit_path):
        raise ValueError(f"Existing matched fit record changed: {key}")
    if not model_path.is_file() or result.get("model_sha256") != digest(model_path):
        raise ValueError(f"Existing matched model artifact changed: {key}")
    if output is not None:
        output = Path(output).resolve()
        expected_files = {
            "fit_record": str(fit_path.relative_to(output)),
            "model_file": str(model_path.relative_to(output)),
            "prediction_file": str(prediction_path.relative_to(output)),
        }
        for field, expected in expected_files.items():
            if result.get(field) != expected:
                raise ValueError(f"Existing matched {field} path changed: {key}")
    if task_record is not None:
        task_name, variant, seed_name = key.split("|")
        seed = int(seed_name.removeprefix("seed"))
        expected_identity = {
            "task": task_name, "site": task_record["site"],
            "mode": task_record["mode"], "role": task_record["role"],
            "variant": variant, "seed": seed, "tau": TAU,
            "shared_artifacts": task_record["shared_artifacts"],
            "feature_map": task_record["shared_artifacts"][f"seed{seed}_feature_map"],
            "data_folder": str(Path(task_record["shared_artifacts"]["preprocessing"]).parent),
        }
        for field, expected in expected_identity.items():
            if result.get(field) != expected:
                raise ValueError(f"Existing matched result {field} changed: {key}")
        if (base_selection_lock_sha256 is not None and
                result.get("base_selection_lock_sha256") != base_selection_lock_sha256):
            raise ValueError(f"Existing matched result used a different base lock: {key}")

        fit_record = _read_json(fit_path)
        expected_fit = {
            "task": task_name, "method": variant, "config": config,
            "seed": seed, "tau": TAU,
            "candidate_id": "matched_ablation_v2_locked",
            "feature_map": expected_identity["feature_map"],
            "data_folder": expected_identity["data_folder"],
        }
        for field, expected in expected_fit.items():
            if fit_record.get(field) != expected:
                raise ValueError(f"Existing matched fit {field} changed: {key}")
        if fit_record.get("diagnostics", {}).get("converged") is not True:
            raise ValueError(f"Existing matched fit lacks the solver certificate: {key}")

        frame = pd.read_csv(prediction_path)
        if tuple(frame.columns) != PREDICTION_COLUMNS:
            raise ValueError(f"Existing matched prediction schema changed: {key}")
        numeric = frame[["window_index", "raw_target_row", "y_true", "prediction"]].to_numpy()
        if not len(frame) or not np.isfinite(numeric).all():
            raise ValueError(f"Existing matched prediction values changed: {key}")
        point = selection.point_metrics(frame["y_true"].to_numpy(),
                                        frame["prediction"].to_numpy())
        recorded_point = result.get("point", {})
        if point["n"] != recorded_point.get("n"):
            raise ValueError(f"Existing matched prediction count changed: {key}")
        for metric in ("rmse", "mae", "r2"):
            observed, recorded = point[metric], recorded_point.get(metric)
            if ((observed is None) != (recorded is None) or
                    (observed is not None and not np.isclose(
                        observed, recorded, rtol=1e-12, atol=1e-12))):
                raise ValueError(f"Existing matched point metric changed: {key}/{metric}")
        if result.get("counts", {}).get("n_test") != len(frame):
            raise ValueError(f"Existing matched test count changed: {key}")


def _preflight_existing_evaluation(output: Path, configurations: dict,
                                   matched_lock_sha256: str,
                                   base_selection_lock_sha256: str) -> None:
    """Reject stale, extra, or cross-lock evaluation artifacts before resuming."""
    config_by_key = {
        f"{task['task']}|{variant}|seed{seed}": task["variants"][variant]
        for task in configurations["tasks"] for variant in VARIANTS for seed in SEEDS
    }
    task_by_name = {task["task"]: task for task in configurations["tasks"]}
    path_by_key = {
        key: output / "evaluation" / key.split("|")[0] / key.split("|")[1] /
        key.split("|")[2] / "result.json"
        for key in config_by_key
    }
    actual_paths = set((output / "evaluation").rglob("result.json")) \
        if (output / "evaluation").exists() else set()
    expected_paths = set(path_by_key.values())
    extra = sorted(actual_paths - expected_paths)
    if extra:
        raise ValueError(f"Unexpected old matched evaluation artifact: {extra[0]}")
    records = {}
    for key, path in path_by_key.items():
        if not path.exists():
            continue
        record = _read_json(path)
        prediction = path.parent / "test_predictions.csv"
        _validate_existing_result(
            record, key=key, matched_lock_sha256=matched_lock_sha256,
            config=config_by_key[key], prediction_path=prediction, output=output,
            task_record=task_by_name[key.split("|")[0]],
            base_selection_lock_sha256=base_selection_lock_sha256)
        records[key] = record
    aggregate_path = output / "results.json"
    if aggregate_path.exists():
        aggregate = _read_json(aggregate_path)
        if not isinstance(aggregate, list):
            raise ValueError("Existing matched results.json is not a result list.")
        aggregate_by_key = {row.get("evaluation_key"): row for row in aggregate}
        if (len(aggregate_by_key) != len(aggregate) or
                set(aggregate_by_key) != set(config_by_key)):
            raise ValueError("Existing matched results.json is incomplete or contains old keys.")
        if set(records) != set(config_by_key) or any(
                records[key] != aggregate_by_key[key] for key in records):
            raise ValueError("Existing matched results.json disagrees with per-key artifacts.")


def evaluate(output: Path, selection_output: Path, *, resume: bool) -> list[dict]:
    if not resume:
        raise RuntimeError("Matched evaluation requires a later explicit evaluate --resume call.")
    output = Path(output).resolve()
    selection_output = Path(selection_output).resolve()
    verified = verify_matched_lock(output, selection_output)
    configurations = verified["configurations"]
    base = verified["base"]
    lock_sha = verified["matched_lock_sha256"]
    manifest = verified["manifest"]
    _preflight_existing_evaluation(
        output, configurations, lock_sha, base["selection_lock_sha256"])
    if manifest.get("heldout_unlocked") is False:
        manifest["heldout_unlocked"] = True
        manifest["status"] = "evaluating"
        manifest["evaluation_started_utc"] = datetime.now(timezone.utc).isoformat()
        selection.save_json(output / "manifest.json", manifest)

    # This is the first point in the workflow at which a Study is constructed
    # and its held-out gate is opened.  Its output remains the base selection
    # directory so all variants reuse the exact locked Full maps and contexts;
    # fitted matched models and predictions are written only under `output`.
    study = selection.Study(selection_output, base["protocol"],
                            write_global_provenance=False)
    study.unlock_heldout(base["manifest"])
    tasks_by_name = {task["name"]: task for task in base["protocol"]["tasks"]}
    results: list[dict] = []
    observed_keys: set[str] = set()

    for frozen_task in configurations["tasks"]:
        task = tasks_by_name[frozen_task["task"]]
        context = study.context(task, "Full")
        test_ids = study._heldout_ids(context, "test")
        y_test = context["data"].y_raw[test_ids]
        artifact_paths = frozen_task["shared_artifacts"]
        for relative in artifact_paths.values():
            expected_hash = verified["lock"]["fit_only_artifact_hashes"][relative]
            if digest(selection_output / relative) != expected_hash:
                raise ValueError(f"A shared locked artifact changed during evaluation: {relative}")

        for variant in VARIANTS:
            config = frozen_task["variants"][variant]
            for seed in SEEDS:
                key = f"{task['name']}|{variant}|seed{seed}"
                if key in observed_keys:
                    raise ValueError(f"Duplicate matched evaluation key: {key}")
                observed_keys.add(key)
                folder = output / "evaluation" / task["name"] / variant / f"seed{seed}"
                result_path = folder / "result.json"
                prediction_path = folder / "test_predictions.csv"
                if result_path.exists():
                    result = _read_json(result_path)
                    _validate_existing_result(
                        result, key=key, matched_lock_sha256=lock_sha,
                        config=config, prediction_path=prediction_path, output=output,
                        task_record=frozen_task,
                        base_selection_lock_sha256=base["selection_lock_sha256"])
                    results.append(result)
                    continue

                fit_record = study.fit(
                    task, variant, config, seed, TAU, folder,
                    "matched_ablation_v2_locked")
                if fit_record["diagnostics"].get("converged") is not True:
                    raise RuntimeError(f"Matched locked fit missed the solver certificate: {key}")
                expected_map = artifact_paths[f"seed{seed}_feature_map"]
                if fit_record.get("feature_map") != expected_map:
                    raise ValueError(f"Matched variants did not reuse the locked Full map: {key}")
                expected_data_folder = str(Path(artifact_paths["preprocessing"]).parent)
                if fit_record.get("data_folder") != expected_data_folder:
                    raise ValueError(f"Matched variants did not reuse the locked Full fit IDs: {key}")
                prediction = study.predict(context, folder, "test")
                frame = pd.DataFrame({
                    "window_index": test_ids,
                    "raw_target_row": context["data"].target_row_indices[test_ids],
                    "time": context["data"].timestamps[test_ids].astype(str),
                    "y_true": y_test,
                    "prediction": prediction,
                })
                _save_csv(prediction_path, frame)
                model_path = folder / "model.npz"
                if not model_path.is_file():
                    raise RuntimeError(f"Matched fit did not produce its model artifact: {key}")
                result = {
                    "evaluation_key": key,
                    "task": task["name"], "site": task["site"],
                    "mode": task["mode"], "role": task["role"],
                    "variant": variant, "seed": seed, "tau": TAU,
                    "config": config,
                    "matched_lock_sha256": lock_sha,
                    "base_selection_lock_sha256": base["selection_lock_sha256"],
                    "shared_artifacts": artifact_paths,
                    "feature_map": fit_record["feature_map"],
                    "data_folder": fit_record["data_folder"],
                    "fit_record": str((folder / "fit.json").relative_to(output)),
                    "fit_record_sha256": digest(folder / "fit.json"),
                    "model_file": str(model_path.relative_to(output)),
                    "model_sha256": digest(model_path),
                    "prediction_file": str(prediction_path.relative_to(output)),
                    "prediction_sha256": digest(prediction_path),
                    "counts": {**context["counts"], "n_test": len(test_ids)},
                    "point": selection.point_metrics(y_test, prediction),
                    "diagnostics": fit_record["diagnostics"],
                }
                selection.save_json(result_path, result)
                results.append(result)
                print(json.dumps({"phase": "matched_evaluation", "key": key,
                                  "rmse": result["point"]["rmse"]}), flush=True)
        study.context_cache.clear()
        study.feature_cache.clear()

    expected = set(verified["lock"]["expected_evaluation_keys"])
    actual = {row["evaluation_key"] for row in results}
    if actual != expected or len(results) != len(expected):
        missing = sorted(expected - actual)
        raise RuntimeError("Matched evaluation is incomplete" +
                           (f"; first missing key: {missing[0]}" if missing else "."))
    results.sort(key=lambda row: verified["lock"]["expected_evaluation_keys"].index(
        row["evaluation_key"]))
    selection.save_json(output / "results.json", results)
    manifest = _read_json(output / "manifest.json")
    manifest.update({
        "status": "complete", "heldout_unlocked": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "results_sha256": digest(output / "results.json"),
        "completed_evaluation_keys": len(results),
    })
    selection.save_json(output / "manifest.json", manifest)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-output", type=Path,
                        default=ROOT / "review/selection_runs/selection_v2")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "review/selection_runs/matched_ablation_v2")
    parser.add_argument("--phase", choices=("freeze", "evaluate"), default="freeze")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "evaluate" and (not args.resume or not args.output.exists()):
        parser.error(
            "Evaluation requires a later --phase evaluate --resume call with an existing matched lock.")
    if args.phase == "freeze":
        freeze(args.output, args.selection_output, resume=args.resume)
    else:
        with threadpool_limits(limits=1):
            evaluate(args.output, args.selection_output, resume=args.resume)


if __name__ == "__main__":
    main()
