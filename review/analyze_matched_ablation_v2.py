#!/usr/bin/env python3
"""Pre-frozen analysis for the selection-v2 matched component ablation.

Run ``--freeze-only`` after the matched-ablation lock exists and before either
the base selection-v2 or matched held-out evaluation begins.  A later
``--resume`` invocation verifies the
unchanged analysis and experiment locks, all 144 result identities, and every
prediction-file hash before producing the predeclared summaries and tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from review.selection_data import digest
import review.run_matched_ablation_v2 as matched
import review.run_selection_study_v2 as selection


COMPARATORS = ("NoVariance", "NoDiscrepancy", "WeightedSVQR")
REQUIRED_PREDICTION_COLUMNS = (
    "window_index", "raw_target_row", "time", "y_true", "prediction",
)


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{name: package_version(name) for name in ("numpy", "pandas", "scipy")},
    }


def analysis_protocol() -> dict:
    experiment = matched.protocol()
    return {
        "version": "matched_ablation_v2_analysis_1",
        "experiment_version": experiment["version"],
        "analysis_plan": experiment["analysis_plan"],
        "capacities_mw": {str(site): capacity for site, capacity in selection.CAPACITY.items()},
        "variant_order": list(matched.VARIANTS),
        "comparator_order": list(COMPARATORS),
        "rmse_rule": "compute RMSE separately per seed; average the three seed RMSE values",
        "rank_rule": "average ranks for exact ties; lower RMSE receives the lower rank",
        "prediction_alignment": (
            "within each task all variants and seeds must have identical window_index, "
            "raw_target_row, timestamp, and y_true arrays"
        ),
        "output_files": [
            "task_metrics.csv", "aggregate_metrics.csv", "paired_inference.csv",
            "task_metrics.tex", "aggregate_metrics.tex", "paired_inference.tex",
            "summary.json",
        ],
        "environment_versions": _environment_versions(),
    }


def _source_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "review/run_matched_ablation_v2.py",
             ROOT / "review/run_selection_study_v2.py"]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def _evaluation_has_started(matched_output: Path, manifest: dict) -> bool:
    return bool(
        manifest.get("heldout_unlocked") or
        manifest.get("status") in ("evaluating", "complete") or
        (matched_output / "results.json").exists() or
        ((matched_output / "evaluation").exists() and
         any(path.is_file() for path in (matched_output / "evaluation").rglob("*")))
    )


def freeze(output: Path, matched_output: Path) -> dict:
    """Freeze analysis before any base or matched held-out artifact exists."""
    output, matched_output = Path(output).resolve(), Path(matched_output).resolve()
    if output.exists():
        raise FileExistsError("Existing matched-analysis output cannot be frozen again.")
    verified = matched.verify_matched_lock(
        matched_output, Path(_read_json(matched_output / "matched_lock.json")
                             ["base_selection"]["output"]),
        require_base_preregistered=True)
    if _evaluation_has_started(matched_output, verified["manifest"]):
        raise RuntimeError("Analysis must be frozen before matched held-out evaluation begins.")
    config = analysis_protocol()
    staging = output.with_name(output.name + f".tmp.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        selection.save_json(staging / "analysis_protocol.json", config)
        lock = {
            "version": "matched_ablation_v2_analysis_lock_1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_protocol_sha256": digest(staging / "analysis_protocol.json"),
            "source_hashes": _source_hashes(),
            "environment_versions": _environment_versions(),
            "matched_output": str(matched_output),
            "matched_lock_sha256": verified["matched_lock_sha256"],
            "matched_protocol_sha256": verified["lock"]["protocol_sha256"],
            "matched_configurations_sha256": verified["lock"]["configurations_sha256"],
            "expected_evaluation_keys": verified["lock"]["expected_evaluation_keys"],
            "heldout_policy": (
                "analysis lock created before base selection-v2 and matched held-out "
                "evaluation; --resume is required after all 144 prediction records exist"
            ),
        }
        selection.save_json(staging / "analysis_lock.json", lock)
        manifest = {
            "version": "matched_ablation_v2_analysis_manifest_1",
            "status": "analysis_locked",
            "created_utc": lock["created_utc"],
            "analysis_lock_sha256": digest(staging / "analysis_lock.json"),
            "heldout_results_read": False,
        }
        selection.save_json(staging / "manifest.json", manifest)
        staging.replace(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    # Recheck both the base and matched held-out gates after atomically
    # publishing the analysis lock, closing the corresponding race window.
    try:
        return verify_analysis_lock(output, matched_output, require_preregistered=True)
    except BaseException:
        # A failed post-publication gate check must not remain resumable as a
        # valid pre-evaluation analysis lock.
        if (output / "manifest.json").is_file():
            failed = _read_json(output / "manifest.json")
            failed.update({"status": "invalid_preregistration_postcheck",
                           "preregistration_valid": False})
            selection.save_json(output / "manifest.json", failed)
        raise


def verify_analysis_lock(output: Path, matched_output: Path, *,
                         require_preregistered: bool = False) -> dict:
    output, matched_output = Path(output).resolve(), Path(matched_output).resolve()
    for name in ("analysis_protocol.json", "analysis_lock.json", "manifest.json"):
        if not (output / name).is_file():
            raise RuntimeError(f"Missing frozen analysis file: {name}")
    protocol = _read_json(output / "analysis_protocol.json")
    lock = _read_json(output / "analysis_lock.json")
    manifest = _read_json(output / "manifest.json")
    if protocol != analysis_protocol():
        raise ValueError("Frozen matched-analysis protocol changed.")
    if lock.get("version") != "matched_ablation_v2_analysis_lock_1":
        raise ValueError("Unsupported matched-analysis lock version.")
    if manifest.get("version") != "matched_ablation_v2_analysis_manifest_1":
        raise ValueError("Unsupported matched-analysis manifest version.")
    state = (manifest.get("status"), manifest.get("heldout_results_read"))
    if state not in (("analysis_locked", False), ("analyzing", True),
                     ("complete", True)):
        raise ValueError("Matched-analysis manifest has an invalid held-out gate state.")
    if lock.get("analysis_protocol_sha256") != digest(output / "analysis_protocol.json"):
        raise ValueError("Matched-analysis protocol hash mismatch.")
    if lock.get("source_hashes") != _source_hashes():
        raise ValueError("Frozen matched-analysis source hashes changed.")
    if lock.get("environment_versions") != _environment_versions():
        raise ValueError("Frozen matched-analysis environment changed.")
    if Path(lock.get("matched_output", "")).resolve() != matched_output:
        raise ValueError("Analysis lock points to a different matched experiment.")
    matched_lock = _read_json(matched_output / "matched_lock.json")
    base_output = Path(matched_lock["base_selection"]["output"])
    verified = matched.verify_matched_lock(
        matched_output, base_output,
        require_base_preregistered=require_preregistered)
    if require_preregistered and _evaluation_has_started(
            matched_output, verified["manifest"]):
        raise RuntimeError(
            "Analysis lock publication overlapped a matched held-out evaluation start.")
    if lock.get("matched_lock_sha256") != verified["matched_lock_sha256"]:
        raise ValueError("Matched experiment lock changed after analysis freeze.")
    if lock.get("matched_protocol_sha256") != verified["lock"]["protocol_sha256"]:
        raise ValueError("Matched experiment protocol changed after analysis freeze.")
    if lock.get("matched_configurations_sha256") != verified["lock"]["configurations_sha256"]:
        raise ValueError("Matched configurations changed after analysis freeze.")
    if lock.get("expected_evaluation_keys") != verified["lock"]["expected_evaluation_keys"]:
        raise ValueError("Matched expected result identities changed after analysis freeze.")
    if manifest.get("analysis_lock_sha256") != digest(output / "analysis_lock.json"):
        raise ValueError("Matched-analysis manifest lock hash mismatch.")
    if "output_hashes" in manifest:
        expected_outputs = set(protocol["output_files"])
        if set(manifest["output_hashes"]) != expected_outputs:
            raise ValueError("Completed matched-analysis output list changed.")
        for name, expected_hash in manifest["output_hashes"].items():
            if not (output / name).is_file() or digest(output / name) != expected_hash:
                raise ValueError(f"Completed matched-analysis output changed: {name}")
    return {"protocol": protocol, "lock": lock, "manifest": manifest,
            "matched": verified}


def _verify_results(matched_output: Path, verified: dict) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    manifest = _read_json(matched_output / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("completed_evaluation_keys") != 144:
        raise RuntimeError("Matched evaluation is not complete for all 144 locked keys.")
    results_path = matched_output / "results.json"
    if not results_path.is_file() or manifest.get("results_sha256") != digest(results_path):
        raise ValueError("Matched aggregate results file is missing or changed.")
    rows = _read_json(results_path)
    expected = verified["lock"]["expected_evaluation_keys"]
    keys = [row.get("evaluation_key") for row in rows]
    if len(rows) != 144 or len(set(keys)) != 144 or set(keys) != set(expected):
        raise ValueError("Matched results must contain exactly the 144 locked identities.")
    by_key = {row["evaluation_key"]: row for row in rows}
    task_by_name = {task["task"]: task
                    for task in verified["configurations"]["tasks"]}
    frames = {}
    for key in expected:
        row = by_key[key]
        task, variant, seed_name = key.split("|")
        standard_result = matched_output / "evaluation" / task / variant / seed_name / "result.json"
        if not standard_result.is_file() or _read_json(standard_result) != row:
            raise ValueError(f"Aggregate and per-key result disagree: {key}")
        prediction = standard_result.parent / "test_predictions.csv"
        matched._validate_existing_result(
            row, key=key, matched_lock_sha256=verified["matched_lock_sha256"],
            config=task_by_name[task]["variants"][variant],
            prediction_path=prediction, output=matched_output,
            task_record=task_by_name[task],
            base_selection_lock_sha256=verified["base"]["selection_lock_sha256"])
        frame = pd.read_csv(prediction)
        if tuple(frame.columns) != REQUIRED_PREDICTION_COLUMNS:
            raise ValueError(f"Prediction schema mismatch: {key}")
        numeric = frame[["window_index", "raw_target_row", "y_true", "prediction"]].to_numpy()
        if len(frame) != row["point"]["n"] or not np.isfinite(numeric).all():
            raise ValueError(f"Prediction length or numeric values are invalid: {key}")
        computed = selection.point_metrics(frame["y_true"].to_numpy(),
                                           frame["prediction"].to_numpy())
        if computed["n"] != row["point"]["n"]:
            raise ValueError(f"Saved point count differs from predictions: {key}")
        for metric in ("rmse", "mae", "r2"):
            observed, recorded = computed[metric], row["point"][metric]
            if ((observed is None) != (recorded is None) or
                    (observed is not None and not np.isclose(
                        observed, recorded, rtol=1e-12, atol=1e-12))):
                raise ValueError(f"Saved point metric differs from predictions: {key}/{metric}")
        frames[key] = frame
    return [by_key[key] for key in expected], frames


def _aligned_task_frames(task: str, frames: dict[str, pd.DataFrame]) -> dict[str, list[pd.DataFrame]]:
    grouped = {
        variant: [frames[f"{task}|{variant}|seed{seed}"] for seed in matched.SEEDS]
        for variant in matched.VARIANTS
    }
    reference = grouped["Full"][0]
    identity_columns = ["window_index", "raw_target_row", "time", "y_true"]
    reference_identity = reference[identity_columns]
    for variant_frames in grouped.values():
        for frame in variant_frames:
            if not frame[identity_columns].equals(reference_identity):
                raise ValueError(f"Prediction rows are not exactly aligned for task {task}.")
    return grouped


def _block_summaries(times: pd.Series, differences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parsed = pd.to_datetime(times, errors="raise")
    if parsed.isna().any():
        raise ValueError("A prediction timestamp is missing.")
    origin = pd.Timestamp("1970-01-05")
    block = ((parsed.dt.normalize() - origin).dt.days // 7).to_numpy()
    unique, inverse = np.unique(block, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(differences, dtype=float),
                       minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    return sums, counts


def _rng_for_key(seed: int, key: str) -> np.random.Generator:
    identity = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(identity[:8], "little"))


def paired_block_inference(differences: np.ndarray, times: pd.Series, *, key: str,
                           random_seed: int, bootstrap_replicates: int,
                           sign_flip_replicates: int,
                           confidence_level: float = 0.95) -> dict:
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not len(differences) or not np.isfinite(differences).all():
        raise ValueError("Paired loss differences must be finite and nonempty.")
    sums, counts = _block_summaries(times, differences)
    rng = _rng_for_key(random_seed, key)
    block_count = len(sums)
    sampled = rng.integers(0, block_count, size=(bootstrap_replicates, block_count))
    bootstrap = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    alpha = 1 - confidence_level
    lower, upper = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    observed = float(sums.sum() / counts.sum())
    extreme = 0
    remaining = sign_flip_replicates
    batch = 10000
    denominator = counts.sum()
    while remaining:
        size = min(batch, remaining)
        signs = rng.integers(0, 2, size=(size, block_count), dtype=np.int8) * 2 - 1
        randomized = (signs @ sums) / denominator
        extreme += int(np.count_nonzero(np.abs(randomized) >= abs(observed)))
        remaining -= size
    p_value = float((extreme + 1) / (sign_flip_replicates + 1))
    return {
        "n": int(counts.sum()), "blocks": block_count,
        "mean_squared_error_difference_mw2": observed,
        "ci_lower_mw2": float(lower), "ci_upper_mw2": float(upper),
        "p_unadjusted": p_value,
    }


def _holm_adjust(values: list[float]) -> list[float]:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array, kind="mergesort")
    adjusted_sorted = np.maximum.accumulate(
        (len(values_array) - np.arange(len(values_array))) * values_array[order])
    adjusted = np.empty(len(values_array), dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def _format_latex(frame: pd.DataFrame, caption: str, label: str,
                  float_format: str = "%.4f") -> str:
    return frame.to_latex(index=False, escape=True, caption=caption, label=label,
                          float_format=float_format, na_rep="--")


def analyze(output: Path, matched_output: Path) -> dict:
    output, matched_output = Path(output).resolve(), Path(matched_output).resolve()
    verified = verify_analysis_lock(output, matched_output)
    manifest = _read_json(output / "manifest.json")
    if manifest.get("status") == "complete":
        # Completed analysis artifacts are immutable.  Revalidate their source
        # result bundle and return the frozen summary without rewriting files.
        _verify_results(matched_output, verified["matched"])
        frozen_summary = _read_json(output / "summary.json")
        if frozen_summary.get("matched_results_sha256") != digest(
                matched_output / "results.json"):
            raise ValueError("Matched results changed after analysis completion.")
        return frozen_summary
    if manifest.get("heldout_results_read") is False:
        # Persist the held-out access marker before opening results so a killed
        # process cannot leave a misleading pre-analysis manifest behind.
        manifest.update({
            "status": "analyzing", "heldout_results_read": True,
            "analysis_started_utc": datetime.now(timezone.utc).isoformat(),
        })
        selection.save_json(output / "manifest.json", manifest)
    rows, frames = _verify_results(matched_output, verified["matched"])
    plan = verified["protocol"]["analysis_plan"]
    configurations = verified["matched"]["configurations"]

    result_by_key = {row["evaluation_key"]: row for row in rows}
    task_metric_rows = []
    inference_rows = []
    task_order = [task["task"] for task in configurations["tasks"]]
    task_metadata = {task["task"]: task for task in configurations["tasks"]}
    for task_name in task_order:
        metadata = task_metadata[task_name]
        capacity = float(selection.CAPACITY[metadata["site"]])
        grouped = _aligned_task_frames(task_name, frames)
        variant_rmse = {}
        for variant in matched.VARIANTS:
            seed_rmse = [result_by_key[f"{task_name}|{variant}|seed{seed}"]["point"]["rmse"]
                         for seed in matched.SEEDS]
            variant_rmse[variant] = float(np.mean(seed_rmse))
        ranks = rankdata([variant_rmse[variant] for variant in matched.VARIANTS],
                         method="average")
        for variant, rank in zip(matched.VARIANTS, ranks):
            seed_rmse = [result_by_key[f"{task_name}|{variant}|seed{seed}"]["point"]["rmse"]
                         for seed in matched.SEEDS]
            task_metric_rows.append({
                "scope": "within" if metadata["mode"] == "within" else "cross",
                "task": task_name, "site": metadata["site"], "variant": variant,
                "capacity_mw": capacity,
                "seed0_rmse_mw": seed_rmse[0], "seed1_rmse_mw": seed_rmse[1],
                "seed2_rmse_mw": seed_rmse[2],
                "mean_seed_rmse_mw": variant_rmse[variant],
                "normalized_rmse": variant_rmse[variant] / capacity,
                "rank": float(rank),
            })

        full_losses = np.mean([
            (frame["y_true"].to_numpy() - frame["prediction"].to_numpy()) ** 2
            for frame in grouped["Full"]
        ], axis=0)
        for comparator in COMPARATORS:
            comparator_losses = np.mean([
                (frame["y_true"].to_numpy() - frame["prediction"].to_numpy()) ** 2
                for frame in grouped[comparator]
            ], axis=0)
            inference = paired_block_inference(
                full_losses - comparator_losses, grouped["Full"][0]["time"],
                key=f"{task_name}|Full-minus-{comparator}",
                random_seed=int(plan["random_seed"]),
                bootstrap_replicates=int(plan["bootstrap"]["replicates"]),
                sign_flip_replicates=int(plan["sign_flip"]["replicates"]),
                confidence_level=float(plan["bootstrap"]["confidence_level"]),
            )
            inference_rows.append({
                "scope": "within" if metadata["mode"] == "within" else "cross",
                "task": task_name, "site": metadata["site"],
                "comparison": f"Full - {comparator}", **inference,
            })

    task_metrics = pd.DataFrame(task_metric_rows)
    aggregate = (task_metrics.groupby(["scope", "variant"], sort=False)
                 .agg(sites=("site", "nunique"),
                      macro_normalized_rmse=("normalized_rmse", "mean"),
                      mean_rank=("rank", "mean"))
                 .reset_index())
    inference = pd.DataFrame(inference_rows)
    inference["p_holm"] = np.nan
    for scope in ("within", "cross"):
        ids = inference.index[inference["scope"] == scope].tolist()
        if len(ids) != 18:
            raise ValueError(f"Holm family {scope} must contain exactly 18 tests.")
        inference.loc[ids, "p_holm"] = _holm_adjust(
            inference.loc[ids, "p_unadjusted"].tolist())
    inference["reject_holm_0_05"] = inference["p_holm"] <= .05

    _atomic_csv(output / "task_metrics.csv", task_metrics)
    _atomic_csv(output / "aggregate_metrics.csv", aggregate)
    _atomic_csv(output / "paired_inference.csv", inference)
    _atomic_text(output / "task_metrics.tex", _format_latex(
        task_metrics[["scope", "task", "variant", "mean_seed_rmse_mw",
                      "normalized_rmse", "rank"]],
        "Matched component ablation by task (three-seed means).",
        "tab:matched_ablation_task"))
    _atomic_text(output / "aggregate_metrics.tex", _format_latex(
        aggregate, "Site-equal matched-ablation aggregates.",
        "tab:matched_ablation_aggregate"))
    _atomic_text(output / "paired_inference.tex", _format_latex(
        inference[["scope", "task", "comparison", "mean_squared_error_difference_mw2",
                   "ci_lower_mw2", "ci_upper_mw2", "p_unadjusted", "p_holm"]],
        "Predeclared paired seven-day-block inference for the matched ablation.",
        "tab:matched_ablation_inference"))
    summary = {
        "version": "matched_ablation_v2_analysis_results_1",
        "analysis_lock_sha256": digest(output / "analysis_lock.json"),
        "matched_lock_sha256": verified["lock"]["matched_lock_sha256"],
        "matched_results_sha256": digest(matched_output / "results.json"),
        "complete_evaluation_keys": 144,
        "task_metric_rows": len(task_metrics),
        "paired_tests": {"within": 18, "cross": 18},
        "aggregate_metrics": aggregate.to_dict(orient="records"),
        "interpretation": plan["interpretation"],
    }
    selection.save_json(output / "summary.json", summary)
    output_hashes = {name: digest(output / name)
                     for name in verified["protocol"]["output_files"]}
    manifest = _read_json(output / "manifest.json")
    manifest.update({
        "status": "complete", "heldout_results_read": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "output_hashes": output_hashes,
    })
    selection.save_json(output / "manifest.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-output", type=Path,
                        default=ROOT / "review/selection_runs/matched_ablation_v2")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "review/selection_runs/matched_ablation_v2_analysis")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        freeze(args.output, args.matched_output)
    else:
        if not args.output.exists():
            parser.error("--resume requires an existing pre-evaluation analysis lock.")
        analyze(args.output, args.matched_output)


if __name__ == "__main__":
    main()
