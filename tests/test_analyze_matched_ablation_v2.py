"""Tests for the pre-frozen matched-ablation analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from review.selection_data import digest
import review.analyze_matched_ablation_v2 as analysis
import review.run_matched_ablation_v2 as matched
import review.run_selection_study_v2 as selection


def _configurations() -> dict:
    tasks = []
    for task in matched.protocol()["task_order"]:
        full = {
            "gamma": .002, "total_loss": 1000., "target_fraction": .5,
            "rank": 128, "lambda1": .1, "lambda2": .2,
        }
        shared = {
            "preprocessing": f"{task['name']}/pooled_data/preprocessing.json",
        }
        for seed in matched.SEEDS:
            shared[f"seed{seed}_feature_map"] = (
                f"{task['name']}/pooled_data/landmarks/seed{seed}/rank128/"
                "gamma_0.002/map.npz")
        tasks.append({
            "task": task["name"], "site": task["site"], "mode": task["mode"],
            "role": task["role"], "full_config": full,
            "variants": matched.derive_variants(full), "shared_artifacts": shared,
        })
    return {"version": "test", "tasks": tasks}


def test_analysis_protocol_contains_exact_preregistered_inference():
    plan = analysis.analysis_protocol()["analysis_plan"]
    assert "six-within-site equal-weight" in plan["primary_aggregate_estimands"][0]
    assert "six-cross-site equal-weight" in plan["secondary_aggregate_estimands"][0]
    assert plan["task_comparisons"] == [
        "Full minus NoVariance", "Full minus NoDiscrepancy",
        "Full minus WeightedSVQR",
    ]
    assert "average squared error across seeds 0,1,2" in plan["paired_loss"]
    assert "Monday-Sunday" in plan["blocks"]
    assert plan["bootstrap"] == {
        "replicates": 10000, "confidence_level": .95,
        "rule": "sample whole blocks with equal probability and replacement; each replicate divides sampled loss-difference sums by sampled counts",
    }
    assert plan["sign_flip"]["replicates"] == 99999
    assert plan["sign_flip"]["sidedness"] == "two-sided"
    assert "18 within tests and 18 cross tests" in plan["multiplicity"]
    assert "not evidence of equivalence" in plan["interpretation"]
    assert "12 tasks x 4 variants x 3 seeds" in plan["complete_reporting"]


def test_monday_sunday_blocks_use_fixed_calendar_origin():
    times = pd.Series(["2020-01-05 23:00:00", "2020-01-06 00:00:00",
                       "2020-01-12 23:00:00", "2020-01-13 00:00:00"])
    sums, counts = analysis._block_summaries(times, np.array([1., 2., 3., 4.]))
    np.testing.assert_allclose(sums, [1., 5., 4.])
    np.testing.assert_allclose(counts, [1., 2., 1.])


def test_paired_block_inference_is_deterministic_and_uses_full_minus_removal():
    times = pd.Series(pd.date_range("2020-07-01", periods=21, freq="D").astype(str))
    differences = np.linspace(-4., -1., 21)
    first = analysis.paired_block_inference(
        differences, times, key="within_site1|Full-minus-NoVariance",
        random_seed=4, bootstrap_replicates=200, sign_flip_replicates=999)
    second = analysis.paired_block_inference(
        differences, times, key="within_site1|Full-minus-NoVariance",
        random_seed=4, bootstrap_replicates=200, sign_flip_replicates=999)
    assert first == second
    assert first["mean_squared_error_difference_mw2"] == pytest.approx(-2.5)
    assert first["blocks"] == 4
    assert 0 < first["p_unadjusted"] <= 1


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    raw = [.01, .04, .03, .20]
    adjusted = analysis._holm_adjust(raw)
    ordered = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[ordered]) >= 0)
    np.testing.assert_allclose(adjusted, [.04, .09, .09, .20])


def test_analysis_freeze_precedes_predictions_and_does_not_read_results(tmp_path,
                                                                        monkeypatch):
    matched_output = tmp_path / "matched"
    matched_output.mkdir()
    matched.selection.save_json(matched_output / "matched_lock.json", {
        "base_selection": {"output": str(tmp_path / "selection")}})
    expected = matched.expected_evaluation_keys(_configurations())
    verified = {
        "manifest": {"status": "matched_locked", "heldout_unlocked": False},
        "lock": {"protocol_sha256": "p", "configurations_sha256": "c",
                 "expected_evaluation_keys": expected},
        "matched_lock_sha256": "m",
    }
    calls = []
    monkeypatch.setattr(
        analysis.matched, "verify_matched_lock",
        lambda *args, **kwargs: calls.append((args, kwargs)) or verified)
    monkeypatch.setattr(analysis, "_source_hashes", lambda: {"analysis": "hash"})
    output = tmp_path / "analysis"
    locked = analysis.freeze(output, matched_output)
    assert len(calls) == 2  # initial freeze check and post-write lock verification
    assert all(call[1].get("require_base_preregistered") is True for call in calls)
    assert locked["manifest"]["heldout_results_read"] is False
    assert not (matched_output / "results.json").exists()
    assert not (output / "summary.json").exists()


def test_analysis_freeze_rejects_started_evaluation():
    assert analysis._evaluation_has_started(
        Path("/tmp/does-not-matter"), {"status": "evaluating", "heldout_unlocked": True})


def test_verify_results_requires_all_144_hashed_predictions(tmp_path):
    configurations = _configurations()
    task_by_name = {task["task"]: task for task in configurations["tasks"]}
    expected = matched.expected_evaluation_keys(configurations)
    matched_lock_sha = "m" * 64
    base_lock_sha = "b" * 64
    rows = []
    for key in expected:
        task, variant, seed_name = key.split("|")
        seed = int(seed_name.removeprefix("seed"))
        task_record = task_by_name[task]
        config = task_record["variants"][variant]
        folder = tmp_path / "evaluation" / task / variant / seed_name
        folder.mkdir(parents=True)
        fit_path = folder / "fit.json"
        model_path = folder / "model.npz"
        matched.selection.save_json(fit_path, {
            "task": task, "method": variant, "config": config,
            "seed": seed, "tau": matched.TAU,
            "candidate_id": "matched_ablation_v2_locked",
            "feature_map": task_record["shared_artifacts"][f"seed{seed}_feature_map"],
            "data_folder": str(Path(task_record["shared_artifacts"]["preprocessing"]).parent),
            "diagnostics": {"converged": True},
        })
        np.savez_compressed(model_path, w=np.array([0.]), b=np.array(0.))
        prediction = folder / "test_predictions.csv"
        frame = pd.DataFrame({
            "window_index": [0, 1], "raw_target_row": [100, 101],
            "time": ["2020-07-01", "2020-07-02"],
            "y_true": [1., 2.], "prediction": [1.1, 1.9],
        })
        frame.to_csv(prediction, index=False)
        point = selection.point_metrics(frame["y_true"].to_numpy(),
                                        frame["prediction"].to_numpy())
        row = {
            "evaluation_key": key, "task": task, "site": task_record["site"],
            "mode": task_record["mode"], "role": task_record["role"],
            "variant": variant, "seed": seed, "tau": matched.TAU,
            "config": config, "matched_lock_sha256": matched_lock_sha,
            "base_selection_lock_sha256": base_lock_sha,
            "shared_artifacts": task_record["shared_artifacts"],
            "feature_map": task_record["shared_artifacts"][f"seed{seed}_feature_map"],
            "data_folder": str(Path(task_record["shared_artifacts"]["preprocessing"]).parent),
            "fit_record": str(fit_path.relative_to(tmp_path)),
            "fit_record_sha256": digest(fit_path),
            "model_file": str(model_path.relative_to(tmp_path)),
            "model_sha256": digest(model_path),
            "counts": {"n_test": len(frame)}, "point": point,
            "prediction_file": str(prediction.relative_to(tmp_path)),
            "prediction_sha256": digest(prediction),
        }
        matched.selection.save_json(folder / "result.json", row)
        rows.append(row)
    matched.selection.save_json(tmp_path / "results.json", rows)
    matched.selection.save_json(tmp_path / "manifest.json", {
        "status": "complete", "completed_evaluation_keys": 144,
        "results_sha256": digest(tmp_path / "results.json"),
    })
    verified = {
        "lock": {"expected_evaluation_keys": expected},
        "configurations": configurations,
        "matched_lock_sha256": matched_lock_sha,
        "base": {"selection_lock_sha256": base_lock_sha},
    }
    checked, frames = analysis._verify_results(tmp_path, verified)
    assert len(checked) == len(frames) == 144
    damaged = tmp_path / rows[0]["prediction_file"]
    damaged.write_text(damaged.read_text() + "\n")
    with pytest.raises(ValueError, match="prediction artifact changed"):
        analysis._verify_results(tmp_path, verified)


def test_analysis_writes_csv_latex_and_json_for_complete_synthetic_results(tmp_path,
                                                                           monkeypatch):
    configurations = _configurations()
    rows, frames = [], {}
    for task in configurations["tasks"]:
        for variant_index, variant in enumerate(matched.VARIANTS):
            for seed in matched.SEEDS:
                key = f"{task['task']}|{variant}|seed{seed}"
                frame = pd.DataFrame({
                    "window_index": [0, 1, 2], "raw_target_row": [100, 101, 102],
                    "time": ["2020-07-01", "2020-07-08", "2020-07-15"],
                    "y_true": [1., 2., 3.],
                    "prediction": np.array([1., 2., 3.]) + .1 * variant_index + .01 * seed,
                })
                point = selection.point_metrics(frame["y_true"].to_numpy(),
                                                frame["prediction"].to_numpy())
                rows.append({"evaluation_key": key, "point": point})
                frames[key] = frame
    plan = json.loads(json.dumps(analysis.analysis_protocol()["analysis_plan"]))
    plan["bootstrap"]["replicates"] = 20
    plan["sign_flip"]["replicates"] = 39
    verified = {
        "protocol": {"analysis_plan": plan,
                     "output_files": analysis.analysis_protocol()["output_files"]},
        "lock": {"matched_lock_sha256": "m"},
        "matched": {"configurations": configurations},
    }
    monkeypatch.setattr(analysis, "verify_analysis_lock", lambda *args: verified)
    output, matched_output = tmp_path / "analysis", tmp_path / "matched"
    output.mkdir()
    matched_output.mkdir()
    (output / "analysis_lock.json").write_text("{}\n")
    matched.selection.save_json(output / "manifest.json", {
        "status": "analysis_locked", "heldout_results_read": False,
    })
    matched.selection.save_json(matched_output / "results.json", rows)

    def verify_after_access_marker(*args):
        access_manifest = json.loads((output / "manifest.json").read_text())
        assert access_manifest["status"] in ("analyzing", "complete")
        assert access_manifest["heldout_results_read"] is True
        return rows, frames

    monkeypatch.setattr(analysis, "_verify_results", verify_after_access_marker)
    summary = analysis.analyze(output, matched_output)
    assert summary["complete_evaluation_keys"] == 144
    assert summary["task_metric_rows"] == 48
    assert summary["paired_tests"] == {"within": 18, "cross": 18}
    for name in analysis.analysis_protocol()["output_files"]:
        assert (output / name).is_file()
    assert len(pd.read_csv(output / "aggregate_metrics.csv")) == 8
    assert len(pd.read_csv(output / "paired_inference.csv")) == 36
    frozen_bytes = {name: (output / name).read_bytes()
                    for name in analysis.analysis_protocol()["output_files"]}
    assert analysis.analyze(output, matched_output) == summary
    assert frozen_bytes == {name: (output / name).read_bytes()
                            for name in analysis.analysis_protocol()["output_files"]}


def test_cli_resume_requires_frozen_analysis_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis.sys, "argv", [
        "analyze_matched_ablation_v2.py", "--resume", "--output", str(tmp_path / "missing")])
    with pytest.raises(SystemExit) as exc:
        analysis.main()
    assert exc.value.code == 2
