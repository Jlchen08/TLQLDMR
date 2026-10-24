"""Synthetic checks for the frozen selection-v2 analysis."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import review.analyze_selection_study_v2 as analysis


def synthetic_frames(days: int = 28) -> dict:
    frames = {}
    times = pd.date_range("2020-07-01", periods=days, freq="D")
    offsets = {
        "Full": 1.0,
        "NoVariance": 1.15,
        "NoDiscrepancy": .9,
        "WeightedSVQR": 1.3,
        "TargetSVQR": 1.7,
        "HistGBR": 1.5,
    }
    for tasks in analysis.SCOPES.values():
        for task in tasks:
            site = analysis._site_from_task(task)
            y = np.linspace(10, 20, days) + site
            for method in analysis.METHODS:
                seeds = (0,) if method == "HistGBR" else (0, 1, 2)
                for seed in seeds:
                    # The three kernel seeds have symmetric magnitudes around
                    # the method value. HistGBR is deterministic and has one.
                    error = offsets[method] * (1 + .02 * (seed - 1))
                    if method == "HistGBR":
                        error = offsets[method]
                    frames[(task, method, seed)] = pd.DataFrame({
                        "window_index": np.arange(days),
                        "time": times.astype(str),
                        "y_true": y,
                        "prediction": y + error,
                    })
    return frames


def test_holm_is_monotone_in_ordered_p_values():
    raw = np.array([.01, .04, .03, .8])
    adjusted = analysis.holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= 0)
    np.testing.assert_allclose(adjusted, [.04, .09, .09, .8])


def test_calendar_blocks_are_fixed_monday_to_sunday():
    times = pd.to_datetime(["2020-01-05", "2020-01-06", "2020-01-12",
                            "2020-01-13"])
    blocks = analysis._calendar_blocks(times)
    assert blocks[0] != blocks[1]
    assert blocks[1] == blocks[2]
    assert blocks[2] != blocks[3]


def _write_three_hist_stages(root, *, repeat_across_stages=False):
    task, method = "within_site1", "HistGBR"
    folder = root / task / method
    folder.mkdir(parents=True)
    for stage_index, stage in enumerate(("base", "middle", "joint")):
        offset = 0 if (repeat_across_stages and stage == "joint") else 18 * stage_index
        configs = [{
            "learning_rate": .01 + (offset + index) / 1000,
            "max_leaf_nodes": 7 + offset + index,
            "min_samples_leaf": 10,
            "l2_regularization": 0.,
            "max_iter": 100,
            "target_fraction": .5,
            "origin": f"{stage}-{index}",
        } for index in range(18)]
        record = {
            "task": task, "method": method, "stage": stage,
            "candidate_count": 18, "queried_effective_unique_count": 18,
            "candidates": [{"config": config} for config in configs],
        }
        (folder / f"{stage}_selection.json").write_text(json.dumps(record))
    return task, method


def test_effective_signatures_and_three_stage_54_query_audit(tmp_path):
    task, method = _write_three_hist_stages(tmp_path)
    assert len(analysis.validate_task_method_query_budget(tmp_path, task, method)) == 54

    target = {"gamma": .02, "total_loss": 100., "rank": 64,
              "target_fraction": .1, "lambda1": 20., "lambda2": 30.}
    changed_inactive = {**target, "target_fraction": .99,
                        "lambda1": 2., "lambda2": 3., "origin": "different"}
    assert (analysis.effective_model_signature("TargetSVQR", target) ==
            analysis.effective_model_signature("TargetSVQR", changed_inactive))
    assert (analysis.effective_model_signature("Full", target) !=
            analysis.effective_model_signature("Full", changed_inactive))


def test_three_stage_audit_rejects_cross_stage_effective_duplicate(tmp_path):
    task, method = _write_three_hist_stages(tmp_path, repeat_across_stages=True)
    with pytest.raises(ValueError, match="54 globally distinct"):
        analysis.validate_task_method_query_budget(tmp_path, task, method)


def _minimal_lock_metadata(reference, *, version="selection_v2_lock_2",
                           total_queries=12 * 6 * 54):
    provenance = reference / "data_provenance.json"
    provenance.write_text("{}")
    environment = {"synthetic": "1"}
    validation_budget = {"total_per_task_method": 54}
    protocol = {
        "methods": list(analysis.METHODS),
        "tasks": [{"name": task} for task in
                  analysis.SCOPES["within"] + analysis.SCOPES["cross"]],
        "evaluation_seeds_by_method": {
            method: ([0] if method == "HistGBR" else [0, 1, 2])
            for method in analysis.METHODS},
        "validation_query_budget": validation_budget,
        "environment_versions": environment,
    }
    protocol_path = reference / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    existing_relative = "review/analyze_selection_study_v2.py"
    existing_hash = analysis.digest(analysis.ROOT / existing_relative)
    frozen = {existing_relative: existing_hash}
    manifest = {
        "protocol_sha256": analysis.digest(protocol_path), "protocol": protocol,
        "source_hashes": frozen, "data_hashes": frozen,
        "environment_versions": environment,
    }
    (reference / "manifest.json").write_text(json.dumps(manifest))
    lock = {
        "version": version, "protocol_sha256": manifest["protocol_sha256"],
        "source_hashes": frozen, "data_hashes": frozen,
        "environment_versions": environment,
        "data_provenance_sha256": analysis.digest(provenance),
        "total_configuration_queries": total_queries,
        "validation_query_budget": validation_budget,
    }
    (reference / "selection_lock.json").write_text(json.dumps(lock))


def test_lock_guard_rejects_old_version_and_wrong_global_query_count(tmp_path):
    assert analysis.analysis_protocol()["required_selection_lock_version"] == (
        "selection_v2_lock_2")
    reference = tmp_path / "selection"
    reference.mkdir()
    _minimal_lock_metadata(reference, version="selection_v2_lock_1")
    with pytest.raises(ValueError, match="lock version"):
        analysis.validate_selection_lock(reference)
    _minimal_lock_metadata(reference, total_queries=12 * 6 * 54 - 1)
    with pytest.raises(ValueError, match="12 x 6 x 54"):
        analysis.validate_selection_lock(reference)


def test_point_summary_retains_all_tasks_methods_and_method_specific_seeds():
    frames = synthetic_frames()
    rows, summary = analysis.summarize_point_results(frames)
    assert len(rows) == 72 and len(summary) == 12
    assert {(row["scope"], row["task"], row["method"])
            for row in rows} == {
                (scope, task, method)
                for scope, tasks in analysis.SCOPES.items()
                for task in tasks for method in analysis.METHODS
            }
    full = next(row for row in rows
                if row["task"] == "within_site1" and row["method"] == "Full")
    hist = next(row for row in rows
                if row["task"] == "within_site1" and row["method"] == "HistGBR")
    assert full["evaluation_seeds"] == 3 and len(full["seed_rmse_mw"]) == 3
    assert hist["evaluation_seeds"] == 1 and hist["seed_rmse_mw"] == pytest.approx([1.5])
    assert full["seed_mean_rmse_mw"] == pytest.approx(1.0)
    assert full["seed_mean_capacity_normalized_rmse"] == pytest.approx(1 / 99)
    best = next(row for row in summary
                if row["scope"] == "within" and row["method"] == "NoDiscrepancy")
    assert best["average_task_rank"] == pytest.approx(1.)


def test_full_inference_has_separate_complete_holm_families():
    rows = analysis.full_pairwise_inference(
        synthetic_frames(), bootstrap_replicates=199,
        sign_flip_replicates=399, random_seed=17)
    assert len(rows) == 60
    for scope in analysis.SCOPES:
        selected = [row for row in rows if row["scope"] == scope]
        assert len(selected) == 30
        assert {row["holm_family"] for row in selected} == {
            f"{scope}_30_full_comparisons"}
        assert all(0 <= row["block_sign_flip_p_holm_within_scope_30"] <= 1
                   for row in selected)
    hist = next(row for row in rows
                if row["task"] == "within_site1" and row["comparator"] == "HistGBR")
    assert hist["full_evaluation_seeds_averaged_within_timestamp"] == 3
    assert hist["comparator_evaluation_seeds_averaged_within_timestamp"] == 1
    assert hist["full_minus_comparator_mse_mw2"] == pytest.approx(
        np.mean(np.square([.98, 1., 1.02])) - 2.25)


def test_conservative_tier_uses_all_pairs_and_does_not_claim_equivalence():
    frames = synthetic_frames()
    _, summary = analysis.summarize_point_results(frames)
    pairs, members = analysis.conservative_first_tier(
        frames, summary, sign_flip_replicates=399, random_seed=29)
    assert len(pairs) == 30 and len(members) == 12
    for scope in analysis.SCOPES:
        scoped_pairs = [row for row in pairs if row["scope"] == scope]
        assert len(scoped_pairs) == 15
        assert {row["holm_family"] for row in scoped_pairs} == {
            f"{scope}_15_all_method_pairs"}
        scoped_members = [row for row in members if row["scope"] == scope]
        assert {row["empirical_endpoint_winner"] for row in scoped_members} == {
            "NoDiscrepancy"}
        winner = next(row for row in scoped_members if row["method"] == "NoDiscrepancy")
        assert winner["in_conservative_nonrejection_set"] is True
        assert all("equivalence" in row["interpretation"]
                   for row in scoped_members if row["in_conservative_nonrejection_set"])


def test_invalid_lock_fails_before_any_prediction_csv_is_opened(tmp_path, monkeypatch):
    reference = tmp_path / "selection"
    reference.mkdir()
    (reference / "selection_lock.json").write_text(json.dumps({"version": "bad"}))
    (reference / "protocol.json").write_text("{}")
    (reference / "manifest.json").write_text("{}")
    (reference / "data_provenance.json").write_text("{}")
    (reference / "results.json").write_text("this is deliberately not JSON")

    def forbidden(*args, **kwargs):
        raise AssertionError("A held-out CSV was opened before lock validation.")

    monkeypatch.setattr(analysis.pd, "read_csv", forbidden)
    with pytest.raises(ValueError, match="lock version"):
        analysis.load_locked_predictions(reference)


def test_csv_and_latex_artifacts_cover_the_complete_synthetic_design(tmp_path):
    frames = synthetic_frames()
    point, scope = analysis.summarize_point_results(frames)
    inference = analysis.full_pairwise_inference(
        frames, bootstrap_replicates=39, sign_flip_replicates=79, random_seed=3)
    _, membership = analysis.conservative_first_tier(
        frames, scope, sign_flip_replicates=79, random_seed=5)
    analysis._write_csv(tmp_path / "point.csv", point)
    analysis._write_latex(tmp_path, point, scope, inference, membership)
    assert len(pd.read_csv(tmp_path / "point.csv")) == 72
    for name in ("task_seed_mean_rmse.tex", "scope_summary.tex",
                 "full_pairwise_inference.tex", "conservative_first_tier.tex"):
        text = (tmp_path / name).read_text()
        assert "\\begin" in text and "\\end" in text


def test_strict_json_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError):
        analysis.save_json(tmp_path / "bad.json", {"value": float("nan")})
