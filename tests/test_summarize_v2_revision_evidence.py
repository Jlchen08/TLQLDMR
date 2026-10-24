"""Tests for complete, analysis-only revision evidence summaries."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

import review.render_selection_study_v2 as renderer
import review.summarize_v2_revision_evidence as summary
from tests import test_render_selection_study_v2 as fixtures


FROZEN_INTERPRETATION = (
    "component effects are conditional on each task's locked Full configuration "
    "and are not independently retuned procedure comparisons; failure to reject "
    "is not evidence of equivalence"
)


def test_script_help_runs_without_external_pythonpath(tmp_path):
    result = subprocess.run(
        [sys.executable, str(Path(summary.__file__).resolve()), "--help"],
        cwd=tmp_path, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "revision-evidence digest" in result.stdout


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _refresh_selection_manifest(root: Path, *, protocol: dict | None = None) -> None:
    path = root / "analysis_manifest.json"
    manifest = json.loads(path.read_text())
    if protocol is not None:
        manifest["protocol"] = protocol
    names = (*renderer.SELECTION_FILES, *summary.SELECTION_EXTRA_FILES)
    manifest["artifact_hashes"] = {name: renderer.digest(root / name) for name in names}
    _write_json(path, manifest)


def _complete_inputs(tmp_path: Path) -> tuple[Path, Path]:
    selection, matched = fixtures._inputs(tmp_path)

    scope = pd.read_csv(selection / "scope_summary.csv")
    inference = pd.read_csv(selection / "full_pairwise_inference.csv")
    # Exercise complete preservation of favorable and adverse observed directions.
    positive = inference.index % 4 == 0
    inference.loc[positive, "full_minus_comparator_mse_mw2"] *= -1
    inference.loc[positive, "time_block_bootstrap_95_ci_low_mw2"] = -.02
    inference.loc[positive, "time_block_bootstrap_95_ci_high_mw2"] = .20
    inference.loc[inference.index % 3 == 0,
                  "block_sign_flip_p_holm_within_scope_30"] = .01
    inference["full_evaluation_seeds_averaged_within_timestamp"] = 3
    inference["comparator_evaluation_seeds_averaged_within_timestamp"] = 3
    inference.loc[inference["comparator"] == "HistGBR",
                  "comparator_evaluation_seeds_averaged_within_timestamp"] = 1
    inference["calendar_blocks"] = 15
    inference["block_size_min"] = 5
    inference["block_size_max"] = 7
    inference["full_minus_comparator_normalized_mse"] = (
        inference["full_minus_comparator_mse_mw2"] /
        inference["site"].map(renderer.CAPACITY).pow(2)
    )
    inference["block_sign_flip_p_unadjusted"] = .05
    inference["role"] = inference["scope"].map({
        "within": "primary", "cross": "secondary_stress",
    })
    inference["holm_family"] = inference["scope"] + "_30_full_comparisons"
    inference.to_csv(selection / "full_pairwise_inference.csv", index=False)

    pair_rows = []
    member_rows = []
    for scope_name in renderer.SCOPES:
        scope_rows = scope[scope.scope == scope_name].set_index("method")
        winner = min(renderer.METHODS, key=lambda method: (
            scope_rows.loc[method, "site_equal_macro_capacity_normalized_rmse_percent"],
            renderer.METHODS.index(method),
        ))
        macro = {
            method: float(scope_rows.loc[
                method, "site_equal_macro_capacity_normalized_rmse_percent"]) / 100
            for method in renderer.METHODS
        }
        for index, method_a in enumerate(renderer.METHODS):
            for method_b in renderer.METHODS[index + 1:]:
                pair_rows.append({
                    "scope": scope_name,
                    "holm_family": f"{scope_name}_15_all_method_pairs",
                    "method_a": method_a,
                    "method_b": method_b,
                    "macro_nrmse_a_minus_b": macro[method_a] - macro[method_b],
                    "block_swap_p_unadjusted_two_sided": .2,
                    "block_swap_p_holm_all_15": .6,
                })
        for method in renderer.METHODS:
            member_rows.append({
                "scope": scope_name,
                "method": method,
                "empirical_endpoint_winner": winner,
                "macro_capacity_normalized_rmse": macro[method],
                "difference_from_winner": macro[method] - macro[winner],
                "holm_p_vs_winner": 1.0 if method == winner else .6,
                "in_conservative_nonrejection_set": True,
                "interpretation": "membership is non-rejection of inferiority, not equivalence",
            })
    pairs = pd.DataFrame(pair_rows)
    members = pd.DataFrame(member_rows)
    pairs.to_csv(selection / "first_tier_all_pair_tests.csv", index=False)
    members.to_csv(selection / "conservative_first_tier.csv", index=False)

    report_path = selection / "selection_v2_analysis.json"
    report = json.loads(report_path.read_text())
    report.update({
        "version": "selection_v2_analysis_2",
        "reporting_counts": {
            **report["reporting_counts"],
            "evaluation_seeds_by_method": {
                **{method: [0, 1, 2] for method in renderer.METHODS[:-1]},
                "HistGBR": [0],
            },
            "locked_result_records": 192,
        },
        "primary_within_summary": scope[scope.scope == "within"].to_dict("records"),
        "secondary_cross_summary": scope[scope.scope == "cross"].to_dict("records"),
        "full_pairwise_inference": [{
            **row,
            "time_block_bootstrap_95_ci_mw2": [
                row.pop("time_block_bootstrap_95_ci_low_mw2"),
                row.pop("time_block_bootstrap_95_ci_high_mw2"),
            ],
        } for row in inference.to_dict("records")],
        "first_tier_all_pair_tests": pairs.to_dict("records"),
        "conservative_first_tier": members.to_dict("records"),
        "first_tier_caveat": "frozen caveat",
    })
    _write_json(report_path, report)
    _refresh_selection_manifest(selection, protocol={
        "first_tier": {"interpretation": "frozen caveat"},
    })

    matched_inference = pd.read_csv(matched / "paired_inference.csv")
    positive = matched_inference.index % 5 == 0
    matched_inference.loc[positive, "mean_squared_error_difference_mw2"] = .12
    matched_inference.loc[matched_inference.index % 4 == 0, "p_holm"] = .02
    matched_inference["reject_holm_0_05"] = matched_inference["p_holm"] <= .05
    matched_inference.to_csv(matched / "paired_inference.csv", index=False)
    aggregate = pd.read_csv(matched / "aggregate_metrics.csv")
    matched_report_path = matched / "summary.json"
    matched_report = json.loads(matched_report_path.read_text())
    matched_report.update({
        "version": "matched_ablation_v2_analysis_results_1",
        "aggregate_metrics": aggregate.to_dict("records"),
        "interpretation": FROZEN_INTERPRETATION,
    })
    _write_json(matched_report_path, matched_report)
    matched_manifest_path = matched / "manifest.json"
    matched_manifest = json.loads(matched_manifest_path.read_text())
    matched_manifest["output_hashes"] = {
        name: renderer.digest(matched / name) for name in renderer.MATCHED_FILES
    }
    _write_json(matched_manifest_path, matched_manifest)
    return selection, matched


def test_complete_summary_preserves_every_result_and_direction(tmp_path):
    selection, matched = _complete_inputs(tmp_path)
    output = tmp_path / "evidence"
    manifest = summary.summarize(selection, matched, output)

    assert set(path.name for path in output.iterdir()) == {
        "revision_evidence.json", "revision_evidence.md", "manifest.json",
    }
    evidence = json.loads((output / "revision_evidence.json").read_text())
    assert [len(item["methods_sorted_by_macro_nrmse"])
            for item in evidence["scope_method_results"]] == [6, 6]
    assert len(evidence["full_scope_relations"]) == 10
    assert len(evidence["full_task_comparisons"]) == 60
    assert len(evidence["matched_ablation"]["aggregate_results"]) == 8
    assert len(evidence["matched_ablation"]["all_task_comparisons"]) == 36
    assert {row["observed_direction"] for row in evidence["full_task_comparisons"]} == {
        "Full_lower_loss", "Full_higher_loss",
    }
    assert {row["significant_at_0_05"] for row in evidence["full_task_comparisons"]} == {
        True, False,
    }
    assert {row["observed_direction"]
            for row in evidence["matched_ablation"]["all_task_comparisons"]} == {
        "Full_lower_loss", "Full_higher_loss",
    }
    assert evidence["counts"]["configuration_queries_total"] == 3888
    assert evidence["counts"]["q1_selection_fit_records_total"] == 10368
    assert evidence["counts"]["locked_main_evaluation_records"] == 192
    assert evidence["counts"]["matched_evaluation_records"] == 144
    assert evidence["counts"]["heldout_evaluation_records_total"] == 336
    assert len(evidence["manuscript_ready_numeric_sentences"]) == 14
    markdown = (output / "revision_evidence.md").read_text()
    assert "All 60 Full task comparisons" in markdown
    assert "All 36 matched task comparisons" in markdown
    assert "Full_lower_loss" in markdown and "Full_higher_loss" in markdown
    for name in summary.OUTPUT_FILES:
        assert manifest["outputs"][name] == summary.digest(output / name)
    # Invalid held-out files created by the shared fixture prove they were not read.
    assert (selection / "results.json").read_text() == "not json"
    assert (matched / "results.json").read_text() == "not json"


def test_validly_rehashed_but_inconsistent_membership_is_rejected_atomically(tmp_path):
    selection, matched = _complete_inputs(tmp_path)
    path = selection / "conservative_first_tier.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "difference_from_winner"] = 99
    frame.to_csv(path, index=False)
    # Keep JSON in sync with the tampered CSV so the independent endpoint check must catch it.
    report_path = selection / "selection_v2_analysis.json"
    report = json.loads(report_path.read_text())
    report["conservative_first_tier"][0]["difference_from_winner"] = 99
    _write_json(report_path, report)
    _refresh_selection_manifest(selection)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="endpoint mismatch"):
        summary.summarize(selection, matched, output)
    assert not output.exists()


def test_incomplete_first_tier_family_is_rejected_even_with_fresh_hashes(tmp_path):
    selection, matched = _complete_inputs(tmp_path)
    path = selection / "first_tier_all_pair_tests.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    report_path = selection / "selection_v2_analysis.json"
    report = json.loads(report_path.read_text())
    report["first_tier_all_pair_tests"] = report["first_tier_all_pair_tests"][:-1]
    _write_json(report_path, report)
    _refresh_selection_manifest(selection)
    with pytest.raises(ValueError, match="complete 15-test families"):
        summary.summarize(selection, matched, tmp_path / "evidence")


def test_json_csv_inference_disagreement_is_rejected_with_valid_hash(tmp_path):
    selection, matched = _complete_inputs(tmp_path)
    report_path = selection / "selection_v2_analysis.json"
    report = json.loads(report_path.read_text())
    report["full_pairwise_inference"][0]["full_minus_comparator_mse_mw2"] = 999
    _write_json(report_path, report)
    _refresh_selection_manifest(selection)
    with pytest.raises(ValueError, match="JSON/CSV inference mismatch"):
        summary.summarize(selection, matched, tmp_path / "evidence")


def test_existing_output_is_not_overwritten(tmp_path):
    selection, matched = _complete_inputs(tmp_path)
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must not already exist"):
        summary.summarize(selection, matched, output)
    assert sentinel.read_text() == "keep"


def test_matched_conditional_interpretation_is_required(tmp_path):
    selection, matched = _complete_inputs(tmp_path)
    path = matched / "summary.json"
    value = json.loads(path.read_text())
    value["interpretation"] = "component effects"
    _write_json(path, value)
    manifest_path = matched / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_hashes"]["summary.json"] = renderer.digest(path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="conditional interpretation"):
        summary.summarize(selection, matched, tmp_path / "evidence")
