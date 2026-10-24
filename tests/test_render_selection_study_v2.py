"""Tests for complete, presentation-only selection-v2 rendering."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

import review.render_selection_study_v2 as renderer


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _selection_inputs(root: Path) -> None:
    root.mkdir(parents=True)
    base = {
        "Full": 1.00,
        "NoVariance": 1.18,
        "NoDiscrepancy": .92,
        "WeightedSVQR": 1.31,
        "TargetSVQR": 1.62,
        "HistGBR": 1.08,
    }
    point_rows = []
    for scope_index, scope in enumerate(renderer.SCOPES):
        for site in range(1, 7):
            task = f"{scope}_site{site}"
            task_rows = []
            for method in renderer.METHODS:
                mean = renderer.CAPACITY[site] * .01 * base[method] * (
                    1 + .04 * (site - 1) + .08 * scope_index)
                seeds = [mean] if method == "HistGBR" else [mean * .98, mean, mean * 1.02]
                task_rows.append({
                    "scope": scope, "task": task, "site": site, "method": method,
                    "n_test": 100, "evaluation_seeds": len(seeds),
                    "seed_rmse_mw": seeds, "seed_mean_rmse_mw": float(np.mean(seeds)),
                    "capacity_mw": renderer.CAPACITY[site],
                    "seed_mean_capacity_normalized_rmse": float(np.mean(seeds)) /
                        renderer.CAPACITY[site],
                    "seed_mean_capacity_normalized_rmse_percent": 100 * float(np.mean(seeds)) /
                        renderer.CAPACITY[site],
                })
            ranks = pd.Series([row["seed_mean_rmse_mw"] for row in task_rows]).rank(
                method="average").to_numpy()
            for row, rank in zip(task_rows, ranks):
                row["task_rank"] = float(rank)
                point_rows.append(row)
    point = pd.DataFrame(point_rows)
    point.to_csv(root / "task_method_seed_mean_rmse.csv", index=False)

    scope_rows = []
    for scope_name in renderer.SCOPES:
        for method in renderer.METHODS:
            selected = point[(point.scope == scope_name) & (point.method == method)]
            scope_rows.append({
                "scope": scope_name,
                "role": "primary" if scope_name == "within" else "secondary_stress",
                "method": method, "tasks": 6,
                "site_equal_macro_capacity_normalized_rmse":
                    selected.seed_mean_capacity_normalized_rmse.mean(),
                "site_equal_macro_capacity_normalized_rmse_percent":
                    selected.seed_mean_capacity_normalized_rmse_percent.mean(),
                "average_task_rank": selected.task_rank.mean(),
                "task_wins_including_ties": int((selected.task_rank == 1).sum()),
            })
    pd.DataFrame(scope_rows).to_csv(root / "scope_summary.csv", index=False)

    inference = []
    for scope_name in renderer.SCOPES:
        for site in range(1, 7):
            for comparator_index, comparator in enumerate(renderer.COMPARATORS):
                difference = -.1 + .02 * comparator_index + .005 * site
                inference.append({
                    "scope": scope_name, "role": "primary", "holm_family": "test",
                    "task": f"{scope_name}_site{site}", "site": site,
                    "comparator": comparator, "n_test": 100,
                    "full_minus_comparator_mse_mw2": difference,
                    "time_block_bootstrap_95_ci_low_mw2": difference - .1,
                    "time_block_bootstrap_95_ci_high_mw2": difference + .1,
                    "block_sign_flip_p_holm_within_scope_30": .2,
                })
    pd.DataFrame(inference).to_csv(root / "full_pairwise_inference.csv", index=False)
    _write_json(root / "selection_v2_analysis.json", {
        "version": "synthetic",
        "reporting_counts": {"tasks": 12, "methods": 6,
                             "task_method_rows": 72, "full_comparisons": 60},
    })
    hashes = {name: renderer.digest(root / name) for name in renderer.SELECTION_FILES}
    _write_json(root / "analysis_manifest.json", {
        "status": "complete", "artifact_hashes": hashes,
    })


def _matched_inputs(root: Path) -> None:
    root.mkdir(parents=True)
    base = {"Full": 1.0, "NoVariance": 1.1,
            "NoDiscrepancy": 1.2, "WeightedSVQR": 1.35}
    task_rows = []
    for scope_index, scope in enumerate(renderer.SCOPES):
        for site in range(1, 7):
            rows = []
            for variant in renderer.VARIANTS:
                mean = renderer.CAPACITY[site] * .01 * base[variant] * (
                    1 + .03 * site + .08 * scope_index)
                seeds = [mean * .98, mean, mean * 1.02]
                rows.append({
                    "scope": scope, "task": f"{scope}_site{site}", "site": site,
                    "variant": variant, "capacity_mw": renderer.CAPACITY[site],
                    "seed0_rmse_mw": seeds[0], "seed1_rmse_mw": seeds[1],
                    "seed2_rmse_mw": seeds[2], "mean_seed_rmse_mw": np.mean(seeds),
                    "normalized_rmse": np.mean(seeds) / renderer.CAPACITY[site],
                })
            ranks = pd.Series([row["mean_seed_rmse_mw"] for row in rows]).rank(
                method="average").to_numpy()
            for row, rank in zip(rows, ranks):
                row["rank"] = float(rank)
                task_rows.append(row)
    task = pd.DataFrame(task_rows)
    task.to_csv(root / "task_metrics.csv", index=False)
    aggregate = (task.groupby(["scope", "variant"], sort=False)
                 .agg(sites=("site", "nunique"),
                      macro_normalized_rmse=("normalized_rmse", "mean"),
                      mean_rank=("rank", "mean")).reset_index())
    aggregate.to_csv(root / "aggregate_metrics.csv", index=False)

    inference = []
    for scope_name in renderer.SCOPES:
        for site in range(1, 7):
            for variant in renderer.ABLATION_COMPARATORS:
                inference.append({
                    "scope": scope_name, "task": f"{scope_name}_site{site}", "site": site,
                    "comparison": f"Full - {variant}", "n": 100, "blocks": 15,
                    "mean_squared_error_difference_mw2": -.1,
                    "ci_lower_mw2": -.2, "ci_upper_mw2": .05,
                    "p_unadjusted": .1, "p_holm": .3, "reject_holm_0_05": False,
                })
    pd.DataFrame(inference).to_csv(root / "paired_inference.csv", index=False)
    _write_json(root / "summary.json", {
        "version": "synthetic", "complete_evaluation_keys": 144,
        "task_metric_rows": 48, "paired_tests": {"within": 18, "cross": 18},
    })
    hashes = {name: renderer.digest(root / name) for name in renderer.MATCHED_FILES}
    _write_json(root / "manifest.json", {
        "status": "complete", "heldout_results_read": True, "output_hashes": hashes,
    })


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    selection, matched = tmp_path / "selection_analysis", tmp_path / "matched_analysis"
    _selection_inputs(selection)
    _matched_inputs(matched)
    # These are deliberately invalid.  A presentation-only renderer must never read them.
    (selection / "results.json").write_text("not json", encoding="utf-8")
    (matched / "results.json").write_text("not json", encoding="utf-8")
    return selection, matched


def test_render_complete_synthetic_design_with_fixed_academic_outputs(tmp_path):
    selection, matched = _inputs(tmp_path)
    output = tmp_path / "render"
    manifest = renderer.render(selection, matched, output)

    assert manifest["validated_counts"] == {
        "selection_task_method_rows": 72,
        "selection_scope_rows": 12,
        "selection_full_inference_rows": 60,
        "matched_task_variant_rows": 48,
        "matched_aggregate_rows": 8,
        "matched_inference_rows": 36,
    }
    expected = {*renderer.OUTPUT_FILES, "manifest.json"}
    assert {path.name for path in output.iterdir()} == expected
    assert (output / "within_site_normalized_rmse.pdf").read_bytes().startswith(b"%PDF")
    assert (output / "within_site_normalized_rmse.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n")
    for name in ("primary_within_rmse.tex", "scope_macro_summary.tex",
                 "full_paired_inference.tex", "matched_ablation.tex"):
        text = (output / name).read_text(encoding="utf-8")
        assert r"\small" in text
        assert "adjustbox" not in text.lower()
        assert not re.search(
            r"\\begin\{(?:table\*?|figure\*?)\}\[(?:!?[htbp]+)\]", text)
    primary = (output / "primary_within_rmse.tex").read_text(encoding="utf-8")
    assert r"\begin{table*}[pos=t,width=\FullWidth]" in primary
    assert primary.index("TL-QLDMR") < primary.index("No variance")
    assert all(f"\nFarm {site} &" in primary for site in range(1, 7))
    matched_table = (output / "matched_ablation.tex").read_text(encoding="utf-8")
    assert r"\begin{table}[pos=t]" in matched_table
    assert matched_table.count("\nWithin & Farm ") == 1
    assert matched_table.count("\nCross & Farm ") == 1
    assert all(matched_table.count(f" & Farm {site} &") == 2 for site in range(1, 7))
    recorded = json.loads((output / "manifest.json").read_text())
    for name in renderer.OUTPUT_FILES:
        assert recorded["outputs"][name] == renderer.digest(output / name)


def test_selection_completeness_is_checked_after_valid_hash(tmp_path):
    selection, matched = _inputs(tmp_path)
    path = selection / "task_method_seed_mean_rmse.csv"
    pd.read_csv(path).iloc[:-1].to_csv(path, index=False)
    manifest = json.loads((selection / "analysis_manifest.json").read_text())
    manifest["artifact_hashes"][path.name] = renderer.digest(path)
    _write_json(selection / "analysis_manifest.json", manifest)
    with pytest.raises(ValueError, match="incomplete or unexpected"):
        renderer.render(selection, matched, tmp_path / "render")


def test_input_hash_mismatch_and_unfinished_analysis_are_rejected(tmp_path):
    selection, matched = _inputs(tmp_path)
    (selection / "scope_summary.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.load_validated_inputs(selection, matched)

    selection, matched = _inputs(tmp_path / "second")
    manifest = json.loads((matched / "manifest.json").read_text())
    manifest["status"] = "analysis_locked"
    _write_json(matched / "manifest.json", manifest)
    with pytest.raises(ValueError, match="not complete"):
        renderer.load_validated_inputs(selection, matched)


def test_existing_output_is_never_overwritten(tmp_path):
    selection, matched = _inputs(tmp_path)
    output = tmp_path / "render"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        renderer.render(selection, matched, output)
    assert sentinel.read_text() == "keep"
