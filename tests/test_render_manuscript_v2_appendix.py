"""Synthetic-fixture tests for the selection-v2 manuscript appendix renderer."""
from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import pytest

import review.render_manuscript_v2_appendix as renderer


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fake_hash(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def _config(method: str, index: int, stage_index: int) -> dict:
    if method == "HistGBR":
        return {
            "learning_rate": .01 + .002 * (index + 18 * stage_index),
            "max_leaf_nodes": (7, 15, 31, 63)[index % 4],
            "min_samples_leaf": (5, 10, 20, 40, 80)[index % 5],
            "l2_regularization": (0.0, 0.1, 1.0)[stage_index],
            "max_iter": (100, 200, 400, 600)[index % 4],
            "target_fraction": .5,
            "origin": "synthetic",
        }
    return {
        "gamma": .001 * (1 + index + 20 * stage_index),
        "total_loss": 1000. + index + 100 * stage_index,
        "rank": ((32, 64, 128) if method == "TargetSVQR" else
                 (128, 256, 512))[index % 3],
        "target_fraction": 1. if method == "TargetSVQR" else .5,
        "lambda1": 0. if method in ("NoVariance", "WeightedSVQR", "TargetSVQR") else 1.,
        "lambda2": 0. if method in ("NoDiscrepancy", "WeightedSVQR", "TargetSVQR") else 1.,
        "origin": "synthetic",
    }


def _selection_fixture(root: Path, input_root: Path) -> tuple[dict, str]:
    root.mkdir(parents=True)
    for relative in renderer.SELECTION_SOURCE_FILES:
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic frozen source: {relative}\n", encoding="utf-8")
    for relative in renderer.SELECTION_DATA_FILES:
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic workbook: " + relative + "\n").encode())
    (input_root / "review/analyze_selection_study_v2.py").write_text(
        "# synthetic frozen analysis source\n", encoding="utf-8")
    chronological = input_root / "review/replied_comment_runs/chronological_v1/manifest.json"
    chronological.parent.mkdir(parents=True, exist_ok=True)
    chronological.write_text('{"fixture": true}\n', encoding="utf-8")
    environment = {"python": "synthetic"}
    seed_map = {method: ([0] if method == "HistGBR" else [0, 1, 2])
                for method in renderer.METHODS}
    protocol = {
        "version": "selection_v2", "methods": list(renderer.METHODS),
        "tasks": [{"name": task} for scope in renderer.SCOPES
                  for task in renderer.TASKS[scope]],
        "validation_query_budget": {"base": 18, "middle": 18, "joint": 18,
                                    "total_per_task_method": 54},
        "evaluation_seeds_by_method": seed_map,
        "environment_versions": environment,
    }
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "data_provenance.json", {"fixture": True})
    source_hashes = {name: renderer.digest(input_root / name)
                     for name in renderer.SELECTION_SOURCE_FILES}
    data_hashes = {name: renderer.digest(input_root / name)
                   for name in renderer.SELECTION_DATA_FILES}
    manifest = {
        "status": "complete", "last_stable_status": "complete", "protocol": protocol,
        "protocol_sha256": renderer.digest(root / "protocol.json"),
        "source_hashes": source_hashes, "data_hashes": data_hashes,
        "reference_manifest_sha256": renderer.digest(chronological),
        "environment_versions": environment,
    }
    _write_json(root / "manifest.json", manifest)

    file_hashes = {}
    final_records = []
    for task, method in renderer._expected_keys():
        stage_records = {}
        prior_winner = None
        for stage_index, stage in enumerate(renderer.STAGES):
            candidates = []
            for index in range(18):
                failed = [0] if method == "Full" and index == 0 else []
                candidate_rmse = 10. - 2 * stage_index + index
                seeds = []
                for seed in seed_map[method]:
                    if method == "HistGBR":
                        diagnostics = {"converged": None, "iterations": 100,
                                       "fit_seconds": .25}
                    else:
                        diagnostics = {
                            "converged": seed not in failed,
                            "iterations": 10,
                            "attempt_count": 1,
                            "attempt_summaries": [{"iterations": 10}],
                            "fit_seconds_including_feature_access": .5,
                        }
                    seeds.append({"seed": seed, "diagnostics": diagnostics,
                                  "metrics": {"rmse": candidate_rmse}})
                candidates.append({
                    "candidate_index": index, "candidate_id": f"{stage}_c{index:03d}",
                    "queried_stage": stage, "config": _config(method, index, stage_index),
                    "seed_results": seeds,
                    "mean_validation_rmse": candidate_rmse,
                    "eligible_for_selection": not failed,
                    "failed_certificate_seeds": failed,
                    "fit_folders": {
                        str(seed):
                            f"{task}/{method}/{stage}/candidates/c{index:03d}/seed{seed}"
                        for seed in seed_map[method]
                    },
                })
            winner = candidates[1 if method == "Full" else 0]
            record = {
                "task": task, "method": method, "stage": stage,
                "candidate_count": 18, "queried_effective_unique_count": 18,
                "carried_incumbent": prior_winner,
                "candidates": candidates, "selected_candidate": winner,
                "selected_config": winner["config"], "selected_source_stage": stage,
                "selected_candidate_index": winner["candidate_index"],
                "selected_candidate_id": winner["candidate_id"],
                "selected_mean_validation_rmse": winner["mean_validation_rmse"],
                "selected_fit_folders": winner["fit_folders"],
            }
            stage_records[stage] = record
            path = root / task / method / f"{stage}_selection.json"
            _write_json(path, record)
            file_hashes[str(path.relative_to(root))] = renderer.digest(path)
            prior_winner = winner
        middle = stage_records["middle"]
        middle_kind = (
            "active_raw_regularizer_search"
            if renderer.ACTIVE_REGULARIZERS.get(method) else
            ("regularizer_equivalent_kernel_base_refinement"
             if method in renderer.KERNEL_METHODS else
             "regularizer_equivalent_tree_refinement")
        )
        regularized = {
            **middle, "source_stage": "middle", "middle_kind": middle_kind,
            "base_selection_sha256": renderer.digest(
                root / task / method / "base_selection.json"),
        }
        path = root / task / method / "regularized_selection.json"
        _write_json(path, regularized)
        file_hashes[str(path.relative_to(root))] = renderer.digest(path)
        final = {
            **stage_records["joint"], "source_stage": "joint",
            "regularized_selection_sha256": renderer.digest(path),
        }
        path = root / task / method / "final_selection.json"
        _write_json(path, final)
        file_hashes[str(path.relative_to(root))] = renderer.digest(path)
        final_records.append(final)
    lock = {
        "version": "selection_v2_lock_2",
        "protocol_sha256": renderer.digest(root / "protocol.json"),
        "source_hashes": source_hashes, "data_hashes": data_hashes,
        "environment_versions": environment,
        "selection_files": file_hashes, "final_selections": final_records,
        "validation_query_budget": protocol["validation_query_budget"],
        "total_configuration_queries": 12 * 6 * 54,
        "data_provenance_sha256": renderer.digest(root / "data_provenance.json"),
        "heldout_policy": "no Q2 calibration or Q3-Q4 prediction was calculated before this lock",
    }
    _write_json(root / "selection_lock.json", lock)
    return protocol, renderer.digest(root / "selection_lock.json")


def _analysis_fixture(root: Path, selection: Path, protocol: dict, lock_hash: str,
                      input_root: Path) -> None:
    root.mkdir(parents=True)
    analysis_protocol = renderer._expected_analysis_protocol()
    _write_json(root / "analysis_protocol.json", analysis_protocol)
    rows = []
    for scope in renderer.SCOPES:
        for site, task in enumerate(renderer.TASKS[scope], start=1):
            for method_index, method in enumerate(renderer.METHODS):
                rows.append({"scope": scope, "task": task, "site": site,
                             "method": method,
                             "seed_mean_rmse_mw": 10 * site + method_index / 10})
    with (root / "task_method_seed_mean_rmse.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "version": "selection_v2_analysis_2",
        "selection_lock_sha256": lock_hash,
        "reporting_counts": {
            "tasks": 12, "methods": 6, "task_method_rows": 72,
            "locked_result_records": 192,
            "full_comparisons": 60,
            "evaluation_seeds_by_method": protocol["evaluation_seeds_by_method"],
        },
        "task_method_seed_mean_rmse": rows,
    }
    _write_json(root / "selection_v2_analysis.json", report)
    input_names = (
        {"selection_lock.json", "results.json"} |
        {f"{task}/{method}/evaluation/seed{seed}/{leaf}"
         for task, method in renderer._expected_keys()
         for seed in protocol["evaluation_seeds_by_method"][method]
         for leaf in ("result.json", "test_predictions.csv")})
    for name in input_names - {"selection_lock.json"}:
        path = selection / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("opaque heldout bytes: " + name + "\n").encode())
    inputs = {name: renderer.digest(selection / name) for name in input_names}
    manifest = {
        "status": "complete", "selection_root": str(selection.resolve()),
        "protocol": analysis_protocol,
        "protocol_sha256": renderer.digest(root / "analysis_protocol.json"),
        "analysis_source_sha256": renderer.digest(
            input_root / "review/analyze_selection_study_v2.py"),
        "reporting_counts": report["reporting_counts"],
        "selection_input_hashes": inputs,
        "artifact_hashes": {},
    }
    for name in renderer.ANALYSIS_ARTIFACT_FILES:
        path = root / name
        if not path.exists():
            path.write_text(f"synthetic artifact: {name}\n", encoding="utf-8")
        manifest["artifact_hashes"][name] = renderer.digest(path)
    _write_json(root / "analysis_manifest.json", manifest)
    # All held-out files are intentionally non-JSON opaque bytes. Hashing is
    # allowed, while attempting to parse them would fail.


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    input_root = tmp_path / "root"
    selection = tmp_path / "selection"
    analysis = tmp_path / "analysis"
    protocol, lock_hash = _selection_fixture(selection, input_root)
    _analysis_fixture(analysis, selection, protocol, lock_hash, input_root)
    return input_root, selection, analysis


def _refresh_selection_chain(selection: Path, analysis: Path, relative: str) -> None:
    """Re-sign a deliberately mutated selection record for semantic tests."""
    lock_path = selection / "selection_lock.json"
    lock = json.loads(lock_path.read_text())
    lock["selection_files"][relative] = renderer.digest(selection / relative)
    _write_json(lock_path, lock)
    lock_hash = renderer.digest(lock_path)
    report_path = analysis / "selection_v2_analysis.json"
    report = json.loads(report_path.read_text())
    report["selection_lock_sha256"] = lock_hash
    _write_json(report_path, report)
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_input_hashes"]["selection_lock.json"] = lock_hash
    manifest["artifact_hashes"]["selection_v2_analysis.json"] = renderer.digest(report_path)
    _write_json(manifest_path, manifest)


def test_complete_appendix_render_is_auditable_and_uses_fixed_academic_tables(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    output = tmp_path / "appendix"
    manifest = renderer.render(selection, analysis, output, input_root=input_root)

    assert manifest["validated_counts"] == {
        "locked_configs": 72, "secondary_cross_rmse_rows": 36,
        "optimization_scope_method_rows": 12, "queried_candidates": 3888,
    }
    assert {path.name for path in output.iterdir()} == {*renderer.OUTPUT_FILES, "manifest.json"}
    for name in (item for item in renderer.OUTPUT_FILES if item.endswith(".tex")):
        text = (output / name).read_text(encoding="utf-8")
        assert r"\small" in text and r"\setlength{\tabcolsep}" in text
        assert r"\toprule" in text and r"\bottomrule" in text
        assert "resizebox" not in text.lower() and "adjustbox" not in text.lower()
        assert r"\begin{table*}[pos=t,width=\FullWidth]" in text
        assert not re.search(
            r"\\begin\{(?:table\*?|figure\*?)\}\[(?:!?[htbp]+)\]", text)
        assert manifest["outputs"][name] == renderer.digest(output / name)
    configs = (output / renderer.OUTPUT_FILES[0]).read_text(encoding="utf-8")
    assert configs.count("TL-QLDMR") == 12
    assert configs.count("HistGBR") == 1  # one caption/header, rows omit repeated method name
    cross = (output / renderer.OUTPUT_FILES[1]).read_text(encoding="utf-8")
    assert "Farm 6 & 60.000 & 60.100 & 60.200 & 60.300 & 60.400 & 60.500" in cross
    optimization = (output / renderer.OUTPUT_FILES[2]).read_text(encoding="utf-8")
    optimization_csv = list(csv.DictReader(
        (output / "selection_v2_optimization_summary.csv").read_text(
            encoding="utf-8").splitlines()))
    assert manifest["outputs"]["selection_v2_optimization_summary.csv"] == \
        renderer.digest(output / "selection_v2_optimization_summary.csv")
    assert len(optimization_csv) == 12
    assert {(row["scope"], row["method"]) for row in optimization_csv} == {
        (scope, method) for scope in renderer.SCOPES for method in renderer.METHODS
    }
    assert all(int(row["total_queries"]) == 324 for row in optimization_csv)
    # Full has one failed candidate seed in each of 6 tasks x 3 stages per scope.
    assert "Within & TL-QLDMR & 108 & 108 & 108 & 306 & 18" in optimization
    assert len(manifest["inputs"]) == 777  # complete selection, analysis, frozen, renderer, and heldout byte graph


def test_incomplete_hash_chains_and_unfinished_analysis_are_rejected_without_publish(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection_input_hashes"].pop("results.json")
    _write_json(manifest_path, manifest)
    output = tmp_path / "appendix"
    with pytest.raises(ValueError, match="input hash chain"):
        renderer.render(selection, analysis, output, input_root=input_root)
    assert not output.exists() and not list(tmp_path.glob("appendix.tmp.*"))

    input_root, selection, analysis = _fixture(tmp_path / "second")
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "running"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="not complete"):
        renderer.render(selection, analysis, tmp_path / "other", input_root=input_root)


def test_lock_and_analysis_artifact_mutations_are_rejected(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    target = selection / "cross_site1" / "Full" / "base_selection.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.render(selection, analysis, tmp_path / "first", input_root=input_root)

    input_root, selection, analysis = _fixture(tmp_path / "second")
    csv_path = analysis / "task_method_seed_mean_rmse.csv"
    csv_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        renderer.render(selection, analysis, tmp_path / "second-out", input_root=input_root)


def test_existing_output_is_never_replaced(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    output = tmp_path / "appendix"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        renderer.render(selection, analysis, output, input_root=input_root)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_formal_analysis_protocol_source_and_heldout_bytes_are_all_verified(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    protocol_path = analysis / "analysis_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["version"] = "synthetic"
    _write_json(protocol_path, protocol)
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"] = protocol
    manifest["protocol_sha256"] = renderer.digest(protocol_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="formal analysis protocol"):
        renderer.render(selection, analysis, tmp_path / "bad-protocol", input_root=input_root)

    input_root, selection, analysis = _fixture(tmp_path / "source")
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["analysis_source_sha256"] = _fake_hash("wrong")
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="actual analysis script"):
        renderer.render(selection, analysis, tmp_path / "bad-source", input_root=input_root)

    input_root, selection, analysis = _fixture(tmp_path / "heldout")
    (selection / "within_site1/Full/evaluation/seed0/test_predictions.csv").write_bytes(
        b"changed opaque bytes\n")
    with pytest.raises(ValueError, match="input byte hash mismatch"):
        renderer.render(selection, analysis, tmp_path / "bad-heldout", input_root=input_root)


def test_csv_must_match_analysis_json_row_by_row(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    report_path = analysis / "selection_v2_analysis.json"
    report = json.loads(report_path.read_text())
    report["task_method_seed_mean_rmse"][0]["seed_mean_rmse_mw"] += .25
    _write_json(report_path, report)
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"][report_path.name] = renderer.digest(report_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="CSV/analysis JSON value differs"):
        renderer.render(selection, analysis, tmp_path / "mismatch", input_root=input_root)


def test_declared_twelve_significant_digit_csv_rounding_matches_json(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    precise = 22.012213166746022
    report_path = analysis / "selection_v2_analysis.json"
    report = json.loads(report_path.read_text())
    report["task_method_seed_mean_rmse"][0]["seed_mean_rmse_mw"] = precise
    _write_json(report_path, report)
    csv_path = analysis / "task_method_seed_mean_rmse.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    rows[0]["seed_mean_rmse_mw"] = format(precise, ".12g")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in (report_path, csv_path):
        manifest["artifact_hashes"][path.name] = renderer.digest(path)
    _write_json(manifest_path, manifest)
    renderer.render(selection, analysis, tmp_path / "rounded", input_root=input_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("stage_duplicate", "Stage effective signatures are not unique"),
        ("cross_stage_duplicate", "repeated across stages"),
        ("candidate_id", "Candidate ID/index/stage mismatch"),
        ("incumbent", "Broken carried-incumbent chain"),
        ("noninteger_index", "candidate index must be an integer"),
        ("noninteger_selected_index", "selected candidate index must be an integer"),
        ("noninteger_rank", "rank must be an integer"),
        ("noninteger_iterations", "ADMM attempt iterations must be an integer"),
        ("numeric_string", "is not numeric"),
        ("out_of_space", "outside the frozen search space"),
        ("mean_rmse", "mean validation RMSE disagrees"),
        ("fit_folder", "fit-folder chain is inconsistent"),
        ("regularized_extra", "Regularized wrapper differs"),
    ],
)
def test_query_semantics_and_integer_parameters_reject_signed_tampering(
        tmp_path, mutation, message):
    input_root, selection, analysis = _fixture(tmp_path)
    if mutation in {"stage_duplicate", "candidate_id", "noninteger_index",
                    "noninteger_selected_index", "noninteger_rank",
                    "noninteger_iterations", "numeric_string", "mean_rmse",
                    "out_of_space", "fit_folder"}:
        relative = "within_site1/NoVariance/base_selection.json"
    elif mutation == "cross_stage_duplicate":
        relative = "within_site1/NoVariance/middle_selection.json"
    elif mutation == "regularized_extra":
        relative = "within_site1/NoVariance/regularized_selection.json"
    else:
        relative = "within_site1/NoVariance/middle_selection.json"
    path = selection / relative
    record = json.loads(path.read_text())
    if mutation == "stage_duplicate":
        record["candidates"][1]["config"] = record["candidates"][0]["config"]
    elif mutation == "cross_stage_duplicate":
        base = json.loads((selection / "within_site1/NoVariance/base_selection.json").read_text())
        record["candidates"][1]["config"] = base["candidates"][1]["config"]
    elif mutation == "candidate_id":
        record["candidates"][2]["candidate_id"] = "base_c999"
    elif mutation == "incumbent":
        record["carried_incumbent"] = None
    elif mutation == "noninteger_index":
        record["candidates"][2]["candidate_index"] = 2.0
    elif mutation == "noninteger_selected_index":
        record["selected_candidate_index"] = 0.0
    elif mutation == "noninteger_rank":
        record["candidates"][2]["config"]["rank"] = 130.5
    elif mutation == "noninteger_iterations":
        record["candidates"][2]["seed_results"][0]["diagnostics"][
            "attempt_summaries"][0]["iterations"] = 10.0
    elif mutation == "numeric_string":
        record["candidates"][17]["config"]["gamma"] = "0.123"
    elif mutation == "out_of_space":
        record["candidates"][17]["config"]["gamma"] = -1.0
    elif mutation == "mean_rmse":
        record["candidates"][17]["mean_validation_rmse"] += .5
    elif mutation == "fit_folder":
        record["candidates"][17]["fit_folders"]["0"] = "redirected/fit"
    elif mutation == "regularized_extra":
        record["unfrozen_extra"] = True
    _write_json(path, record)
    _refresh_selection_chain(selection, analysis, relative)
    with pytest.raises(ValueError, match=message):
        renderer.render(selection, analysis, tmp_path / "bad-query", input_root=input_root)


def test_exact_frozen_and_analysis_artifact_inventories_are_required(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    lock_path = selection / "selection_lock.json"
    selection_manifest_path = selection / "manifest.json"
    lock = json.loads(lock_path.read_text())
    selection_manifest = json.loads(selection_manifest_path.read_text())
    omitted = renderer.SELECTION_SOURCE_FILES[-1]
    lock["source_hashes"].pop(omitted)
    selection_manifest["source_hashes"].pop(omitted)
    _write_json(lock_path, lock)
    _write_json(selection_manifest_path, selection_manifest)
    lock_hash = renderer.digest(lock_path)
    report_path = analysis / "selection_v2_analysis.json"
    report = json.loads(report_path.read_text())
    report["selection_lock_sha256"] = lock_hash
    _write_json(report_path, report)
    analysis_manifest_path = analysis / "analysis_manifest.json"
    analysis_manifest = json.loads(analysis_manifest_path.read_text())
    analysis_manifest["selection_input_hashes"]["selection_lock.json"] = lock_hash
    analysis_manifest["artifact_hashes"][report_path.name] = renderer.digest(report_path)
    _write_json(analysis_manifest_path, analysis_manifest)
    with pytest.raises(ValueError, match="source_hashes inventory"):
        renderer.render(selection, analysis, tmp_path / "bad-frozen", input_root=input_root)

    input_root, selection, analysis = _fixture(tmp_path / "artifacts")
    analysis_manifest_path = analysis / "analysis_manifest.json"
    analysis_manifest = json.loads(analysis_manifest_path.read_text())
    analysis_manifest["artifact_hashes"].pop("scope_summary.csv")
    _write_json(analysis_manifest_path, analysis_manifest)
    with pytest.raises(ValueError, match="artifact hash inventory"):
        renderer.render(selection, analysis, tmp_path / "bad-artifacts", input_root=input_root)


def test_duplicate_csv_headers_are_rejected_even_when_resigned(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    csv_path = analysis / "task_method_seed_mean_rmse.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0]) + ["site"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in rows:
            writer.writerow([row[name] for name in fields])
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifact_hashes"][csv_path.name] = renderer.digest(csv_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="lacks a header"):
        renderer.render(selection, analysis, tmp_path / "duplicate-header", input_root=input_root)


def test_latex_escape_is_single_pass_and_complete():
    assert renderer._escape(r"a_b&c\{}%#$~^") == (
        r"a\_b\&c\textbackslash{}\{\}\%\#\$\textasciitilde{}\textasciicircum{}")


def test_publish_revalidates_all_inputs_and_detects_mid_render_change(tmp_path, monkeypatch):
    input_root, selection, analysis = _fixture(tmp_path)
    original = renderer._load_validated
    calls = 0

    def changing_load(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            (input_root / renderer.SELECTION_DATA_FILES[0]).write_text(
                "changed\n", encoding="utf-8")
        return result

    monkeypatch.setattr(renderer, "_load_validated", changing_load)
    output = tmp_path / "toctou"
    with pytest.raises(ValueError, match="data_hashes hash mismatch"):
        renderer.render(selection, analysis, output, input_root=input_root)
    assert not output.exists()
    assert not list(tmp_path.glob(".toctou.tmp-*"))


def test_final_snapshot_closes_the_post_validation_toctou_window(tmp_path, monkeypatch):
    input_root, selection, analysis = _fixture(tmp_path)
    original = renderer._input_snapshot
    calls = 0

    def changing_snapshot(*args, **kwargs):
        nonlocal calls
        snapshot = original(*args, **kwargs)
        calls += 1
        # The second snapshot is made by the second complete semantic
        # validation. Mutate immediately after it returns; the independent
        # pre-publication snapshot must still catch the change.
        if calls == 2:
            (analysis / "scope_summary.csv").write_text(
                "changed after second validation\n", encoding="utf-8")
        return snapshot

    monkeypatch.setattr(renderer, "_input_snapshot", changing_snapshot)
    output = tmp_path / "late-toctou"
    with pytest.raises(RuntimeError, match="immediately before appendix publication"):
        renderer.render(selection, analysis, output, input_root=input_root)
    assert not output.exists()
    assert not list(tmp_path.glob(".late-toctou.tmp-*"))


def test_concurrent_publishers_cannot_replace_the_same_target(tmp_path):
    input_root, selection, analysis = _fixture(tmp_path)
    output = tmp_path / "concurrent"

    def invoke():
        try:
            renderer.render(selection, analysis, output, input_root=input_root)
            return "published"
        except FileExistsError:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: invoke(), range(2)))
    assert sorted(outcomes) == ["published", "refused"]
    assert (output / "manifest.json").is_file()
    assert not list(tmp_path.glob(".concurrent.tmp-*"))
