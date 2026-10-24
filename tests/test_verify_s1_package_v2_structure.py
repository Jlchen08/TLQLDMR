"""End-to-end synthetic check of S1's nested v2 integrity audit."""
from __future__ import annotations

import hashlib
import csv
import gzip
import io
import json
from pathlib import Path

import review.build_reproducibility_supplement as builder
import review.verify_s1_package as verifier


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _file(path: Path, value: str = "record\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _manifest(root: Path) -> None:
    path = root / "MANIFEST.sha256"
    lines = [f"{_digest(item)}  {item.stat().st_size:12d}  {item.relative_to(root)}"
             for item in sorted(root.rglob("*")) if item.is_file() and item != path]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_split_assignments(root: Path) -> list[dict]:
    timestamps = (
        ("2019-07-01T00:00:00", "train"),
        ("2020-02-01T00:00:00", "validation"),
        ("2020-05-01T00:00:00", "calibration"),
        ("2020-08-01T00:00:00", "test"),
    )
    records = []
    for site in range(1, 7):
        path = root / "split_assignments" / f"farm{site}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    writer = csv.writer(text)
                    writer.writerow(verifier.SPLIT_ASSIGNMENT_HEADER)
                    for index, (timestamp, split) in enumerate(timestamps):
                        target = index % 2
                        writer.writerow((index, 24 + index, timestamp, split,
                                         "target" if target else "source", 1.0,
                                         target, 0, 0, 0))
        records.append({"site": site, "rows": len(timestamps),
                        "file": f"split_assignments/farm{site}.csv.gz",
                        "sha256": _digest(path)})
    return records


def _candidate(method: str, value: int) -> dict:
    if method == "HistGBR":
        return {"learning_rate": value / 1000, "max_leaf_nodes": 7,
                "min_samples_leaf": 10, "l2_regularization": 0.0,
                "max_iter": 100, "target_fraction": 0.5}
    result = {"gamma": value / 1000, "total_loss": 100.0, "rank": 1,
              "target_fraction": 0.5, "lambda1": 1.0, "lambda2": 1.0}
    if method == "NoVariance":
        result["lambda1"] = 0.0
    elif method == "NoDiscrepancy":
        result["lambda2"] = 0.0
    elif method in ("WeightedSVQR", "TargetSVQR"):
        result["lambda1"] = result["lambda2"] = 0.0
    return result


def _selection_fit_record(task: str, method: str, config: dict, seed: int,
                          tau: float, candidate_id: str,
                          diagnostics: dict) -> dict:
    data_subfolder = "target_data" if method == "TargetSVQR" else "pooled_data"
    return {
        "task": task,
        "site": int(task.rsplit("site", 1)[1]),
        "method": method,
        "candidate_id": candidate_id,
        "config": config,
        "seed": seed,
        "tau": tau,
        "feature_map": (None if method == "HistGBR" else
                        f"{task}/{data_subfolder}/landmarks/seed{seed}/rank1/"
                        f"gamma_{float(config.get('gamma', 0.001)):.17g}/map.npz"),
        "data_folder": f"{task}/{data_subfolder}",
        "diagnostics": diagnostics,
        "selection_counts": {
            "n_source_fit": 0 if method == "TargetSVQR" else 1,
            "n_target_fit": 1, "n_validation": 4,
            "fit_period": "2019", "validation_period": "2020 Q1",
        },
        "effective_target_fraction": (
            1.0 if method == "TargetSVQR" else
            float(config.get("target_fraction", 1.0))
        ),
    }


def _selection_candidate(selection: Path, task: str, method: str, stage: str,
                         index: int, value: int, seeds: list[int]) -> dict:
    rmse = float(value)
    mae = rmse / 2.0
    config = _candidate(method, value)
    candidate_id = f"{stage}_c{index:03d}"
    seed_results = []
    fit_folders = {}
    for seed in seeds:
        diagnostics = {
            "solver": ("histogram gradient boosting" if method == "HistGBR" else "ADMM"),
            "converged": None if method == "HistGBR" else True,
        }
        metrics = {"n": 4, "rmse": rmse, "mae": mae, "r2": 0.1}
        relative = f"{task}/{method}/{stage}/candidates/c{index:03d}/seed{seed}"
        fit_folders[str(seed)] = relative
        _json(selection / relative / "fit.json", _selection_fit_record(
            task, method, config, seed, 0.5, candidate_id, diagnostics))
        _json(selection / relative / "validation_metrics.json", metrics)
        seed_results.append({
            "seed": seed, "metrics": metrics, "diagnostics": diagnostics,
        })
    return {
        "candidate_index": index,
        "candidate_id": candidate_id,
        "queried_stage": stage,
        "config": config,
        "seed_results": seed_results,
        "mean_validation_rmse": rmse,
        "std_validation_rmse": 0.0,
        "mean_validation_mae": mae,
        "eligible_for_selection": True,
        "failed_certificate_seeds": [],
        "fit_folders": fit_folders,
    }


def _add_chronological(root: Path) -> Path:
    protocol = {
        "version": 1,
        "seeds": [0, 1, 2],
        "components": ["Full", "NoVariance", "NoMMD", "TargetOnly"],
        "label_budgets": [16, 32, 64, 128, 256, 512, "all"],
        "weight_schemes": ["domain_balanced", "sample_equal", "historical_ratio", "target_10x"],
        "threshold_quantiles": [95, 98, 99],
        "landmark_grid": [32, 64, 128, 256, 512, 1024],
        "model": {"landmarks": 256},
        "source_fractions": [0.1, 0.25, 0.5, 1.0],
        "cross_farm_sites": [1, 2, 3, 4, 5, 6],
        "point_tau": 0.5,
        "interval_taus": [0.025, 0.975],
    }
    _json(root / "review/replied_study_protocol.json", protocol)
    base = root / verifier.CHRONOLOGICAL
    jobs, aliases = builder._planned_chronological(protocol)
    assert (len(jobs), len(aliases)) == (94, 6)
    _file(base / "summary.csv", "name,value\nsynthetic,1\n")
    _json(base / "threshold_definitions.json", [
        {"q": q, "cutout_ms": 25, "identical_domain_mask_to_25": True}
        for q in protocol["threshold_quantiles"]
    ])
    for site in range(1, 7):
        _json(base / "data" / f"farm{site}_audit.json", {"site": site})
        path = base / "data" / f"farm{site}_ids.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"farm{site}-ids".encode())
    for name in jobs:
        folder = base / name
        _json(folder / "result.json", {"name": name})
        _file(folder / "predictions.csv", "row,prediction\n0,0\n")
        _json(folder / "preprocessing.json", {"name": name})
        (folder / "fit_ids.npz").write_bytes((name + "-ids").encode())
        _file(folder / "training.log", "synthetic log\n")
        if "interval" in name:
            _file(folder / "calibration.csv", "row,score\n0,0\n")
            taus = protocol["interval_taus"]
        else:
            taus = [protocol["point_tau"]]
        for tau in taus:
            (folder / f"model_tau_{tau}.npz").write_bytes(
                f"{name}-{tau}-model".encode())
    for name, target in aliases.items():
        _json(base / name / "alias.json", {"name": name, "target": target})
    _json(base / "verification.json", {
        "status": "passed", "pending": [],
        "branches": [{"name": name, "status": "passed"} for name in jobs],
        "aliases": [{"name": name, "status": "passed"} for name in aliases],
        "global_checks": [{"name": "synthetic", "status": "passed"}],
    })
    data_hashes = {}
    for site in range(1, 7):
        source = root / "data" / f"chronological_farm{site}.xlsx"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"farm{site}-public-data".encode())
        data_hashes[str(source.relative_to(root))] = _digest(source)
    source_hashes = {
        relative: _digest(root / relative) for relative in (
            "review/run_replied_comment_study.py",
            "review/temporal_data.py",
        )
    }
    _json(base / "manifest.json", {
        "status": "complete", "protocol": protocol, "completed_fits": 94,
        "source_hashes": source_hashes, "data_hashes": data_hashes,
    })
    return base / "manifest.json"


def _add_probability(root: Path) -> None:
    v1 = {
        "version": "selection_v1",
        "methods": list(verifier.METHODS),
        "candidate_count_per_method": 16,
        "tasks": ([{"name": "within_site6", "site": 6, "mode": "within"}] +
                  [{"name": f"cross_site{site}", "site": site, "mode": "cross"}
                   for site in range(1, 7)]),
    }
    v1_path = root / "review/selection_runs/selection_v1/protocol.json"
    _json(v1_path, v1)
    _file(root / "review/run_selection_study.py",
          "def protocol():\n    return " + repr(v1) + "\n")

    base = root / verifier.PROBABILITY
    protocol = {
        "version": "probabilistic_selection_v2", "candidate_budget": 16,
        "kernel_seeds": [0, 1, 2], "tree_seeds": [0],
    }
    _json(base / "protocol.json", protocol)
    selections = []
    selection_files = {}
    for method in verifier.METHODS:
        row = {"task": "within_site6", "site": 6, "method": method,
               "selected_index": 0, "candidates": []}
        relative = f"within_site6/{method}/selection.json"
        _json(base / relative, row)
        selections.append(row)
        selection_files[relative] = _digest(base / relative)
        for index in range(16):
            _file(base / "within_site6" / method / "candidates" / f"c{index:02d}" /
                  "validation_quantiles.csv", "window_index,q500\n0,0\n")
    _json(base / "selection_lock.json", {
        "protocol_sha256": _digest(base / "protocol.json"),
        "selections": selections, "selection_files": selection_files,
    })
    results = []
    for method in verifier.METHODS:
        for seed in verifier.PROBABILITY_METHOD_SEEDS[method]:
            folder = base / "within_site6" / method / f"final_seed{seed}"
            row = {"task": "within_site6", "site": 6, "method": method,
                   "seed": seed, "selected_index": 0}
            _json(folder / "result.json", row)
            _file(folder / "calibration_quantiles.csv", "window_index,q500\n0,0\n")
            _file(folder / "test_quantiles.csv", "window_index,q500\n0,0\n")
            results.append(row)
    _json(base / "results.json", results)
    selection_sources = {
        relative: _digest(root / relative) for relative in (
            "review/run_probabilistic_selection_study.py",
            "review/run_selection_study.py",
        )
    }
    _json(base / "manifest.json", {
        "status": "complete", "protocol": protocol,
        "selection_records": 6, "probabilistic_records": 16,
        "base_candidate_protocol_sha256": _digest(v1_path),
        "source_hashes": selection_sources,
    })

    analysis = root / verifier.PROBABILITY_ANALYSIS
    analysis_protocol = {"version": "probabilistic_analysis_v2"}
    _json(analysis / "protocol.json", analysis_protocol)
    _json(analysis / "aggregate_results.json", [{"method": method}
                                                  for method in verifier.METHODS])
    _json(analysis / "paired_probability_inference.json", {"comparisons": 5})
    for name in verifier.PROBABILITY_ANALYSIS_FILES:
        if name in {"protocol.json", "manifest.json", "aggregate_results.json",
                    "paired_probability_inference.json", "posthoc_integrity_inventory.json"}:
            continue
        _file(analysis / name, "synthetic probability output\n")
    input_hashes = {
        str(verifier.PROBABILITY / name): _digest(base / name)
        for name in ("manifest.json", "selection_lock.json", "results.json")
    }
    for method in verifier.METHODS:
        for seed in verifier.PROBABILITY_METHOD_SEEDS[method]:
            relative = (verifier.PROBABILITY / "within_site6" / method /
                        f"final_seed{seed}" / "test_quantiles.csv")
            input_hashes[str(relative)] = _digest(root / relative)
    _json(analysis / "manifest.json", {
        "status": "complete", "protocol": analysis_protocol,
        "experiment_protocol_sha256": _digest(base / "protocol.json"),
        "experiment_runner_sha256": _digest(
            root / "review/run_probabilistic_selection_study.py"),
        "methods": 6, "paired_comparisons": 5,
        "source_hashes": {
            "review/run_probabilistic_analysis_v2.py": _digest(
                root / "review/run_probabilistic_analysis_v2.py")},
        "input_hashes": input_hashes,
    })
    validation, calibration, test = builder._probability_csv_paths()
    controls = {
        verifier.PROBABILITY / name for name in
        ("protocol.json", "manifest.json", "selection_lock.json", "results.json")
    } | {
        verifier.PROBABILITY_ANALYSIS / name for name in
        ("protocol.json", "manifest.json", "aggregate_results.json",
         "paired_probability_inference.json")
    } | {Path(name) for name in (
        "review/run_probabilistic_selection_study.py",
        "review/run_probabilistic_analysis_v2.py",
        "review/render_probabilistic_figures.py",
        "review/create_probabilistic_integrity_inventory.py",
    )}
    inventory_paths = controls | {
        verifier.PROBABILITY / path for path in validation | calibration | test
    }
    _json(analysis / "posthoc_integrity_inventory.json", {
        "role": "post-completion integrity inventory; not preregistration",
        "counts": {"candidate_validation_csv": 96,
                   "selected_calibration_csv": 16, "selected_test_csv": 16},
        "sha256": {str(path): _digest(root / path)
                   for path in sorted(inventory_paths, key=str)},
    })

    presentation = root / verifier.PROBABILITY_PRESENTATION
    output_names = verifier.PROBABILITY_PRESENTATION_FILES - {"manifest.json"}
    for name in output_names:
        _file(presentation / name, "synthetic probability figure\n")
    _json(presentation / "manifest.json", {
        "input": str(verifier.PROBABILITY_ANALYSIS / "aggregate_results.json"),
        "input_sha256": _digest(analysis / "aggregate_results.json"),
        "source": "review/render_probabilistic_figures.py",
        "source_sha256": _digest(root / "review/render_probabilistic_figures.py"),
        "outputs": {name: _digest(presentation / name) for name in output_names},
    })


def _complete_package(root: Path, *, retain_source_fit_records: bool = False) -> Path:
    sources = {
        "review/run_selection_study.py": "probability v1 dependency\n",
        "review/run_probabilistic_selection_study.py": "probability source\n",
        "review/run_selection_study_v2.py": "selection source\n",
        "review/analyze_selection_study_v2.py": "selection analysis source\n",
        "review/run_matched_ablation_v2.py": "matched source\n",
        "review/analyze_matched_ablation_v2.py": "matched analysis source\n",
        "review/render_selection_study_v2.py": "render source\n",
    }
    for relative in builder.CODE_FILES:
        sources.setdefault(relative, f"synthetic source: {relative}\n")
    for relative, text in sources.items():
        _file(root / relative, text)
    _file(root / "TLQLDMR_R1/event_composition.tex", "synthetic event composition\n")
    _json(root / "review/extreme_event_summary.json", {"synthetic": True})

    selection = root / verifier.SELECTION
    protocol = {
        "version": "selection_v2", "methods": list(verifier.METHODS),
        "tasks": [{
            "name": task,
            "site": int(task.rsplit("site", 1)[1]),
            "mode": "within" if task.startswith("within") else "cross",
            "role": "primary" if task.startswith("within") else "stress",
        } for task in verifier.TASKS],
        "evaluation_seeds_by_method": {
            method: ([0] if method == "HistGBR" else [0, 1, 2])
            for method in verifier.METHODS
        },
        "selection_seeds_by_method": {
            method: ([0] if method == "HistGBR" else [0, 1, 2])
            for method in verifier.METHODS
        },
        "taus": [0.5, 0.025, 0.975],
        "validation_query_budget": {"total_per_task_method": 54},
    }
    _json(selection / "protocol.json", protocol)
    _json(selection / "data_provenance.json", {"version": "synthetic"})
    selection_files = {}
    final = []
    for task in verifier.TASKS:
        for method in verifier.METHODS:
            selected = (_candidate("Full", 1) if method == "Full" else {"rank": 1})
            final_row = {"task": task, "method": method, "selected_config": selected}
            for stage in verifier.STAGES:
                relative = f"{task}/{method}/{stage}_selection.json"
                record = (dict(final_row) if stage == "final" else
                          {"task": task, "method": method, "stage": stage})
                if stage in ("base", "middle", "joint"):
                    query_stage = ("base", "middle", "joint").index(stage)
                    seeds = protocol["selection_seeds_by_method"][method]
                    record.update({
                        "candidate_count": 18, "queried_effective_unique_count": 18,
                        "candidates": [
                            _selection_candidate(
                                selection, task, method, stage, i,
                                query_stage * 18 + i + 1, seeds)
                            for i in range(18)
                        ],
                    })
                _json(selection / relative, record)
                selection_files[relative] = _digest(selection / relative)
            final.append(final_row)
    selection_source_hashes = {
        "review/run_selection_study_v2.py": _digest(root / "review/run_selection_study_v2.py")
    }
    environment = {"python": "synthetic"}
    workbook = root / "data/public-workbook.xlsx"
    workbook.parent.mkdir(parents=True, exist_ok=True)
    workbook.write_bytes(b"synthetic public workbook")
    data_hashes = {"data/public-workbook.xlsx": _digest(workbook)}
    lock = {
        "version": "selection_v2_lock_2",
        "protocol_sha256": _digest(selection / "protocol.json"),
        "source_hashes": selection_source_hashes, "data_hashes": data_hashes,
        "environment_versions": environment, "selection_files": selection_files,
        "final_selections": final,
        "validation_query_budget": protocol["validation_query_budget"],
        "total_configuration_queries": 3888,
        "data_provenance_sha256": _digest(selection / "data_provenance.json"),
    }
    _json(selection / "selection_lock.json", lock)
    selection_lock_hash = _digest(selection / "selection_lock.json")
    selection_results = []
    selection_input_hashes = {
        "selection_lock.json": selection_lock_hash,
    }
    for task in verifier.TASKS:
        for method in verifier.METHODS:
            selected_config = next(row["selected_config"] for row in final
                                   if row["task"] == task and row["method"] == method)
            for seed in protocol["evaluation_seeds_by_method"][method]:
                folder = selection / task / method / "evaluation" / f"seed{seed}"
                fits = []
                for tau in protocol["taus"]:
                    relative = (f"{task}/{method}/evaluation/seed{seed}/tau_" +
                                format(float(tau), ".17g"))
                    diagnostics = {
                        "solver": ("histogram gradient boosting"
                                   if method == "HistGBR" else "ADMM"),
                        "converged": None if method == "HistGBR" else True,
                    }
                    fit_record = _selection_fit_record(
                        task, method, selected_config, seed, tau,
                        "locked_final", diagnostics)
                    _json(selection / relative / "fit.json", fit_record)
                    fits.append({"path": relative, **fit_record})
                row = {"task": task, "method": method, "seed": seed,
                       "selected_config": selected_config,
                       "selection_lock_sha256": selection_lock_hash,
                       "counts": {"n_calibration": 1},
                       "fits": fits}
                _json(folder / "result.json", row)
                _file(folder / "test_predictions.csv", "window_index,prediction\n0,0\n")
                _file(folder / "calibration_predictions.csv",
                      "window_index,raw_target_row,time,y_true,q025,q975,score\n"
                      "0,24,2020-05-01T00:00:00,0,0,0,0\n")
                selection_results.append(row)
                for name in ("result.json", "test_predictions.csv"):
                    relative = str((folder / name).relative_to(selection))
                    selection_input_hashes[relative] = _digest(folder / name)
    _json(selection / "results.json", selection_results)
    selection_input_hashes["results.json"] = _digest(selection / "results.json")
    _json(selection / "manifest.json", {
        "status": "complete", "last_stable_status": "complete", "protocol": protocol,
        "protocol_sha256": lock["protocol_sha256"], "source_hashes": selection_source_hashes,
        "data_hashes": data_hashes, "environment_versions": environment,
    })

    selection_analysis = root / verifier.SELECTION_ANALYSIS
    selection_analysis_protocol = {"version": "selection_v2_analysis_2"}
    _json(selection_analysis / "analysis_protocol.json", selection_analysis_protocol)
    selection_analysis_files = sorted(verifier.SELECTION_ANALYSIS_FILES)
    for name in selection_analysis_files:
        if name.endswith(".json"):
            _json(selection_analysis / name, {"synthetic": True})
        else:
            _file(selection_analysis / name, "synthetic\n")
    _json(selection_analysis / "analysis_manifest.json", {
        "status": "complete",
        "protocol": selection_analysis_protocol,
        "protocol_sha256": _digest(selection_analysis / "analysis_protocol.json"),
        "analysis_source_sha256": _digest(root / "review/analyze_selection_study_v2.py"),
        "reporting_counts": {"tasks": 12, "methods": 6, "locked_result_records": 192,
                             "task_method_rows": 72, "full_comparisons": 60},
        "artifact_hashes": {name: _digest(selection_analysis / name)
                            for name in selection_analysis_files},
        "selection_input_hashes": selection_input_hashes,
    })

    fit_only_hashes = {}
    configuration_tasks = []
    shared = None
    for task in verifier.TASKS:
        _json(selection / task / "target_data" / "selection_counts.json", {
            "n_source_fit": 0, "n_target_fit": 1, "n_validation": 4,
            "fit_period": "2019", "validation_period": "2020 Q1",
        })
        shared_artifacts = {
            "fit_and_validation_ids": f"{task}/pooled_data/fit_and_validation_ids.npz",
            "sampling": f"{task}/pooled_data/sampling.json",
            "preprocessing": f"{task}/pooled_data/preprocessing.json",
            "selection_counts": f"{task}/pooled_data/selection_counts.json",
        }
        for seed in range(3):
            shared_artifacts[f"seed{seed}_landmarks"] = (
                f"{task}/pooled_data/landmarks/seed{seed}/rank1/landmarks.npz")
            shared_artifacts[f"seed{seed}_feature_map"] = (
                f"{task}/pooled_data/landmarks/seed{seed}/rank1/gamma_0.001/map.npz")
        for relative in shared_artifacts.values():
            path = selection / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "selection_counts.json":
                _json(path, {
                    "n_source_fit": 1, "n_target_fit": 1, "n_validation": 4,
                    "fit_period": "2019", "validation_period": "2020 Q1",
                })
            else:
                path.write_bytes((relative + "\n").encode())
            fit_only_hashes[relative] = _digest(path)
            if shared is None:
                shared = path
        task_protocol = next(row for row in protocol["tasks"] if row["name"] == task)
        full_row = next(row for row in final
                        if row["task"] == task and row["method"] == "Full")
        full_config = full_row["selected_config"]
        full_path = selection / task / "Full" / "final_selection.json"
        configuration_tasks.append({
            "task": task, "site": task_protocol["site"], "mode": task_protocol["mode"],
            "role": task_protocol["role"],
            "base_full_selection_file": f"{task}/Full/final_selection.json",
            "base_full_selection_sha256": _digest(full_path),
            "full_config": full_config,
            "variants": builder._derive_matched_variants(full_config),
            "shared_artifacts": shared_artifacts,
        })
    matched = root / verifier.MATCHED
    matched_protocol = {
        "version": "matched_ablation_v2", "base_experiment": "selection_v2",
        "task_order": protocol["tasks"], "variants": list(verifier.VARIANTS),
        "evaluation_seeds": [0, 1, 2], "tau": 0.5,
        "environment_versions": environment,
    }
    _json(matched / "protocol.json", matched_protocol)
    _json(matched / "configurations.json", {
        "version": "matched_ablation_v2_configurations_1",
        "base_selection_lock_sha256": selection_lock_hash,
        "tasks": configuration_tasks})
    matched_source_hashes = {
        "review/run_matched_ablation_v2.py": _digest(root / "review/run_matched_ablation_v2.py")
    }
    expected_keys = [f"{task}|{variant}|seed{seed}" for task in verifier.TASKS
                     for variant in verifier.VARIANTS for seed in range(3)]
    matched_lock = {
        "version": "matched_ablation_v2_lock_1",
        "protocol_sha256": _digest(matched / "protocol.json"),
        "configurations_sha256": _digest(matched / "configurations.json"),
        "source_hashes": matched_source_hashes,
        "environment_versions": environment,
        "base_selection": {
            "output": "/execution/selection_v2",
            "selection_lock_sha256": selection_lock_hash,
            "protocol_sha256": lock["protocol_sha256"],
            "source_hashes": selection_source_hashes, "data_hashes": data_hashes,
            "environment_versions": environment,
            "data_provenance_sha256": lock["data_provenance_sha256"],
        },
        "fit_only_artifact_hashes": fit_only_hashes,
        "expected_evaluation_keys": expected_keys,
    }
    _json(matched / "matched_lock.json", matched_lock)
    matched_lock_hash = _digest(matched / "matched_lock.json")
    matched_results = []
    for key in expected_keys:
        task, variant, seed_name = key.split("|")
        seed = int(seed_name.removeprefix("seed"))
        task_record = next(row for row in configuration_tasks if row["task"] == task)
        config = task_record["variants"][variant]
        folder = matched / "evaluation" / task / variant / seed_name
        diagnostics = {"converged": True}
        data_folder = str(Path(task_record["shared_artifacts"]["preprocessing"]).parent)
        feature_map = task_record["shared_artifacts"][f"seed{seed}_feature_map"]
        _json(folder / "fit.json", {
            "task": task, "method": variant, "config": config, "seed": seed, "tau": 0.5,
            "candidate_id": "matched_ablation_v2_locked", "feature_map": feature_map,
            "data_folder": data_folder, "diagnostics": diagnostics,
        })
        _file(folder / "test_predictions.csv",
              "window_index,raw_target_row,time,y_true,prediction\n"
              "0,24,2020-07-01T00:00:00,0,0\n")
        (folder / "model.npz").write_bytes((key + " model").encode())
        row = {
            "evaluation_key": key, "matched_lock_sha256": matched_lock_hash,
            "base_selection_lock_sha256": selection_lock_hash,
            "task": task, "site": task_record["site"], "mode": task_record["mode"],
            "role": task_record["role"], "variant": variant, "seed": seed, "tau": 0.5,
            "config": config, "shared_artifacts": task_record["shared_artifacts"],
            "feature_map": feature_map, "data_folder": data_folder,
            "fit_record": str((folder / "fit.json").relative_to(matched)),
            "fit_record_sha256": _digest(folder / "fit.json"),
            "prediction_file": str((folder / "test_predictions.csv").relative_to(matched)),
            "prediction_sha256": _digest(folder / "test_predictions.csv"),
            "model_file": str((folder / "model.npz").relative_to(matched)),
            "model_sha256": _digest(folder / "model.npz"),
            "counts": {"n_test": 1},
            "point": {"n": 1, "rmse": 0.0, "mae": 0.0, "r2": None},
            "diagnostics": diagnostics,
        }
        _json(folder / "result.json", row)
        matched_results.append(row)
    _json(matched / "results.json", matched_results)
    _json(matched / "manifest.json", {
        "version": "matched_ablation_v2_manifest_1",
        "status": "complete", "heldout_unlocked": True,
        "completed_evaluation_keys": 144, "matched_lock_sha256": matched_lock_hash,
        "base_selection_lock_sha256": selection_lock_hash,
        "protocol_sha256": matched_lock["protocol_sha256"],
        "configurations_sha256": matched_lock["configurations_sha256"],
        "results_sha256": _digest(matched / "results.json"),
    })

    matched_analysis = root / verifier.MATCHED_ANALYSIS
    matched_outputs = ["task_metrics.csv", "aggregate_metrics.csv", "paired_inference.csv",
                       "task_metrics.tex", "aggregate_metrics.tex", "paired_inference.tex",
                       "summary.json"]
    matched_analysis_protocol = {
        "version": "matched_ablation_v2_analysis_1", "output_files": matched_outputs,
        "environment_versions": environment,
    }
    _json(matched_analysis / "analysis_protocol.json", matched_analysis_protocol)
    matched_analysis_lock = {
        "version": "matched_ablation_v2_analysis_lock_1",
        "analysis_protocol_sha256": _digest(matched_analysis / "analysis_protocol.json"),
        "source_hashes": {"review/analyze_matched_ablation_v2.py": _digest(
            root / "review/analyze_matched_ablation_v2.py")},
        "environment_versions": environment,
        "matched_lock_sha256": matched_lock_hash,
        "matched_protocol_sha256": matched_lock["protocol_sha256"],
        "matched_configurations_sha256": matched_lock["configurations_sha256"],
        "expected_evaluation_keys": expected_keys,
    }
    _json(matched_analysis / "analysis_lock.json", matched_analysis_lock)
    matched_analysis_lock_hash = _digest(matched_analysis / "analysis_lock.json")
    _json(matched_analysis / "summary.json", {
        "version": "matched_ablation_v2_analysis_results_1",
        "analysis_lock_sha256": matched_analysis_lock_hash,
        "matched_lock_sha256": matched_lock_hash,
        "matched_results_sha256": _digest(matched / "results.json"),
        "complete_evaluation_keys": 144, "task_metric_rows": 48,
        "paired_tests": {"within": 18, "cross": 18},
    })
    for name in matched_outputs:
        if name != "summary.json":
            _file(matched_analysis / name, "synthetic\n")
    _json(matched_analysis / "manifest.json", {
        "version": "matched_ablation_v2_analysis_manifest_1",
        "status": "complete", "heldout_results_read": True,
        "analysis_lock_sha256": matched_analysis_lock_hash,
        "output_hashes": {name: _digest(matched_analysis / name)
                          for name in matched_outputs},
    })

    render = root / verifier.RENDER
    for name in verifier.RENDER_FILES:
        if name == "render_protocol.json":
            _json(render / name, {"version": "synthetic"})
        else:
            _file(render / name, "synthetic output\n")
    _json(render / "manifest.json", {
        "version": "selection_v2_render_manifest_1", "status": "complete",
        "source_sha256": _digest(root / "review/render_selection_study_v2.py"),
        "inputs": {
            "selection_analysis_manifest_sha256": _digest(
                selection_analysis / "analysis_manifest.json"),
            "matched_analysis_manifest_sha256": _digest(matched_analysis / "manifest.json"),
            "files": {
                **{str(selection_analysis / name): _digest(selection_analysis / name)
                   for name in verifier.RENDER_SELECTION_INPUT_FILES},
                **{str(matched_analysis / name): _digest(matched_analysis / name)
                   for name in verifier.RENDER_MATCHED_INPUT_FILES},
            },
        },
        "validated_counts": {
            "selection_task_method_rows": 72, "selection_scope_rows": 12,
            "selection_full_inference_rows": 60, "matched_task_variant_rows": 48,
            "matched_aggregate_rows": 8, "matched_inference_rows": 36,
        },
        "outputs": {name: _digest(render / name) for name in verifier.RENDER_FILES},
    })

    chronological_manifest = _add_chronological(root)
    selection_manifest = json.loads((selection / "manifest.json").read_text())
    selection_manifest["reference_manifest_sha256"] = _digest(chronological_manifest)
    _json(selection / "manifest.json", selection_manifest)
    _add_probability(root)

    # Validate the synthetic source tree once, then turn it into the package view
    # by removing the deliberately omitted chronological logs/models.
    state = builder.validate_completed_v2(root)
    fit_inventory_path = selection / builder.FIT_RECORD_INVENTORY
    _json(fit_inventory_path, state["selection"]["fit_record_inventory"])
    if not retain_source_fit_records:
        inventory = state["selection"]["fit_record_inventory"]
        for record in inventory["selection_fit_records"]:
            (selection / record["fit_record"]).unlink()
            (selection / record["validation_metrics"]).unlink()
        for record in inventory["evaluation_fit_records"]:
            (selection / record["fit_record"]).unlink()
    for relative in state["chronological"]["omitted_paths"]:
        (state["chronological"]["root"] / relative).unlink()
    for model in (root / verifier.MATCHED).glob("evaluation/*/*/seed*/model.npz"):
        model.unlink()

    builder.write_readme(root)
    packaging_environment = {
        "python": "synthetic", "implementation": "synthetic",
        "platform": "synthetic", "machine": "synthetic",
        "processor": None, "packages": {},
    }
    _json(root / "environment.json", {
        "version": "s1_environment_2",
        "frozen_selection_v2_environment": environment,
        "packaging_runtime": packaging_environment,
        "interpretation": (
            "The frozen block records the experiment environment bound by the selection-v2 "
            "lock. The packaging block records only the later archive-construction runtime."
        ),
    })
    builder.write_requirements(root, environment, packaging_environment)
    builder.write_evidence_inventory(root, state)
    builder.write_packaging_provenance(root, state)
    split_assignments = _add_split_assignments(root)
    _json(root / "PACKAGE_INDEX.json", {
        "version": "selection_v2_matched_2",
        "workflow_order": [
            "selection-v2 freeze", "base selection started", "selection-analysis freeze",
            "base selection completed", "regularizer selection", "joint selection", "selection lock",
            "matched-ablation freeze", "matched-analysis freeze", "selection evaluation",
            "matched evaluation", "selection analysis", "matched analysis",
            "publication rendering",
        ],
        "v2_design": {
            "selection_lock_version": "selection_v2_lock_2", "tasks": 12,
            "methods": 6, "queries_per_task_method": 54,
            "total_configuration_queries": 3888, "evaluation_records": 192,
            "selection_fits": 10368,
            "evaluation_tau_fits": 576,
            "matched_lock_version": "matched_ablation_v2_lock_1",
            "matched_evaluation_records": 144,
        },
        "evidence_inventory": str(builder.EVIDENCE_INVENTORY),
        "fit_record_inventory": {
            "path": str(verifier.SELECTION / verifier.FIT_RECORD_INVENTORY),
            "sha256": _digest(fit_inventory_path),
            "version": verifier.FIT_RECORD_INVENTORY_VERSION,
            "hash_authority": verifier.FIT_RECORD_HASH_AUTHORITY,
            "selection_fit_records": 10368,
            "selection_validation_metric_records": 10368,
            "evaluation_fit_records": 576,
        },
        "split_assignments": split_assignments,
        "artifact_file_counts": {
            "common_setting": len(state["chronological"]["paths"]),
            "probabilistic_selection_v2": len(state["probability"]["selection"]["paths"]),
            "probabilistic_analysis_v2": len(state["probability"]["analysis"]["paths"]),
            "probabilistic_presentation_v1": len(
                state["probability"]["presentation"]["paths"]),
            "selection_v2": len(builder._selection_package_paths(state)) + 1,
            "selection_v2_analysis": 2 + len(
                state["selection_analysis"]["artifact_paths"]),
            "matched_ablation_v2": len(builder._matched_package_paths(state)),
            "matched_ablation_v2_analysis": 3 + len(
                state["matched_analysis"]["output_paths"]),
            "selection_v2_render": 1 + len(state["render"]["output_paths"]),
        },
        "probability_provenance": {
            "class": "legacy/posthoc",
            "limitation": builder.LEGACY_PROBABILITY_LIMITATION,
            "current_candidate_protocol_check":
                state["probability"]["current_candidate_protocol_check"],
        },
        "selection_v1_allowlist": ["review/selection_runs/selection_v1/protocol.json"],
        "omitted_artifact_classes": [
            "full selection and main fit histories", "final fitted-model binaries",
        ],
    })
    _manifest(root)
    assert shared is not None
    return shared, state


def test_complete_synthetic_v2_passes_and_inner_fit_only_hash_detects_tamper(tmp_path):
    shared, state = _complete_package(tmp_path)
    assert len(state["selection"]["results"]) == 192
    assert len(state["matched"]["results"]) == 144
    report = verifier.verify(tmp_path)
    assert report["status"] == "passed", report["errors"]

    shared.write_bytes(b"tampered-after-lock")
    _manifest(tmp_path)  # outer manifest is fresh; the nested lock must still reject it.
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("matched fit-only SHA-256 mismatch" in error for error in report["errors"])


def test_matched_result_and_fit_identity_tampering_is_rejected(tmp_path):
    _complete_package(tmp_path)
    matched = tmp_path / verifier.MATCHED
    results_path = matched / "results.json"
    manifest_path = matched / "manifest.json"
    original_results = results_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    original_rows = json.loads(original_results)
    first = original_rows[0]
    per_result = matched / "evaluation" / first["task"] / first["variant"] / \
        f"seed{first['seed']}" / "result.json"
    original_per_result = per_result.read_bytes()

    attacks = {
        "matched_lock_sha256": "0" * 64,
        "base_selection_lock_sha256": "1" * 64,
        "config": {**first["config"], "gamma": 999.0},
        "task": "within_site2",
        "variant": "NoVariance",
        "seed": 99,
    }
    for field, value in attacks.items():
        rows = json.loads(original_results)
        rows[0][field] = value
        _json(per_result, rows[0])
        _json(results_path, rows)
        manifest = json.loads(original_manifest)
        manifest["results_sha256"] = _digest(results_path)
        _json(manifest_path, manifest)
        _manifest(tmp_path)
        report = verifier.verify(tmp_path)
        assert report["status"] == "failed"
        assert any(f"matched result {field} mismatch" in error
                   for error in report["errors"]), report["errors"]
        per_result.write_bytes(original_per_result)
        results_path.write_bytes(original_results)
        manifest_path.write_bytes(original_manifest)

    rows = json.loads(original_results)
    fit_path = matched / rows[0]["fit_record"]
    original_fit = fit_path.read_bytes()
    fit = json.loads(original_fit)
    fit["task"] = "within_site2"
    _json(fit_path, fit)
    rows[0]["fit_record_sha256"] = _digest(fit_path)
    _json(per_result, rows[0])
    _json(results_path, rows)
    manifest = json.loads(original_manifest)
    manifest["results_sha256"] = _digest(results_path)
    _json(manifest_path, manifest)
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("matched fit task mismatch" in error for error in report["errors"])


def test_analysis_protocol_lock_summary_and_render_hash_tampering_is_rejected(tmp_path):
    _complete_package(tmp_path)
    selection_analysis = tmp_path / verifier.SELECTION_ANALYSIS
    matched_analysis = tmp_path / verifier.MATCHED_ANALYSIS

    analysis_protocol = selection_analysis / "analysis_protocol.json"
    analysis_manifest = selection_analysis / "analysis_manifest.json"
    protocol = json.loads(analysis_protocol.read_text())
    protocol["version"] = "forged"
    _json(analysis_protocol, protocol)
    manifest = json.loads(analysis_manifest.read_text())
    manifest["protocol"] = protocol
    manifest["protocol_sha256"] = _digest(analysis_protocol)
    _json(analysis_manifest, manifest)
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert any("selection-v2 analysis protocol/manifest mismatch" in error
               for error in report["errors"])

    # Recreate once to isolate the matched-analysis and render attacks.
    tmp_path2 = tmp_path / "second"
    _complete_package(tmp_path2)
    matched_analysis = tmp_path2 / verifier.MATCHED_ANALYSIS
    manifest_path = matched_analysis / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["analysis_lock_sha256"] = "0" * 64
    _json(manifest_path, manifest)
    _manifest(tmp_path2)
    report = verifier.verify(tmp_path2)
    assert any("matched analysis manifest lock hash mismatch" in error
               for error in report["errors"])

    manifest_path.write_bytes(original_manifest)
    summary_path = matched_analysis / "summary.json"
    original_summary = summary_path.read_bytes()
    for field in ("analysis_lock_sha256", "matched_lock_sha256",
                  "matched_results_sha256"):
        summary = json.loads(original_summary)
        summary[field] = "f" * 64
        _json(summary_path, summary)
        manifest = json.loads(original_manifest)
        manifest["output_hashes"]["summary.json"] = _digest(summary_path)
        _json(manifest_path, manifest)
        _manifest(tmp_path2)
        report = verifier.verify(tmp_path2)
        assert any("matched analysis identity or design counts changed" in error
                   for error in report["errors"]), report["errors"]
        summary_path.write_bytes(original_summary)
        manifest_path.write_bytes(original_manifest)

    render_manifest_path = tmp_path2 / verifier.RENDER / "manifest.json"
    render_manifest = json.loads(render_manifest_path.read_text())
    first_input = next(iter(render_manifest["inputs"]["files"]))
    render_manifest["inputs"]["files"][first_input] = "e" * 64
    _json(render_manifest_path, render_manifest)
    _manifest(tmp_path2)
    report = verifier.verify(tmp_path2)
    assert any("render input-file hashes changed" in error for error in report["errors"])


def test_calibration_chronological_and_probability_hash_chains_reject_tampering(tmp_path):
    _, state = _complete_package(tmp_path)
    selection_calibration = (tmp_path / verifier.SELECTION /
                             sorted(state["selection"]["calibration_paths"], key=str)[0])
    chronological_prediction = next((tmp_path / verifier.CHRONOLOGICAL).glob(
        "*/predictions.csv"))
    probability_calibration = next((tmp_path / verifier.PROBABILITY).glob(
        "within_site6/*/final_seed*/calibration_quantiles.csv"))
    probability_analysis = tmp_path / verifier.PROBABILITY_ANALYSIS / "aggregate_results.json"
    probability_presentation = next((tmp_path / verifier.PROBABILITY_PRESENTATION).glob("*.pdf"))
    attacks = (
        (selection_calibration, "selection calibration evidence SHA-256 mismatch"),
        (chronological_prediction, "chronological evidence SHA-256 mismatch"),
        (probability_calibration, "probability posthoc SHA-256 mismatch"),
        (probability_analysis, "probability posthoc SHA-256 mismatch"),
        (probability_presentation, "probability presentation output SHA-256 mismatch"),
    )
    for path, expected in attacks:
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        _manifest(tmp_path)
        report = verifier.verify(tmp_path)
        assert report["status"] == "failed"
        assert any(expected in error for error in report["errors"]), report["errors"]
        path.write_bytes(original)

    extra = tmp_path / verifier.CHRONOLOGICAL / "stale_result.json"
    _json(extra, {"stale": True})
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert any("chronological evidence subtree contains missing or extra files" in error
               for error in report["errors"])


def test_missing_or_forged_evidence_and_environment_fail_without_crashing(tmp_path):
    _complete_package(tmp_path)
    evidence_path = tmp_path / verifier.EVIDENCE_INVENTORY
    original_evidence = evidence_path.read_bytes()
    evidence = json.loads(original_evidence)
    evidence["selection_v2"]["preregistered"] = True
    _json(evidence_path, evidence)
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("Selection-v2 evidence schema changed" in error
               for error in report["errors"])

    evidence_path.write_bytes(original_evidence)
    evidence = json.loads(original_evidence)
    evidence["probability_v2"]["legacy_provenance_limitation"] = "forged"
    _json(evidence_path, evidence)
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("probability provenance declaration changed" in error or
               "legacy provenance declaration" in error for error in report["errors"])

    evidence_path.unlink()
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("Missing package evidence inventory" in error for error in report["errors"])

    # A malformed nested package must produce a failed report, never an exception.
    _json(evidence_path, {"version": "s1_evidence_inventory_1"})
    environment_path = tmp_path / "environment.json"
    environment = json.loads(environment_path.read_text())
    environment["frozen_selection_v2_environment"] = environment["packaging_runtime"]
    _json(environment_path, environment)
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("environment conflates" in error for error in report["errors"])


def test_noncanonical_evidence_path_alias_is_rejected(tmp_path):
    _complete_package(tmp_path)
    evidence_path = tmp_path / verifier.EVIDENCE_INVENTORY
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    calibration = evidence["selection_v2"]["calibration_predictions"]
    original = next(iter(calibration))
    assert "/" in original
    alias = original.replace("/", "//", 1)
    calibration[alias] = calibration[original]
    _json(evidence_path, evidence)
    _manifest(tmp_path)

    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("Unsafe selection calibration evidence path" in error
               for error in report["errors"])


def test_readme_requires_analysis_freezes_before_evaluations(tmp_path):
    _complete_package(tmp_path)
    readme_path = tmp_path / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    analysis_freeze = verifier.README_POINT_WORKFLOW[5]
    matched_freeze = verifier.README_POINT_WORKFLOW[6]
    assert f"{analysis_freeze}\n{matched_freeze}" in readme
    readme = readme.replace(
        f"{analysis_freeze}\n{matched_freeze}",
        f"{matched_freeze}\n{analysis_freeze}",
        1,
    )
    readme_path.write_text(readme, encoding="utf-8")
    _manifest(tmp_path)

    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("README point-workflow freeze/evaluation order changed" in error
               for error in report["errors"])


def test_required_appendix_and_evidence_tools_are_released_and_documented(tmp_path):
    _complete_package(tmp_path)
    renderer = tmp_path / verifier.REQUIRED_PUBLICATION_SCRIPTS[0]
    original_renderer = renderer.read_bytes()
    renderer.unlink()
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("Missing required publication/revision script" in error
               for error in report["errors"])

    renderer.parent.mkdir(parents=True, exist_ok=True)
    renderer.write_bytes(original_renderer)
    readme_path = tmp_path / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    command = verifier.README_POINT_WORKFLOW[-1]
    assert readme.splitlines().count(command) == 1
    readme_path.write_text(readme.replace(command + "\n", "", 1), encoding="utf-8")
    _manifest(tmp_path)
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("README point-workflow command inventory changed" in error
               for error in report["errors"])


def test_split_assignment_semantics_survive_refreshed_outer_hashes(tmp_path):
    _complete_package(tmp_path)
    split_path = tmp_path / "split_assignments/farm1.csv.gz"
    with gzip.open(split_path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    rows[1][4] = "target"  # flags remain all zero, so this is a semantic forgery.
    with split_path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                csv.writer(text).writerows(rows)
    index_path = tmp_path / "PACKAGE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["split_assignments"][0]["sha256"] = _digest(split_path)
    _json(index_path, index)
    _manifest(tmp_path)

    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("meteorological flags/domain assignment changed" in error
               for error in report["errors"]), report["errors"]


def test_split_assignment_inventory_rejects_alias_and_extra_file(tmp_path):
    _complete_package(tmp_path)
    index_path = tmp_path / "PACKAGE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["split_assignments"][0]["file"] = "split_assignments//farm1.csv.gz"
    _json(index_path, index)
    extra = tmp_path / "split_assignments/stale.csv.gz"
    extra.write_bytes(b"stale")
    _manifest(tmp_path)

    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("Unsafe split assignment path" in error for error in report["errors"])
    assert any("split-assignment subtree inventory changed" in error
               for error in report["errors"])


def test_selection_fit_count_requires_seed_evidence_and_canonical_folder(tmp_path):
    _complete_package(tmp_path)
    selection = tmp_path / verifier.SELECTION
    record_path = selection / "within_site1/Full/base_selection.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["candidates"][0]["seed_results"][0].pop("metrics")
    record["candidates"][0]["fit_folders"]["0"] = "unbound/seed0"
    _json(record_path, record)
    errors: list[str] = []
    verifier._verify_query_records(selection, errors)
    assert any("malformed fit identity" in error or "malformed fit evidence" in error
               for error in errors), errors


def test_selection_seed_records_must_match_canonical_fit_and_metric_files(tmp_path):
    _complete_package(tmp_path, retain_source_fit_records=True)
    selection = tmp_path / verifier.SELECTION
    record_path = selection / "within_site1/Full/base_selection.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    seed_result = record["candidates"][0]["seed_results"][0]
    fit_folder = selection / record["candidates"][0]["fit_folders"]["0"]
    fit_path = fit_folder / "fit.json"
    metrics_path = fit_folder / "validation_metrics.json"
    original_fit = fit_path.read_bytes()
    original_metrics = metrics_path.read_bytes()

    attacks = (
        (metrics_path, {**seed_result["metrics"], "rmse": 999.0},
         "validation metrics disagree"),
        (fit_path, {**json.loads(original_fit),
                    "diagnostics": {**seed_result["diagnostics"], "attempt_count": 99}},
         "fit diagnostics disagree"),
        (fit_path, {**json.loads(original_fit), "candidate_id": "forged"},
         "fit-record identity changed"),
    )
    for path, forged, expected in attacks:
        fit_path.write_bytes(original_fit)
        metrics_path.write_bytes(original_metrics)
        _json(path, forged)
        try:
            builder._validate_query_records(selection)
        except ValueError as error:
            assert expected in str(error).lower(), error
        else:
            raise AssertionError("source selection fit forgery was accepted")


def _set_selection_candidate_certificate(selection: Path, *, consistent: bool) -> None:
    record_path = selection / "within_site1/Full/base_selection.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    candidate = record["candidates"][7]
    seed_result = candidate["seed_results"][0]
    seed_result["diagnostics"]["converged"] = False
    if consistent:
        candidate["failed_certificate_seeds"] = [seed_result["seed"]]
        candidate["eligible_for_selection"] = False
    fit_path = selection / candidate["fit_folders"][str(seed_result["seed"])] / "fit.json"
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    fit["diagnostics"]["converged"] = False
    _json(fit_path, fit)
    _json(record_path, record)


def test_uncertified_selection_candidate_is_retained_as_ineligible(tmp_path):
    _, state = _complete_package(tmp_path, retain_source_fit_records=True)
    selection = tmp_path / verifier.SELECTION
    _set_selection_candidate_certificate(selection, consistent=True)

    fit_count, source_inventory = builder._validate_query_records(selection)
    assert fit_count == 10368
    source_row = next(row for row in source_inventory
                      if row["key"] == "within_site1|Full|base|c007|seed0")
    assert source_row["certificate_status"] == "failed_ineligible_candidate"

    errors: list[str] = []
    verified_count, expected_selection = verifier._verify_query_records(selection, errors)
    assert verified_count == 10368
    assert errors == []
    expected_row = next(row for row in expected_selection
                        if row["key"] == source_row["key"])
    assert expected_row["certificate_status"] == "failed_ineligible_candidate"

    inventory_path = selection / verifier.FIT_RECORD_INVENTORY
    compact = json.loads(inventory_path.read_text(encoding="utf-8"))
    compact["selection_fit_records"] = source_inventory
    _json(inventory_path, compact)
    expected_evaluation = [
        {key: value for key, value in row.items() if key != "fit_record_sha256"}
        for row in state["selection"]["fit_record_inventory"]["evaluation_fit_records"]
    ]
    inventory_errors: list[str] = []
    verifier._verify_fit_record_inventory(
        selection, expected_selection, expected_evaluation, inventory_errors,
    )
    assert inventory_errors == []


def test_uncertified_selection_candidate_requires_consistent_ineligibility(tmp_path):
    _complete_package(tmp_path, retain_source_fit_records=True)
    selection = tmp_path / verifier.SELECTION
    _set_selection_candidate_certificate(selection, consistent=False)

    try:
        builder._validate_query_records(selection)
    except ValueError as error:
        assert "inconsistent selection-fit summary" in str(error).lower(), error
    else:
        raise AssertionError("inconsistent selection-candidate eligibility was accepted")

    errors: list[str] = []
    verifier._verify_query_records(selection, errors)
    assert any("inconsistent fit summary" in error for error in errors), errors


def test_main_embedded_fits_must_match_files_and_retain_kernel_certificate(tmp_path):
    _complete_package(tmp_path, retain_source_fit_records=True)
    selection = tmp_path / verifier.SELECTION
    results_path = selection / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    row = next(item for item in results
               if item["task"] == "within_site1" and item["method"] == "Full" and
               item["seed"] == 0)
    per_seed_path = selection / "within_site1/Full/evaluation/seed0/result.json"
    fit_path = selection / row["fits"][0]["path"] / "fit.json"
    original_fit = fit_path.read_bytes()
    original_row = json.loads(per_seed_path.read_text(encoding="utf-8"))
    original_results = results_path.read_bytes()

    fit = json.loads(original_fit)
    fit["diagnostics"] = {**fit["diagnostics"], "attempt_count": 99}
    _json(fit_path, fit)
    try:
        builder._validate_selection(tmp_path)
    except ValueError as error:
        assert "embedded fit disagrees with fit.json" in str(error).lower(), error
    else:
        raise AssertionError("source main-fit forgery was accepted")

    fit_path.write_bytes(original_fit)
    row = json.loads(json.dumps(original_row))
    fit = json.loads(original_fit)
    fit["diagnostics"]["converged"] = False
    row["fits"][0]["diagnostics"]["converged"] = False
    _json(fit_path, fit)
    _json(per_seed_path, row)
    results = json.loads(original_results)
    for index, item in enumerate(results):
        if (item["task"], item["method"], item["seed"]) == (
                row["task"], row["method"], row["seed"]):
            results[index] = row
            break
    _json(results_path, results)
    try:
        builder._validate_selection(tmp_path)
    except ValueError as error:
        assert "kernel fit lacks the solver certificate" in str(error).lower(), error
    else:
        raise AssertionError("nonconverged source main fit was accepted")


def test_compact_fit_inventory_must_match_locked_embeddings_after_outer_rehash(tmp_path):
    _complete_package(tmp_path)
    inventory_path = tmp_path / verifier.SELECTION / verifier.FIT_RECORD_INVENTORY
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["selection_fit_records"][0]["diagnostics_sha256"] = "0" * 64
    _json(inventory_path, inventory)

    index_path = tmp_path / "PACKAGE_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["fit_record_inventory"]["sha256"] = _digest(inventory_path)
    _json(index_path, index)
    evidence_path = tmp_path / verifier.EVIDENCE_INVENTORY
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["selection_v2"]["fit_record_inventory_sha256"] = _digest(inventory_path)
    _json(evidence_path, evidence)
    _manifest(tmp_path)

    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("fit-record inventory embedding changed" in error
               for error in report["errors"]), report["errors"]


def test_calibration_semantics_reject_forgery_after_hash_refresh(tmp_path):
    _, state = _complete_package(tmp_path)
    relative = sorted(state["selection"]["calibration_paths"], key=str)[0]
    calibration = tmp_path / verifier.SELECTION / relative
    lines = calibration.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[-1] = "9"  # inconsistent with y=q025=q975=0
    calibration.write_text(lines[0] + "\n" + ",".join(fields) + "\n", encoding="utf-8")

    evidence_path = tmp_path / verifier.EVIDENCE_INVENTORY
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_key = str(verifier.SELECTION / relative)
    evidence["selection_v2"]["calibration_predictions"][evidence_key] = _digest(calibration)
    _json(evidence_path, evidence)
    _manifest(tmp_path)

    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert any("Malformed selection calibration CSV" in error
               for error in report["errors"]), report["errors"]


def test_matched_144_records_require_prediction_rows_and_metrics(tmp_path):
    _complete_package(tmp_path)
    prediction = next((tmp_path / verifier.MATCHED).glob(
        "evaluation/*/*/seed*/test_predictions.csv"))
    prediction.write_text(
        "window_index,raw_target_row,time,y_true,prediction\n"
        "0,24,2020-07-01T00:00:00,0,4\n", encoding="utf-8")
    errors: list[str] = []
    verifier._verify_matched_prediction_csv(
        prediction, 1, {"n": 1, "rmse": 0.0, "mae": 0.0, "r2": None}, errors)
    assert any("metric mismatch" in error for error in errors), errors
