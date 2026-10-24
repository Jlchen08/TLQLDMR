"""Isolation tests for the locked selection-v2 matched ablation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from review.selection_data import digest
import review.run_matched_ablation_v2 as matched


def _full_config(task_index: int = 0) -> dict:
    return {
        "gamma": 0.002 + task_index * 1e-7,
        "total_loss": 562500.0,
        "target_fraction": 0.8,
        "rank": 128,
        "lambda1": 0.25,
        "lambda2": 4.0,
        "origin": "locked_full",
    }


def _fake_base(selection_output: Path, *, with_artifacts: bool = False) -> dict:
    base_protocol = matched.selection.protocol()
    tasks = []
    artifact_hashes = {}
    for index, task in enumerate(base_protocol["tasks"]):
        full = _full_config(index)
        variants = matched.derive_variants(full)
        paths = {
            "fit_and_validation_ids": f"{task['name']}/pooled_data/fit_and_validation_ids.npz",
            "sampling": f"{task['name']}/pooled_data/sampling.json",
            "preprocessing": f"{task['name']}/pooled_data/preprocessing.json",
            "selection_counts": f"{task['name']}/pooled_data/selection_counts.json",
        }
        for seed in matched.SEEDS:
            paths[f"seed{seed}_landmarks"] = (
                f"{task['name']}/pooled_data/landmarks/seed{seed}/rank128/landmarks.npz")
            paths[f"seed{seed}_feature_map"] = (
                f"{task['name']}/pooled_data/landmarks/seed{seed}/rank128/"
                f"gamma_{format(full['gamma'], '.17g')}/map.npz")
        if with_artifacts:
            for relative in paths.values():
                path = selection_output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
                artifact_hashes[relative] = digest(path)
        tasks.append({
            "task": task["name"], "site": task["site"], "mode": task["mode"],
            "role": task["role"],
            "base_full_selection_file": f"{task['name']}/Full/final_selection.json",
            "base_full_selection_sha256": "f" * 64,
            "full_config": full, "variants": variants,
            "shared_artifacts": paths,
        })
    lock = {
        "version": "selection_v2_lock_2",
        "protocol_sha256": "p" * 64,
        "source_hashes": {"source": "s"},
        "data_hashes": {"data": "d"},
        "environment_versions": matched.selection.environment_versions(),
        "data_provenance_sha256": "v" * 64,
    }
    return {
        "selection_output": str(selection_output.resolve()),
        "manifest": {"source_hashes": lock["source_hashes"],
                     "data_hashes": lock["data_hashes"]},
        "protocol": base_protocol, "lock": lock,
        "selection_lock_sha256": "l" * 64,
        "tasks": tasks,
        "fit_only_artifact_hashes": artifact_hashes,
    }


def test_derive_variants_changes_only_declared_lambdas():
    full = _full_config()
    variants = matched.derive_variants(full)
    matched._assert_only_lambda_changes(full, variants)
    assert variants["Full"] == full
    assert variants["NoVariance"]["lambda1"] == 0
    assert variants["NoVariance"]["lambda2"] == full["lambda2"]
    assert variants["NoDiscrepancy"]["lambda1"] == full["lambda1"]
    assert variants["NoDiscrepancy"]["lambda2"] == 0
    assert variants["WeightedSVQR"]["lambda1"] == 0
    assert variants["WeightedSVQR"]["lambda2"] == 0
    fixed = set(full) - {"lambda1", "lambda2"}
    assert all({key: config[key] for key in fixed} == {key: full[key] for key in fixed}
               for config in variants.values())


def test_full_requires_both_components_active():
    invalid = {**_full_config(), "lambda1": 0.0}
    with pytest.raises(ValueError, match="both components active"):
        matched.derive_variants(invalid)


def test_freeze_never_constructs_study_or_unlocks_heldout(tmp_path, monkeypatch):
    selection_output = tmp_path / "selection"
    selection_output.mkdir()
    base = _fake_base(selection_output)
    calls = []

    def fake_verify(path, *, require_preregistered):
        calls.append(require_preregistered)
        return base

    class ForbiddenStudy:
        def __init__(self, *args, **kwargs):
            pytest.fail("Freeze must not construct Study or read held-out data")

    monkeypatch.setattr(matched, "_verify_base_selection", fake_verify)
    monkeypatch.setattr(matched, "_source_hashes", lambda: {"matched": "hash"})
    monkeypatch.setattr(matched.selection, "Study", ForbiddenStudy)
    output = tmp_path / "matched"
    verified = matched.freeze(output, selection_output)
    assert calls == [True, True]  # initial gate and post-publication race check
    assert verified["manifest"]["heldout_unlocked"] is False
    assert len(verified["lock"]["expected_evaluation_keys"]) == 12 * 4 * 3
    assert not (output / "results.json").exists()
    assert not (output / "evaluation").exists()


def test_expected_keys_are_complete_and_unique():
    base = _fake_base(Path("/tmp/synthetic_selection"))
    configurations = matched._configuration_document(base)
    keys = matched.expected_evaluation_keys(configurations)
    assert len(keys) == 12 * 4 * 3
    assert len(set(keys)) == len(keys)
    assert sum(key.startswith("within_site") for key in keys) == 6 * 4 * 3
    assert sum(key.startswith("cross_site") for key in keys) == 6 * 4 * 3


def test_evaluation_requires_a_separate_resume_call(tmp_path, monkeypatch):
    class ForbiddenStudy:
        def __init__(self, *args, **kwargs):
            pytest.fail("Study must not be constructed before the resume gate")

    monkeypatch.setattr(matched.selection, "Study", ForbiddenStudy)
    with pytest.raises(RuntimeError, match="later explicit evaluate --resume"):
        matched.evaluate(tmp_path / "matched", tmp_path / "selection", resume=False)


@pytest.mark.parametrize("arguments", [
    ["--phase", "evaluate"],
    ["--phase", "evaluate", "--resume", "--output", "/path/that/does/not/exist"],
])
def test_cli_rejects_evaluation_without_prior_lock(monkeypatch, arguments):
    monkeypatch.setattr(matched.sys, "argv", ["run_matched_ablation_v2.py", *arguments])
    with pytest.raises(SystemExit) as exc:
        matched.main()
    assert exc.value.code == 2


def test_freeze_detects_prior_base_heldout_evaluation(tmp_path):
    (tmp_path / "results.json").write_text("[]\n")
    assert matched._heldout_artifacts(tmp_path) == ["results.json"]
    evaluation = tmp_path / "within_site1" / "Full" / "evaluation" / "seed0"
    evaluation.mkdir(parents=True)
    (evaluation / "result.json").write_text("{}\n")
    found = matched._heldout_artifacts(tmp_path)
    assert "results.json" in found
    assert "within_site1/Full/evaluation" in found


@pytest.mark.parametrize(
    ("lock_version", "total_queries", "message"),
    [
        ("selection_v2_lock_1", 12 * 6 * 54, "only the audited selection_v2_lock_2"),
        ("selection_v2_lock_2", 1, "12 x 6 x 54 queries"),
    ],
)
def test_base_verification_requires_lock2_and_full_query_count(
        tmp_path, monkeypatch, lock_version, total_queries, message):
    config = matched.selection.protocol()
    matched.selection.save_json(tmp_path / "protocol.json", config)
    (tmp_path / "data_provenance.json").write_text("{}\n")
    manifest = {
        "source_hashes": {"source": "hash"},
        "data_hashes": {"data": "hash"},
        "environment_versions": matched.selection.environment_versions(),
    }
    lock = {
        "version": lock_version,
        "protocol_sha256": digest(tmp_path / "protocol.json"),
        **manifest,
        "selection_files": {},
        "data_provenance_sha256": digest(tmp_path / "data_provenance.json"),
        "total_configuration_queries": total_queries,
        "validation_query_budget": {"total_per_task_method": 54},
        "final_selections": [],
    }
    matched.selection.save_json(tmp_path / "selection_lock.json", lock)
    monkeypatch.setattr(matched.selection, "_initialize",
                        lambda output, current, resume: manifest)
    monkeypatch.setattr(matched, "_expected_selection_files", lambda current: set())
    with pytest.raises(ValueError, match=message):
        matched._verify_base_selection(tmp_path, require_preregistered=False)


def test_matched_evaluation_shares_ids_maps_and_landmarks_for_all_keys(tmp_path,
                                                                      monkeypatch):
    selection_output = tmp_path / "selection"
    selection_output.mkdir()
    base = _fake_base(selection_output, with_artifacts=True)
    configurations = matched._configuration_document(base)
    expected = matched.expected_evaluation_keys(configurations)
    lock = {
        "fit_only_artifact_hashes": base["fit_only_artifact_hashes"],
        "expected_evaluation_keys": expected,
    }
    output = tmp_path / "matched"
    output.mkdir()
    manifest = {"status": "matched_locked", "heldout_unlocked": False}
    matched.selection.save_json(output / "manifest.json", manifest)
    verified = {
        "protocol": matched.protocol(), "configurations": configurations,
        "lock": lock, "manifest": manifest, "base": base,
        "matched_lock_sha256": "m" * 64,
    }
    monkeypatch.setattr(matched, "verify_matched_lock", lambda *args, **kwargs: verified)

    visits = []

    class FakeStudy:
        def __init__(self, study_output, config, write_global_provenance=False):
            assert Path(study_output) == selection_output
            self.output = Path(study_output)
            self.context_cache, self.feature_cache = {}, {}
            self.unlocked = False

        def unlock_heldout(self, manifest):
            self.unlocked = True

        def context(self, task, method):
            assert method == "Full"
            data = SimpleNamespace(
                y_raw=np.array([10., 20.]),
                target_row_indices=np.array([100, 101]),
                timestamps=np.array(["2020-07-01", "2020-07-02"], dtype="datetime64[ns]"),
            )
            return {"data": data, "counts": {"n_source_fit": 3, "n_target_fit": 2}}

        def _heldout_ids(self, context, split):
            assert self.unlocked and split == "test"
            return np.array([0, 1])

        def fit(self, task, method, config, seed, tau, folder, candidate_id):
            assert self.unlocked and tau == 0.5
            frozen = next(row for row in configurations["tasks"] if row["task"] == task["name"])
            assert config == frozen["variants"][method]
            folder.mkdir(parents=True, exist_ok=True)
            record = {
                "task": task["name"], "method": method, "config": config,
                "seed": seed, "tau": tau, "candidate_id": candidate_id,
                "feature_map": frozen["shared_artifacts"][f"seed{seed}_feature_map"],
                "data_folder": str(Path(frozen["shared_artifacts"]["preprocessing"]).parent),
                "diagnostics": {"converged": True},
            }
            matched.selection.save_json(folder / "fit.json", record)
            np.savez_compressed(folder / "model.npz", w=np.array([0.]))
            visits.append((task["name"], method, seed, record["feature_map"],
                           record["data_folder"], dict(config)))
            return record

        def predict(self, context, folder, split):
            assert self.unlocked and split == "test"
            return np.array([10.5, 19.5])

    monkeypatch.setattr(matched.selection, "Study", FakeStudy)
    results = matched.evaluate(output, selection_output, resume=True)
    assert len(results) == 12 * 4 * 3
    assert {row["evaluation_key"] for row in results} == set(expected)
    assert len(visits) == 12 * 4 * 3
    for task in configurations["tasks"]:
        for seed in matched.SEEDS:
            group = [row for row in visits if row[0] == task["task"] and row[2] == seed]
            assert len(group) == 4
            assert len({row[3] for row in group}) == 1  # exact Full feature map/landmarks
            assert len({row[4] for row in group}) == 1  # exact fit IDs/preprocessing folder
            for _, variant, _, _, _, config in group:
                assert config == task["variants"][variant]
                matched._assert_only_lambda_changes(task["full_config"], task["variants"])
    assert json.loads((output / "manifest.json").read_text())["completed_evaluation_keys"] == 144
    assert len(json.loads((output / "results.json").read_text())) == 144
    resumed = matched.evaluate(output, selection_output, resume=True)
    assert len(resumed) == 144
    assert len(visits) == 12 * 4 * 3  # completed fits were verified, not rerun


def test_resume_rejects_result_from_another_lock(tmp_path):
    prediction = tmp_path / "prediction.csv"
    prediction.write_text("y_true,prediction\n1,1\n")
    record = {
        "evaluation_key": "within_site1|Full|seed0",
        "matched_lock_sha256": "old",
        "config": _full_config(),
        "prediction_sha256": digest(prediction),
    }
    with pytest.raises(ValueError, match="different lock"):
        matched._validate_existing_result(
            record, key="within_site1|Full|seed0", matched_lock_sha256="new",
            config=_full_config(), prediction_path=prediction)
