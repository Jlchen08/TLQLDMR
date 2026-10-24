import json
from types import SimpleNamespace

import numpy as np
import pytest

from review import resume_solver_certificate_v2 as amendment
from review import run_selection_study_v2 as selection


def test_extension_schedule_preserves_certificate_and_only_increases_caps():
    assert [row["max_iter"] for row in amendment.EXTENDED_ATTEMPTS] == [96000]
    assert {row["gap_rtol"] for row in amendment.EXTENDED_ATTEMPTS} == {0.001}
    assert {row["gap_atol"] for row in amendment.EXTENDED_ATTEMPTS} == {1e-7}
    assert {row["check_every"] for row in amendment.EXTENDED_ATTEMPTS} == {25}


def test_amended_study_calls_continuation_only_for_an_uncertified_fit(monkeypatch, tmp_path):
    calls = []

    class FakeStudy:
        def fit(self, task, method, config, seed, tau, folder, candidate_id):
            return {"diagnostics": {"converged": method != "Full"}}

    monkeypatch.setattr(amendment.selection, "Study", FakeStudy)
    monkeypatch.setattr(
        amendment, "extend_fit_certificate",
        lambda *args: calls.append(args[2]) or {"continued": True},
    )
    cls = amendment.amended_study_class(tmp_path / "amendment.json")
    instance = cls()
    assert instance.fit({}, "Full", {}, 0, .5, tmp_path, "locked") == {"continued": True}
    assert instance.fit({}, "HistGBR", {}, 0, .5, tmp_path, "locked") == {
        "diagnostics": {"converged": True}}
    assert calls == ["Full"]


def test_extension_archives_original_and_records_unchanged_tolerance(monkeypatch, tmp_path):
    monkeypatch.setattr(amendment, "ROOT", tmp_path)
    selection_output = tmp_path / "selection"
    matched_output = tmp_path / "matched"
    folder = selection_output / "within_site1" / "Full" / "evaluation" / "seed0" / "tau_0.5"
    map_path = selection_output / "within_site1" / "pooled_data" / "map.npz"
    map_path.parent.mkdir(parents=True)
    np.savez(map_path, placeholder=np.array([1]))
    folder.mkdir(parents=True)
    z = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    config = {"gamma": .1, "rank": 1}
    record = {
        "task": "within_site1", "method": "Full", "seed": 0, "tau": .5,
        "config": config, "candidate_id": "locked_final",
        "feature_map": str(map_path.relative_to(selection_output)),
        "diagnostics": {
            "solver": "ADMM", "converged": False, "attempt_count": 2,
            "attempt_summaries": [
                {"iterations": 3000, "gap_rtol": .001, "gap_atol": 1e-7},
                {"iterations": 12000, "gap_rtol": .001, "gap_atol": 1e-7},
            ],
            "fit_seconds_including_feature_access": 2.0,
            "requested_rank": 1, "effective_rank": 1,
            "regularizer_components": {"normalization": "test"},
        },
    }
    selection.save_json(folder / "fit.json", record)
    selection.save_npz(
        folder / "model.npz", w=np.zeros(1), b=np.asarray(0.), dual=np.zeros(3),
        R=np.eye(1), c=np.zeros(1), C=np.ones(3), y=y,
    )
    original_fit_sha = selection.digest(folder / "fit.json")
    original_model_sha = selection.digest(folder / "model.npz")
    frozen = {
        "selection": {"output": "selection"}, "matched": {"output": "matched"},
        "rule": {"original_attempts": [
            {"max_iter": 3000, "gap_rtol": .001, "gap_atol": 1e-7, "check_every": 25},
            {"max_iter": 12000, "gap_rtol": .001, "gap_atol": 1e-7, "check_every": 25},
        ], "conditional_warm_restart": list(amendment.EXTENDED_ATTEMPTS)},
    }
    amendment_path = tmp_path / "amendment.json"
    selection.save_json(amendment_path, frozen)
    selection.save_json(selection_output / "selection_lock.json", {
        "final_selections": [{"task": "within_site1", "method": "Full",
                              "selected_config": config}],
    })
    monkeypatch.setattr(amendment, "validate_amendment", lambda *args: frozen)
    seen_options = []

    def fake_refine(Z, y_arg, R, c, C, tau, *, w0, b0, **options):
        seen_options.append(options)
        report = {
            "initial_primal": 2.0, "primal": 1.0, "dual_bound": .9995,
            "gap": .0005, "relative_gap": .0005, "converged": True,
            "iterations": 25, "elapsed_seconds": .01, "stop_reason": "duality_gap",
            "dual_sum_residual": 0., "dual_box_violation": 0.,
            "gap_atol": options["gap_atol"], "gap_rtol": options["gap_rtol"],
            "history": [],
        }
        return SimpleNamespace(
            w=np.ones(1), b=.5, dual_variable=np.zeros(3), converged=True,
            report=lambda: report,
        )

    monkeypatch.setattr(amendment, "refine_pinball", fake_refine)

    class FakeStudy:
        output = selection_output
        def context(self, task, method):
            return {"y": y}
        def features(self, context, seed, gamma, rank):
            return {"z": z, "path": map_path}

    amended = amendment.extend_fit_certificate(
        FakeStudy(), {"name": "within_site1"}, "Full", config, 0, .5,
        folder, record, amendment_path, "locked_final")
    assert amended["diagnostics"]["converged"] is True
    assert amended["diagnostics"]["attempt_count"] == 3
    assert seen_options == [dict(amendment.EXTENDED_ATTEMPTS[0])]
    assert selection.digest(folder / "fit.pre_amendment.json") == original_fit_sha
    assert selection.digest(folder / "model.pre_amendment.npz") == original_model_sha
    completion = json.loads((folder / "solver_amendment.json").read_text())
    assert completion["converged"] is True
    assert completion["original_fit_sha256"] == original_fit_sha
    assert amended["diagnostics"]["gap_rtol"] == .001
    # A completed resume validates and reuses the same amended artifacts.
    assert amendment.extend_fit_certificate(
        FakeStudy(), {"name": "within_site1"}, "Full", config, 0, .5,
        folder, amended, amendment_path, "locked_final") == amended

    # A crash after atomically replacing fit/model but before the completion
    # marker restores the committed originals and reruns the same rule.
    (folder / "solver_amendment.json").unlink()
    resumed = amendment.extend_fit_certificate(
        FakeStudy(), {"name": "within_site1"}, "Full", config, 0, .5,
        folder, amended, amendment_path, "locked_final")
    assert resumed["diagnostics"]["converged"] is True
    assert len(seen_options) == 2
    assert selection.digest(folder / "fit.pre_amendment.json") == original_fit_sha
    assert selection.digest(folder / "model.pre_amendment.npz") == original_model_sha

    # Neither an unlocked candidate nor an already completed result may enter
    # the recovery path.
    with pytest.raises(ValueError, match="identity changed"):
        amendment.extend_fit_certificate(
            FakeStudy(), {"name": "within_site1"}, "Full", config, 0, .5,
            folder, resumed, amendment_path, "not_locked")
    selection.save_json(folder.parent / "result.json", {"complete": True})
    with pytest.raises(ValueError, match="must never be amended"):
        amendment.extend_fit_certificate(
            FakeStudy(), {"name": "within_site1"}, "Full", config, 0, .5,
            folder, resumed, amendment_path, "locked_final")


def test_completion_inventory_includes_fit_and_model_dependencies(monkeypatch, tmp_path):
    monkeypatch.setattr(amendment, "ROOT", tmp_path)
    main, matched = tmp_path / "main", tmp_path / "matched"
    completed = main / "task" / "method" / "evaluation" / "seed0"
    tau = completed / "tau_0.5"
    tau.mkdir(parents=True)
    for path in (completed / "result.json", completed / "test_predictions.csv",
                 tau / "fit.json", tau / "model.npz"):
        path.write_bytes(path.name.encode())
    # A failed/incomplete sibling has no result and is deliberately outside
    # the commitment set used for already completed results.
    incomplete = main / "task" / "method" / "evaluation" / "seed1" / "tau_0.5"
    incomplete.mkdir(parents=True)
    (incomplete / "fit.json").write_text("incomplete")
    observed = amendment._completion_hashes(main, matched)
    assert set(observed) == {
        str(path.relative_to(tmp_path)) for path in
        (completed / "result.json", completed / "test_predictions.csv",
         tau / "fit.json", tau / "model.npz")
    }
