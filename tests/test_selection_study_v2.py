"""Isolation tests for selection v2; no real held-out outcome is evaluated."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from review.selection_data_v2 import append_common_pressure_humidity
import review.run_selection_study_v2 as study


def test_protocol_covers_historical_region_and_uses_six_plus_six_tasks():
    config = study.protocol()
    tasks = config["tasks"]
    assert [task["name"] for task in tasks[:6]] == [f"within_site{i}" for i in range(1, 7)]
    assert [task["name"] for task in tasks[6:]] == [f"cross_site{i}" for i in range(1, 7)]
    candidates = config["kernel_base_candidates"]
    assert {candidate["rank"] for candidate in candidates} == {128, 256, 512}
    assert max(candidate["target_fraction"] for candidate in candidates) == pytest.approx(.995)
    assert max(candidate["total_loss"] for candidate in candidates) >= 2.39e6
    assert any(candidate["gamma"] == .002 and candidate["total_loss"] == 562500
               and candidate["target_fraction"] == .99413 for candidate in candidates)
    for method in ("Full", "NoVariance", "NoDiscrepancy"):
        regularizers = config["regularizer_candidates"][method]
        active = study.ACTIVE_REGULARIZERS[method]
        assert len(regularizers) == 18
        assert all(min(candidate[name] for candidate in regularizers) <= 1e-5
                   for name in active)
    assert config["base_regularizer_anchors"] == {
        "Full": {"lambda1": 1.0, "lambda2": 1.0},
        "NoVariance": {"lambda1": 0.0, "lambda2": 1.0},
        "NoDiscrepancy": {"lambda1": 1.0, "lambda2": 0.0},
        "WeightedSVQR": {"lambda1": 0.0, "lambda2": 0.0},
        "TargetSVQR": {"lambda1": 0.0, "lambda2": 0.0},
    }
    full_pairs = {(candidate["lambda1"], candidate["lambda2"])
                  for candidate in config["regularizer_candidates"]["Full"]}
    for pair in ((1e-5, 1e-5), (5e-4, 1e-4), (1., 1.), (1e3, 1e3),
                 (1e-4, 1e2), (1e2, 1e-4)):
        assert pair in full_pairs
    center = {**candidates[3], "lambda1": 5e-4, "lambda2": 1e-4}
    joint = study._joint_kernel_candidates("Full", center)
    assert len(joint) == 18 and joint[0]["lambda1"] == 5e-4
    assert {candidate["rank"] for candidate in joint} == {128, 256, 512}
    assert len(study._joint_tree_candidates(config["tree_candidates"][0])) == 18
    assert len(study._middle_kernel_candidates("WeightedSVQR", center)) == 18
    assert len(study._middle_tree_candidates(config["tree_candidates"][0])) == 18
    assert config["validation_query_budget"]["total_per_task_method"] == 54
    assert config["within_target_fit_budget"] == config["cross_target_fit_budget"] == 128


def test_pressure_humidity_are_strictly_past_and_left_unscaled():
    # Two windows use original rows [0,1] and [1,2]. Row 3 is an outcome/future
    # value and must never enter either window.
    data = SimpleNamespace(
        X=np.array([[10., 11.], [11., 12.]], dtype=np.float32),
        y_raw=np.array([1., 2.]),
        input_row_indices=np.array([[0, 1], [1, 2]]),
        input_missing=np.zeros((2, 2, 1), dtype=bool),
        feature_names=("wind",),
        preprocessing={"protocol": {"window_size": 2}, "feature_names": ["wind"],
                       "feature_imputation_fitted": False,
                       "feature_scaler_fitted": False, "label_scaler_fitted": False},
        statistics={"original_rows": 4, "all_meteorological_zero_raw_indices": [1],
                    "invalid_or_missing_by_variable": {}, "windows_with_missing_inputs": 0},
        provenance={},
    )
    # dataclasses.replace is used by the loader, so provide a tiny real
    # TemporalFarmData-like dataclass through the production type.
    from review.temporal_data import TemporalFarmData
    real = TemporalFarmData(
        X=data.X, y_raw=data.y_raw,
        timestamps=np.array(["2019-01-01", "2019-01-02"], dtype="datetime64[ns]"),
        split=np.array(["train", "train"]), is_extreme=np.array([False, True]),
        event_masks={}, target_row_indices=np.array([2, 3]),
        input_row_indices=data.input_row_indices, input_missing=data.input_missing,
        feature_names=data.feature_names, target_weather={}, training_weather={},
        preprocessing=data.preprocessing, statistics=data.statistics, provenance=data.provenance,
    )
    frame = pd.DataFrame({
        "Atmosphere (hpa)": [1000., 0., -99., 999999.],
        "Relative humidity (%)": [50., 0., 70., 999999.],
    })
    augmented = append_common_pressure_humidity(real, frame)
    values = augmented.X.reshape(2, 2, 3)
    np.testing.assert_allclose(values[0, 0], [10., 1000., 50.])
    assert np.isnan(values[0, 1, 1:]).all()  # frozen complete-zero row
    assert np.isnan(values[1, 1, 1]) and values[1, 1, 2] == 70.
    assert not np.any(values == 999999.)
    assert augmented.preprocessing["feature_scaler_fitted"] is False


def test_fit_only_preprocessing_ignores_future_values():
    x = np.array([[1., np.nan], [3., 7.], [5., 11.]])
    y = np.array([.1, .3, .8])
    transformed, scaled_y, transform, state = study.preprocess(x, y, ("a", "b"))
    np.testing.assert_allclose(state["mean"], [3., 9.])
    np.testing.assert_allclose(transformed.mean(axis=0), 0., atol=1e-15)
    before = json.dumps(state, sort_keys=True)
    heldout = transform(np.array([[1e12, 1e12]]))
    assert np.all(heldout > 1e9)
    assert json.dumps(state, sort_keys=True) == before
    np.testing.assert_allclose(scaled_y, (y - y.mean()) / y.std())


def test_pooled_preprocessing_gives_each_domain_equal_mass():
    # Ten source rows around zero and two target rows around 100 must have a
    # domain-balanced mean of 50 rather than the sample-count mean of 16.67.
    x = np.r_[np.zeros((10, 1)), np.full((2, 1), 100.)]
    y = x[:, 0].copy()
    _, _, _, state = study.preprocess(x, y, ("x",), source_n=10)
    assert state["mean"] == pytest.approx([50.])
    assert state["label_mean"] == pytest.approx(50.)
    assert state["weighting"] == "domain_balanced_source_0.5_target_0.5"


def test_regularizer_components_preserve_raw_manuscript_objective_and_record_scale():
    rng = np.random.default_rng(9)
    z = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    covariance, cross, discrepancy, audit = study.regularizer_components(z, y, 12)
    raw_centered = z[:12] - z[:12].mean(axis=0)
    raw_covariance = raw_centered.T @ raw_centered / 12
    raw_cross = raw_centered.T @ (y[:12] - y[:12].mean()) / 12
    delta = z[:12].mean(axis=0) - z[12:].mean(axis=0)
    np.testing.assert_allclose(covariance, raw_covariance)
    np.testing.assert_allclose(cross, raw_cross)
    np.testing.assert_allclose(discrepancy, np.outer(delta, delta))
    assert audit["variance_trace_raw"] == pytest.approx(np.trace(raw_covariance))
    assert audit["normalization"] == "none_manuscript_raw_objective"


def test_rank_specific_landmarks_and_domain_balanced_pool_weights(tmp_path):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    instance.config["landmark_pool_source_cap"] = 6
    instance.config["landmark_pool_target_cap"] = 3
    instance.feature_cache = {}
    rng = np.random.default_rng(4)
    context = {"folder": tmp_path / "pooled_data", "x": rng.normal(size=(12, 4)),
               "source_n": 8}
    context["folder"].mkdir()
    rank3 = instance.features(context, seed=0, gamma=.02, requested_rank=3)
    rank5 = instance.features(context, seed=0, gamma=.02, requested_rank=5)
    assert rank3["z"].shape == (12, 3) and rank5["z"].shape == (12, 5)
    assert rank3["path"] != rank5["path"]
    for rank in (3, 5):
        stored = np.load(context["folder"] / "landmarks" / "seed0" /
                         f"rank{rank}" / "landmarks.npz")
        positions, weights = stored["pool_positions"], stored["pool_weight"]
        np.testing.assert_allclose(weights[positions < 8].sum(), .5)
        np.testing.assert_allclose(weights[positions >= 8].sum(), .5)


def test_validation_selection_averages_three_seeds_without_heldout_access(tmp_path):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    instance.config["selection_seeds"] = [0, 1, 2]
    instance.heldout_unlocked = False
    validation_ids = np.array([0, 1])
    context = {
        "validation_ids": validation_ids,
        "data": SimpleNamespace(
            y_raw=np.array([1., 2.]), target_row_indices=np.array([20, 21]),
            timestamps=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        ),
    }
    instance.context = lambda task, method: context
    visits = []

    def fake_fit(task, method, config, seed, tau, folder, candidate_id):
        folder.mkdir(parents=True)
        study.save_json(folder / "fit.json", {
            "task": task["name"], "method": method, "config": config,
            "seed": seed, "tau": tau, "diagnostics": {"converged": seed != 2},
        })
        visits.append((config["marker"], seed))
        return {"diagnostics": {"converged": True}}

    def fake_predict(ctx, folder, split):
        assert split == "validation" and not instance.heldout_unlocked
        record = json.loads((folder / "fit.json").read_text())
        # Candidate zero has errors [0, 3, 3], mean RMSE=2. Candidate one has
        # [1.5, 1.5, 1.5], mean RMSE=1.5 and must be selected.
        error = ([0., 3., 3.] if record["config"]["marker"] == 0 else [1.5] * 3)[record["seed"]]
        return np.array([1., 2.]) + error

    instance.fit, instance.predict = fake_fit, fake_predict
    task = {"name": "fake", "site": 1, "mode": "within", "role": "primary_breadth"}
    configs = [{**candidate, "marker": index} for index, candidate in enumerate(instance._base_configs("Full"))]
    result = instance._tune_configs(task, "Full", configs, "base")
    assert result["selected_candidate_index"] == 1
    assert len(visits) == 54
    assert all(len(candidate["seed_results"]) == 3 for candidate in result["candidates"])
    with pytest.raises(RuntimeError, match="forbidden"):
        instance._heldout_ids({"data": SimpleNamespace()}, "test")


def test_uncertified_candidate_cannot_win_validation_selection(tmp_path):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    instance.config["selection_seeds"] = [0, 1, 2]
    validation_ids = np.array([0, 1])
    context = {
        "validation_ids": validation_ids,
        "data": SimpleNamespace(
            y_raw=np.array([1., 2.]), target_row_indices=np.array([0, 1]),
            timestamps=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")),
    }
    instance.context = lambda task, method: context

    def fake_fit(task, method, config, seed, tau, folder, candidate_id):
        folder.mkdir(parents=True)
        study.save_json(folder / "identity.json", {"bad": config["bad"], "seed": seed})
        return {"diagnostics": {"converged": not (config["bad"] and seed == 2)}}

    def fake_predict(ctx, folder, split):
        identity = json.loads((folder / "identity.json").read_text())
        return np.array([1., 2.]) + (0. if identity["bad"] else 1.)

    instance.fit, instance.predict = fake_fit, fake_predict
    task = {"name": "fake", "site": 1, "mode": "within", "role": "primary_breadth"}
    configs = [{**candidate, "bad": index == 0} for index, candidate in enumerate(instance._base_configs("Full"))]
    result = instance._tune_configs(task, "Full", configs, "base")
    assert result["selected_candidate_index"] == 1
    assert result["candidates"][0]["eligible_for_selection"] is False


@pytest.mark.parametrize("method", study.KERNEL_METHODS)
def test_method_valid_rank_search_preserves_all_stage_budgets(method):
    instance = study.Study.__new__(study.Study)
    instance.config = study.protocol()
    base = instance._base_configs(method)
    expected = {32, 64, 128} if method == "TargetSVQR" else {128, 256, 512}
    assert len(base) == 18
    assert {candidate["rank"] for candidate in base} == expected
    center = base[1]
    if study.ACTIVE_REGULARIZERS[method]:
        middle = [{**center, **regularizer}
                  for regularizer in instance.config["regularizer_candidates"][method]]
    else:
        middle = study._middle_kernel_candidates(method, center)
        assert {candidate["rank"] for candidate in middle} == expected
    joint = study._joint_kernel_candidates(method, middle[0])
    assert {candidate["rank"] for candidate in joint} == expected
    assert len(base) == len(middle) == len(joint) == 18
    assert len(base) + len(middle) + len(joint) == 54
    # All non-rank base coordinates stay paired across methods.
    if method != "TargetSVQR":
        for actual, pooled in zip(base, instance._base_configs("Full")):
            for key in ("gamma", "total_loss", "target_fraction", "origin"):
                assert actual[key] == pooled[key]
    else:
        assert all("target_fraction" not in candidate for candidate in base + middle + joint)


def test_target_three_requested_ranks_are_three_effective_feature_dimensions(tmp_path):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    instance.feature_cache = {}
    context = {"folder": tmp_path / "target_data", "source_n": 0,
               "x": np.random.default_rng(4).normal(size=(128, 4))}
    context["folder"].mkdir()
    with study.threadpool_limits(limits=1):
        dimensions = [instance.features(context, seed=0, gamma=.02, requested_rank=rank)["z"].shape[1]
                      for rank in study._kernel_ranks("TargetSVQR")]
    assert dimensions == [32, 64, 128]


def test_histgbr_uses_one_seed_per_query_and_kernel_uses_three(tmp_path):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    for stage in ("selection", "evaluation"):
        assert instance.config[f"{stage}_seeds_by_method"]["HistGBR"] == [0]
        assert all(instance.config[f"{stage}_seeds_by_method"][method] == [0, 1, 2]
                   for method in study.KERNEL_METHODS)
    context = {"validation_ids": np.array([0, 1]),
               "data": SimpleNamespace(y_raw=np.array([1., 2.]),
                   target_row_indices=np.array([0, 1]),
                   timestamps=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"))}
    instance.context = lambda task, method: context
    visits = []
    def fake_fit(task, method, config, seed, tau, folder, candidate_id):
        folder.mkdir(parents=True)
        visits.append(seed)
        return {"diagnostics": {"converged": True}}
    instance.fit = fake_fit
    instance.predict = lambda *args: np.array([1., 2.])
    task = {"name": "fake", "site": 1, "mode": "within", "role": "primary_breadth"}
    result = instance._tune_configs(task, "HistGBR", instance._base_configs("HistGBR"), "base")
    assert visits == [0] * 18
    assert result["candidate_count"] == 18
    assert all(len(candidate["seed_results"]) == 1 for candidate in result["candidates"])


def test_preregistered_analysis_covers_all_sites_and_comparisons():
    plan = study.protocol()["analysis_plan"]
    assert plan["comparators"] == list(study.METHODS[1:])
    assert len(plan["primary_estimands"]) == 2
    assert plan["paired_inference"]["bootstrap_replicates"] == 10000
    assert plan["paired_inference"]["sign_flip_replicates"] == 99999
    assert "30 tests" in plan["paired_inference"]["holm_families"]
    assert "not equivalence" in plan["paired_inference"]["interpretation"]


def test_resume_rejects_changed_environment_even_with_unchanged_code(tmp_path, monkeypatch):
    config = study.protocol()
    assert {"python", "numpy", "scipy", "pandas", "scikit-learn", "joblib"} <= set(config["environment_versions"])
    monkeypatch.setattr(study, "_experiment_hashes", lambda: ({"code": "abc"}, {"data": "xyz"}, "reference"))
    output = tmp_path / "synthetic_run"
    manifest = study._initialize(output, config, False)
    assert manifest["environment_versions"] == config["environment_versions"]
    assert study._initialize(output, config, True) == manifest
    changed = {**config["environment_versions"], "scipy": "different-version"}
    monkeypatch.setattr(study, "environment_versions", lambda: changed)
    with pytest.raises(ValueError, match="Protocol environment"):
        study._initialize(output, config, True)
    with pytest.raises(ValueError, match="Frozen.*environment"):
        study._initialize(output, {**config, "environment_versions": changed}, True)


@pytest.mark.parametrize("arguments", [["--phase", "all"], ["--phase", "evaluate"]])
def test_cli_cannot_cross_selection_and_evaluation_gate_in_one_call(monkeypatch, arguments):
    monkeypatch.setattr(study.sys, "argv", ["run_selection_study_v2.py", *arguments])
    def forbidden_initialize(*args):
        pytest.fail("CLI must reject this invocation before initializing or reading experiment data")
    monkeypatch.setattr(study, "_initialize", forbidden_initialize)
    with pytest.raises(SystemExit) as exc:
        study.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("method", study.METHODS)
@pytest.mark.parametrize("boundary", ["lower", "upper"])
def test_boundary_searches_replenish_to_54_globally_unique_models(method, boundary):
    instance = study.Study.__new__(study.Study)
    instance.config = study.protocol()
    base = instance._base_configs(method)
    # Put the local center at a clipping boundary; nominal samples often collapse.
    if method == "HistGBR":
        center = {**base[0], "learning_rate": .01 if boundary == "lower" else .3,
                  "target_fraction": .05 if boundary == "lower" else .995}
        base[0] = center
        middle = study._middle_tree_candidates(center, excluded=base)
        joint = study._joint_tree_candidates(center, excluded=base + middle)
    else:
        center = {**base[0], "gamma": 1e-4 if boundary == "lower" else .08,
                  "total_loss": 100. if boundary == "lower" else 2.4e6}
        if method != "TargetSVQR":
            center["target_fraction"] = .05 if boundary == "lower" else .995
        for name in study.ACTIVE_REGULARIZERS[method]:
            center[name] = 1e-5 if boundary == "lower" else 1e3
        # Base itself is made unique again because boundary anchors may coincide.
        base = study._complete_unique_candidates(method, [center, *base], 778)
        active = study.ACTIVE_REGULARIZERS[method]
        if active:
            proposals = [{**center, **reg} for reg in study._regularizer_candidates(method)]
            middle = study._complete_unique_candidates(method, proposals, 779,
                excluded=base, regularizer_center=center)
            assert all(all(candidate[key] == center[key] for key in
                       ("gamma", "total_loss", "rank", "target_fraction")) for candidate in middle)
        else:
            middle = study._middle_kernel_candidates(method, center, excluded=base)
        joint = study._joint_kernel_candidates(method, center, excluded=base + middle)
        if method == "TargetSVQR":
            assert all("target_fraction" not in candidate for candidate in base + middle + joint)
    for stage in (base, middle, joint):
        study._assert_unique_configs(method, stage)
    study._assert_unique_configs(method, base + middle + joint, count=54)


def test_signature_uses_effective_parameters_and_deduplicates_nominal_duplicates():
    base = {"gamma": .002, "total_loss": 562500., "rank": 64,
            "target_fraction": .99413, "lambda1": 0., "lambda2": 0., "origin": "a"}
    nominal_duplicate = {**base, "target_fraction": .9808, "lambda1": 999.,
                         "lambda2": 3., "origin": "b"}
    assert study.effective_model_signature("TargetSVQR", base) == study.effective_model_signature("TargetSVQR", nominal_duplicate)
    assert study.effective_model_signature("WeightedSVQR", base) != study.effective_model_signature("WeightedSVQR", nominal_duplicate)
    assert study.effective_model_signature("WeightedSVQR", base) == study.effective_model_signature("WeightedSVQR", {**base, "lambda1": 4.})
    candidates = study._complete_unique_candidates("TargetSVQR", [base, nominal_duplicate] * 9, 2026)
    study._assert_unique_configs("TargetSVQR", candidates)
    assert candidates == study._complete_unique_candidates("TargetSVQR", [base, nominal_duplicate] * 9, 2026)
    assert all("target_fraction" not in candidate and candidate["lambda1"] == 0 for candidate in candidates)


def test_duplicate_stage_is_rejected_before_context_or_fitting(tmp_path):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    config = instance._base_configs("TargetSVQR")[0]
    with pytest.raises(ValueError, match="effective unique"):
        instance._tune_configs({}, "TargetSVQR", [config] * 18, "base")
    configs = instance._base_configs("TargetSVQR")
    with pytest.raises(ValueError, match="earlier effective"):
        instance._tune_configs({}, "TargetSVQR", configs, "middle", previous_configs=configs)


def test_full_three_stage_selection_carries_incumbent_without_refitting(tmp_path):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    task = instance.config["tasks"][0]
    instance.config["tasks"] = [task]
    instance.config["methods"] = ["TargetSVQR"]
    instance.context_cache, instance.feature_cache = {}, {}
    context = {"validation_ids": np.array([0, 1]),
               "data": SimpleNamespace(y_raw=np.array([1., 2.]),
                    target_row_indices=np.array([0, 1]),
                    timestamps=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"))}
    instance.context = lambda *args: context
    visits = []
    def fake_fit(task, method, config, seed, tau, folder, candidate_id):
        folder.mkdir(parents=True)
        visits.append((candidate_id, seed))
        return {"diagnostics": {"converged": True}}
    def fake_predict(context, folder, split):
        if "base" in folder.parts:
            error = .1 if "c000" in folder.parts else 1.
        elif "middle" in folder.parts:
            error = 2.
        else:
            error = .05 if "c001" in folder.parts else 3.
        return context["data"].y_raw + error
    instance.fit, instance.predict = fake_fit, fake_predict
    base = instance.tune_base()[0]
    middle = instance.tune_regularizers()[0]
    joint = instance.tune_joint()[0]
    assert middle["selected_source_stage"] == "base"
    assert middle["selected_fit_folders"] == base["selected_fit_folders"]
    assert joint["selected_source_stage"] == "joint"
    assert joint["selected_candidate_index"] == 1
    assert middle["carried_incumbent"] == base["selected_candidate"]
    assert joint["carried_incumbent"] == base["selected_candidate"]
    assert len(visits) == 54 * 3  # only new candidates were fitted, no incumbent refit
    configs = [row["config"] for stage in (base, middle, joint) for row in stage["candidates"]]
    study._assert_unique_configs("TargetSVQR", configs, count=54)
    assert all(stage["candidate_count"] == stage["queried_effective_unique_count"] == 18 for stage in (base, middle, joint))
    # Resuming uses the identical records and does not issue new queries.
    assert instance.tune_regularizers()[0] == middle
    assert instance.tune_joint()[0] == joint
    assert len(visits) == 54 * 3


def test_selection_lock_rejects_repeated_task_method_keys(tmp_path):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    instance.config["tasks"][1] = instance.config["tasks"][0]
    with pytest.raises(ValueError, match="72 unique"):
        instance.create_selection_lock({})


@pytest.mark.parametrize("failure", ["duplicate_query", "invented_winner"])
def test_lock_rechecks_cross_stage_uniqueness_and_incumbent_score(tmp_path, failure):
    instance = study.Study.__new__(study.Study)
    instance.output = tmp_path
    instance.config = study.protocol()
    # All expected files exist so locking reaches the first task's actual checks.
    for relative in instance._expected_selection_files():
        study.save_json(tmp_path / relative, {})
    task = instance.config["tasks"][0]
    method = "Full"
    base = instance._base_configs(method)
    middle = study._complete_unique_candidates(method,
        [{**base[0], **reg} for reg in study._regularizer_candidates(method)], 454,
        excluded=base, regularizer_center=base[0])
    joint = study._joint_kernel_candidates(method, base[0], excluded=base + middle)
    if failure == "duplicate_query":
        joint[0] = dict(base[1])  # distinct within joint, but already queried in base
    incumbent = None
    for stage, configs in zip(("base", "middle", "joint"), (base, middle, joint)):
        rows = [{"config": config, "eligible_for_selection": True,
                 "candidate_id": f"{stage}_c{index:03d}", "candidate_index": index,
                 "queried_stage": stage, "mean_validation_rmse": float(1 + index),
                 "fit_folders": {}} for index, config in enumerate(configs)]
        best = incumbent if incumbent is not None else rows[0]
        record = {"task": task["name"], "method": method, "stage": stage,
                  "candidates": rows, "candidate_count": 18,
                  "queried_effective_unique_count": 18, "carried_incumbent": incumbent,
                  "selected_candidate": best, "selected_config": best["config"],
                  "selected_fit_folders": {}, "selected_mean_validation_rmse": 1.}
        if failure == "invented_winner" and stage == "middle":
            record["selected_mean_validation_rmse"] = .001
        study.save_json(tmp_path / task["name"] / method / f"{stage}_selection.json", record)
        incumbent = best
    message = "54 effective unique" if failure == "duplicate_query" else "inconsistent"
    with pytest.raises(ValueError, match=message):
        instance.create_selection_lock({})
