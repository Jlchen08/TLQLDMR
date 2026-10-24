"""Synthetic checks of study fairness, preprocessing, and calibration isolation.

The recording solver deliberately produces crossed quantile outputs. It tests
the experiment orchestration without training on, or inspecting, real data.
The actual objective/optimizer is checked in test_centered_residual_objective.
"""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from review import run_replied_comment_study as study_module


@pytest.mark.parametrize("ns,nt", [(7, 2), (10000, 128), (12, 27)])
@pytest.mark.parametrize("task", ["point", "interval"])
def test_weight_schemes_match_total_scale_and_domain_ratios(ns, nt, task):
    total = 1000.0
    schemes = ["domain_balanced", "sample_equal", "historical_ratio", "target_10x"]
    for scheme in schemes:
        cs, ct = study_module.loss_weights(ns, nt, total, scheme, task)
        assert ns * cs + nt * ct == pytest.approx(total)
        assert cs > 0 and ct > 0
        if scheme == "domain_balanced":
            assert ns * cs == pytest.approx(nt * ct)
            assert ct / cs == pytest.approx(ns / nt)
            # Explicit inverse-count compensation, normalized to the same sum.
            ratio = ns / nt
            inverse_count_cs = total / (ns + ratio * nt)
            np.testing.assert_allclose([cs, ct], [inverse_count_cs, ratio * inverse_count_cs])
        elif scheme == "sample_equal":
            assert cs == ct
        elif scheme == "historical_ratio":
            assert ct / cs == pytest.approx(3000 if task == "point" else 8000)
        else:
            assert nt * ct / (ns * cs) == pytest.approx(10)
    cs, ct = study_module.loss_weights(0, nt, total, "domain_balanced", task)
    assert cs == 0 and nt * ct == pytest.approx(total)


def test_missing_features_and_label_scaling_use_only_supplied_fit_arrays():
    xs = np.array([[1, np.nan, 4, np.nan], [3, 2, 4, np.inf]])
    xt = np.array([[5, 6, 4, np.nan]])
    ys, yt = np.array([0.1, 0.3]), np.array([0.5])
    xs_before, xt_before = xs.copy(), xt.copy()
    zs, ysc, zt, ytc, transform, state = study_module.preprocess(xs, ys, xt, yt)
    np.testing.assert_allclose(state["mean"], [3, 4, 4, 0])
    np.testing.assert_allclose(state["scale"], [np.sqrt(8 / 3), np.sqrt(8 / 3), 1, 1])
    np.testing.assert_allclose(np.r_[zs, zt].mean(axis=0), 0, atol=1e-7)
    assert np.isfinite(np.r_[zs, zt]).all()
    np.testing.assert_allclose(np.r_[ysc, ytc] * state["label_scale"] + state["label_mean"],
                               np.r_[ys, yt], atol=1e-8)
    # Held-out extremes are transformed, never used to refit or clip statistics.
    heldout = transform(np.array([[np.nan, 100, 4, np.nan]]))
    assert heldout[0, 0] == 0 and heldout[0, 1] > 50
    np.testing.assert_array_equal(xs, xs_before)
    np.testing.assert_array_equal(xt, xt_before)


def test_target_only_constant_labels_have_finite_invertible_scaling():
    _, empty_y, zt, ytc, _, state = study_module.preprocess(
        np.empty((0, 2)), np.empty(0), np.array([[2, np.nan], [4, np.nan]]), np.array([0.2, 0.2]))
    assert len(empty_y) == 0
    assert state["fit_source_count"] == 0
    assert state["label_scale"] == 1
    np.testing.assert_array_equal(ytc, [0, 0])
    assert np.isfinite(zt).all()


def test_interval_metrics_have_inclusive_coverage_and_two_sided_penalties():
    result = study_module.interval_metrics(np.array([-2., 1., 6.]),
                                           np.array([0., 0., 2.]), np.array([2., 2., 4.]), 0.2)
    assert result["picp"] == pytest.approx(1 / 3)
    assert result["mpiw"] == 2
    assert result["pinaw"] == 0.25
    assert result["winkler"] == pytest.approx(46 / 3)
    boundary = study_module.interval_metrics(np.array([0., 2.]), np.zeros(2), np.full(2, 2.), 0.2)
    assert boundary["picp"] == 1 and boundary["winkler"] == 2


class RecordingSolver:
    calls = []

    def __init__(self, **options):
        self.options = options

    def fit(self, xs, ys, xt, yt):
        self.calls.append({"options": self.options.copy(), "arrays": [a.copy() for a in (xs, ys, xt, yt)]})
        width = xt.shape[1]
        self.w = np.zeros(width)
        # Reverses the raw quantile order on purpose.
        self.b = 1 - self.options["tau"]
        self.feature_map = SimpleNamespace(
            components_=xt.copy(), normalization_=np.eye(width),
            transform=lambda x: np.asarray(x, dtype=float))
        self.regularizer_matrix_ = np.eye(width)
        self.regularizer_linear_ = np.zeros(width)
        return True


def make_synthetic_study(tmp_path, monkeypatch, name):
    monkeypatch.setattr(study_module, "Fast_TL_QLDMR", RecordingSolver)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "review/replied_study_protocol.json").read_text())
    cfg["model"].update(epochs=3, batch_size=3, landmarks=2)
    cfg.update(alpha=0.1, interval_taus=[0.2, 0.8])
    result = study_module.Study.__new__(study_module.Study)
    result.out = tmp_path / name
    result.out.mkdir()
    result.cfg, result.jobs, result.capacity = cfg, [], {1: 100., 6: 200.}
    source = SimpleNamespace(X=np.array([[1., np.nan], [3., 4.]]),
                             y_raw=np.array([10., 30.]), target_row_indices=np.array([101, 102]))
    x = np.column_stack([np.arange(26, dtype=float) + 5, np.ones(26)])
    x[2] = [1e8, -1e8]  # An unused training window.
    y = np.r_[40., 80., 1e9, np.linspace(-20, 120, 20), 30., 50., 90.]
    target = SimpleNamespace(X=x, y_raw=y, target_row_indices=np.arange(26) + 200,
                             timestamps=np.datetime64("2020-01-01") + np.arange(26) * np.timedelta64(15, "m"))
    result.data = {1: source, 6: target}
    return result


def run_synthetic(study, name, **kwargs):
    return study.run(name, 6, [(1, np.array([0, 1]))], np.array([0, 1]),
                     np.array([23, 24, 25]), np.arange(3, 23), **kwargs)


def test_orchestration_excludes_unused_training_and_all_heldout_values(tmp_path, monkeypatch):
    RecordingSolver.calls = []
    first = make_synthetic_study(tmp_path, monkeypatch, "first")
    run_synthetic(first, "point")
    second = make_synthetic_study(tmp_path, monkeypatch, "changed")
    second.data[6].X[2:] += 100000
    second.data[6].y_raw[2:] -= 50000
    run_synthetic(second, "point")
    first_prep = json.loads((first.out / "point/preprocessing.json").read_text())
    second_prep = json.loads((second.out / "point/preprocessing.json").read_text())
    assert first_prep == second_prep
    assert first_prep["label_mean"] == pytest.approx(0.25)
    assert first_prep["label_scale"] == pytest.approx(np.std([0.1, 0.3, 0.2, 0.4]))
    for a, b in zip(RecordingSolver.calls[-2]["arrays"], RecordingSolver.calls[-1]["arrays"]):
        np.testing.assert_array_equal(a, b)
    with np.load(first.out / "point/fit_ids.npz") as ids:
        np.testing.assert_array_equal(ids["source_site"], [1, 1])
        np.testing.assert_array_equal(ids["target_raw_row"], [200, 201])
        assert not set(ids["target_window_index"]) & set(ids["test_window_index"])
        assert not set(ids["target_window_index"]) & set(ids["calibration_window_index"])


def test_calibration_rank_crossing_capacity_inverse_and_test_label_isolation(tmp_path, monkeypatch):
    first = make_synthetic_study(tmp_path, monkeypatch, "interval")
    result = run_synthetic(first, "base", task="interval")
    prep = json.loads((first.out / "base/preprocessing.json").read_text())
    predictions = pd.read_csv(first.out / "base/predictions.csv")
    calibration = pd.read_csv(first.out / "base/calibration.csv")
    # Inversion must use the target capacity after pooled per-unit label scaling.
    raw_low = 200 * (prep["label_mean"] + 0.8 * prep["label_scale"])
    raw_high = 200 * (prep["label_mean"] + 0.2 * prep["label_scale"])
    np.testing.assert_allclose(predictions["raw_low"], raw_low)
    np.testing.assert_allclose(predictions["raw_high"], raw_high)
    assert raw_low > raw_high
    ycal = first.data[6].y_raw[3:23]
    scores = np.maximum(np.maximum(raw_high - ycal, ycal - raw_low), 0)
    np.testing.assert_allclose(calibration["score"], scores)
    assert result["calibration"]["rank"] == 19
    q = sorted(scores)[18]
    assert result["calibration"]["q_hat"] == pytest.approx(q)
    np.testing.assert_allclose(predictions["lower"], raw_high - q)
    np.testing.assert_allclose(predictions["upper"], raw_low + q)
    second = make_synthetic_study(tmp_path, monkeypatch, "altered_test")
    second.data[6].y_raw[23:] += 1e5
    changed = run_synthetic(second, "base", task="interval")
    assert changed["calibration"] == result["calibration"]
    changed_predictions = pd.read_csv(second.out / "base/predictions.csv")
    np.testing.assert_allclose(changed_predictions[["lower", "upper"]], predictions[["lower", "upper"]])
    assert changed["metrics"]["picp"] == 0


def test_target_only_uses_no_source_and_matches_reference_update_budget(tmp_path, monkeypatch):
    RecordingSolver.calls = []
    study = make_synthetic_study(tmp_path, monkeypatch, "updates")
    result = run_synthetic(study, "target", variant="TargetOnly")
    fit = result["fits"][0]
    assert result["source_n"] == 0 and result["target_n"] == 2
    assert fit["reference_full_updates"] == 6  # ceil(4/3) batches * 3 epochs
    assert fit["adam_updates"] == 6
    assert fit["options"]["epochs"] == 6
    assert fit["options"]["lambda2"] == 0
    assert fit["options"]["lambda1"] == study.cfg["model"]["lambda1"]
    arrays = RecordingSolver.calls[-1]["arrays"]
    assert len(arrays[0]) == len(arrays[1]) == 0
    prep = json.loads((study.out / "target/preprocessing.json").read_text())
    assert prep["label_mean"] == pytest.approx(0.3)
    assert prep["fit_source_count"] == 0


def test_cutout_sensitivity_counts_and_fits_the_changed_domain(tmp_path):
    study = study_module.Study.__new__(study_module.Study)
    study.out = tmp_path
    study.cfg = {"seeds": [], "components": [], "label_budgets": [], "weight_schemes": [],
                 "threshold_quantiles": [98], "cutout_sensitivity_ms": [20, 25, 30],
                 "landmark_grid": [], "source_fractions": [], "cross_farm_sites": []}
    split = np.repeat(["train", "calibration", "test"], 3)
    wind = np.tile([10., 22., 27.], 3)
    domain = wind > 25
    data = SimpleNamespace(split=split, target_weather={"wind_hub": wind, "temperature": np.full(9, 15.)},
                           training_weather={"wind_hub": np.linspace(0, 100, 101),
                                             "temperature": np.linspace(0, 30, 101)},
                           mask=lambda part, which: (split == part) & (domain if which == "target" else ~domain))
    study.data = {6: data}
    calls = []
    study.run = lambda *args, **kwargs: calls.append((args, kwargs))
    study.alias = lambda *args: None
    study.execute()
    rows = json.loads((tmp_path / "threshold_definitions.json").read_text())
    assert [r["n_train_target"] for r in rows] == [2, 1, 0]
    assert [r["n_test"] for r in rows] == [2, 1, 0]
    assert [r["identical_domain_mask_to_25"] for r in rows] == [False, True, False]
    assert len(calls) == 4  # Two tasks for each changed mask; 25m/s is an alias.
    for args, kwargs in calls:
        cutoff = kwargs["extra"]["cutout_ms"]
        np.testing.assert_array_equal(args[3], np.flatnonzero((split == "train") & (wind > cutoff)))

