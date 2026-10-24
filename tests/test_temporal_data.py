"""Chronological exclusion and leakage checks independent of the trainer."""

import numpy as np
import pandas as pd
import pytest

from review.temporal_data import TemporalProtocol, prepare_temporal_frame


def frame(n=240):
    t = np.arange(n, dtype=float)
    result = {"Time(year-month-day h:m:s)": pd.date_range("2019-01-01", periods=n, freq="15min")}
    for height in (10, 30, 50):
        result[f"Wind speed at height of {height} meters (m/s)"] = 3 + t / 100 + height / 100
        result[f"Wind direction at height of {height} meters (˚)"] = (t * 3 + height) % 360
    result.update({
        "Wind speed - at the height of wheel hub  (m/s)": 4 + t / 100,
        "Wind speed - at the height of wheel hub  (˚)": t * 2 % 360,
        "Air temperature  (°C) ": 10 + np.sin(t / 10),
        "Atmosphere (hpa)": np.full(n, 1000.), "Relative humidity (%)": np.full(n, 50.),
        "Power (MW)": 30 + np.sin(t / 5),
    })
    return pd.DataFrame(result)


def protocol(window=4):
    return TemporalProtocol(train_end="2019-01-02", validation_end="2019-01-02 12:00",
                            calibration_end="2019-01-03", test_end="2019-01-04",
                            window_size=window)


def test_heldout_features_labels_cannot_affect_training_or_thresholds():
    data = frame()
    first = prepare_temporal_frame(data, protocol())
    changed = data.copy()
    changed.loc[96:, "Power (MW)"] = 10000
    changed.loc[96:, "Wind speed - at the height of wheel hub  (m/s)"] = 1000
    changed.loc[96:, "Air temperature  (°C) "] = 300
    second = prepare_temporal_frame(changed, protocol())
    assert first.preprocessing == second.preprocessing
    np.testing.assert_array_equal(first.X[first.mask("train")], second.X[second.mask("train")])
    np.testing.assert_array_equal(first.y_raw[first.mask("train")], second.y_raw[second.mask("train")])
    np.testing.assert_array_equal(first.is_extreme[first.mask("train")], second.is_extreme[second.mask("train")])
    assert first.X.shape[1] == 4 * 13


def test_missing_inputs_remain_missing_and_do_not_copy_next_row():
    data = frame()
    speed = "Wind speed at height of 10 meters (m/s)"
    data.loc[205, speed] = -99
    first = prepare_temporal_frame(data, protocol())
    changed = data.copy()
    changed.loc[206:, speed] += 500
    second = prepare_temporal_frame(changed, protocol())
    a = np.flatnonzero(first.target_row_indices == 206)[0]
    b = np.flatnonzero(second.target_row_indices == 206)[0]
    np.testing.assert_array_equal(first.X[a], second.X[b])
    assert first.input_missing[a, -1, 3]
    assert np.isnan(first.X[a].reshape(4, 13)[-1, 3])
    assert not first.preprocessing["feature_imputation_fitted"]
    assert not first.preprocessing["feature_scaler_fitted"]
    assert not first.preprocessing["label_scaler_fitted"]


def test_duplicate_and_gap_windows_are_removed_before_power_filtering():
    data = frame()
    data.loc[20, data.columns[0]] = data.loc[19, data.columns[0]]
    data = data.drop(index=30).reset_index(drop=True)
    result = prepare_temporal_frame(data, protocol())
    assert result.statistics["ambiguous_duplicate_timestamp_rows_removed"] == 2
    assert 19 not in result.input_row_indices
    assert 20 not in result.input_row_indices
    for input_rows, target_row in zip(result.input_row_indices, result.target_row_indices):
        stamps = pd.to_datetime(data.iloc[np.r_[input_rows, target_row], 0]).to_numpy()
        assert (np.diff(stamps) == np.timedelta64(15, "m")).all()
    assert result.statistics["by_split"]["train"]["excluded_first_reason"]["non_contiguous"] > 0


def test_hour24_normalization_sorting_and_raw_row_provenance():
    data = frame()
    data[data.columns[0]] = data.iloc[:, 0].astype(str)
    data.loc[96, data.columns[0]] = "2019-01-01 24:00:00"
    shuffled = data.sample(frac=1, random_state=4).reset_index(drop=True)
    result = prepare_temporal_frame(shuffled, protocol())
    assert result.statistics["normalized_hour24_rows"] == 1
    assert (np.diff(result.timestamps) > np.timedelta64(0, "ns")).all()
    for i, raw in enumerate(result.target_row_indices):
        assert result.y_raw[i] == shuffled.iloc[raw]["Power (MW)"]


def test_power_policy_and_missing_weather_domains_are_predeclared():
    data = frame()
    data.loc[200, "Power (MW)"] = -0.9
    data.loc[201, "Power (MW)"] = 999
    data.loc[202, "Power (MW)"] = -99
    data.loc[203, "Power (MW)"] = np.nan
    data.loc[204, "Air temperature  (°C) "] = np.nan
    weather = [c for c in data if c not in (data.columns[0], "Power (MW)")]
    data.loc[205, weather] = 0
    result = prepare_temporal_frame(data, protocol())
    assert 200 in result.target_row_indices and 201 in result.target_row_indices
    for row in (202, 203, 204, 205):
        assert row not in result.target_row_indices
    assert result.statistics["all_meteorological_zero_rows"] == 1
    assert result.statistics["by_split"]["test"]["excluded_first_reason"]["invalid_power"] == 2
    assert result.statistics["by_split"]["test"]["excluded_first_reason"]["missing_domain_weather"] == 2


def test_missing_train_weather_cannot_silently_disable_temperature_rule():
    data = frame()
    data.loc[:95, "Air temperature  (°C) "] = np.nan
    with pytest.raises(ValueError, match="valid hub wind and temperature"):
        prepare_temporal_frame(data, protocol())


def test_boundaries_are_disjoint_and_source_target_partition_each_split():
    result = prepare_temporal_frame(frame(), protocol())
    boundaries = protocol().boundaries()
    for i, name in enumerate(("train", "validation", "calibration", "test")):
        mask = result.mask(name)
        assert ((result.timestamps[mask] >= boundaries[i]) &
                (result.timestamps[mask] < boundaries[i + 1])).all()
        assert np.array_equal(result.mask(name, "source") | result.mask(name, "target"), mask)
        assert not (result.mask(name, "source") & result.mask(name, "target")).any()
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_temporal_frame(frame(), TemporalProtocol(train_end="2021-01-02"))
