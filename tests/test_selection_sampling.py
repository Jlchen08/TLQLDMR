"""Budget allocation, chronological selection, and held-out noninterference."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from review.selection_sampling import _time_uniform_indices, stratified_target_indices


def example():
    times = np.array(["2019-01-01", "2019-01-02", "2019-01-20", "2019-01-31",
                      "2019-02-01", "2019-02-02", "2019-02-10", "2019-02-20", "2019-02-28",
                      "2019-03-01", "2019-03-02", "2019-03-15", "2019-03-31",
                      "2020-02-01", "2020-05-01", "2020-08-01"], dtype="datetime64[ns]")
    # Cell capacities: January cold=4, February hot=5, March wind=4.
    temperature = np.array([-2.] * 4 + [32.] * 5 + [10.] * 4 + [10.] * 3)
    wind = np.array([5.] * 9 + [20.] * 7)
    return SimpleNamespace(
        timestamps=times, split=np.array(["train"] * 13 + ["validation", "calibration", "test"]),
        is_extreme=np.ones(16, dtype=bool), target_weather={"wind_hub": wind, "temperature": temperature},
        target_row_indices=np.arange(100, 116), y_raw=np.arange(16.),
        preprocessing={"protocol": {"train_start": "2019-01-01", "train_end": "2020-01-01"},
                       "domain_thresholds": {"wind_ms": 15., "temperature_low_c": 0., "temperature_high_c": 30., "cutout_ms": 25.}},
    )


def test_exact_largest_remainder_and_cell_coverage():
    ids, audit = stratified_target_indices(example(), 8)
    assert len(ids) == len(np.unique(ids)) == 8
    # Initial 1 each; extras 5 proportional to capacities [3,4,3] -> [2,2,1].
    assert [c["selected"] for c in audit["cells"]] == [3, 3, 2]
    assert sum(c["selected"] for c in audit["cells"]) == 8
    assert all(c["selected"] >= 1 for c in audit["cells"])
    assert not audit["uses_power_label_values"]


def test_future_weather_outcomes_and_all_power_values_cannot_change_selection():
    first = example()
    expected_ids, expected_audit = stratified_target_indices(first, 8)
    altered = deepcopy(first)
    altered.y_raw[:] = -999999
    altered.target_weather["wind_hub"][13:] = np.nan
    altered.target_weather["temperature"][13:] = 1000
    altered.is_extreme[13:] = False
    ids, audit = stratified_target_indices(altered, 8)
    np.testing.assert_array_equal(ids, expected_ids)
    assert audit == expected_audit


def test_exclusive_event_priority_and_cutout_fallback():
    data = example()
    data.target_weather["wind_hub"][:4] = 40  # Cold wins even when also high wind.
    data.target_weather["wind_hub"][4:9] = 40  # Hot wins over high wind.
    data.preprocessing["domain_thresholds"]["wind_ms"] = 50  # March is cut-out only.
    data.target_weather["wind_hub"][9:13] = 30
    ids, audit = stratified_target_indices(data, 8)
    assert [c["exclusive_event"] for c in audit["cells"]] == ["cold", "hot", "wind"]
    assert audit["available_training_target_count"] == 13


def test_actual_time_spacing_and_single_label_midpoint():
    times = np.array(["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-17", "2019-01-31"], dtype="datetime64[ns]")
    # Actual-clock middle is near January 17, unlike the middle row January 3.
    np.testing.assert_array_equal(_time_uniform_indices(np.arange(5), times, 1), [3])
    np.testing.assert_array_equal(_time_uniform_indices(np.arange(5), times, 3), [0, 3, 4])


def test_budget_and_training_time_validation():
    with pytest.raises(ValueError, match="nonempty cells"):
        stratified_target_indices(example(), 2)
    for budget in (0, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            stratified_target_indices(example(), budget)
    data = example()
    data.split[13] = "train"
    with pytest.raises(ValueError, match="outside the declared training interval"):
        stratified_target_indices(data, 8)


def test_saturating_budget_returns_every_training_target_once():
    data = example()
    ids, audit = stratified_target_indices(data, 100)
    np.testing.assert_array_equal(ids, np.arange(13))
    assert audit["requested_budget"] == 100 and audit["effective_budget"] == 13


def test_row_reordering_preserves_chosen_original_observations():
    data = example()
    ids, _ = stratified_target_indices(data, 8)
    order = np.random.default_rng(20).permutation(16)
    shuffled = deepcopy(data)
    for field in ("timestamps", "split", "is_extreme", "target_row_indices", "y_raw"):
        setattr(shuffled, field, getattr(data, field)[order])
    shuffled.target_weather = {key: values[order] for key, values in data.target_weather.items()}
    other, _ = stratified_target_indices(shuffled, 8)
    np.testing.assert_array_equal(data.target_row_indices[ids], shuffled.target_row_indices[other])
