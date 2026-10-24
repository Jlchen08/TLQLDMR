"""Checks on the data separation and calibration used by the repair diagnostic."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def runner():
    path = Path(__file__).resolve().parents[1] / "review/run_corrected_validation.py"
    spec = importlib.util.spec_from_file_location("corrected_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_uses_finite_sample_rank(runner):
    y = np.arange(20.0)
    q, scores, rank = runner.conformal_radius(y, np.zeros(20), np.zeros(20), 0.1)
    assert rank == 19
    assert q == 18
    assert np.array_equal(scores, y)
    with pytest.raises(ValueError, match="Too few"):
        runner.conformal_radius(y[:2], np.zeros(2), np.zeros(2), 0.05)


def test_test_and_calibration_labels_cannot_change_training_preprocessing(runner):
    raw_x = np.arange(400, dtype=float).reshape(100, 4)
    raw_y = np.sin(np.arange(100.0))
    source_ids, fit_ids = np.arange(20), np.arange(20, 40)
    test_ids, cal_ids = np.arange(60, 70), np.arange(50, 60)
    window = 3
    original = runner.prepare_arrays(raw_x, raw_y, window, source_ids, fit_ids, test_ids, cal_ids)
    altered_y = raw_y.copy()
    altered_y[test_ids + window] += 10000
    altered_y[cal_ids + window] -= 10000
    altered = runner.prepare_arrays(raw_x, altered_y, window, source_ids, fit_ids, test_ids, cal_ids)
    assert original[7] == altered[7]
    for original_array, altered_array in zip(original[:6], altered[:6]):
        np.testing.assert_array_equal(original_array, altered_array)


def test_heldout_only_features_do_not_change_training_statistics(runner):
    raw_x = np.arange(400, dtype=float).reshape(100, 4)
    raw_y = np.sin(np.arange(100.0))
    source_ids, fit_ids = np.arange(20), np.arange(20, 40)
    test_ids, cal_ids = np.arange(60, 70), np.arange(50, 60)
    original = runner.prepare_arrays(raw_x, raw_y, 3, source_ids, fit_ids, test_ids, cal_ids)
    altered_x = raw_x.copy()
    altered_x[50:] += 10000
    altered = runner.prepare_arrays(altered_x, raw_y, 3, source_ids, fit_ids, test_ids, cal_ids)
    assert original[7] == altered[7]
    for original_array, altered_array in zip(original[:4], altered[:4]):
        np.testing.assert_array_equal(original_array, altered_array)
    assert not np.array_equal(original[4], altered[4])
