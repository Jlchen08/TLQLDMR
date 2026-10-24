"""Versioned feature extension for the selection-v2 experiments.

The frozen selection-v1 loader deliberately exposes only the common wind and
temperature covariates.  Selection v2 keeps that loader and all of its caches
immutable, then appends the two other meteorological measurements that are
present in every supplied workbook: atmospheric pressure and relative
humidity.  No imputation or scaling is performed here.  The runner must fit
those operations on the actual model-fitting rows only.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd

from review.selection_data import ROOT, load_selection_farm
from review.temporal_data import TemporalFarmData


LOADER_VERSION = "selection_v2_common_pressure_humidity_1"
EXTRA_COLUMNS = {
    "pressure_hpa": "Atmosphere (hpa)",
    "relative_humidity_pct": "Relative humidity (%)",
}


def _normal_name(name: object) -> str:
    return "".join(str(name).lower().split())


def append_common_pressure_humidity(data: TemporalFarmData,
                                    raw_frame: pd.DataFrame) -> TemporalFarmData:
    """Append strictly past pressure and humidity values to existing windows.

    ``input_row_indices`` point to original workbook rows, so this operation
    cannot introduce outcome-time or future covariates.  Original ``-99`` and
    nonfinite values, plus rows already identified as complete meteorological
    zero records by the frozen loader, remain missing until fit-only
    preprocessing in the runner.
    """
    if any(name in data.feature_names for name in EXTRA_COLUMNS):
        raise ValueError("Pressure/humidity have already been appended.")
    if len(raw_frame) != data.statistics["original_rows"]:
        raise ValueError("Raw frame length differs from the frozen loader input.")

    available = {_normal_name(column): column for column in raw_frame.columns}
    resolved = {}
    for name, expected in EXTRA_COLUMNS.items():
        key = _normal_name(expected)
        if key not in available:
            raise ValueError(f"Missing common v2 feature column: {expected}")
        resolved[name] = available[key]

    all_zero = np.zeros(len(raw_frame), dtype=bool)
    zero_rows = np.asarray(data.statistics.get("all_meteorological_zero_raw_indices", []), dtype=int)
    if len(zero_rows):
        if zero_rows.min() < 0 or zero_rows.max() >= len(raw_frame):
            raise ValueError("Frozen all-zero row index lies outside the workbook.")
        all_zero[zero_rows] = True

    histories, missing_histories, invalid_counts = [], [], {}
    for name in EXTRA_COLUMNS:
        values = pd.to_numeric(raw_frame[resolved[name]], errors="coerce").to_numpy(dtype=float, copy=True)
        missing = (~np.isfinite(values)) | (values == -99) | all_zero
        values[missing] = np.nan
        history = values[data.input_row_indices]
        histories.append(history)
        missing_histories.append(~np.isfinite(history))
        invalid_counts[name] = int(missing.sum())

    window = int(data.preprocessing["protocol"]["window_size"])
    base = np.asarray(data.X).reshape(len(data.y_raw), window, len(data.feature_names))
    extra = np.stack(histories, axis=2)
    combined = np.concatenate((base, extra), axis=2).astype(np.float32, copy=False)
    extra_missing = np.stack(missing_histories, axis=2)
    missing = np.concatenate((np.asarray(data.input_missing), extra_missing), axis=2)

    names = tuple(data.feature_names) + tuple(EXTRA_COLUMNS)
    preprocessing = copy.deepcopy(data.preprocessing)
    preprocessing["feature_names"] = list(names)
    preprocessing["selection_v2_feature_extension"] = {
        "version": LOADER_VERSION,
        "features": list(EXTRA_COLUMNS),
        "source": "unaltered values at the frozen strictly preceding input rows",
        "missing_policy": "-99, nonfinite, and frozen complete-meteorological-zero rows become NaN",
        "fit_requirement": "imputation and scaling are fitted later on actual 2019 model-fit rows only",
    }
    statistics = copy.deepcopy(data.statistics)
    statistics["invalid_or_missing_by_variable"].update(invalid_counts)
    statistics["windows_with_missing_inputs"] = int(missing.any(axis=(1, 2)).sum())
    provenance = copy.deepcopy(data.provenance)
    provenance["input_track"] = "common_meteorology_15_variables_pressure_humidity"
    provenance["selection_v2_loader"] = LOADER_VERSION

    return replace(
        data,
        X=np.ascontiguousarray(combined.reshape(len(data.y_raw), -1)),
        input_missing=np.ascontiguousarray(missing),
        feature_names=names,
        preprocessing=preprocessing,
        statistics=statistics,
        provenance=provenance,
    )


def load_selection_farm_v2(site: int, **kwargs) -> TemporalFarmData:
    """Load one immutable v1 temporal cache and add the versioned v2 inputs."""
    data = load_selection_farm(site, include_past_power=False, **kwargs)
    matches = list((ROOT / "data/wind_farm_data/data_original/wind_farms").glob(
        f"Wind farm site {int(site)} (Nominal capacity-*MW).xlsx"
    ))
    if len(matches) != 1:
        raise ValueError(f"Expected one original workbook for site {site}, got {len(matches)}.")
    return append_common_pressure_humidity(data, pd.read_excel(matches[0]))


def load_all_selection_data_v2(**kwargs) -> dict[int, TemporalFarmData]:
    """Load all farms and require an identical augmented feature schema."""
    result = {site: load_selection_farm_v2(site, **kwargs) for site in range(1, 7)}
    schemas = {data.feature_names for data in result.values()}
    if len(schemas) != 1:
        raise ValueError("Selection-v2 cross-farm feature schemas differ.")
    return result
