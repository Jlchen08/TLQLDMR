"""Traceable chronological windows from the supplied *original* wind workbooks.

No filling, scaling, interpolation, clipping, or target-error filtering is used.
Domain thresholds use the declared training period. Input history may cross
a split boundary; all 24 input timestamps precede the prediction timestamp.
These are observational chronological splits, not an exchangeability guarantee
for conformal intervals or a proof that the upstream original data are causal.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalProtocol:
    train_start: str = "2019-01-01"
    train_end: str = "2019-10-01"
    validation_end: str = "2020-01-01"
    calibration_end: str = "2020-04-01"
    test_end: str = "2021-01-01"
    window_size: int = 24
    frequency_minutes: int = 15
    wind_quantile: float = 98.0
    temperature_low_quantile: float = 2.0
    temperature_high_quantile: float = 98.0
    cutout_ms: float = 25.0
    use_wind: bool = True
    use_temperature: bool = True
    use_cutout: bool = True

    def boundaries(self):
        values = pd.to_datetime([self.train_start, self.train_end, self.validation_end,
                                 self.calibration_end, self.test_end], format="mixed")
        if not all(values[i] < values[i + 1] for i in range(4)):
            raise ValueError("Temporal split boundaries must be strictly increasing.")
        if self.window_size < 1 or self.frequency_minutes < 1:
            raise ValueError("Window size and sampling interval must be positive.")
        if not (0 < self.wind_quantile < 100 and
                0 < self.temperature_low_quantile < self.temperature_high_quantile < 100):
            raise ValueError("Invalid meteorological quantiles.")
        if not np.isfinite(self.cutout_ms) or self.cutout_ms <= 0:
            raise ValueError("Cut-out wind threshold must be positive and finite.")
        return values.to_numpy(dtype="datetime64[ns]")


@dataclass
class TemporalFarmData:
    # Missing covariates remain NaN; fit imputation/scalers on actual model-fit
    # windows in the runner, especially for source-only/few-shot experiments.
    X: np.ndarray
    y_raw: np.ndarray
    timestamps: np.ndarray
    split: np.ndarray
    is_extreme: np.ndarray
    event_masks: dict[str, np.ndarray]
    target_row_indices: np.ndarray
    input_row_indices: np.ndarray
    input_missing: np.ndarray
    feature_names: tuple[str, ...]
    target_weather: dict[str, np.ndarray]
    training_weather: dict[str, np.ndarray]
    preprocessing: dict
    statistics: dict
    provenance: dict

    def mask(self, split: str, domain: str | None = None) -> np.ndarray:
        if split not in ("train", "validation", "calibration", "test"):
            raise ValueError(f"Unknown split: {split}")
        result = self.split == split
        if domain == "source":
            result &= ~self.is_extreme
        elif domain == "target":
            result &= self.is_extreme
        elif domain is not None:
            raise ValueError("Domain must be 'source', 'target', or None.")
        return result


def _normal_name(name):
    return "".join(str(name).lower().split())


def _schema(df):
    available = {_normal_name(c): c for c in df.columns}
    names = {
        "time": "Time(year-month-day h:m:s)",
        "power": "Power (MW)",
        "temperature": "Air temperature (°C)",
        "wind_hub": "Wind speed - at the height of wheel hub (m/s)",
        "direction_hub": "Wind speed - at the height of wheel hub (˚)",
    }
    for height in (10, 30, 50):
        names[f"wind_{height}"] = f"Wind speed at height of {height} meters (m/s)"
        names[f"direction_{height}"] = f"Wind direction at height of {height} meters (˚)"
    missing = [v for v in names.values() if _normal_name(v) not in available]
    if missing:
        raise ValueError(f"Missing required common feature columns: {missing}")
    return {k: available[_normal_name(v)] for k, v in names.items()}


def _timestamps(values):
    strings = values.astype(str).str.strip()
    hour24 = strings.str.contains(r"\s24:00(?::00(?:\.0+)?)?$", regex=True)
    corrected = strings.str.replace(r"\s24:00(?::00(?:\.0+)?)?$", " 00:00:00", regex=True)
    parsed = pd.to_datetime(corrected, errors="coerce", format="mixed")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError("Supply naive local timestamps with a documented time basis.")
    parsed = parsed + pd.to_timedelta(hour24.astype(int), unit="D")
    return parsed, int(hour24.sum())


def _split_names(times, boundaries):
    result = np.full(len(times), "outside", dtype="U11")
    for i, name in enumerate(("train", "validation", "calibration", "test")):
        result[(times >= boundaries[i]) & (times < boundaries[i + 1])] = name
    return result


def prepare_temporal_frame(df: pd.DataFrame, protocol: TemporalProtocol | None = None,
                           *, provenance: dict | None = None) -> TemporalFarmData:
    """Prepare raw windows without changing the frame or fitting any scaler.

    The common 13 features are four wind speeds, temperature, and sine/cosine
    for four directions. Pressure/humidity are used only to recognize complete
    meteorological zero records when those columns exist. Feature-missing masks
    are returned separately and are not appended as model inputs.
    """
    protocol = protocol or TemporalProtocol()
    boundaries = protocol.boundaries()
    columns = _schema(df)
    parsed, hour24_count = _timestamps(df[columns["time"]])
    invalid_time = parsed.isna().to_numpy()
    duplicate_time = (parsed.notna() & parsed.duplicated(keep=False)).to_numpy()
    kept = np.flatnonzero(~invalid_time & ~duplicate_time)
    order = np.argsort(parsed.iloc[kept].to_numpy(), kind="stable")
    raw_indices = kept[order]
    frame = df.iloc[raw_indices].reset_index(drop=True)
    times = parsed.iloc[raw_indices].to_numpy(dtype="datetime64[ns]")
    if len(frame) <= protocol.window_size:
        raise ValueError("Too few unambiguous timestamps for the requested window.")

    weather_columns = [columns[key] for key in columns if key not in ("time", "power")]
    for candidate in ("Atmosphere (hpa)", "Relative humidity (%)"):
        weather_columns.extend(c for c in frame if _normal_name(c) == _normal_name(candidate))
    weather = frame[weather_columns].apply(pd.to_numeric, errors="coerce")
    all_weather_zero = weather.eq(0).all(axis=1).to_numpy()
    raw_values = {}
    invalid_counts = {}
    for key, col in columns.items():
        if key == "time":
            continue
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float, copy=True)
        missing = ~np.isfinite(values) | (values == -99)
        if key != "power":
            missing |= all_weather_zero
        if key.startswith("wind_"):
            missing |= values < 0
        if key.startswith("direction_"):
            missing |= (values < 0) | (values > 360)
        values[missing] = np.nan
        raw_values[key] = values
        invalid_counts[key] = int(missing.sum())

    train_time = (times >= boundaries[0]) & (times < boundaries[1])
    hub, temperature = raw_values["wind_hub"], raw_values["temperature"]
    train_wind = hub[train_time & np.isfinite(hub)]
    train_temperature = temperature[train_time & np.isfinite(temperature)]
    if not len(train_wind) or not len(train_temperature):
        raise ValueError("Training requires valid hub wind and temperature for domain thresholds.")
    thresholds = {
        "wind_ms": float(np.percentile(train_wind, protocol.wind_quantile)),
        "temperature_low_c": float(np.percentile(train_temperature, protocol.temperature_low_quantile)),
        "temperature_high_c": float(np.percentile(train_temperature, protocol.temperature_high_quantile)),
        "cutout_ms": float(protocol.cutout_ms),
        "training_valid_wind_count": len(train_wind),
        "training_valid_temperature_count": len(train_temperature),
        "fit_scope": "valid original meteorological observations in the training time interval only",
    }
    events = {
        "wind": (hub > thresholds["wind_ms"]) & protocol.use_wind,
        "temperature": ((temperature < thresholds["temperature_low_c"]) |
                        (temperature > thresholds["temperature_high_c"])) & protocol.use_temperature,
        "cutout": (hub > protocol.cutout_ms) & protocol.use_cutout,
    }
    domain_valid = np.isfinite(hub) & np.isfinite(temperature)
    extreme = events["wind"] | events["temperature"] | events["cutout"]

    features, names = [], []
    for key in ("wind_hub", "wind_50", "wind_30", "wind_10", "temperature"):
        features.append(raw_values[key])
        names.append(key)
    for key in ("direction_hub", "direction_50", "direction_30", "direction_10"):
        radians = np.deg2rad(raw_values[key])
        features.extend((np.sin(radians), np.cos(radians)))
        names.extend((f"sin_{key}", f"cos_{key}"))
    x_raw = np.column_stack(features)
    missing_features = ~np.isfinite(x_raw)
    window = protocol.window_size
    target_positions = np.arange(window, len(frame))
    bad_interval = np.diff(times) != np.timedelta64(protocol.frequency_minutes, "m")
    bad_prefix = np.r_[0, np.cumsum(bad_interval)]
    contiguous = bad_prefix[window:] - bad_prefix[:-window] == 0
    label_valid = np.isfinite(raw_values["power"][target_positions])
    valid_domain = domain_valid[target_positions]
    splits = _split_names(times[target_positions], boundaries)
    accepted = contiguous & label_valid & valid_domain & (splits != "outside")
    train_accepted = accepted & (splits == "train")
    if not train_accepted.any():
        raise ValueError("No valid contiguous training windows remain after exclusions.")
    accepted_positions = target_positions[accepted]
    input_positions = accepted_positions[:, None] - window + np.arange(window)
    y = raw_values["power"][accepted_positions]

    rejection = np.full(len(target_positions), "accepted", dtype="U32")
    for reason, bad in (("non_contiguous", ~contiguous), ("invalid_power", ~label_valid),
                        ("missing_domain_weather", ~valid_domain), ("outside_period", splits == "outside")):
        rejection[(rejection == "accepted") & bad] = reason
    by_split = {}
    for split in ("train", "validation", "calibration", "test"):
        candidate = splits == split
        selected = candidate & accepted
        by_split[split] = {
            "candidate_windows": int(candidate.sum()), "accepted_windows": int(selected.sum()),
            "source_windows": int((selected & ~extreme[target_positions]).sum()),
            "target_windows": int((selected & extreme[target_positions]).sum()),
            "excluded_first_reason": {r: int((candidate & (rejection == r)).sum()) for r in
                                      ("non_contiguous", "invalid_power", "missing_domain_weather")},
        }
    stats = {
        "original_rows": len(df), "normalized_hour24_rows": hour24_count,
        "invalid_timestamp_rows_removed": int(invalid_time.sum()),
        "invalid_timestamp_raw_indices": np.flatnonzero(invalid_time).tolist(),
        "ambiguous_duplicate_timestamp_rows_removed": int(duplicate_time.sum()),
        "ambiguous_duplicate_timestamp_raw_indices": np.flatnonzero(duplicate_time).tolist(),
        "retained_sorted_rows": len(frame), "all_meteorological_zero_rows": int(all_weather_zero.sum()),
        "all_meteorological_zero_raw_indices": raw_indices[all_weather_zero].tolist(),
        "invalid_or_missing_by_variable": invalid_counts,
        "nonconforming_adjacent_intervals": int(bad_interval.sum()),
        "required_interval_minutes": protocol.frequency_minutes,
        "candidate_windows": len(target_positions), "accepted_windows": int(accepted.sum()),
        "windows_with_missing_inputs": int(missing_features[input_positions].any(axis=(1, 2)).sum()),
        "excluded_target_raw_indices_by_first_reason": {
            reason: raw_indices[target_positions[rejection == reason]].tolist()
            for reason in ("non_contiguous", "invalid_power", "missing_domain_weather", "outside_period")
        },
        "by_split": by_split,
    }
    prep = {
        "protocol": asdict(protocol), "domain_thresholds": thresholds,
        "feature_names": names, "feature_imputation_fitted": False,
        "feature_scaler_fitted": False, "label_scaler_fitted": False,
        "downstream_fit_requirement": "fit imputation, feature scaling, and label scaling on the actual model-fitting training windows only; exclude unused few-shot target labels",
        "missing_policy": "-99/nonfinite, complete meteorological zero rows, negative wind speeds, and directions outside [0,360] become NaN; no filling in loader",
        "label_policy": "exclude nonfinite and -99 only; retain all other powers, including negative and above nominal capacity",
        "window_policy": "24 preceding rows by default; all input-to-target intervals must be exactly the configured 15 minutes; no future filling",
    }
    return TemporalFarmData(
        X=np.ascontiguousarray(x_raw[input_positions].reshape(len(y), -1), dtype=np.float32),
        y_raw=y.copy(),
        timestamps=times[accepted_positions], split=splits[accepted],
        is_extreme=extreme[accepted_positions],
        event_masks={key: value[accepted_positions] for key, value in events.items()},
        target_row_indices=raw_indices[accepted_positions], input_row_indices=raw_indices[input_positions],
        input_missing=missing_features[input_positions], feature_names=tuple(names),
        target_weather={"wind_hub": hub[accepted_positions], "temperature": temperature[accepted_positions]},
        training_weather={"wind_hub": train_wind.copy(), "temperature": train_temperature.copy()},
        preprocessing=prep, statistics=stats, provenance=provenance or {},
    )


def load_temporal_farm(path: str | Path, protocol: TemporalProtocol | None = None) -> TemporalFarmData:
    """Read one original workbook and retain its absolute path and SHA-256."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return prepare_temporal_frame(pd.read_excel(path), protocol, provenance={
        "data_file": str(path), "data_sha256": digest.hexdigest(),
        "input_kind": "caller-supplied original workbook; never overwritten",
    })
