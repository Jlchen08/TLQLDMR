"""Immutable caches for validation-selected follow-up experiments.

The default time boundaries are explicitly the frozen v1 study boundaries, not
the temporal loader's illustrative defaults. Optional power history is a common
input track for every compared model. Its observed past powers are additional
information: a 128-supervised-target-label budget does not mean that only 128
target power measurements are available in this track.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.temporal_data import TemporalFarmData, TemporalProtocol, _timestamps, load_temporal_farm


REFERENCE_MANIFEST = ROOT / "review/replied_comment_runs/chronological_v1/manifest.json"
DEFAULT_CACHE = ROOT / "review/selection_cache"
YEAR_PROTOCOL = {
    "train_start": "2019-01-01", "train_end": "2020-01-01",
    "validation_end": "2020-04-01", "calibration_end": "2020-07-01", "test_end": "2021-01-01",
    "window_size": 24, "frequency_minutes": 15,
}
CACHE_VERSION = 1
ARRAY_FIELDS = ("X", "y_raw", "timestamps", "split", "is_extreme", "target_row_indices",
                "input_row_indices", "input_missing")
MAPPING_FIELDS = ("event_masks", "target_weather", "training_weather")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def append_past_power(data: TemporalFarmData, raw_frame: pd.DataFrame,
                      capacity_mw: float) -> TemporalFarmData:
    """Append only the original powers at the existing strictly past input rows.

    This function does not change data/raw_frame, fit statistics, fill missing
    powers, clip power values, or add outcome-time power as an input.
    """
    if not np.isfinite(capacity_mw) or capacity_mw <= 0:
        raise ValueError("Nominal capacity must be finite and positive.")
    if "past_power_per_unit" in data.feature_names:
        raise ValueError("Power history has already been appended.")
    normalized = {"".join(str(c).lower().split()): c for c in raw_frame}
    try:
        power_col = normalized["power(mw)"]
        time_col = normalized["time(year-month-dayh:m:s)"]
    except KeyError as error:
        raise ValueError("Raw data must provide power and original timestamps.") from error
    if len(raw_frame) != data.statistics["original_rows"]:
        raise ValueError("Raw frame length differs from the frozen loader's input.")
    protocol = data.preprocessing["protocol"]
    window = int(protocol["window_size"])
    frequency = int(protocol["frequency_minutes"])
    if data.input_row_indices.shape != (len(data.y_raw), window):
        raise ValueError("Unexpected input-row index shape.")
    parsed, _ = _timestamps(raw_frame[time_col])
    input_times = parsed.to_numpy(dtype="datetime64[ns]")[data.input_row_indices]
    expected = data.timestamps[:, None] - np.arange(window, 0, -1)[None, :] * np.timedelta64(frequency, "m")
    if not np.array_equal(input_times, expected):
        raise ValueError("Power-history rows are not the exact preceding chronological inputs.")
    power = pd.to_numeric(raw_frame[power_col], errors="coerce").to_numpy(dtype=float, copy=True)
    power[(~np.isfinite(power)) | (power == -99)] = np.nan
    history = (power[data.input_row_indices] / capacity_mw).astype(np.float32)
    base = data.X.reshape(len(data.y_raw), window, len(data.feature_names))
    combined = np.concatenate((base, history[:, :, None]), axis=2)
    missing = np.concatenate((data.input_missing, ~np.isfinite(history[:, :, None])), axis=2)
    prep, stats, provenance = copy.deepcopy(data.preprocessing), copy.deepcopy(data.statistics), copy.deepcopy(data.provenance)
    names = tuple(data.feature_names) + ("past_power_per_unit",)
    prep["feature_names"] = list(names)
    prep["power_history"] = {
        "included": True, "nominal_capacity_mw": capacity_mw,
        "source": "unaltered original power at each existing strictly past input row, divided by nominal capacity",
        "missing_policy": "-99 and nonfinite become NaN; no filling or clipping; retain other negative and above-capacity observations",
        "availability_assumption": "past power measurements are available at each forecast origin; this is additional information beyond supervised target fitting labels",
        "fit_statistics": "none; all downstream imputation/scaling must use actual model-fit windows only",
    }
    stats["windows_with_missing_inputs"] = int(missing.any(axis=(1, 2)).sum())
    stats["windows_with_missing_past_power"] = int((~np.isfinite(history)).any(axis=1).sum())
    stats["past_power_available_input_rows_by_split"] = {}
    for split in ("train", "validation", "calibration", "test"):
        used = np.unique(data.input_row_indices[data.split == split])
        stats["past_power_available_input_rows_by_split"][split] = {
            "distinct_input_rows": len(used), "finite_observations": int(np.isfinite(power[used]).sum()),
        }
    provenance["input_track"] = "meteorology_and_strictly_past_power"
    return replace(data, X=np.ascontiguousarray(combined.reshape(len(data.y_raw), -1)),
                   input_missing=missing, feature_names=names, preprocessing=prep,
                   statistics=stats, provenance=provenance)


def _fingerprint(site, path, protocol, include_past_power, reference_manifest):
    reference = json.loads(Path(reference_manifest).read_text())
    loader_path = ROOT / "review/temporal_data.py"
    loader_hash = digest(loader_path)
    expected_loader = reference["source_hashes"]["review/temporal_data.py"]
    if loader_hash != expected_loader:
        raise ValueError("Frozen temporal loader hash changed; do not silently change follow-up data processing.")
    data_hash = digest(path)
    relative = str(path.relative_to(ROOT))
    if reference["data_hashes"].get(relative) != data_hash:
        raise ValueError("Original data differ from the frozen study; declare a new data revision before caching.")
    specification = {
        "cache_version": CACHE_VERSION, "site": int(site), "original_file": str(path.resolve()),
        "original_sha256": data_hash, "temporal_loader_sha256": loader_hash,
        "selection_helper_sha256": digest(Path(__file__)), "protocol": asdict(protocol),
        "include_past_power": bool(include_past_power),
    }
    key = hashlib.sha256(json.dumps(specification, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return key, specification


def _write_cache(folder, data, specification):
    arrays = {name: getattr(data, name) for name in ARRAY_FIELDS}
    mappings = {}
    for field in MAPPING_FIELDS:
        mappings[field] = list(getattr(data, field))
        arrays.update({f"{field}__{name}": values for name, values in getattr(data, field).items()})
    files = {}
    for name, values in arrays.items():
        path = folder / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        files[name] = {"file": path.name, "sha256": digest(path), "shape": list(values.shape), "dtype": str(values.dtype)}
    metadata = {"specification": specification, "arrays": files, "mapping_fields": mappings,
                "feature_names": data.feature_names, "preprocessing": data.preprocessing,
                "statistics": data.statistics, "provenance": data.provenance}
    (folder / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _read_cache(folder, specification, mmap_mode):
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata["specification"] != specification:
        raise ValueError(f"Cache specification mismatch: {folder}")
    values = {}
    for name, entry in metadata["arrays"].items():
        path = folder / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Cache array hash mismatch: {path}")
        array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
            raise ValueError(f"Cache array schema mismatch: {path}")
        array.flags.writeable = False
        values[name] = array
    fields = {name: values[name] for name in ARRAY_FIELDS}
    fields.update({field: {key: values[f"{field}__{key}"] for key in keys}
                   for field, keys in metadata["mapping_fields"].items()})
    provenance = copy.deepcopy(metadata["provenance"])
    provenance["selection_cache"] = {"directory": str(folder.resolve()), "specification": specification}
    return TemporalFarmData(**fields, feature_names=tuple(metadata["feature_names"]),
                            preprocessing=metadata["preprocessing"], statistics=metadata["statistics"], provenance=provenance)


def load_selection_farm(site: int, *, cache_dir: str | Path = DEFAULT_CACHE,
                        include_past_power: bool = False, mmap_mode="r",
                        protocol: TemporalProtocol | None = None,
                        reference_manifest: str | Path = REFERENCE_MANIFEST) -> TemporalFarmData:
    """Load/create one content-addressed immutable cache; no model is fitted."""
    if int(site) != site or int(site) not in range(1, 7):
        raise ValueError("Site must be an integer from 1 through 6.")
    if mmap_mode not in (None, "r"):
        raise ValueError("Only read-only mmap_mode='r' or an in-memory copy is supported.")
    protocol = protocol or TemporalProtocol(**YEAR_PROTOCOL)
    protocol.boundaries()
    matches = list((ROOT / "data/wind_farm_data/data_original/wind_farms").glob(f"Wind farm site {int(site)} (Nominal capacity-*MW).xlsx"))
    if len(matches) != 1:
        raise ValueError(f"Expected one original workbook for site {site}, got {len(matches)}.")
    path = matches[0]
    key, specification = _fingerprint(site, path, protocol, include_past_power, reference_manifest)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    folder = cache_dir / f"farm{int(site)}_{key}"
    if not folder.exists():
        data = load_temporal_farm(path, protocol)
        if include_past_power:
            capacity = float(re.search(r"(\d+)MW", path.name).group(1))
            data = append_past_power(data, pd.read_excel(path), capacity)
        else:
            data.provenance = {**data.provenance, "input_track": "meteorology_only"}
        # Build in a private staging directory; never replace an existing cache.
        with TemporaryDirectory(prefix=f".building-farm{int(site)}-", dir=cache_dir) as temporary:
            staging = Path(temporary)
            _write_cache(staging, data, specification)
            try:
                os.rename(staging, folder)
            except OSError:
                if not folder.exists():
                    raise
                # Another process may have completed this exact immutable key.
                # The full specification and array hashes are checked below.
    return _read_cache(folder, specification, mmap_mode)


def load_all_selection_data(**kwargs) -> dict[int, TemporalFarmData]:
    """Return all six farms with the same feature track and temporal protocol."""
    result = {site: load_selection_farm(site, **kwargs) for site in range(1, 7)}
    schemas = {data.feature_names for data in result.values()}
    if len(schemas) != 1:
        raise ValueError("Cross-farm feature schemas differ.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--site", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--include-past-power", action="store_true")
    args = parser.parse_args()
    for site in args.site:
        data = load_selection_farm(site, cache_dir=args.cache_dir, include_past_power=args.include_past_power)
        print(json.dumps({"site": site, "shape": list(data.X.shape), "feature_names": data.feature_names,
                          "cache_directory": data.provenance["selection_cache"]["directory"],
                          "split_counts": {split: int(data.mask(split).sum()) for split in
                                           ("train", "validation", "calibration", "test")}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
