#!/usr/bin/env python3
"""Frozen paired temporal-block inference for the selected point forecasts.

The estimand, bootstrap interval, and sign-flip test all use the same
per-observation squared-error difference.  Algorithmic landmark seeds are
averaged within an observation and are never treated as independent samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "review/selection_runs/selection_v1"
DEFAULT_OUT = ROOT / "review/inference_runs/paired_point_v3"
COMPARATORS = ["WeightedSVQR", "TargetSVQR", "HistGBR"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")


def protocol():
    return {
        "version": "paired_point_v3",
        "test_status": "exploratory follow-up on previously inspected 2020 Q3-Q4",
        "scope": "Full versus WeightedSVQR, TargetSVQR, and HistGBR on sites 1--6",
        "unit": "non-overlapping seven-day calendar blocks anchored at the calendar date of the first retained test observation",
        "estimand": (
            "mean paired squared-error difference in MW^2 after averaging algorithmic "
            "seeds within each forecast; Full minus comparator"
        ),
        "bootstrap": (
            "10,000 equal-probability block resamples; each replicate is the sum of "
            "sampled block losses divided by the sum of sampled block counts"
        ),
        "test": (
            "99,999 two-sided block sign flips of block loss sums divided by the fixed "
            "total observation count"
        ),
        "multiplicity": "Holm adjustment over all 18 site-comparator tests",
        "random_seed": 20261022,
        "interpretation": (
            "negative effects favor Full; temporal blocks are an analysis assumption, "
            "not independent-event or new-dataset replication"
        ),
    }


def prediction_files():
    files = []
    for site in range(1, 7):
        task = f"cross_site{site}"
        for method in ["Full", *COMPARATORS]:
            seeds = [0] if method == "HistGBR" else [0, 1, 2]
            files += [REFERENCE / task / method / f"final_seed{s}" /
                      "predictions.csv" for s in seeds]
    return files


def load_errors(site, method):
    task = f"cross_site{site}"
    seeds = [0] if method == "HistGBR" else [0, 1, 2]
    frames = [pd.read_csv(REFERENCE / task / method / f"final_seed{s}" /
                          "predictions.csv") for s in seeds]
    ids = frames[0].window_index.to_numpy()
    times = pd.to_datetime(frames[0].time).to_numpy()
    y = frames[0].y_true.to_numpy(float)
    for frame in frames[1:]:
        if not np.array_equal(ids, frame.window_index.to_numpy()):
            raise ValueError("Seed prediction identifiers differ")
        if not np.array_equal(times, pd.to_datetime(frame.time).to_numpy()):
            raise ValueError("Seed prediction times differ")
        if not np.allclose(y, frame.y_true.to_numpy(float), rtol=0, atol=0):
            raise ValueError("Seed labels differ")
    errors = np.mean([(frame.prediction.to_numpy(float)-y)**2
                      for frame in frames], axis=0)
    return ids, times, y, errors


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def analyse():
    rng = np.random.default_rng(20261022)
    reports = []
    for site in range(1, 7):
        ids, times, y, full = load_errors(site, "Full")
        first_day = times.min().astype("datetime64[D]")
        block = ((times.astype("datetime64[D]")-first_day).astype(int)//7).astype(int)
        block_values = np.unique(block)
        indices = [np.flatnonzero(block == value) for value in block_values]
        for comparator in COMPARATORS:
            ids2, times2, y2, other = load_errors(site, comparator)
            if not (np.array_equal(ids, ids2) and np.array_equal(times, times2)
                    and np.allclose(y, y2, rtol=0, atol=0)):
                raise ValueError("Comparator rows are not aligned")
            difference = full-other
            block_sums = np.array([difference[ii].sum() for ii in indices])
            block_counts = np.array([len(ii) for ii in indices])
            effect = float(difference.mean())
            bootstrap = np.empty(10000)
            for iteration in range(len(bootstrap)):
                draw = rng.integers(0, len(indices), len(indices))
                bootstrap[iteration] = (block_sums[draw].sum() /
                                        block_counts[draw].sum())
            observed = abs(effect)
            tolerance = 1e-12 * max(1.0, observed)
            exceed = 0
            remaining = 99999
            chunk = 10000
            while remaining:
                count = min(chunk, remaining)
                signs = rng.choice(np.array([-1.0, 1.0]),
                                   size=(count, len(indices)))
                permuted = np.abs((signs @ block_sums) / block_counts.sum())
                exceed += int(np.sum(permuted >= observed-tolerance))
                remaining -= count
            pvalue = float((1+exceed)/100000)
            reports.append({
                "site": site, "comparator": comparator, "n": int(len(y)),
                "calendar_blocks": int(len(indices)),
                "block_size_min": int(block_counts.min()),
                "block_size_max": int(block_counts.max()),
                "full_minus_comparator_mse_mw2": effect,
                "cluster_bootstrap_95_ci_mw2": [
                    float(np.quantile(bootstrap, 0.025)),
                    float(np.quantile(bootstrap, 0.975)),
                ],
                "block_sign_flip_p_unadjusted": pvalue,
            })
    adjusted = holm([row["block_sign_flip_p_unadjusted"] for row in reports])
    for row, value in zip(reports, adjusted):
        row["block_sign_flip_p_holm_18"] = float(value)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    p = protocol()
    sources = [Path(__file__)]
    inputs = prediction_files() + [REFERENCE / "selection_lock.json"]
    manifest = {
        "status": "frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": p,
        "source_hashes": {str(x.relative_to(ROOT)): digest(x) for x in sources},
        "input_hashes": {str(x.relative_to(ROOT)): digest(x) for x in inputs},
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    if args.output.exists():
        old = json.loads((args.output / "manifest.json").read_text())
        if not args.resume:
            raise FileExistsError("Use --resume for the frozen analysis")
        for key in ["protocol", "source_hashes", "input_hashes"]:
            if old[key] != manifest[key]:
                raise ValueError(f"Frozen {key} changed")
        manifest = old
    else:
        args.output.mkdir(parents=True)
        save(args.output / "protocol.json", p)
        save(args.output / "manifest.json", manifest)
    if args.freeze_only:
        return
    manifest["status"] = "running"
    save(args.output / "manifest.json", manifest)
    try:
        results = analyse()
        save(args.output / "paired_point_inference.json", {
            "interpretation": p["interpretation"], "comparisons": results})
        manifest["status"] = "complete"
        manifest["comparisons"] = len(results)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
