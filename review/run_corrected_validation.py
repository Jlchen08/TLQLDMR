#!/usr/bin/env python3
"""Fixed-configuration validation of the corrected centered-residual solver.

Use a Python environment with NumPy, pandas, scikit-learn, openpyxl and CVXOPT.
This script never overwrites an existing output directory or historical result.
The historical random-window split and full-record extreme-event definition are
retained. Consequently this is NOT a chronological/leakage-safe evaluation.
Feature imputation/scaling and label scaling use model-training data only.
Intervals use a held-out part of target training for split-conformal calibration;
test labels never select parameters, calibration, clipping, or post-processing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment3.metrics import interval_metrics
from wind_farm_Data_Utils import WindFarmDataGenerator

VARIANTS = ("Full", "NoVariance", "NoMMD", "TargetOnly")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def conformal_radius(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float):
    """Conservative split-conformal expansion using calibration data only."""
    if not 0 < alpha < 1 or not len(y):
        raise ValueError("Calibration requires nonempty labels and 0 < alpha < 1.")
    scores = np.maximum.reduce((lower - y, y - upper, np.zeros(len(y))))
    rank = math.ceil((len(y) + 1) * (1 - alpha))
    if rank > len(y):
        raise ValueError("Too few calibration observations for the requested coverage.")
    return float(np.sort(scores)[rank - 1]), scores, rank


def extract_raw(df: pd.DataFrame, generator: WindFarmDataGenerator, cfg: dict):
    """Match the historical full_dir_cyclic feature construction without scaling."""
    if cfg["feature_set"] != "full_dir_cyclic":
        raise ValueError("This validation runner explicitly supports full_dir_cyclic only.")
    extreme_cfg = cfg.get("extreme_cfg", {})
    if extreme_cfg.get("stat_on_power", True) or extreme_cfg.get("use_ramp", True):
        raise ValueError("This diagnostic requires the historical covariate-only extreme rules; power/ramp screening is unsupported.")

    def column(*words):
        return next((c for c in df.columns if all(w.lower() in c.lower() for w in words)), None)

    hub = column("wind speed", "wheel hub", "m/s")
    temp = column("air temperature", "°C")
    power = column("power", "MW")
    if hub is None or temp is None or power is None:
        raise ValueError("Required hub-wind, temperature, or power column is absent.")
    base = [hub, column("wind speed", "50 meters", "m/s"),
            column("wind speed", "30 meters", "m/s"), column("wind speed", "10 meters", "m/s"),
            temp, column("atmosphere", "hpa"), column("relative humidity", "%")]
    directions = [column("wind speed", "wheel hub", "˚"), column("wind direction", "50 meters"),
                  column("wind direction", "30 meters"), column("wind direction", "10 meters")]
    arrays, names = [], []
    for col in base:
        if col is not None:
            arrays.append(df[col].to_numpy(dtype=float))
            names.append(col)
    for col in directions:
        if col is not None:
            radians = np.deg2rad(df[col].to_numpy(dtype=float))
            arrays.extend((np.sin(radians), np.cos(radians)))
            names.extend((f"sin({col})", f"cos({col})"))
    y = df[power].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite labels require a declared exclusion policy; labels are not imputed.")
    extreme = generator._identify_extreme_weather(df, y, hub, temp, extreme_cfg=cfg.get("extreme_cfg"))
    time_col = column("time")
    timestamps = df[time_col].astype(str).to_numpy() if time_col else np.arange(len(df)).astype(str)
    return np.column_stack(arrays), y, extreme, timestamps, names


def prepare_arrays(raw_x, raw_y, window, source_ids, target_fit_ids, target_test_ids, cal_ids):
    """Fit preprocessing exclusively on the windows/labels actually used to fit this variant."""
    train_ids = np.concatenate((source_ids, target_fit_ids))
    # Window i contains raw feature rows [i, i+window), with target at i+window.
    counts = np.zeros(len(raw_x) + 1, dtype=np.int64)
    np.add.at(counts, train_ids, 1)
    np.add.at(counts, train_ids + window, -1)
    input_rows = np.cumsum(counts[:-1]) > 0
    train_x = np.where(np.isfinite(raw_x[input_rows]), raw_x[input_rows], np.nan)
    feature_means = np.nanmean(train_x, axis=0)
    feature_means = np.where(np.isfinite(feature_means), feature_means, 0.0)
    filled_x = np.where(np.isfinite(raw_x), raw_x, feature_means)
    x_scaler = StandardScaler().fit(filled_x[input_rows])
    y_scaler = StandardScaler().fit(raw_y[train_ids + window, None])
    x_scaled = x_scaler.transform(filled_x).astype(np.float32)
    y_scaled = y_scaler.transform(raw_y[:, None]).ravel().astype(np.float32)
    all_windows = np.lib.stride_tricks.sliding_window_view(x_scaled, window, axis=0)
    all_windows = all_windows[:-1].transpose(0, 2, 1).reshape(len(raw_x) - window, -1)

    def take(ids):
        return np.ascontiguousarray(all_windows[ids]), y_scaled[ids + window]

    x_s, y_s = take(source_ids)
    x_t, y_t = take(target_fit_ids)
    x_test, _ = take(target_test_ids)
    x_cal, _ = take(cal_ids)
    preprocessing = {
        "feature_imputation_mean": feature_means.tolist(), "feature_mean": x_scaler.mean_.tolist(),
        "feature_scale": x_scaler.scale_.tolist(), "label_mean": float(y_scaler.mean_[0]),
        "label_scale": float(y_scaler.scale_[0]), "unique_training_input_rows": int(input_rows.sum()),
        "fit_label_count": len(train_ids), "fit_scope": "model-training inputs and labels only",
    }
    return x_s, y_s, x_t, y_t, x_test, x_cal, y_scaler, preprocessing


def fit_model(params, tau, variant, seed, arrays):
    from Train_TL_QLDMR import Fast_TL_QLDMR

    options = {
        "lambda1": float(params["lambda1"]), "lambda2": float(params["lambda2"]),
        "C_S": float(params["C_S"]), "C_T": float(params["C_T"]), "tau": tau,
        "kernel_gamma": float(params["kernel_gamma"]),
        "n_components": int(params["nystrom_n_components"]),
        "learning_rate": float(params["nystrom_lr"]), "epochs": int(params["nystrom_epochs"]),
        "batch_size": int(params["nystrom_batch_size"]),
        "variance_mode": "centered_residual", "asym_scale": 0.0,
    }
    if variant == "NoVariance":
        options["lambda1"] = 0.0
    if variant in ("NoMMD", "TargetOnly"):
        options["lambda2"] = 0.0
    if variant == "TargetOnly":
        options["C_S"] = 0.0
    np.random.seed(seed)
    solver = Fast_TL_QLDMR(**options)
    start = time.perf_counter()
    solver.fit(*arrays[:4])
    elapsed = time.perf_counter() - start
    effective = len(solver.feature_map.components_)

    def predict(x):
        scaled = solver.feature_map.transform(x) @ solver.w + solver.b
        return arrays[6].inverse_transform(np.asarray(scaled)[:, None]).ravel()

    return solver, predict, options, elapsed, effective


def point_metrics(y, prediction):
    squared_error = float(np.sum((y - prediction) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {"rmse": float(np.sqrt(squared_error / len(y))),
            "mae": float(np.mean(np.abs(y - prediction))), "r2": 1 - squared_error / total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/wind_farm_data/data_processed/wind_farms")
    parser.add_argument("--farm-file", type=Path, help="Optional single-farm Excel file; overrides --data-dir.")
    parser.add_argument("--point-config", type=Path, default=ROOT / "TLQLDMR_R1/experiment4/tlqldmr_ablation_config.json")
    parser.add_argument("--interval-config", type=Path, default=ROOT / "TLQLDMR_R1/experiment4/tlqldmr_interval_ablation_config.json")
    parser.add_argument("--tasks", nargs="+", choices=("point", "interval"), default=["point", "interval"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--calibration-ratio", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "review/corrected_validation_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--check-data", action="store_true", help="Report availability without fitting or writing results.")
    args = parser.parse_args()
    if not 0 < args.calibration_ratio < 1 or not 0 < args.alpha < 1 or args.threads < 1:
        parser.error("Calibration ratio and alpha must be in (0,1), and threads must be positive.")
    configs = {task: json.loads(getattr(args, f"{task}_config").read_text()) for task in args.tasks}
    first = next(iter(configs.values()))
    for cfg in configs.values():
        for key in ("farm_idx", "feature_set", "window_size", "target_train_ratio", "split_seed", "extreme_cfg"):
            if cfg.get(key) != first.get(key):
                raise ValueError(f"Task configurations disagree on shared data protocol: {key}")
    generator = WindFarmDataGenerator(args.data_dir)
    farms = generator.get_available_farms()
    farm_file = args.farm_file
    if farm_file is None and int(first["farm_idx"]) < len(farms):
        farm_file = farms[int(first["farm_idx"])]["path"]
    exists = farm_file is not None and farm_file.is_file()
    if args.check_data:
        print(json.dumps({"data_available": exists, "farm_file": str(farm_file) if farm_file else None,
                          "data_dir": str(args.data_dir), "available_farms": len(farms),
                          "training_executed": False}))
        return
    if not exists:
        raise FileNotFoundError(f"Raw Excel data are unavailable in {args.data_dir}; supply --farm-file or --data-dir. No experiment has run.")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output}")
    match = re.search(r"(\d+)MW", farm_file.stem)
    if not match:
        raise ValueError("Farm filename must contain its nominal capacity, e.g. 96MW.")
    generator.nominal_capacity = int(match.group(1))
    df = pd.read_excel(farm_file)
    raw_x, raw_y, extreme, timestamps, feature_names = extract_raw(df, generator, first)
    window = int(first["window_size"])
    source_ids = np.flatnonzero(~extreme[window:])
    target_ids = np.flatnonzero(extreme[window:])
    order = np.random.default_rng(int(first.get("split_seed", 0))).permutation(len(target_ids))
    split_at = int(len(target_ids) * float(first["target_train_ratio"]))
    train_ids, test_ids = target_ids[order[:split_at]], target_ids[order[split_at:]]
    if len(train_ids) < 2 or len(test_ids) < 2:
        raise ValueError("At least two training and test target samples are required.")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        revision = None
    source_files = [Path(__file__), ROOT / "Train_TL_QLDMR.py", ROOT / "Kernel_Function.py",
                    ROOT / "wind_farm_Data_Utils.py", ROOT / "experiment3/metrics.py"]
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": str(farm_file.resolve()), "data_sha256": digest(farm_file), "git_head": revision,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "packages": {name: version(name) for name in ("scikit-learn", "scipy", "cvxopt", "openpyxl", "threadpoolctl")},
        "threads": args.threads, "train_seed": args.train_seed, "configs": configs,
        "tasks": args.tasks, "variants": args.variants, "feature_names": feature_names,
        "counts": {"source": len(source_ids), "target_train_pool": len(train_ids), "test": len(test_ids)},
        "protocol": {
            "split": "historical random-window permutation with fixed split_seed",
            "extreme_definition": "historical rules with full-record covariate quantiles retained",
            "preprocessing": "fit on each variant's model-training windows and labels only",
            "tuning": "none; historical hyperparameters fixed before this run",
            "historical_parameter_selection": "original tuning may have used this test set; this is a fixed-parameter repair diagnostic, not a new untouched holdout",
            "interval_calibration": "last 20% (or declared calibration_ratio) of target training pool; excluded from fitting/scaling labels",
            "calibration_ratio": args.calibration_ratio, "alpha": args.alpha,
            "postprocessing": "order endpoints then one calibration-only nonnegative finite-sample conformal expansion; no clipping or q_scale search",
            "limitation": "neighboring overlapping windows and future source windows remain possible; no chronological validity or distribution-free temporal coverage claim",
        },
    }
    save_json(args.output / "manifest.json", manifest)
    np.savez_compressed(args.output / "split_ids.npz", source_window_ids=source_ids,
                        target_training_window_ids=train_ids, test_window_ids=test_ids,
                        test_raw_target_rows=test_ids + window, test_times=timestamps[test_ids + window])
    rows = []
    try:
        with threadpool_limits(limits=args.threads):
            for task, cfg in configs.items():
                n_cal = max(1, int(len(train_ids) * args.calibration_ratio)) if task == "interval" else 0
                fit_ids = train_ids[:-n_cal] if n_cal else train_ids
                cal_ids = train_ids[-n_cal:] if n_cal else np.array([], dtype=int)
                for variant in args.variants:
                    out = args.output / f"{task}_{variant}"
                    out.mkdir()
                    active_source = source_ids if variant != "TargetOnly" else np.array([], dtype=int)
                    arrays = prepare_arrays(raw_x, raw_y, window, active_source, fit_ids, test_ids, cal_ids)
                    save_json(out / "preprocessing.json", arrays[7])
                    y_test = raw_y[test_ids + window]
                    base = {"window_id": test_ids, "raw_target_row": test_ids + window,
                            "time": timestamps[test_ids + window], "y_true": y_test}
                    taus = [0.5] if task == "point" else [float(cfg.get("tau_low", 0.2)), float(cfg.get("tau_high", 0.8))]
                    test_predictions, cal_predictions, fits = [], [], []
                    for tau in taus:
                        solver, predict, options, seconds, effective = fit_model(cfg["params"], tau, variant, args.train_seed, arrays)
                        test_predictions.append(predict(arrays[4]))
                        if n_cal:
                            cal_predictions.append(predict(arrays[5]))
                        fits.append({"options": options, "fit_seconds": seconds, "effective_landmarks": effective})
                        np.savez_compressed(out / f"model_tau_{tau}.npz", w=solver.w, b=solver.b,
                                            landmarks=solver.feature_map.components_, normalization=solver.feature_map.normalization_)
                    if task == "point":
                        base["prediction"] = test_predictions[0]
                        metrics = point_metrics(y_test, test_predictions[0])
                        calibration = None
                    else:
                        lower = np.minimum(*test_predictions)
                        upper = np.maximum(*test_predictions)
                        cal_lower, cal_upper = np.minimum(*cal_predictions), np.maximum(*cal_predictions)
                        y_cal = raw_y[cal_ids + window]
                        q, scores, rank = conformal_radius(y_cal, cal_lower, cal_upper, args.alpha)
                        base.update({"raw_tau_low": test_predictions[0], "raw_tau_high": test_predictions[1],
                                     "ordered_lower": lower, "ordered_upper": upper,
                                     "lower": lower - q, "upper": upper + q})
                        metrics = interval_metrics(y_test, lower - q, upper + q, alpha=args.alpha)
                        calibration = {"q_hat": q, "rank": rank, "n_cal": n_cal, "q_scale": 1.0,
                                       "raw_metrics": interval_metrics(y_test, lower, upper, alpha=args.alpha)}
                        pd.DataFrame({"window_id": cal_ids, "raw_target_row": cal_ids + window,
                                      "y_true": y_cal, "raw_tau_low": cal_predictions[0], "raw_tau_high": cal_predictions[1],
                                      "ordered_lower": cal_lower, "ordered_upper": cal_upper, "score": scores}).to_csv(out / "calibration.csv", index=False)
                    pd.DataFrame(base).to_csv(out / "test_predictions.csv", index=False)
                    np.savez_compressed(out / "fit_ids.npz", source_window_ids=active_source,
                                        target_fit_window_ids=fit_ids, calibration_window_ids=cal_ids)
                    save_json(out / "result.json", {"task": task, "variant": variant, "fits": fits,
                                                    "metrics": metrics, "calibration": calibration})
                    rows.append({"task": task, "variant": variant, **metrics})
                    pd.DataFrame(rows).to_csv(args.output / "summary_metrics.csv", index=False)
                    print(json.dumps(rows[-1]), flush=True)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
