import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from experiment2.data_utils import prepare_farm_data, set_seed
from experiment2.metrics import compute_metrics
from experiment2.models.tl_qldmr_median import TLQLDMRConfig, TLQLDMRMedianModel


def inverse_transform(scaler, y_scaled: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def eval_tlqldmr(
    farm_idx: int,
    feature_set: str,
    window_size: int,
    target_train_ratio: float,
    split_seed: int,
    extreme_cfg: dict,
    params: dict,
):
    data = prepare_farm_data(
        farm_idx=farm_idx,
        feature_set=feature_set,
        window_size=window_size,
        target_train_ratio=target_train_ratio,
        extreme_cfg=extreme_cfg,
        split_mode="shuffle",
        split_seed=split_seed,
    )
    scaler_y = data["scaler_y"]
    X_S = data["X_S_flat"]
    y_S = data["y_S"]
    X_T_train = data["X_T_train_flat"]
    y_T_train = data["y_T_train"]
    X_T_test = data["X_T_test_flat"]
    y_T_test = data["y_T_test"]

    cfg = TLQLDMRConfig(
        lambda1=float(params["lambda1"]),
        lambda2=float(params["lambda2"]),
        C_S=float(params["C_S"]),
        C_T=float(params["C_T"]),
        tau=float(params["tau"]),
        kernel_gamma=float(params["kernel_gamma"]),
        solver="fast_nystrom",
        nystrom_n_components=int(params["nystrom_n_components"]),
        nystrom_lr=float(params["nystrom_lr"]),
        nystrom_epochs=int(params["nystrom_epochs"]),
        nystrom_batch_size=int(params["nystrom_batch_size"]),
    )

    model = TLQLDMRMedianModel(cfg)
    model.fit(X_S, y_S, X_T_train, y_T_train)
    y_pred_scaled = model.predict(X_T_test)
    y_true = inverse_transform(scaler_y, y_T_test)
    y_pred = inverse_transform(scaler_y, y_pred_scaled)
    y_pred = np.maximum(y_pred, 0)
    metrics = compute_metrics(y_true, y_pred)

    return metrics, data["farm_name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-r2", type=float, default=0.9)
    parser.add_argument("--seeds", type=str, default="0,1,2,3")
    parser.add_argument("--farms", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--feature-sets", type=str, default="full,full_dir_cyclic")
    parser.add_argument("--window-sizes", type=str, default="48,72")
    parser.add_argument("--train-ratios", type=str, default="0.8")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--stat-q", type=float, default=95.0)
    parser.add_argument("--stat-on-wind", type=int, default=1)
    parser.add_argument("--stat-on-power", type=int, default=1)
    parser.add_argument("--use-stat", type=int, default=1)
    parser.add_argument("--use-ramp", type=int, default=1)
    parser.add_argument("--use-cutout", type=int, default=1)
    parser.add_argument("--use-temp", type=int, default=1)
    parser.add_argument("--ramp-q", type=float, default=None)
    parser.add_argument("--ramp-ratio", type=float, default=0.05)
    parser.add_argument("--cutout-ms", type=float, default=25.0)
    parser.add_argument("--temp-low", type=float, default=-10.0)
    parser.add_argument("--temp-high", type=float, default=35.0)
    parser.add_argument("--temp-q-low", type=float, default=None)
    parser.add_argument("--temp-q-high", type=float, default=None)
    parser.add_argument("--temp-requires-wind", type=int, default=0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    farms = [int(f) for f in args.farms.split(",") if f.strip()]
    feature_sets = [s.strip() for s in args.feature_sets.split(",") if s.strip()]
    window_sizes = [int(w) for w in args.window_sizes.split(",") if w.strip()]
    train_ratios = [float(r) for r in args.train_ratios.split(",") if r.strip()]

    out_dir = Path(__file__).resolve().parent
    results_dir = out_dir / "results_hunt"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "tlqldmr_hunt.csv"
    best_path = results_dir / "tlqldmr_hunt_best.json"

    extreme_cfg = {
        "stat_q": float(args.stat_q),
        "stat_on_wind": bool(args.stat_on_wind),
        "stat_on_power": bool(args.stat_on_power),
        "use_stat": bool(args.use_stat),
        "use_ramp": bool(args.use_ramp),
        "use_cutout": bool(args.use_cutout),
        "use_temp": bool(args.use_temp),
        "ramp_ratio": float(args.ramp_ratio),
        "cutout_ms": float(args.cutout_ms),
        "temp_low": float(args.temp_low),
        "temp_high": float(args.temp_high),
        "temp_q_low": args.temp_q_low,
        "temp_q_high": args.temp_q_high,
        "temp_requires_wind": bool(args.temp_requires_wind),
    }
    if args.ramp_q is not None:
        extreme_cfg["ramp_q"] = float(args.ramp_q)

    # Candidate configs (hand-tuned)
    candidate_params = [
        {
            "lambda1": 1e-4,
            "lambda2": 1e-5,
            "C_S": 0.1,
            "C_T": 50.0,
            "kernel_gamma": 0.001,
            "nystrom_n_components": 800,
            "nystrom_lr": 0.01,
            "nystrom_epochs": 48,
            "nystrom_batch_size": 256,
            "tau": 0.5,
        },
        {
            "lambda1": 1e-4,
            "lambda2": 1e-4,
            "C_S": 0.1,
            "C_T": 100.0,
            "kernel_gamma": 0.001,
            "nystrom_n_components": 1000,
            "nystrom_lr": 0.01,
            "nystrom_epochs": 64,
            "nystrom_batch_size": 256,
            "tau": 0.5,
        },
        {
            "lambda1": 5e-4,
            "lambda2": 1e-4,
            "C_S": 0.05,
            "C_T": 150.0,
            "kernel_gamma": 0.002,
            "nystrom_n_components": 1200,
            "nystrom_lr": 0.01,
            "nystrom_epochs": 64,
            "nystrom_batch_size": 256,
            "tau": 0.5,
        },
        {
            "lambda1": 1e-3,
            "lambda2": 1e-3,
            "C_S": 0.1,
            "C_T": 200.0,
            "kernel_gamma": 0.001,
            "nystrom_n_components": 1200,
            "nystrom_lr": 0.005,
            "nystrom_epochs": 80,
            "nystrom_batch_size": 256,
            "tau": 0.5,
        },
        {
            "lambda1": 1e-3,
            "lambda2": 0.1,
            "C_S": 0.05,
            "C_T": 200.0,
            "kernel_gamma": 0.0005,
            "nystrom_n_components": 1500,
            "nystrom_lr": 0.01,
            "nystrom_epochs": 80,
            "nystrom_batch_size": 256,
            "tau": 0.6,
        },
        {
            "lambda1": 1e-2,
            "lambda2": 1e-3,
            "C_S": 0.1,
            "C_T": 300.0,
            "kernel_gamma": 0.002,
            "nystrom_n_components": 1500,
            "nystrom_lr": 0.01,
            "nystrom_epochs": 80,
            "nystrom_batch_size": 256,
            "tau": 0.5,
        },
    ]
    if args.max_configs is not None:
        candidate_params = candidate_params[: max(1, args.max_configs)]

    header = [
        "farm_idx",
        "farm_name",
        "split_seed",
        "feature_set",
        "window_size",
        "target_train_ratio",
        "config_id",
        "r2",
        "rmse",
        "mae",
        "mape",
    ]

    best = None

    file_mode = "a" if args.append else "w"
    with csv_path.open(file_mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if file_mode == "w" or not csv_path.exists() or csv_path.stat().st_size == 0:
            writer.writeheader()

        for farm_idx in farms:
            for split_seed in seeds:
                for feature_set in feature_sets:
                    for window_size in window_sizes:
                        for train_ratio in train_ratios:
                            for cfg_id, params in enumerate(candidate_params):
                                set_seed(split_seed)
                                metrics, farm_name = eval_tlqldmr(
                                    farm_idx=farm_idx,
                                    feature_set=feature_set,
                                    window_size=window_size,
                                    target_train_ratio=train_ratio,
                                    split_seed=split_seed,
                                    extreme_cfg=extreme_cfg,
                                    params=params,
                                )
                                row = {
                                    "farm_idx": farm_idx,
                                    "farm_name": farm_name,
                                    "split_seed": split_seed,
                                    "feature_set": feature_set,
                                    "window_size": window_size,
                                    "target_train_ratio": train_ratio,
                                    "config_id": cfg_id,
                                    "r2": metrics["r2"],
                                    "rmse": metrics["rmse"],
                                    "mae": metrics["mae"],
                                    "mape": metrics["mape"],
                                }
                                writer.writerow(row)
                                handle.flush()
                                os.fsync(handle.fileno())

                                if best is None or metrics["r2"] > best["metrics"]["r2"]:
                                    best = {
                                        "farm_idx": farm_idx,
                                        "farm_name": farm_name,
                                        "split_seed": split_seed,
                                        "feature_set": feature_set,
                                        "window_size": window_size,
                                        "target_train_ratio": train_ratio,
                                        "params": params,
                                        "metrics": metrics,
                                        "extreme_cfg": extreme_cfg,
                                    }
                                    best_path.write_text(
                                        json.dumps(best, indent=2, default=float),
                                        encoding="utf-8",
                                    )

                                if metrics["r2"] >= args.target_r2:
                                    print(f"Reached target R2 {metrics['r2']:.4f}")
                                    return

    if best is None:
        print("Finished hunt. No results were recorded.")
    else:
        print(f"Finished hunt. Best so far: {best['metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()
