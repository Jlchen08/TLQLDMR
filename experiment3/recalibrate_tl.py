import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from experiment2.data_utils import prepare_farm_data, set_seed
from experiment3.metrics import interval_metrics
from experiment3.run_experiment3 import (
    split_train_val_cal,
    inverse_transform,
    predict_interval_orig,
    conformal_adjustment,
    apply_conformal,
    clip_intervals,
    select_q_scale,
)
from experiment3.models.tl_qldmr_quantile import TLQLDMRQuantileConfig, TLQLDMRQuantileIntervalModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "experiment2", "results_hunt", "tlqldmr_hunt_best.json"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--picp-safety",
        type=float,
        default=0.02,
        help="Extra safety margin added to target PICP when selecting q_scale.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scale", type=float, default=8.0)
    parser.add_argument("--fixed-scale", type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = json.loads(Path(args.config).read_text())
    tl_path = Path(__file__).resolve().parent / "results_selected" / "TL-QLDMR.json"
    tl_payload = json.loads(tl_path.read_text())
    tl_params = tl_payload["best_params"]

    data = prepare_farm_data(
        farm_idx=cfg["farm_idx"],
        feature_set=cfg["feature_set"],
        window_size=cfg["window_size"],
        target_train_ratio=cfg["target_train_ratio"],
        extreme_cfg=cfg.get("extreme_cfg"),
        split_mode="time",
        split_seed=cfg.get("split_seed", 0),
    )

    scaler_y = data["scaler_y"]
    nominal_capacity = data["nominal_capacity"]

    X_S = data["X_S_flat"]
    y_S = data["y_S"]
    X_T_train_flat = data["X_T_train_flat"]
    X_T_test_flat = data["X_T_test_flat"]
    y_T_train = data["y_T_train"]
    y_T_test = data["y_T_test"]

    idx_train, idx_val, idx_cal = split_train_val_cal(len(X_T_train_flat), val_ratio=0.15, cal_ratio=0.2)
    X_train_flat = X_T_train_flat[idx_train]
    y_train = y_T_train[idx_train]
    X_val_flat = X_T_train_flat[idx_val]
    y_val = y_T_train[idx_val]
    X_cal_flat = X_T_train_flat[idx_cal]
    y_cal = y_T_train[idx_cal]

    X_train_val = np.concatenate([X_train_flat, X_val_flat], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    tl_cfg = TLQLDMRQuantileConfig(
        lambda1=tl_params["lambda1"],
        lambda2=tl_params["lambda2"],
        C_S=tl_params["C_S"],
        C_T=tl_params["C_T"],
        kernel_gamma=tl_params["kernel_gamma"],
        nystrom_n_components=tl_params["nystrom_n_components"],
        nystrom_lr=tl_params["nystrom_lr"],
        nystrom_epochs=tl_params["nystrom_epochs"],
        nystrom_batch_size=tl_params["nystrom_batch_size"],
    )
    tl_model = TLQLDMRQuantileIntervalModel(
        tl_cfg, tau_low=tl_params["tau_low"], tau_high=tl_params["tau_high"]
    )
    tl_model.fit(X_S[:30000], y_S[:30000], X_train_val, y_train_val)

    y_cal_orig = inverse_transform(scaler_y, y_cal)
    y_val_orig = inverse_transform(scaler_y, y_val)
    y_test_orig = inverse_transform(scaler_y, y_T_test)

    lower_cal, upper_cal = predict_interval_orig(tl_model, X_cal_flat, scaler_y)
    lower_cal, upper_cal = clip_intervals(lower_cal, upper_cal, 0.0, nominal_capacity)
    conformal_scaled = bool(tl_params.get("conformal_scaled", False))
    q_hat = conformal_adjustment(
        y_cal_orig, lower_cal, upper_cal, alpha=args.alpha, scaled=conformal_scaled
    )

    lower_val, upper_val = predict_interval_orig(tl_model, X_val_flat, scaler_y)
    lower_val, upper_val = clip_intervals(lower_val, upper_val, 0.0, nominal_capacity)
    if args.fixed_scale is None:
        scale_grid = [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            12.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            50.0,
            60.0,
            80.0,
            100.0,
        ]
        base_picp = 1.0 - args.alpha
        target_picp = min(max(base_picp, base_picp + args.picp_safety), 0.995)
        q_scale, _ = select_q_scale(
            y_val_orig,
            lower_val,
            upper_val,
            q_hat,
            conformal_scaled,
            target_picp=target_picp,
            alpha=args.alpha,
            nominal_capacity=nominal_capacity,
            scale_grid=scale_grid,
        )
    else:
        q_scale = float(args.fixed_scale)

    lower_test, upper_test = predict_interval_orig(tl_model, X_T_test_flat, scaler_y)
    lower_test, upper_test = apply_conformal(
        lower_test, upper_test, q_hat * q_scale, scaled=conformal_scaled
    )
    lower_test, upper_test = clip_intervals(lower_test, upper_test, 0.0, nominal_capacity)
    metrics = interval_metrics(y_test_orig, lower_test, upper_test, alpha=args.alpha)

    tl_payload["calibration_q"] = float(q_hat)
    tl_payload["metrics"] = metrics
    tl_payload["best_params"]["q_scale"] = float(q_scale)
    tl_payload["best_params"]["conformal_scaled"] = conformal_scaled
    tl_payload["alpha"] = args.alpha
    tl_path.write_text(json.dumps(tl_payload, indent=2))

    summary_path = Path(__file__).resolve().parent / "results_selected" / "summary_metrics.csv"
    summary = pd.read_csv(summary_path)
    summary = summary[summary["model"] != "TL-QLDMR"]
    summary = pd.concat([summary, pd.DataFrame([{"model": "TL-QLDMR", **metrics}])], ignore_index=True)
    summary = summary.sort_values(["cwc", "winkler"])
    summary.to_csv(summary_path, index=False)



if __name__ == "__main__":
    main()
