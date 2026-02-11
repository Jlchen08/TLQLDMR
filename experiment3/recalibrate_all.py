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
)
from experiment3.models.tl_qldmr_quantile import TLQLDMRQuantileConfig, TLQLDMRQuantileIntervalModel
from experiment3.models.tsvqr_model import TSVQRConfig, TSVQRIntervalModel
from experiment3.models.ssvqr_model import SSVQRConfig, SparseSVQRIntervalModel
from experiment3.models.nfs_svqr_model import NFSSVQRConfig, NFSSVQRIntervalModel
from experiment3.models.nu_svr_model import NuSVRConfig, NuSVRIntervalModel
from experiment3.models.quantile_gbr_model import QuantileGBRConfig, QuantileGBRIntervalModel
from experiment3.models.mlp_quantile_model import MLPQuantileConfig, MLPQuantileEnsembleIntervalModel
from experiment3.models.ensemble_kde_model import EnsembleKDEConfig, EnsembleKDEIntervalModel


def apply_width_scale(lower: np.ndarray, upper: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    lower_s = center - scale * half
    upper_s = center + scale * half
    return lower_s, upper_s


def select_width_scale(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_picp: float,
    alpha: float,
    nominal_capacity: float | None,
    scale_grid: list[float],
) -> tuple[float, dict]:
    best = None
    best_key = None
    best_infeasible = None
    best_infeasible_key = None
    for scale in scale_grid:
        l_s, u_s = apply_width_scale(lower, upper, scale)
        l_s, u_s = clip_intervals(l_s, u_s, 0.0, nominal_capacity)
        metrics = interval_metrics(y_true, l_s, u_s, alpha=alpha)
        if metrics["picp"] >= target_picp:
            key = (metrics["pinaw"], metrics["winkler"], scale)
            if best_key is None or key < best_key:
                best_key = key
                best = (scale, metrics)
        else:
            key = (-metrics["picp"], metrics["pinaw"], scale)
            if best_infeasible_key is None or key < best_infeasible_key:
                best_infeasible_key = key
                best_infeasible = (scale, metrics)
    if best is not None:
        return best
    if best_infeasible is not None:
        return best_infeasible
    return 1.0, interval_metrics(y_true, lower, upper, alpha=alpha)


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
        help="Extra safety margin added to target PICP when selecting width scale.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = json.loads(Path(args.config).read_text())

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

    y_cal_orig = inverse_transform(scaler_y, y_cal)
    y_val_orig = inverse_transform(scaler_y, y_val)
    y_test_orig = inverse_transform(scaler_y, y_T_test)

    results_dir = Path(__file__).resolve().parent / "results_selected"
    summary_rows = []

    width_scale_grid = [
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.1,
        1.2,
        1.3,
        1.5,
        1.8,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
        6.0,
        8.0,
        10.0,
        12.0,
        15.0,
        20.0,
    ]
    conformal_candidates = [False, True]
    base_picp = 1.0 - args.alpha
    target_picp = min(max(base_picp, base_picp + args.picp_safety), 0.995)

    def _save(name: str, payload: dict) -> None:
        with open(results_dir / f"{name}.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _best_calibration(
        model,
        lower_cal,
        upper_cal,
        lower_val,
        upper_val,
        lower_test,
        upper_test,
    ) -> tuple[dict, dict, float, float, bool]:
        best = None
        best_key = None
        best_meta = None
        for conformal_scaled in conformal_candidates:
            q_hat = conformal_adjustment(
                y_cal_orig,
                lower_cal,
                upper_cal,
                alpha=args.alpha,
                scaled=conformal_scaled,
            )
            lower_val_adj, upper_val_adj = apply_conformal(
                lower_val,
                upper_val,
                q_hat,
                scaled=conformal_scaled,
            )
            lower_val_adj, upper_val_adj = clip_intervals(lower_val_adj, upper_val_adj, 0.0, nominal_capacity)
            width_scale, metrics_val = select_width_scale(
                y_val_orig,
                lower_val_adj,
                upper_val_adj,
                target_picp=target_picp,
                alpha=args.alpha,
                nominal_capacity=nominal_capacity,
                scale_grid=width_scale_grid,
            )
            if metrics_val["picp"] >= target_picp:
                key = (metrics_val["pinaw"], metrics_val["winkler"], width_scale)
            else:
                key = (-metrics_val["picp"], metrics_val["pinaw"], width_scale)
            if best_key is None or key < best_key:
                best_key = key
                best = metrics_val
                best_meta = (q_hat, width_scale, conformal_scaled)
        if best_meta is None:
            raise RuntimeError("Failed to select calibration scale.")
        q_hat, width_scale, conformal_scaled = best_meta
        lower_test_adj, upper_test_adj = apply_conformal(
            lower_test,
            upper_test,
            q_hat,
            scaled=conformal_scaled,
        )
        lower_test_adj, upper_test_adj = clip_intervals(lower_test_adj, upper_test_adj, 0.0, nominal_capacity)
        lower_test_adj, upper_test_adj = apply_width_scale(lower_test_adj, upper_test_adj, width_scale)
        lower_test_adj, upper_test_adj = clip_intervals(lower_test_adj, upper_test_adj, 0.0, nominal_capacity)
        metrics_test = interval_metrics(y_test_orig, lower_test_adj, upper_test_adj, alpha=args.alpha)
        return metrics_test, best, q_hat, width_scale, conformal_scaled

    # ---- TL-QLDMR ----
    tl_payload = json.loads((results_dir / "TL-QLDMR.json").read_text())
    tl_params = tl_payload["best_params"]
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
    tl_lower_cal, tl_upper_cal = predict_interval_orig(tl_model, X_cal_flat, scaler_y)
    tl_lower_test, tl_upper_test = predict_interval_orig(tl_model, X_T_test_flat, scaler_y)
    tl_lower_cal, tl_upper_cal = clip_intervals(tl_lower_cal, tl_upper_cal, 0.0, nominal_capacity)
    tl_lower_test, tl_upper_test = clip_intervals(tl_lower_test, tl_upper_test, 0.0, nominal_capacity)
    tl_lower_val, tl_upper_val = predict_interval_orig(tl_model, X_val_flat, scaler_y)
    tl_lower_val, tl_upper_val = clip_intervals(tl_lower_val, tl_upper_val, 0.0, nominal_capacity)
    metrics, metrics_val, q_hat, width_scale, conformal_scaled = _best_calibration(
        tl_model, tl_lower_cal, tl_upper_cal, tl_lower_val, tl_upper_val, tl_lower_test, tl_upper_test
    )
    tl_payload["alpha"] = args.alpha
    tl_payload["metrics"] = metrics
    tl_payload["calibration_q"] = float(q_hat)
    tl_payload["best_params"]["width_scale"] = float(width_scale)
    tl_payload["best_params"]["conformal_scaled"] = conformal_scaled
    tl_payload["search_metrics"] = metrics_val
    _save("TL-QLDMR", tl_payload)
    summary_rows.append({"model": "TL-QLDMR", **metrics})

    def _calibrate_and_save(name: str, model, params: dict, train_limit: int | None = None, model_type: str = "quantile"):
        X_tr = X_train_val
        y_tr = y_train_val
        if train_limit is not None and len(X_tr) > train_limit:
            X_tr = X_tr[:train_limit]
            y_tr = y_tr[:train_limit]
        if model_type == "kde":
            model.fit(X_tr, y_tr, X_cal_flat, y_cal)
        else:
            model.fit(X_tr, y_tr)
        lower_cal, upper_cal = predict_interval_orig(model, X_cal_flat, scaler_y)
        lower_test, upper_test = predict_interval_orig(model, X_T_test_flat, scaler_y)
        lower_cal, upper_cal = clip_intervals(lower_cal, upper_cal, 0.0, nominal_capacity)
        lower_test, upper_test = clip_intervals(lower_test, upper_test, 0.0, nominal_capacity)
        lower_val, upper_val = predict_interval_orig(model, X_val_flat, scaler_y)
        lower_val, upper_val = clip_intervals(lower_val, upper_val, 0.0, nominal_capacity)
        metrics, metrics_val, q_hat, width_scale, conformal_scaled = _best_calibration(
            model, lower_cal, upper_cal, lower_val, upper_val, lower_test, upper_test
        )
        payload = {
            "model": name,
            "farm_name": data["farm_name"],
            "alpha": args.alpha,
            "calibration_q": float(q_hat),
            "metrics": metrics,
            "train_samples": int(len(X_tr)),
            "val_samples": int(len(X_val_flat)),
            "cal_samples": int(len(X_cal_flat)),
            "test_samples": int(len(X_T_test_flat)),
            "best_params": {**params, "width_scale": float(width_scale), "conformal_scaled": conformal_scaled},
            "search_metrics": metrics_val,
        }
        _save(name, payload)
        summary_rows.append({"model": name, **metrics})

    # ---- TSVQR ----
    tsvqr_params = json.loads((results_dir / "TSVQR.json").read_text())["best_params"]
    _calibrate_and_save(
        "TSVQR",
        TSVQRIntervalModel(TSVQRConfig(C=tsvqr_params["C"], gamma=tsvqr_params["gamma"], max_samples=tsvqr_params["max_samples"])),
        tsvqr_params,
        train_limit=tsvqr_params["max_samples"],
    )

    # ---- UQSVM-SSVQR ----
    ssvqr_params = json.loads((results_dir / "UQSVM-SSVQR.json").read_text())["best_params"]
    _calibrate_and_save(
        "UQSVM-SSVQR",
        SparseSVQRIntervalModel(
            SSVQRConfig(gamma=ssvqr_params["gamma"], n_components=ssvqr_params["n_components"], alpha=ssvqr_params["alpha"])
        ),
        ssvqr_params,
        train_limit=2500,
    )

    # ---- NFS-SVQR ----
    nfs_params = json.loads((results_dir / "NFS-SVQR.json").read_text())["best_params"]
    _calibrate_and_save(
        "NFS-SVQR",
        NFSSVQRIntervalModel(
            NFSSVQRConfig(gamma=nfs_params["gamma"], n_components=nfs_params["n_components"], max_features=nfs_params["max_features"])
        ),
        nfs_params,
        train_limit=2500,
    )

    # ---- NuSVR-CI ----
    nu_params = json.loads((results_dir / "NuSVR-CI.json").read_text())["best_params"]
    _calibrate_and_save(
        "NuSVR-CI",
        NuSVRIntervalModel(NuSVRConfig(nu=nu_params["nu"], C=nu_params["C"], gamma=nu_params["gamma"])),
        nu_params,
        train_limit=2200,
    )

    # ---- NCQ-GBR ----
    gbr_params = json.loads((results_dir / "NCQ-GBR.json").read_text())["best_params"]
    _calibrate_and_save(
        "NCQ-GBR",
        QuantileGBRIntervalModel(
            QuantileGBRConfig(
                n_estimators=gbr_params["n_estimators"],
                learning_rate=gbr_params["learning_rate"],
                max_depth=gbr_params["max_depth"],
                min_samples_leaf=gbr_params["min_samples_leaf"],
            )
        ),
        gbr_params,
    )

    # ---- NESCQR ----
    nescqr_params = json.loads((results_dir / "NESCQR.json").read_text())["best_params"]
    _calibrate_and_save(
        "NESCQR",
        MLPQuantileEnsembleIntervalModel(
            input_dim=X_train_flat.shape[1],
            config=MLPQuantileConfig(
                hidden_dims=tuple(nescqr_params["hidden_dims"]),
                epochs=nescqr_params["epochs"],
                n_members=nescqr_params["n_members"],
            ),
        ),
        nescqr_params,
    )

    # ---- ENS-KDE ----
    enskde_params = json.loads((results_dir / "ENS-KDE.json").read_text())["best_params"]
    enskde_model = EnsembleKDEIntervalModel(
        EnsembleKDEConfig(
            n_bins=enskde_params["n_bins"],
            kde_bandwidth=enskde_params["kde_bandwidth"],
            kde_samples=enskde_params["kde_samples"],
        )
    )
    _calibrate_and_save("ENS-KDE", enskde_model, enskde_params, model_type="kde")

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["cwc", "winkler"])
    summary_df.to_csv(results_dir / "summary_metrics.csv", index=False)


if __name__ == "__main__":
    main()
