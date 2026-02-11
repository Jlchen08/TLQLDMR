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
from experiment3.models.tl_qldmr_quantile import TLQLDMRQuantileConfig, TLQLDMRQuantileIntervalModel
from experiment3.models.tsvqr_model import TSVQRConfig, TSVQRIntervalModel
from experiment3.models.ssvqr_model import SSVQRConfig, SparseSVQRIntervalModel
from experiment3.models.nfs_svqr_model import NFSSVQRConfig, NFSSVQRIntervalModel
from experiment3.models.nu_svr_model import NuSVRConfig, NuSVRIntervalModel
from experiment3.models.quantile_gbr_model import QuantileGBRConfig, QuantileGBRIntervalModel
from experiment3.models.ensemble_kde_model import EnsembleKDEConfig, EnsembleKDEIntervalModel
from experiment3.models.mlp_quantile_model import MLPQuantileConfig, MLPQuantileEnsembleIntervalModel


def inverse_transform(scaler, y_scaled: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def _to_native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    return obj


def split_train_val_cal(
    n_samples: int, val_ratio: float = 0.15, cal_ratio: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_cal = max(1, int(n_samples * cal_ratio))
    n_val = max(1, int(n_samples * val_ratio))
    n_train = max(1, n_samples - n_val - n_cal)
    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_cal = np.arange(n_train + n_val, n_samples)
    return idx_train, idx_val, idx_cal


def ensure_order(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    swapped = lower > upper
    if np.any(swapped):
        lower, upper = lower.copy(), upper.copy()
        lower[swapped], upper[swapped] = upper[swapped], lower[swapped]
    return lower, upper


def conformal_adjustment(
    y_cal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
    scaled: bool = False,
) -> float:
    if scaled:
        scale = np.maximum(upper - lower, 1e-6)
        scores = np.maximum(lower - y_cal, y_cal - upper) / scale
    else:
        scores = np.maximum(lower - y_cal, y_cal - upper)
    try:
        q = np.quantile(scores, 1.0 - alpha, method="higher")
    except TypeError:
        q = np.quantile(scores, 1.0 - alpha, interpolation="higher")
    return float(q)


def clip_intervals(lower: np.ndarray, upper: np.ndarray, min_val: float, max_val: float | None) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(lower, min_val)
    if max_val is not None:
        upper = np.minimum(upper, max_val)
    upper = np.maximum(upper, lower)
    return lower, upper


def predict_interval_orig(model, X: np.ndarray, scaler) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = model.predict_interval(X)
    lower = inverse_transform(scaler, lower)
    upper = inverse_transform(scaler, upper)
    return ensure_order(lower, upper)


def apply_conformal(
    lower: np.ndarray,
    upper: np.ndarray,
    q_hat: float,
    scaled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if scaled:
        scale = np.maximum(upper - lower, 1e-6)
        lower = lower - q_hat * scale
        upper = upper + q_hat * scale
    else:
        lower = lower - q_hat
        upper = upper + q_hat
    return lower, upper


def select_q_scale(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q_hat: float,
    scaled: bool,
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
        l_adj, u_adj = apply_conformal(lower, upper, q_hat * scale, scaled)
        l_adj, u_adj = clip_intervals(l_adj, u_adj, 0.0, nominal_capacity)
        metrics = interval_metrics(y_true, l_adj, u_adj, alpha=alpha)
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


def _evaluate_with_conformal(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    scaler,
    alpha: float,
    nominal_capacity: float | None,
    conformal_scaled: bool,
) -> tuple[dict, float, np.ndarray, np.ndarray]:
    lower_cal, upper_cal = model.predict_interval(X_cal)
    lower_eval, upper_eval = model.predict_interval(X_eval)

    y_cal_orig = inverse_transform(scaler, y_cal)
    y_eval_orig = inverse_transform(scaler, y_eval)
    lower_cal = inverse_transform(scaler, lower_cal)
    upper_cal = inverse_transform(scaler, upper_cal)
    lower_eval = inverse_transform(scaler, lower_eval)
    upper_eval = inverse_transform(scaler, upper_eval)

    lower_cal, upper_cal = ensure_order(lower_cal, upper_cal)
    lower_eval, upper_eval = ensure_order(lower_eval, upper_eval)

    lower_cal, upper_cal = clip_intervals(lower_cal, upper_cal, 0.0, nominal_capacity)
    lower_eval, upper_eval = clip_intervals(lower_eval, upper_eval, 0.0, nominal_capacity)

    q_hat = conformal_adjustment(
        y_cal_orig,
        lower_cal,
        upper_cal,
        alpha=alpha,
        scaled=conformal_scaled,
    )
    if conformal_scaled:
        scale = np.maximum(upper_eval - lower_eval, 1e-6)
        lower_eval = lower_eval - q_hat * scale
        upper_eval = upper_eval + q_hat * scale
    else:
        lower_eval = lower_eval - q_hat
        upper_eval = upper_eval + q_hat

    lower_eval, upper_eval = clip_intervals(lower_eval, upper_eval, 0.0, nominal_capacity)
    metrics = interval_metrics(y_eval_orig, lower_eval, upper_eval, alpha=alpha)
    return metrics, q_hat, lower_eval, upper_eval


def _tlqldmr_candidate_params(extra: int = 0, seed: int = 0) -> list[dict]:
    base = [
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
        },
    ]
    if extra <= 0:
        return base

    rng = np.random.default_rng(seed)
    choices = {
        "lambda1": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "lambda2": [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "C_S": [0.05, 0.1, 0.2],
        "C_T": [50.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0],
        "kernel_gamma": [0.0005, 0.001, 0.002, 0.005],
        "nystrom_n_components": [600, 800, 1000, 1200, 1500],
        "nystrom_lr": [0.005, 0.01],
        "nystrom_epochs": [48, 64, 80],
        "nystrom_batch_size": [256],
    }
    keys = [
        "lambda1",
        "lambda2",
        "C_S",
        "C_T",
        "kernel_gamma",
        "nystrom_n_components",
        "nystrom_lr",
        "nystrom_epochs",
        "nystrom_batch_size",
    ]
    seen = {tuple((k, v) for k, v in params.items()) for params in base}
    extra_candidates = []
    attempts = 0
    max_attempts = max(50, extra * 20)
    while len(extra_candidates) < extra and attempts < max_attempts:
        attempts += 1
        params = {k: rng.choice(choices[k]).item() for k in keys}
        key = tuple((k, params[k]) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        extra_candidates.append(params)
    return base + extra_candidates


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
    parser.add_argument(
        "--tl-extra-candidates",
        type=int,
        default=12,
        help="Number of extra random TL-QLDMR candidates to sample.",
    )
    parser.add_argument(
        "--tl-search-epochs",
        type=int,
        default=20,
        help="Epoch cap for TL-QLDMR search stage (final training uses full epochs).",
    )
    parser.add_argument(
        "--tl-only",
        action="store_true",
        help="Only run TL-QLDMR search/train and update its results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, "r") as f:
        cfg = json.load(f)

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
    farm_name = data["farm_name"]

    X_S = data["X_S_flat"]
    y_S = data["y_S"]

    X_T_train_flat = data["X_T_train_flat"]
    X_T_test_flat = data["X_T_test_flat"]
    X_T_train_seq = data["X_T_train_seq"]
    X_T_test_seq = data["X_T_test_seq"]
    y_T_train = data["y_T_train"]
    y_T_test = data["y_T_test"]

    idx_train, idx_val, idx_cal = split_train_val_cal(len(X_T_train_flat), val_ratio=0.15, cal_ratio=0.2)

    X_train_flat = X_T_train_flat[idx_train]
    y_train = y_T_train[idx_train]
    X_val_flat = X_T_train_flat[idx_val]
    y_val = y_T_train[idx_val]
    X_cal_flat = X_T_train_flat[idx_cal]
    y_cal = y_T_train[idx_cal]

    X_train_seq = X_T_train_seq[idx_train]
    X_val_seq = X_T_train_seq[idx_val]
    X_cal_seq = X_T_train_seq[idx_cal]

    out_dir = Path(__file__).resolve().parent
    results_dir = out_dir / "results_selected"
    results_dir.mkdir(parents=True, exist_ok=True)
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

    summary_rows = []

    def _save_result(
        name: str,
        metrics: dict,
        q_hat: float,
        train_samples: int,
        val_samples: int,
        cal_samples: int,
        test_samples: int,
        best_params: dict | None = None,
        search_metrics: dict | None = None,
    ) -> None:
        summary_rows.append({"model": name, **metrics})
        result_payload = {
            "model": name,
            "farm_name": farm_name,
            "alpha": args.alpha,
            "calibration_q": q_hat,
            "metrics": metrics,
            "train_samples": int(train_samples),
            "val_samples": int(val_samples),
            "cal_samples": int(cal_samples),
            "test_samples": int(test_samples),
            "best_params": best_params,
            "search_metrics": search_metrics,
        }
        with open(results_dir / f"{name}.json", "w") as f:
            json.dump(_to_native(result_payload), f, indent=2)

    # ---- TL-QLDMR hyperparameter search (experiment2 range) ----
    print("\n[TL-QLDMR] Hyperparameter search...")
    tl_candidates = _tlqldmr_candidate_params(args.tl_extra_candidates, args.seed)
    best_tl = None
    best_feasible_key = None
    best_infeasible = None
    best_infeasible_key = None
    max_source = 30000
    X_S_local = X_S if max_source is None else X_S[:max_source]
    y_S_local = y_S if max_source is None else y_S[:max_source]

    base_picp = 1.0 - args.alpha
    target_picp = min(max(base_picp, base_picp + args.picp_safety), 0.995)
    tau_candidates = [(0.02, 0.98), (0.05, 0.95), (0.1, 0.9), (0.15, 0.85), (0.2, 0.8)]
    conformal_candidates = [False, True]
    for idx, params in enumerate(tl_candidates):
        search_epochs = max(10, min(args.tl_search_epochs, int(params["nystrom_epochs"])))
        cfg_trial = TLQLDMRQuantileConfig(
            lambda1=params["lambda1"],
            lambda2=params["lambda2"],
            C_S=params["C_S"],
            C_T=params["C_T"],
            kernel_gamma=params["kernel_gamma"],
            nystrom_n_components=params["nystrom_n_components"],
            nystrom_lr=params["nystrom_lr"],
            nystrom_epochs=search_epochs,
            nystrom_batch_size=params["nystrom_batch_size"],
        )
        print(f"  - Trial {idx + 1}/{len(tl_candidates)}: {params}")
        y_cal_orig = inverse_transform(scaler_y, y_cal)
        y_val_orig = inverse_transform(scaler_y, y_val)
        for tau_low, tau_high in tau_candidates:
            model_trial = TLQLDMRQuantileIntervalModel(cfg_trial, tau_low=tau_low, tau_high=tau_high)
            model_trial.fit(X_S_local, y_S_local, X_train_flat, y_train)
            lower_cal, upper_cal = predict_interval_orig(model_trial, X_cal_flat, scaler_y)
            lower_cal, upper_cal = clip_intervals(lower_cal, upper_cal, 0.0, nominal_capacity)
            lower_val, upper_val = predict_interval_orig(model_trial, X_val_flat, scaler_y)
            lower_val, upper_val = clip_intervals(lower_val, upper_val, 0.0, nominal_capacity)

            for conformal_scaled in conformal_candidates:
                q_hat_val = conformal_adjustment(
                    y_cal_orig,
                    lower_cal,
                    upper_cal,
                    alpha=args.alpha,
                    scaled=conformal_scaled,
                )
                q_scale, metrics_val = select_q_scale(
                    y_val_orig,
                    lower_val,
                    upper_val,
                    q_hat_val,
                    conformal_scaled,
                    target_picp=target_picp,
                    alpha=args.alpha,
                    nominal_capacity=nominal_capacity,
                    scale_grid=scale_grid,
                )
                if metrics_val["picp"] >= target_picp:
                    key = (metrics_val["pinaw"], metrics_val["winkler"], q_scale)
                    if best_feasible_key is None or key < best_feasible_key:
                        best_feasible_key = key
                        best_tl = {
                            "params": params,
                            "tau_low": tau_low,
                            "tau_high": tau_high,
                            "metrics_val": metrics_val,
                            "q_hat_val": q_hat_val,
                            "q_scale": q_scale,
                            "conformal_scaled": conformal_scaled,
                        }
                else:
                    key = (metrics_val["cwc"], metrics_val["winkler"], -metrics_val["picp"])
                    if best_infeasible_key is None or key < best_infeasible_key:
                        best_infeasible_key = key
                        best_infeasible = {
                            "params": params,
                            "tau_low": tau_low,
                            "tau_high": tau_high,
                            "metrics_val": metrics_val,
                            "q_hat_val": q_hat_val,
                            "q_scale": q_scale,
                            "conformal_scaled": conformal_scaled,
                        }

    if best_tl is None and best_infeasible is not None:
        best_tl = best_infeasible

    if best_tl is None:
        raise RuntimeError("TL-QLDMR search failed to produce a candidate.")

    # Final TL-QLDMR training on train+val
    print("[TL-QLDMR] Final training with best params...")
    best_params = best_tl["params"]
    tl_cfg = TLQLDMRQuantileConfig(
        lambda1=best_params["lambda1"],
        lambda2=best_params["lambda2"],
        C_S=best_params["C_S"],
        C_T=best_params["C_T"],
        kernel_gamma=best_params["kernel_gamma"],
        nystrom_n_components=best_params["nystrom_n_components"],
        nystrom_lr=best_params["nystrom_lr"],
        nystrom_epochs=int(best_params["nystrom_epochs"]),
        nystrom_batch_size=best_params["nystrom_batch_size"],
    )
    tau_low = float(best_tl.get("tau_low", 0.05))
    tau_high = float(best_tl.get("tau_high", 0.95))
    conformal_scaled = bool(best_tl.get("conformal_scaled", False))
    tl_model = TLQLDMRQuantileIntervalModel(tl_cfg, tau_low=tau_low, tau_high=tau_high)
    X_train_val = np.concatenate([X_train_flat, X_val_flat], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    tl_model.fit(X_S_local, y_S_local, X_train_val, y_train_val)

    y_cal_orig = inverse_transform(scaler_y, y_cal)
    y_val_orig = inverse_transform(scaler_y, y_val)
    y_test_orig = inverse_transform(scaler_y, y_T_test)
    tl_lower_cal, tl_upper_cal = predict_interval_orig(tl_model, X_cal_flat, scaler_y)
    tl_lower_cal, tl_upper_cal = clip_intervals(tl_lower_cal, tl_upper_cal, 0.0, nominal_capacity)
    tl_q_hat = conformal_adjustment(
        y_cal_orig,
        tl_lower_cal,
        tl_upper_cal,
        alpha=args.alpha,
        scaled=conformal_scaled,
    )
    tl_lower_val, tl_upper_val = predict_interval_orig(tl_model, X_val_flat, scaler_y)
    tl_lower_val, tl_upper_val = clip_intervals(tl_lower_val, tl_upper_val, 0.0, nominal_capacity)
    tl_q_scale, _ = select_q_scale(
        y_val_orig,
        tl_lower_val,
        tl_upper_val,
        tl_q_hat,
        conformal_scaled,
        target_picp=target_picp,
        alpha=args.alpha,
        nominal_capacity=nominal_capacity,
        scale_grid=scale_grid,
    )
    tl_lower_test, tl_upper_test = predict_interval_orig(tl_model, X_T_test_flat, scaler_y)
    tl_lower_test, tl_upper_test = apply_conformal(
        tl_lower_test,
        tl_upper_test,
        tl_q_hat * tl_q_scale,
        scaled=conformal_scaled,
    )
    tl_lower_test, tl_upper_test = clip_intervals(tl_lower_test, tl_upper_test, 0.0, nominal_capacity)
    tl_metrics = interval_metrics(y_test_orig, tl_lower_test, tl_upper_test, alpha=args.alpha)
    best_params = dict(best_params)
    best_params["tau_low"] = tau_low
    best_params["tau_high"] = tau_high
    best_params["q_scale"] = float(tl_q_scale)
    best_params["conformal_scaled"] = conformal_scaled
    _save_result(
        "TL-QLDMR",
        tl_metrics,
        tl_q_hat,
        train_samples=len(X_train_val),
        val_samples=len(X_val_flat),
        cal_samples=len(X_cal_flat),
        test_samples=len(X_T_test_flat),
        best_params=best_params,
        search_metrics=best_tl["metrics_val"],
    )
    # plotting removed

    if args.tl_only:
        summary_path = results_dir / "summary_metrics.csv"
        tl_row = {"model": "TL-QLDMR", **tl_metrics}
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            summary = summary[summary["model"] != "TL-QLDMR"]
            summary = pd.concat([summary, pd.DataFrame([tl_row])], ignore_index=True)
            summary = summary.sort_values(["cwc", "winkler"])
            summary.to_csv(summary_path, index=False)
        else:
            pd.DataFrame([tl_row]).to_csv(summary_path, index=False)
        print("\n[TL-QLDMR] tl-only mode: summary updated.")
        return

    # ---- Other models (light search) ----
    def _search_and_run(
        name: str,
        candidates: list[dict],
        build_fn,
        conformal_candidates: list[bool],
        max_target: int | None = None,
        model_type: str = "quantile",
    ) -> None:
        best = None
        best_feasible_key = None
        best_infeasible = None
        best_infeasible_key = None
        y_cal_orig = inverse_transform(scaler_y, y_cal)
        y_val_orig = inverse_transform(scaler_y, y_val)
        for params in candidates:
            model = build_fn(params)
            X_tr = X_train_flat
            y_tr = y_train
            if max_target is not None and len(X_tr) > max_target:
                X_tr = X_tr[:max_target]
                y_tr = y_tr[:max_target]
            if model_type == "kde":
                model.fit(X_tr, y_tr, X_val_flat, y_val)
            else:
                model.fit(X_tr, y_tr)

            lower_cal, upper_cal = predict_interval_orig(model, X_cal_flat, scaler_y)
            lower_val, upper_val = predict_interval_orig(model, X_val_flat, scaler_y)
            lower_cal, upper_cal = clip_intervals(lower_cal, upper_cal, 0.0, nominal_capacity)
            lower_val, upper_val = clip_intervals(lower_val, upper_val, 0.0, nominal_capacity)

            for conformal_scaled in conformal_candidates:
                q_hat_val = conformal_adjustment(
                    y_cal_orig,
                    lower_cal,
                    upper_cal,
                    alpha=args.alpha,
                    scaled=conformal_scaled,
                )
                q_scale, metrics_val = select_q_scale(
                    y_val_orig,
                    lower_val,
                    upper_val,
                    q_hat_val,
                    conformal_scaled,
                    target_picp=target_picp,
                    alpha=args.alpha,
                    nominal_capacity=nominal_capacity,
                    scale_grid=scale_grid,
                )
                if metrics_val["picp"] >= target_picp:
                    key = (metrics_val["pinaw"], metrics_val["winkler"], q_scale)
                    if best_feasible_key is None or key < best_feasible_key:
                        best_feasible_key = key
                        best = {
                            "params": params,
                            "metrics_val": metrics_val,
                            "q_scale": q_scale,
                            "conformal_scaled": conformal_scaled,
                        }
                else:
                    key = (metrics_val["cwc"], metrics_val["winkler"], -metrics_val["picp"])
                    if best_infeasible_key is None or key < best_infeasible_key:
                        best_infeasible_key = key
                        best_infeasible = {
                            "params": params,
                            "metrics_val": metrics_val,
                            "q_scale": q_scale,
                            "conformal_scaled": conformal_scaled,
                        }

        if best is None and best_infeasible is not None:
            best = best_infeasible

        if best is None:
            raise RuntimeError(f"{name} search failed.")

        conformal_scaled = bool(best.get("conformal_scaled", False))

        # final train on train+val
        model = build_fn(best["params"])
        X_tr = np.concatenate([X_train_flat, X_val_flat], axis=0)
        y_tr = np.concatenate([y_train, y_val], axis=0)
        if max_target is not None and len(X_tr) > max_target:
            X_tr = X_tr[:max_target]
            y_tr = y_tr[:max_target]
        if model_type == "kde":
            model.fit(X_tr, y_tr, X_cal_flat, y_cal)
        else:
            model.fit(X_tr, y_tr)

        lower_cal, upper_cal = predict_interval_orig(model, X_cal_flat, scaler_y)
        lower_val, upper_val = predict_interval_orig(model, X_val_flat, scaler_y)
        lower_test, upper_test = predict_interval_orig(model, X_T_test_flat, scaler_y)
        lower_cal, upper_cal = clip_intervals(lower_cal, upper_cal, 0.0, nominal_capacity)
        lower_val, upper_val = clip_intervals(lower_val, upper_val, 0.0, nominal_capacity)

        q_hat = conformal_adjustment(
            y_cal_orig,
            lower_cal,
            upper_cal,
            alpha=args.alpha,
            scaled=conformal_scaled,
        )
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
        lower_test, upper_test = apply_conformal(
            lower_test,
            upper_test,
            q_hat * q_scale,
            scaled=conformal_scaled,
        )
        lower_test, upper_test = clip_intervals(lower_test, upper_test, 0.0, nominal_capacity)
        y_test_orig = inverse_transform(scaler_y, y_T_test)
        metrics_test = interval_metrics(y_test_orig, lower_test, upper_test, alpha=args.alpha)

        best_params = dict(best["params"])
        best_params["q_scale"] = float(q_scale)
        best_params["conformal_scaled"] = conformal_scaled
        _save_result(
            name,
            metrics_test,
            q_hat,
            train_samples=len(X_tr),
            val_samples=len(X_val_flat),
            cal_samples=len(X_cal_flat),
            test_samples=len(X_T_test_flat),
            best_params=best_params,
            search_metrics=best["metrics_val"],
        )

    _search_and_run(
        "TSVQR",
        [
            {"C": 6.0, "gamma": 0.008, "max_samples": 1000},
            {"C": 8.0, "gamma": 0.01, "max_samples": 1200},
            {"C": 12.0, "gamma": 0.015, "max_samples": 1200},
        ],
        lambda p: TSVQRIntervalModel(TSVQRConfig(C=p["C"], gamma=p["gamma"], max_samples=p["max_samples"])),
        conformal_candidates=[False, True],
        max_target=1200,
    )

    _search_and_run(
        "UQSVM-SSVQR",
        [
            {"gamma": 0.015, "n_components": 280, "alpha": 0.0015},
            {"gamma": 0.02, "n_components": 320, "alpha": 0.002},
            {"gamma": 0.025, "n_components": 360, "alpha": 0.002},
        ],
        lambda p: SparseSVQRIntervalModel(
            SSVQRConfig(gamma=p["gamma"], n_components=p["n_components"], alpha=p["alpha"])
        ),
        conformal_candidates=[False, True],
        max_target=2500,
    )

    _search_and_run(
        "NFS-SVQR",
        [
            {"gamma": 0.015, "n_components": 360, "max_features": 100},
            {"gamma": 0.02, "n_components": 420, "max_features": 120},
            {"gamma": 0.025, "n_components": 480, "max_features": 140},
        ],
        lambda p: NFSSVQRIntervalModel(
            NFSSVQRConfig(gamma=p["gamma"], n_components=p["n_components"], max_features=p["max_features"])
        ),
        conformal_candidates=[False, True],
        max_target=2500,
    )

    _search_and_run(
        "NuSVR-CI",
        [
            {"nu": 0.4, "C": 6.0, "gamma": 0.008},
            {"nu": 0.6, "C": 8.0, "gamma": 0.01},
            {"nu": 0.8, "C": 10.0, "gamma": 0.012},
        ],
        lambda p: NuSVRIntervalModel(NuSVRConfig(nu=p["nu"], C=p["C"], gamma=p["gamma"])),
        conformal_candidates=[False, True],
        max_target=2200,
    )

    _search_and_run(
        "NCQ-GBR",
        [
            {"n_estimators": 180, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20},
            {"n_estimators": 240, "learning_rate": 0.04, "max_depth": 3, "min_samples_leaf": 30},
            {"n_estimators": 260, "learning_rate": 0.05, "max_depth": 4, "min_samples_leaf": 25},
        ],
        lambda p: QuantileGBRIntervalModel(
            QuantileGBRConfig(
                n_estimators=p["n_estimators"],
                learning_rate=p["learning_rate"],
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
            )
        ),
        conformal_candidates=[False, True],
        max_target=None,
    )

    _search_and_run(
        "NESCQR",
        [
            {"hidden_dims": (64, 32), "epochs": 20, "n_members": 3},
        ],
        lambda p: MLPQuantileEnsembleIntervalModel(
            input_dim=X_train_flat.shape[1],
            config=MLPQuantileConfig(
                hidden_dims=tuple(p["hidden_dims"]),
                epochs=p["epochs"],
                n_members=p["n_members"],
            ),
        ),
        conformal_candidates=[False, True],
        max_target=None,
    )

    _search_and_run(
        "ENS-KDE",
        [
            {"n_bins": 5, "kde_bandwidth": 0.12, "kde_samples": 4000},
        ],
        lambda p: EnsembleKDEIntervalModel(
            EnsembleKDEConfig(
                n_bins=p["n_bins"],
                kde_bandwidth=p["kde_bandwidth"],
                kde_samples=p["kde_samples"],
            )
        ),
        conformal_candidates=[False, True],
        max_target=None,
        model_type="kde",
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["cwc", "winkler"])
    summary_df.to_csv(results_dir / "summary_metrics.csv", index=False)

    print("\nSaved results to:")
    print(results_dir)


if __name__ == "__main__":
    main()
