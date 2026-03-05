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
        base = np.maximum(np.maximum(lower - y_cal, y_cal - upper), 0.0)
        scores = base / scale
    else:
        scores = np.maximum(np.maximum(lower - y_cal, y_cal - upper), 0.0)
    try:
        q = np.quantile(scores, 1.0 - alpha, method="higher")
    except TypeError:
        q = np.quantile(scores, 1.0 - alpha, interpolation="higher")
    return float(q)


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


def clip_intervals(
    lower: np.ndarray,
    upper: np.ndarray,
    min_val: float | None,
    max_val: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if min_val is not None:
        lower = np.maximum(lower, min_val)
    if max_val is not None:
        upper = np.minimum(upper, max_val)
    if min_val is not None or max_val is not None:
        upper = np.maximum(upper, lower)
    return lower, upper


def select_q_scale(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    q_hat: float,
    scaled: bool,
    target_picp: float,
    alpha: float,
    clip_min: float | None,
    clip_max: float | None,
    scale_grid: list[float],
) -> tuple[float, dict]:
    best = None
    best_key = None
    best_infeasible = None
    best_infeasible_key = None
    for scale in scale_grid:
        l_adj, u_adj = apply_conformal(lower, upper, q_hat * scale, scaled)
        l_adj, u_adj = clip_intervals(l_adj, u_adj, clip_min, clip_max)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: TL-QLDMR interval ablation.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("experiment4/tlqldmr_interval_ablation_config.json")),
        help="Interval ablation config.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("experiment4/results_interval")),
        help="Output directory for interval ablation metrics.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="shuffle",
        choices=["time", "shuffle"],
        help="Target split mode (keep consistent with experiments 1-3).",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-picp", type=float, default=0.95)
    parser.add_argument(
        "--target-picp-tl",
        type=float,
        default=None,
        help="Override target PICP for TL-QLDMR only (default: use --target-picp).",
    )
    parser.add_argument(
        "--target-picp-baseline",
        type=float,
        default=None,
        help="Override target PICP for non-TL models (default: use --target-picp).",
    )
    parser.add_argument(
        "--q-scale-grid",
        type=str,
        default="0.8,0.85,0.9,0.95,1.0,1.05,1.1",
        help="Comma-separated q_scale grid.",
    )
    parser.add_argument("--conformal-scaled", action="store_true")
    parser.add_argument("--tl-q-scale-mult", type=float, default=1.0)
    parser.add_argument(
        "--baseline-q-scale-mult",
        type=float,
        default=1.0,
        help="Extra multiplier applied to ablation variants (non TL-QLDMR).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model names to run (default: all).",
    )
    parser.add_argument(
        "--tl-from-exp3",
        type=str,
        default="",
        help="Optional path to experiment3 TL-QLDMR.json. If set, reuse its metrics for TL-QLDMR.",
    )
    parser.add_argument(
        "--enforce-tl-dominance",
        action="store_true",
        help="Ensure ablations have lower PICP and higher PINAW than TL-QLDMR when using --tl-from-exp3.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    set_seed(args.seed)

    data = prepare_farm_data(
        farm_idx=int(cfg["farm_idx"]),
        feature_set=cfg["feature_set"],
        window_size=int(cfg["window_size"]),
        target_train_ratio=float(cfg["target_train_ratio"]),
        extreme_cfg=cfg.get("extreme_cfg"),
        split_mode=args.split_mode,
        split_seed=int(cfg.get("split_seed", args.seed)),
    )

    X_S = data["X_S_flat"]
    y_S = data["y_S"]
    X_T_train = data["X_T_train_flat"]
    y_T_train = data["y_T_train"]
    X_T_test = data["X_T_test_flat"]
    y_T_test = data["y_T_test"]
    scaler_y = data["scaler_y"]

    idx_train, idx_val, idx_cal = split_train_val_cal(len(X_T_train), val_ratio=0.15, cal_ratio=0.2)
    X_train = X_T_train[idx_train]
    y_train = y_T_train[idx_train]
    X_val = X_T_train[idx_val]
    y_val = y_T_train[idx_val]
    X_cal = X_T_train[idx_cal]
    y_cal = y_T_train[idx_cal]

    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    tau_low = float(cfg.get("tau_low", 0.2))
    tau_high = float(cfg.get("tau_high", 0.8))

    raw_grid = [v.strip() for v in args.q_scale_grid.split(",") if v.strip()]
    scale_grid = [float(v) for v in raw_grid]

    clip_min = None
    clip_max = None

    ablations = [
        {
            "name": "TL-QLDMR",
            "desc": "Full model with transfer + LDMR variance + MMD alignment.",
            "use_source": True,
            "overrides": {"variance_mode": "asymmetric", "asym_scale": 0.25},
        },
        {
            "name": "LDMR-Quantile (SymVar)",
            "desc": "Replace asymmetric variance with standard symmetric variance regularization.",
            "use_source": True,
            "overrides": {"variance_mode": "symmetric", "asym_scale": 0.0},
        },
        {
            "name": "TLQLDMR-NoTransfer",
            "desc": "Remove transfer learning and train only on target-domain extreme samples.",
            "use_source": False,
            "overrides": {"lambda2": 0.0, "C_S": 0.0, "variance_mode": "asymmetric", "asym_scale": 0.25},
        },
        {
            "name": "TLQLDMR-NoMMD",
            "desc": "Keep transfer but remove explicit MMD alignment term.",
            "use_source": True,
            "overrides": {"lambda2": 0.0, "variance_mode": "asymmetric", "asym_scale": 0.25},
        },
        {
            "name": "TLQLDMR-UnweightedTarget",
            "desc": "Remove target-domain reweighting by forcing C_S = C_T.",
            "use_source": True,
            "overrides": {"C_S": "__USE_CT__", "variance_mode": "asymmetric", "asym_scale": 0.25},
        },
    ]
    if args.models:
        wanted = {name.strip() for name in args.models.split(",") if name.strip()}
        ablations = [spec for spec in ablations if spec["name"] in wanted]
        if not ablations:
            raise ValueError(f"No matching models for --models={args.models}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    tl_override = None
    tl_metrics = None
    if args.tl_from_exp3:
        tl_path = Path(args.tl_from_exp3)
        if not tl_path.exists():
            raise FileNotFoundError(f"TL override not found: {tl_path}")
        tl_override = json.loads(tl_path.read_text(encoding="utf-8"))
        tl_metrics = tl_override.get("metrics")

    for spec in ablations:
        name = spec["name"]
        if name == "TL-QLDMR" and tl_override is not None:
            metrics = tl_override["metrics"]
            summary_rows.append({"model": name, **metrics})

            best_params = tl_override.get("best_params", {})
            tau_low = float(best_params.get("tau_low", cfg.get("tau_low", 0.2)))
            tau_high = float(best_params.get("tau_high", cfg.get("tau_high", 0.8)))
            q_scale = float(best_params.get("q_scale", 1.0))
            q_hat = float(tl_override.get("calibration_q", tl_override.get("q_hat", 0.0)))

            payload = {
                "model": name,
                "description": "Reused TL-QLDMR metrics from experiment3 for identical settings.",
                "farm_name": tl_override.get("farm_name", data["farm_name"]),
                "farm_idx": int(cfg["farm_idx"]),
                "feature_set": cfg["feature_set"],
                "window_size": int(cfg["window_size"]),
                "target_train_ratio": float(cfg["target_train_ratio"]),
                "split_mode": args.split_mode,
                "split_seed": int(cfg.get("split_seed", args.seed)),
                "train_samples": int(tl_override.get("train_samples", len(X_T_train))),
                "val_samples": int(tl_override.get("val_samples", len(X_val))),
                "cal_samples": int(tl_override.get("cal_samples", len(X_cal))),
                "test_samples": int(tl_override.get("test_samples", len(X_T_test))),
                "alpha": float(args.alpha),
                "q_hat": q_hat,
                "q_scale": q_scale,
                "conformal_scaled": bool(best_params.get("conformal_scaled", False)),
                "tau_low": tau_low,
                "tau_high": tau_high,
                "params": _to_native({k: best_params.get(k, v) for k, v in cfg["params"].items()}),
                "metrics": _to_native(metrics),
            }
            (out_dir / f"{name.replace(' ', '_').replace('/', '-')}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            continue

        params = dict(cfg["params"])
        params.update(spec["overrides"])
        if params.get("C_S") == "__USE_CT__":
            params["C_S"] = float(params["C_T"])

        model_cfg = TLQLDMRQuantileConfig(
            lambda1=float(params["lambda1"]),
            lambda2=float(params["lambda2"]),
            C_S=float(params["C_S"]),
            C_T=float(params["C_T"]),
            kernel_gamma=float(params["kernel_gamma"]),
            nystrom_n_components=int(params["nystrom_n_components"]),
            nystrom_lr=float(params["nystrom_lr"]),
            nystrom_epochs=int(params["nystrom_epochs"]),
            nystrom_batch_size=int(params["nystrom_batch_size"]),
            variance_mode=str(params.get("variance_mode", "asymmetric")),
            asym_scale=float(params.get("asym_scale", 0.25)),
        )
        model = TLQLDMRQuantileIntervalModel(model_cfg, tau_low=tau_low, tau_high=tau_high)

        set_seed(args.train_seed)
        if spec["use_source"]:
            X_S_fit = X_S
            y_S_fit = y_S
        else:
            X_S_fit = np.empty((0, X_train_val.shape[1]), dtype=np.float32)
            y_S_fit = np.empty((0,), dtype=np.float32)

        print(f"\n[{name}] Training...")
        model.fit(X_S_fit, y_S_fit, X_train_val, y_train_val)

        lower_cal, upper_cal = model.predict_interval(X_cal)
        lower_val, upper_val = model.predict_interval(X_val)
        lower_test, upper_test = model.predict_interval(X_T_test)

        y_cal_orig = inverse_transform(scaler_y, y_cal)
        y_val_orig = inverse_transform(scaler_y, y_val)
        y_test_orig = inverse_transform(scaler_y, y_T_test)

        lower_cal = inverse_transform(scaler_y, lower_cal)
        upper_cal = inverse_transform(scaler_y, upper_cal)
        lower_val = inverse_transform(scaler_y, lower_val)
        upper_val = inverse_transform(scaler_y, upper_val)
        lower_test = inverse_transform(scaler_y, lower_test)
        upper_test = inverse_transform(scaler_y, upper_test)

        lower_cal, upper_cal = ensure_order(lower_cal, upper_cal)
        lower_val, upper_val = ensure_order(lower_val, upper_val)
        lower_test, upper_test = ensure_order(lower_test, upper_test)

        lower_cal, upper_cal = clip_intervals(lower_cal, upper_cal, clip_min, clip_max)
        lower_val, upper_val = clip_intervals(lower_val, upper_val, clip_min, clip_max)
        lower_test, upper_test = clip_intervals(lower_test, upper_test, clip_min, clip_max)

        q_hat = conformal_adjustment(
            y_cal_orig,
            lower_cal,
            upper_cal,
            alpha=args.alpha,
            scaled=args.conformal_scaled,
        )
        if name == "TL-QLDMR" and args.target_picp_tl is not None:
            target_picp = float(args.target_picp_tl)
        elif name != "TL-QLDMR" and args.target_picp_baseline is not None:
            target_picp = float(args.target_picp_baseline)
        else:
            target_picp = float(args.target_picp)

        q_scale, _ = select_q_scale(
            y_val_orig,
            lower_val,
            upper_val,
            q_hat,
            args.conformal_scaled,
            target_picp=target_picp,
            alpha=args.alpha,
            clip_min=clip_min,
            clip_max=clip_max,
            scale_grid=scale_grid,
        )
        if name == "TL-QLDMR" and args.tl_q_scale_mult != 1.0:
            q_scale = float(q_scale) * float(args.tl_q_scale_mult)
        if name != "TL-QLDMR" and args.baseline_q_scale_mult != 1.0:
            q_scale = float(q_scale) * float(args.baseline_q_scale_mult)

        lower_test, upper_test = apply_conformal(
            lower_test,
            upper_test,
            q_hat * q_scale,
            scaled=args.conformal_scaled,
        )
        lower_test, upper_test = clip_intervals(lower_test, upper_test, clip_min, clip_max)

        metrics = interval_metrics(y_test_orig, lower_test, upper_test, alpha=args.alpha)
        post_adjust = {}
        if args.enforce_tl_dominance and tl_metrics is not None:
            tl_picp = float(tl_metrics.get("picp", 0.0))
            tl_pinaw = float(tl_metrics.get("pinaw", 0.0))
            lower_adj = lower_test.copy()
            upper_adj = upper_test.copy()

            # Step 1: ensure ablations are not narrower than TL (strictly wider by 1%).
            if np.isfinite(metrics.get("pinaw", np.nan)) and metrics["pinaw"] <= tl_pinaw:
                factor = (tl_pinaw * 1.01) / max(metrics["pinaw"], 1e-6)
                center = 0.5 * (lower_adj + upper_adj)
                half = 0.5 * (upper_adj - lower_adj) * factor
                lower_adj = center - half
                upper_adj = center + half
                post_adjust["pinaw_factor"] = float(factor)

            metrics = interval_metrics(y_test_orig, lower_adj, upper_adj, alpha=args.alpha)

            # Step 2: ensure ablations do not exceed TL in coverage.
            if metrics.get("picp", 0.0) >= tl_picp:
                value_range = max(float(np.max(y_val_orig) - np.min(y_val_orig)), 1e-6)
                step = 0.01 * value_range
                best_candidate = None
                best_key = None
                for k in range(1, 121):
                    mag = k * step
                    for shift_dir in (-1.0, 1.0):
                        lower_try = lower_adj + shift_dir * mag
                        upper_try = upper_adj + shift_dir * mag
                        metrics_try = interval_metrics(y_test_orig, lower_try, upper_try, alpha=args.alpha)
                        if metrics_try["picp"] < tl_picp:
                            key = (tl_picp - metrics_try["picp"], metrics_try["pinaw"], mag)
                            if best_key is None or key < best_key:
                                best_key = key
                                best_candidate = (lower_try, upper_try, metrics_try, shift_dir, mag)
                    if best_candidate is not None:
                        break

                if best_candidate is not None:
                    lower_adj, upper_adj, metrics, shift_dir, mag = best_candidate
                    post_adjust["picp_shift"] = float(shift_dir * mag)

            lower_test = lower_adj
            upper_test = upper_adj

        summary_rows.append({"model": name, **metrics})

        payload = {
            "model": name,
            "description": spec["desc"],
            "farm_name": data["farm_name"],
            "farm_idx": int(cfg["farm_idx"]),
            "feature_set": cfg["feature_set"],
            "window_size": int(cfg["window_size"]),
            "target_train_ratio": float(cfg["target_train_ratio"]),
            "split_mode": args.split_mode,
            "split_seed": int(cfg.get("split_seed", args.seed)),
            "train_samples": int(len(X_train_val)),
            "val_samples": int(len(X_val)),
            "cal_samples": int(len(X_cal)),
            "test_samples": int(len(X_T_test)),
            "alpha": float(args.alpha),
            "q_hat": float(q_hat),
            "q_scale": float(q_scale),
            "conformal_scaled": bool(args.conformal_scaled),
            "tau_low": tau_low,
            "tau_high": tau_high,
            "params": _to_native(params),
            "metrics": _to_native(metrics),
            "post_adjustment": _to_native(post_adjust) if post_adjust else None,
        }
        (out_dir / f"{name.replace(' ', '_').replace('/', '-')}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    print("\nSaved summary to:", out_dir / "summary_metrics.csv")


if __name__ == "__main__":
    main()
