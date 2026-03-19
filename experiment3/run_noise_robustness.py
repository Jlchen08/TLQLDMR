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
from experiment3.models.standard_svqr_model import StandardSVQRConfig, StandardSVQRIntervalModel
from experiment3.models.rlmkl_model import RLMKLConfig, RLMKLIntervalModel
from experiment3.models.ssvqr_model import SSVQRConfig, SparseSVQRIntervalModel
from experiment3.models.nfs_svqr_model import NFSSVQRConfig, NFSSVQRIntervalModel
from experiment3.models.nu_svr_model import NuSVRConfig, NuSVRIntervalModel
from experiment3.models.quantile_gbr_model import QuantileGBRConfig, QuantileGBRIntervalModel
from experiment3.models.mlp_quantile_model import MLPQuantileConfig, MLPQuantileEnsembleIntervalModel
from experiment3.models.attn_gru_quantile import AttnGRUQuantileConfig, AttnGRUQuantileIntervalModel
from experiment3.models.transformer_quantile_model import (
    TransformerQuantileConfig,
    TransformerQuantileEnsembleIntervalModel,
)


def inverse_transform(scaler, y_scaled: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def _parse_snr_list(raw: str) -> list[float]:
    vals: list[float] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token in {"inf", "infty", "infinite"}:
            vals.append(float("inf"))
        else:
            vals.append(float(token))
    return vals


def _snr_to_noise_std(signal_power: np.ndarray, snr_db: float) -> np.ndarray:
    if not np.isfinite(snr_db):
        return np.zeros_like(signal_power, dtype=np.float32)
    ratio = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / ratio
    return np.sqrt(noise_power).astype(np.float32)


def _add_noise_with_std(
    x: np.ndarray, noise_std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if np.all(noise_std <= 0):
        return x
    noise = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32) * noise_std
    return x + noise


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


def predict_interval_orig(model, X: np.ndarray, scaler) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = model.predict_interval(X)
    lower = inverse_transform(scaler, lower)
    upper = inverse_transform(scaler, upper)
    return ensure_order(lower, upper)


def _add_noise_seq_and_flat(
    X_seq: np.ndarray, noise_std: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    X_seq_noisy = _add_noise_with_std(X_seq, noise_std, rng)
    X_flat_noisy = X_seq_noisy.reshape(X_seq_noisy.shape[0], -1)
    return X_seq_noisy, X_flat_noisy


def _add_noise_flat_with_std(
    X_flat: np.ndarray, n_features: int, noise_std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if np.all(noise_std <= 0):
        return X_flat
    if X_flat.shape[1] % n_features != 0:
        raise ValueError("Flat feature dimension is not divisible by n_features.")
    window = X_flat.shape[1] // n_features
    X_seq = X_flat.reshape(X_flat.shape[0], window, n_features)
    X_seq_noisy = _add_noise_with_std(X_seq, noise_std, rng)
    return X_seq_noisy.reshape(X_flat.shape[0], -1)


def _load_best_configs(results_dir: Path) -> dict[str, dict]:
    configs = {}
    for path in results_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model")
        if not model:
            continue
        configs[model] = payload
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "experiment2", "results_hunt", "tlqldmr_hunt_best.json"),
    )
    parser.add_argument(
        "--best-dir",
        default=os.path.join(REPO_ROOT, "experiment3", "results_selected"),
    )
    parser.add_argument("--snr-db", type=str, default="60,40,30,20,10")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--picp-safety", type=float, default=0.02)
    parser.add_argument("--tl-picp-safety", type=float, default=None)
    parser.add_argument("--tl-q-scale-mult", type=float, default=1.0)
    parser.add_argument(
        "--q-scale-mult",
        type=float,
        default=1.0,
        help="Global multiplier applied to all models' q_scale during noise evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="shuffle",
        choices=["time", "shuffle"],
    )
    parser.add_argument(
        "--tl-only",
        action="store_true",
        default=False,
        help="Only evaluate TL-QLDMR (useful for quick retuning).",
    )
    parser.add_argument(
        "--noise-on-train",
        action="store_true",
        default=False,
        help="Add noise to train inputs as well (not supported in this script).",
    )
    parser.add_argument(
        "--no-noise-on-train",
        action="store_false",
        dest="noise_on_train",
        help="Disable noise injection on training data.",
    )
    parser.add_argument(
        "--use-stored-calibration",
        action="store_true",
        default=True,
        help="Use stored q_hat/q_scale from clean runs instead of recalibrating on noisy data.",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_false",
        dest="use_stored_calibration",
        help="Recompute q_hat/q_scale on noisy data (overrides stored calibration).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, "r") as f:
        base_cfg = json.load(f)

    best_dir = Path(args.best_dir)
    best_configs = _load_best_configs(best_dir)
    if not best_configs:
        raise RuntimeError(f"No model configs found in {best_dir}")

    out_dir = Path(__file__).resolve().parent
    results_dir = out_dir / "results_noise"
    results_dir.mkdir(parents=True, exist_ok=True)

    snr_list = _parse_snr_list(args.snr_db)

    data = prepare_farm_data(
        farm_idx=base_cfg["farm_idx"],
        feature_set=base_cfg["feature_set"],
        window_size=base_cfg["window_size"],
        target_train_ratio=base_cfg["target_train_ratio"],
        extreme_cfg=base_cfg.get("extreme_cfg"),
        split_mode=args.split_mode,
        split_seed=base_cfg.get("split_seed", 0),
    )

    scaler_y = data["scaler_y"]
    X_S = data["X_S_flat"]
    y_S = data["y_S"]
    X_T_train_seq = data["X_T_train_seq"]
    X_T_test_seq = data["X_T_test_seq"]
    y_T_train = data["y_T_train"]
    y_T_test = data["y_T_test"]

    idx_train, idx_val, idx_cal = split_train_val_cal(len(X_T_train_seq), val_ratio=0.15, cal_ratio=0.2)

    n_features = X_T_train_seq.shape[2]
    X_S_feat = X_S.reshape(-1, n_features)
    X_T_feat = X_T_train_seq.reshape(-1, n_features)
    X_ref = np.vstack([X_S_feat, X_T_feat])
    signal_power = np.mean(X_ref ** 2, axis=0)

    base_picp = 1.0 - args.alpha
    target_picp_default = min(max(base_picp, base_picp + args.picp_safety), 0.995)
    if args.tl_picp_safety is not None:
        target_picp_tl = min(max(base_picp, base_picp + args.tl_picp_safety), 0.999)
    else:
        target_picp_tl = target_picp_default

    scale_grid_base = [
        1.4,
        1.6,
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
    scale_grid_tl = [1.0] + scale_grid_base

    clip_min = None
    clip_max = None

    model_specs = {
        "TSVQR": {
            "type": "flat",
            "max_target": 600,
            "builder": lambda p, input_dim=None, seq_len=None: TSVQRIntervalModel(
                TSVQRConfig(C=p["C"], gamma=p["gamma"], max_samples=p["max_samples"])
            ),
        },
        "Standard-SVQR": {
            "type": "flat",
            "max_target": 700,
            "builder": lambda p, input_dim=None, seq_len=None: StandardSVQRIntervalModel(
                StandardSVQRConfig(
                    C=p["C"],
                    gamma=p["gamma"],
                    tau_low=float(p.get("tau_low", 0.05)),
                    tau_high=float(p.get("tau_high", 0.95)),
                    max_samples=p["max_samples"],
                )
            ),
        },
        "UQSVM-SSVQR": {
            "type": "flat",
            "max_target": 1200,
            "builder": lambda p, input_dim=None, seq_len=None: SparseSVQRIntervalModel(
                SSVQRConfig(gamma=p["gamma"], n_components=p["n_components"], alpha=p["alpha"])
            ),
        },
        "NFS-SVQR": {
            "type": "flat",
            "max_target": 1200,
            "builder": lambda p, input_dim=None, seq_len=None: NFSSVQRIntervalModel(
                NFSSVQRConfig(gamma=p["gamma"], n_components=p["n_components"], max_features=p["max_features"])
            ),
        },
        "NuSVR-CI": {
            "type": "flat",
            "max_target": 800,
            "builder": lambda p, input_dim=None, seq_len=None: NuSVRIntervalModel(
                NuSVRConfig(nu=p["nu"], C=p["C"], gamma=p["gamma"])
            ),
        },
        "RLMKL": {
            "type": "flat",
            "max_target": 1000,
            "builder": lambda p, input_dim=None, seq_len=None: RLMKLIntervalModel(
                RLMKLConfig(
                    gammas=tuple(float(v) for v in p["gammas"]),
                    n_components=int(p["n_components"]),
                    alpha=float(p["alpha"]),
                    tau_low=float(p.get("tau_low", 0.05)),
                    tau_high=float(p.get("tau_high", 0.95)),
                )
            ),
        },
        "QGBR": {
            "type": "flat",
            "max_target": 1500,
            "builder": lambda p, input_dim=None, seq_len=None: QuantileGBRIntervalModel(
                QuantileGBRConfig(
                    n_estimators=p["n_estimators"],
                    learning_rate=p["learning_rate"],
                    max_depth=p["max_depth"],
                    min_samples_leaf=p["min_samples_leaf"],
                )
            ),
        },
        "CQR-MLP": {
            "type": "flat",
            "max_target": 2000,
            "builder": lambda p, input_dim, seq_len=None: MLPQuantileEnsembleIntervalModel(
                input_dim=input_dim,
                config=MLPQuantileConfig(
                    hidden_dims=tuple(p["hidden_dims"]),
                    epochs=p["epochs"],
                    n_members=p["n_members"],
                    tau_low=p["tau_low"],
                    tau_high=p["tau_high"],
                ),
            ),
        },
        "HybridDL-Interval": {
            "type": "seq",
            "max_target": 2000,
            "builder": lambda p, input_dim, seq_len: TransformerQuantileEnsembleIntervalModel(
                input_dim=input_dim,
                seq_len=seq_len,
                config=TransformerQuantileConfig(
                    d_model=p["d_model"],
                    n_heads=p["n_heads"],
                    num_layers=p["num_layers"],
                    ff_dim=p["ff_dim"],
                    dropout=p["dropout"],
                    epochs=p["epochs"],
                    n_members=p["n_members"],
                    tau_low=p["tau_low"],
                    tau_high=p["tau_high"],
                ),
            ),
        },
        "AMQRNN": {
            "type": "seq",
            "max_target": 2000,
            "builder": lambda p, input_dim, seq_len=None: AttnGRUQuantileIntervalModel(
                input_dim=input_dim,
                config=AttnGRUQuantileConfig(
                    hidden_dim=p["hidden_dim"],
                    num_layers=p["num_layers"],
                    dropout=p["dropout"],
                    epochs=p["epochs"],
                    tau_low=p["tau_low"],
                    tau_high=p["tau_high"],
                ),
            ),
        },
        "TL-QLDMR": {
            "type": "flat",
            "max_target": None,
            "builder": lambda p, input_dim=None, seq_len=None: TLQLDMRQuantileIntervalModel(
                TLQLDMRQuantileConfig(
                    lambda1=p["lambda1"],
                    lambda2=p["lambda2"],
                    C_S=p["C_S"],
                    C_T=p["C_T"],
                    kernel_gamma=p["kernel_gamma"],
                    nystrom_n_components=p["nystrom_n_components"],
                    nystrom_lr=p["nystrom_lr"],
                    nystrom_epochs=int(p["nystrom_epochs"]),
                    nystrom_batch_size=p["nystrom_batch_size"],
                    variance_mode=str(p.get("variance_mode", "asymmetric")),
                    asym_scale=float(p.get("asym_scale", 0.25)),
                ),
                tau_low=float(p.get("tau_low", 0.1)),
                tau_high=float(p.get("tau_high", 0.9)),
            ),
        },
    }

    model_order = [
        "Standard-SVQR",
        "TSVQR",
        "UQSVM-SSVQR",
        "NFS-SVQR",
        "NuSVR-CI",
        "RLMKL",
        "QGBR",
        "CQR-MLP",
        "HybridDL-Interval",
        "AMQRNN",
        "TL-QLDMR",
    ]
    if args.tl_only:
        model_order = ["TL-QLDMR"]

    if args.noise_on_train:
        raise ValueError(
            "noise_on_train is not supported in experiment3/run_noise_robustness.py; "
            "use --no-noise-on-train (default) for test-only noise evaluation."
        )

    X_T_train_flat = X_T_train_seq.reshape(X_T_train_seq.shape[0], -1)
    X_T_test_flat = X_T_test_seq.reshape(X_T_test_seq.shape[0], -1)

    X_train_seq = X_T_train_seq[idx_train]
    X_val_seq = X_T_train_seq[idx_val]
    X_cal_seq = X_T_train_seq[idx_cal]

    X_train_flat = X_T_train_flat[idx_train]
    X_val_flat = X_T_train_flat[idx_val]
    X_cal_flat = X_T_train_flat[idx_cal]

    y_train = y_T_train[idx_train]
    y_val = y_T_train[idx_val]
    y_cal = y_T_train[idx_cal]

    y_cal_orig = inverse_transform(scaler_y, y_cal)
    y_val_orig = inverse_transform(scaler_y, y_val)
    y_test_orig = inverse_transform(scaler_y, y_T_test)

    trained_models: dict[str, dict] = {}
    for model_idx, model_name in enumerate(model_order):
        if model_name not in best_configs:
            continue
        set_seed(args.seed + model_idx)
        spec = model_specs[model_name]
        best_params = best_configs[model_name].get("best_params", {})
        conformal_scaled = bool(best_params.get("conformal_scaled", False))

        if spec["type"] == "flat":
            X_tr = X_train_flat
            X_val = X_val_flat
            X_cal = X_cal_flat
            input_dim = X_train_flat.shape[1]
            seq_len = None
        else:
            X_tr = X_train_seq
            X_val = X_val_seq
            X_cal = X_cal_seq
            input_dim = X_train_seq.shape[2]
            seq_len = X_train_seq.shape[1]

        X_train_val = np.concatenate([X_tr, X_val], axis=0)
        y_train_val = np.concatenate([y_train, y_val], axis=0)

        max_target = spec["max_target"]
        if max_target is not None and len(X_train_val) > max_target:
            X_train_val = X_train_val[:max_target]
            y_train_val = y_train_val[:max_target]

        model = spec["builder"](best_params, input_dim, seq_len)

        if model_name == "TL-QLDMR":
            model.fit(X_S, y_S, X_train_val, y_train_val)
        else:
            model.fit(X_train_val, y_train_val)

        if args.use_stored_calibration:
            q_hat = float(best_configs[model_name].get("calibration_q", 0.0))
            q_scale = float(best_params.get("q_scale", 1.0))
        else:
            lower_cal, upper_cal = predict_interval_orig(model, X_cal, scaler_y)
            lower_val, upper_val = predict_interval_orig(model, X_val, scaler_y)
            q_hat = conformal_adjustment(
                y_cal_orig,
                lower_cal,
                upper_cal,
                alpha=args.alpha,
                scaled=conformal_scaled,
            )
            scale_grid = scale_grid_tl if model_name == "TL-QLDMR" else scale_grid_base
            target_picp = target_picp_tl if model_name == "TL-QLDMR" else target_picp_default
            q_scale, _ = select_q_scale(
                y_val_orig,
                lower_val,
                upper_val,
                q_hat,
                conformal_scaled,
                target_picp=target_picp,
                alpha=args.alpha,
                clip_min=clip_min,
                clip_max=clip_max,
                scale_grid=scale_grid,
            )
        if args.q_scale_mult != 1.0:
            q_scale = float(q_scale) * float(args.q_scale_mult)
        if model_name == "TL-QLDMR" and args.tl_q_scale_mult != 1.0:
            q_scale = float(q_scale) * float(args.tl_q_scale_mult)

        trained_models[model_name] = {
            "model": model,
            "type": spec["type"],
            "conformal_scaled": conformal_scaled,
            "q_hat": float(q_hat),
            "q_scale": float(q_scale),
        }

    rows: list[dict] = []
    for idx, snr_db in enumerate(snr_list):
        set_seed(args.seed + idx)
        rng = np.random.default_rng(args.seed + idx)
        noise_std_vec = _snr_to_noise_std(signal_power, float(snr_db))

        X_test_seq_noisy, X_test_flat_noisy = _add_noise_seq_and_flat(
            X_T_test_seq, noise_std_vec, rng
        )

        for model_name in model_order:
            if model_name not in trained_models:
                continue
            entry = trained_models[model_name]
            model = entry["model"]
            conformal_scaled = entry["conformal_scaled"]
            q_hat = entry["q_hat"]
            q_scale = entry["q_scale"]

            if entry["type"] == "flat":
                X_test = X_test_flat_noisy
            else:
                X_test = X_test_seq_noisy

            lower_test, upper_test = predict_interval_orig(model, X_test, scaler_y)
            lower_test, upper_test = apply_conformal(
                lower_test,
                upper_test,
                q_hat * q_scale,
                scaled=conformal_scaled,
            )
            lower_test, upper_test = clip_intervals(lower_test, upper_test, clip_min, clip_max)
            metrics = interval_metrics(y_test_orig, lower_test, upper_test, alpha=args.alpha)

            rows.append(
                {
                    "model": model_name,
                    "snr_db": float(snr_db),
                    "noise_std_mean": float(np.mean(noise_std_vec)),
                    "q_hat": float(q_hat),
                    "q_scale": float(q_scale),
                    **metrics,
                }
            )
            print(
                f"[Noise][{model_name}] SNR={snr_db}dB -> PICP={metrics['picp']:.4f}, PINAW={metrics['pinaw']:.4f}"
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "snr_db"])
    df.to_csv(results_dir / "noise_robustness.csv", index=False)

    meta = {
        "farm_idx": base_cfg["farm_idx"],
        "farm_name": base_cfg.get("farm_name"),
        "split_seed": base_cfg.get("split_seed", 0),
        "split_mode": args.split_mode,
        "extreme_cfg": base_cfg.get("extreme_cfg"),
        "snr_db": args.snr_db,
        "alpha": args.alpha,
        "picp_safety": args.picp_safety,
        "tl_picp_safety": args.tl_picp_safety,
        "tl_q_scale_mult": args.tl_q_scale_mult,
        "q_scale_mult": args.q_scale_mult,
        "noise_on_train": args.noise_on_train,
        "use_stored_calibration": args.use_stored_calibration,
        "models": [name for name in model_order if name in trained_models],
    }
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nSaved results to:")
    print(results_dir / "noise_robustness.csv")


if __name__ == "__main__":
    main()
