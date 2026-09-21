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
from experiment2.metrics import compute_metrics
from experiment2.models.tl_qldmr_median import TLQLDMRConfig, TLQLDMRMedianModel


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


def _split_train_val(
    X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(X)
    n_val = max(1, int(n_total * val_ratio))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def _candidate_params() -> list[dict]:
    # Keep aligned with experiment2/run_tlqldmr_hunt.py candidate settings.
    return [
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(REPO_ROOT, "experiment2", "results_hunt", "tlqldmr_hunt_best.json"),
    )
    parser.add_argument("--snr-db", type=str, default="60,20")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-configs", type=int, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    base_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data = prepare_farm_data(
        farm_idx=base_cfg["farm_idx"],
        feature_set=base_cfg["feature_set"],
        window_size=base_cfg["window_size"],
        target_train_ratio=base_cfg["target_train_ratio"],
        extreme_cfg=base_cfg.get("extreme_cfg"),
        split_mode="shuffle",
        split_seed=base_cfg.get("split_seed", 0),
    )

    scaler_y = data["scaler_y"]
    X_S = data["X_S_flat"]
    y_S = data["y_S"]
    X_T_train = data["X_T_train_flat"]
    y_T_train = data["y_T_train"]

    X_T_tr, y_T_tr, X_T_val, y_T_val = _split_train_val(
        X_T_train, y_T_train, val_ratio=args.val_ratio, seed=args.seed
    )

    X_ref = np.vstack([X_S, X_T_tr])
    signal_power = np.mean(X_ref ** 2, axis=0)

    snr_list = _parse_snr_list(args.snr_db)
    candidates = _candidate_params()
    if args.max_configs is not None:
        candidates = candidates[: max(1, args.max_configs)]

    rows: list[dict] = []
    best = None
    best_key = None

    for idx, params in enumerate(candidates):
        r2_scores = []
        rmse_scores = []
        for jdx, snr_db in enumerate(snr_list):
            set_seed(args.seed + idx * 100 + jdx)
            rng = np.random.default_rng(args.seed + idx * 100 + jdx)
            noise_std_vec = _snr_to_noise_std(signal_power, float(snr_db))

            X_S_use = _add_noise_with_std(X_S, noise_std_vec, rng)
            X_T_tr_use = _add_noise_with_std(X_T_tr, noise_std_vec, rng)
            X_T_val_use = _add_noise_with_std(X_T_val, noise_std_vec, rng)

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
            model.fit(X_S_use, y_S, X_T_tr_use, y_T_tr)

            y_pred_scaled = model.predict(X_T_val_use)
            y_true = inverse_transform(scaler_y, y_T_val)
            y_pred = inverse_transform(scaler_y, y_pred_scaled)
            y_pred = np.maximum(y_pred, 0.0)
            metrics = compute_metrics(y_true, y_pred)
            r2_scores.append(metrics["r2"])
            rmse_scores.append(metrics["rmse"])
            rows.append(
                {
                    "config_id": idx,
                    "snr_db": float(snr_db),
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    **params,
                }
            )

        mean_r2 = float(np.mean(r2_scores))
        mean_rmse = float(np.mean(rmse_scores))
        key = (-mean_r2, mean_rmse)
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "best_metrics": {"mean_r2": mean_r2, "mean_rmse": mean_rmse},
                "best_config": {
                    "feature_set": base_cfg["feature_set"],
                    "window_size": base_cfg["window_size"],
                    "target_train_ratio": base_cfg["target_train_ratio"],
                    **params,
                },
                "farm_idx": base_cfg["farm_idx"],
                "farm_name": base_cfg.get("farm_name"),
                "split_seed": base_cfg.get("split_seed", 0),
                "search_snr_db": args.snr_db,
                "val_ratio": args.val_ratio,
            }

    out_dir = Path(__file__).resolve().parent / "results_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "tlqldmr_noise_search.csv", index=False)
    if best is None:
        raise RuntimeError("No TL-QLDMR configs were evaluated.")
    (out_dir / "tlqldmr_noise_best.json").write_text(
        json.dumps(best, indent=2), encoding="utf-8"
    )

    print("Best TL-QLDMR config under noise:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
