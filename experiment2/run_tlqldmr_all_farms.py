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
from experiment2.models.tl_qldmr_median import TLQLDMRMedianModel, TLQLDMRConfig


def inverse_transform(scaler, y_scaled: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def to_native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


def split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2):
    n_total = len(X)
    n_val = max(1, int(n_total * val_ratio))
    n_train = max(1, n_total - n_val)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def sample_from_space(rng: np.random.Generator, space: dict) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def tune_tlqldmr_for_farm(
    farm_idx: int,
    results_dir: Path,
    seed: int,
    n_trials: int,
    data_space: dict,
    hyperparam_space: dict,
):
    rng = np.random.default_rng(seed)
    data_cache = {}

    best_key = None
    best_record = None

    for trial in range(n_trials):
        data_cond = sample_from_space(rng, data_space)
        cache_key = (
            data_cond["feature_set"],
            int(data_cond["window_size"]),
            float(data_cond["target_train_ratio"]),
        )
        if cache_key in data_cache:
            data = data_cache[cache_key]
        else:
            data = prepare_farm_data(
                farm_idx=farm_idx,
                feature_set=data_cond["feature_set"],
                window_size=int(data_cond["window_size"]),
                target_train_ratio=float(data_cond["target_train_ratio"]),
                max_source_samples=None,
                max_target_train_samples=None,
            )
            data_cache[cache_key] = data

        scaler_y = data["scaler_y"]
        X_S = data["X_S_flat"]
        y_S = data["y_S"]
        X_T_train = data["X_T_train_flat"]
        y_T_train = data["y_T_train"]

        X_T_tr, y_T_tr, X_T_val, y_T_val = split_train_val(X_T_train, y_T_train)

        params = sample_from_space(rng, hyperparam_space)
        cfg = TLQLDMRConfig(
            lambda1=float(params["lambda1"]),
            lambda2=float(params["lambda2"]),
            C_S=float(params["C_S"]),
            C_T=float(params["C_T"]),
            tau=0.5,
            kernel_gamma=float(params["kernel_gamma"]),
            solver="fast_nystrom",
            nystrom_n_components=int(params["nystrom_n_components"]),
            nystrom_lr=float(params["nystrom_lr"]),
            nystrom_epochs=int(params["nystrom_epochs"]),
            nystrom_batch_size=int(params["nystrom_batch_size"]),
        )

        model = TLQLDMRMedianModel(cfg)
        model.fit(X_S, y_S, X_T_tr, y_T_tr)
        y_pred_scaled = model.predict(X_T_val)
        y_true = inverse_transform(scaler_y, y_T_val)
        y_pred = inverse_transform(scaler_y, y_pred_scaled)
        y_pred = np.maximum(y_pred, 0)

        metrics = compute_metrics(y_true, y_pred)
        key = (metrics["rmse"], metrics["mae"], -metrics["r2"])

        if best_key is None or key < best_key:
            best_key = key
            best_record = {
                "data_cond": data_cond,
                "params": params,
                "metrics_val": metrics,
                "farm_name": data["farm_name"],
            }

    # Final retrain with best params on full target train
    data = prepare_farm_data(
        farm_idx=farm_idx,
        feature_set=best_record["data_cond"]["feature_set"],
        window_size=int(best_record["data_cond"]["window_size"]),
        target_train_ratio=float(best_record["data_cond"]["target_train_ratio"]),
        max_source_samples=None,
        max_target_train_samples=None,
    )
    scaler_y = data["scaler_y"]
    X_S = data["X_S_flat"]
    y_S = data["y_S"]
    X_T_train = data["X_T_train_flat"]
    y_T_train = data["y_T_train"]
    X_T_test = data["X_T_test_flat"]
    y_T_test = data["y_T_test"]

    params = dict(best_record["params"])
    params["nystrom_epochs"] = max(int(params["nystrom_epochs"]), 20)

    cfg = TLQLDMRConfig(
        lambda1=float(params["lambda1"]),
        lambda2=float(params["lambda2"]),
        C_S=float(params["C_S"]),
        C_T=float(params["C_T"]),
        tau=0.5,
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

    metrics_test = compute_metrics(y_true, y_pred)

    result = {
        "farm_idx": farm_idx,
        "farm_name": data["farm_name"],
        "best_metrics": metrics_test,
        "best_config": {
            **best_record["data_cond"],
            **params,
        },
        "data_space": data_space,
        "hyperparam_space": hyperparam_space,
        "n_trials": n_trials,
        "seed": seed,
        "selection": "min_rmse_then_mae_then_max_r2",
    }

    out_file = results_dir / f"farm{farm_idx}_TLQLDMR_best.json"
    out_file.write_text(json.dumps(to_native(result), indent=2), encoding="utf-8")

    return result


def main():
    set_seed(42)

    out_dir = Path(__file__).resolve().parent
    results_dir = out_dir / "results_tlqldmr_all"
    results_dir.mkdir(exist_ok=True, parents=True)

    data_space = {
        "feature_set": ["full", "wind_only", "simple"],
        "window_size": [3, 6, 12],
        "target_train_ratio": [0.5, 0.6, 0.7],
    }

    hyperparam_space = {
        "lambda1": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "lambda2": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        "C_S": [0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        "C_T": [1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
        "kernel_gamma": [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        "nystrom_n_components": [100, 150, 200, 300],
        "nystrom_lr": [0.003, 0.005, 0.01, 0.02],
        "nystrom_epochs": [8, 12],
        "nystrom_batch_size": [256],
    }

    n_trials = 8
    seed = 42

    summary_rows = []
    for farm_idx in range(6):
        out_file = results_dir / f"farm{farm_idx}_TLQLDMR_best.json"
        if out_file.exists():
            payload = json.loads(out_file.read_text(encoding="utf-8"))
            summary_rows.append(
                {
                    "farm_idx": payload["farm_idx"],
                    "farm_name": payload["farm_name"],
                    **to_native(payload["best_metrics"]),
                    **to_native(payload["best_config"]),
                }
            )
            continue

        print("\n" + "=" * 70)
        print(f"TL-QLDMR tuning for farm {farm_idx}")
        print("=" * 70)
        result = tune_tlqldmr_for_farm(
            farm_idx=farm_idx,
            results_dir=results_dir,
            seed=seed,
            n_trials=n_trials,
            data_space=data_space,
            hyperparam_space=hyperparam_space,
        )
        summary_rows.append(
            {
                "farm_idx": result["farm_idx"],
                "farm_name": result["farm_name"],
                **to_native(result["best_metrics"]),
                **to_native(result["best_config"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "tlqldmr_all_farms_summary.csv", index=False)

    print("\nSaved results to:")
    print(f"- {results_dir / 'tlqldmr_all_farms_summary.csv'}")
    for farm_idx in range(6):
        print(f"- {results_dir / f'farm{farm_idx}_TLQLDMR_best.json'}")


if __name__ == "__main__":
    main()
