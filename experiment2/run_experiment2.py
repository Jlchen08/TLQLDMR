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
from experiment2.plots import plot_prediction_with_zooms
from experiment2.models.svr_model import SVRModel, SVRConfig
from experiment2.models.elm_model import ELMModel, ELMConfig
from experiment2.models.lstm_model import LSTMModel, LSTMConfig
from experiment2.models.cnn_lstm_model import CNNLSTMModel, CNNLSTMConfig
from experiment2.models.tl_svr_model import TLSVRModel, TLSVRConfig
from experiment2.models.tl_lstm_model import TLLSTMModel, TLLSTMConfig
from experiment2.models.tl_qldmr_median import TLQLDMRMedianModel, TLQLDMRConfig


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


def _split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2):
    n_total = len(X)
    n_val = max(1, int(n_total * val_ratio))
    n_train = max(1, n_total - n_val)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def _sample_from_space(rng: np.random.Generator, space: dict) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def _build_model(model_name: str, params: dict, input_size: int | None):
    if model_name == "SVR":
        return SVRModel(
            SVRConfig(
                C=float(params["C"]),
                epsilon=float(params["epsilon"]),
                gamma=float(params["gamma"]),
                kernel="rbf",
            )
        )
    if model_name == "ELM":
        return ELMModel(
            ELMConfig(
                n_hidden=int(params["n_hidden"]),
                activation=str(params["activation"]),
                reg=float(params["reg"]),
            )
        )
    if model_name == "LSTM":
        return LSTMModel(
            input_size,
            LSTMConfig(
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
                lr=float(params["lr"]),
                epochs=int(params["epochs"]),
                patience=2,
            ),
        )
    if model_name == "CNN-LSTM":
        return CNNLSTMModel(
            input_size,
            CNNLSTMConfig(
                conv_channels=int(params["conv_channels"]),
                kernel_size=int(params["kernel_size"]),
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
                lr=float(params["lr"]),
                epochs=int(params["epochs"]),
                patience=2,
            ),
        )
    if model_name == "TL-SVR":
        return TLSVRModel(
            TLSVRConfig(
                C=float(params["C"]),
                epsilon=float(params["epsilon"]),
                gamma=float(params["gamma"]),
                kernel="rbf",
                source_weight=float(params["source_weight"]),
            )
        )
    if model_name == "TL-LSTM":
        return TLLSTMModel(
            input_size,
            TLLSTMConfig(
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
                lr=float(params["lr"]),
                pretrain_epochs=int(params["pretrain_epochs"]),
                finetune_epochs=int(params["finetune_epochs"]),
                patience=2,
            ),
        )
    if model_name == "TL-QLDMR(Median)":
        return TLQLDMRMedianModel(
            TLQLDMRConfig(
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
        )
    raise ValueError(f"Unknown model: {model_name}")


def _get_data(
    data_cache: dict,
    farm_idx: int,
    feature_set: str,
    window_size: int,
    target_train_ratio: float,
):
    key = (feature_set, window_size, target_train_ratio)
    if key in data_cache:
        return data_cache[key]
    data = prepare_farm_data(
        farm_idx=farm_idx,
        feature_set=feature_set,
        window_size=window_size,
        target_train_ratio=target_train_ratio,
        max_source_samples=None,
        max_target_train_samples=None,
    )
    data_cache[key] = data
    return data


def search_model_best(
    model_name: str,
    input_type: str,
    transfer: bool,
    search_space: dict,
    n_trials: int,
    data_space: dict,
    data_cache: dict,
    farm_idx: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    best_key = None
    best = None

    for trial in range(n_trials):
        data_cond = _sample_from_space(rng, data_space)
        data = _get_data(
            data_cache,
            farm_idx,
            data_cond["feature_set"],
            int(data_cond["window_size"]),
            float(data_cond["target_train_ratio"]),
        )
        scaler_y = data["scaler_y"]

        if input_type == "flat":
            X_T_train = data["X_T_train_flat"]
            X_T_test = data["X_T_test_flat"]
            X_S = data["X_S_flat"]
            input_size = None
        else:
            X_T_train = data["X_T_train_seq"]
            X_T_test = data["X_T_test_seq"]
            X_S = data["X_S_seq"]
            input_size = X_T_train.shape[-1]

        y_T_train = data["y_T_train"]
        y_T_test = data["y_T_test"]
        y_S = data["y_S"]

        # Downsample for heavy kernel models to keep runtime manageable
        if model_name in {"SVR", "TL-SVR"}:
            max_src = 10000
            max_tgt = 5000
            X_S = X_S[:max_src]
            y_S = y_S[:max_src]
            X_T_train = X_T_train[:max_tgt]
            y_T_train = y_T_train[:max_tgt]

        X_T_tr, y_T_tr, X_T_val, y_T_val = _split_train_val(X_T_train, y_T_train)

        params = _sample_from_space(rng, search_space)
        model = _build_model(model_name, params, input_size)

        if transfer:
            model.fit(X_S, y_S, X_T_tr, y_T_tr)
        else:
            model.fit(X_T_tr, y_T_tr)

        y_pred_scaled = model.predict(X_T_val)
        y_true = inverse_transform(scaler_y, y_T_val)
        y_pred = inverse_transform(scaler_y, y_pred_scaled)
        y_pred = np.maximum(y_pred, 0)

        metrics = compute_metrics(y_true, y_pred)
        key = (metrics["rmse"], metrics["mae"], -metrics["r2"])

        if best_key is None or key < best_key:
            best_key = key
            best = {
                "data_cond": data_cond,
                "params": params,
                "metrics_val": metrics,
            }

    # Final retrain on full target train using best
    data = _get_data(
        data_cache,
        farm_idx,
        best["data_cond"]["feature_set"],
        int(best["data_cond"]["window_size"]),
        float(best["data_cond"]["target_train_ratio"]),
    )
    scaler_y = data["scaler_y"]

    if input_type == "flat":
        X_T_train = data["X_T_train_flat"]
        X_T_test = data["X_T_test_flat"]
        X_S = data["X_S_flat"]
        input_size = None
    else:
        X_T_train = data["X_T_train_seq"]
        X_T_test = data["X_T_test_seq"]
        X_S = data["X_S_seq"]
        input_size = X_T_train.shape[-1]

    y_T_train = data["y_T_train"]
    y_T_test = data["y_T_test"]
    y_S = data["y_S"]

    if model_name in {"SVR", "TL-SVR"}:
        max_src = 10000
        max_tgt = 5000
        X_S = X_S[:max_src]
        y_S = y_S[:max_src]
        X_T_train = X_T_train[:max_tgt]
        y_T_train = y_T_train[:max_tgt]

    # For TL-QLDMR, use slightly longer final training
    if model_name == "TL-QLDMR(Median)":
        params = dict(best["params"])
        params["nystrom_epochs"] = max(int(params["nystrom_epochs"]), 20)
    else:
        params = dict(best["params"])

    model = _build_model(model_name, params, input_size)
    if transfer:
        model.fit(X_S, y_S, X_T_train, y_T_train)
    else:
        model.fit(X_T_train, y_T_train)

    y_pred_scaled = model.predict(X_T_test)
    y_true = inverse_transform(scaler_y, y_T_test)
    y_pred = inverse_transform(scaler_y, y_pred_scaled)
    y_pred = np.maximum(y_pred, 0)

    metrics_test = compute_metrics(y_true, y_pred)

    max_source_samples = 10000 if model_name in {"SVR", "TL-SVR"} else None
    max_target_train_samples = 5000 if model_name in {"SVR", "TL-SVR"} else None

    return {
        "model": model_name,
        "best_metrics": metrics_test,
        "best_config": {
            **best["data_cond"],
            **params,
            "max_source_samples": max_source_samples,
            "max_target_train_samples": max_target_train_samples,
        },
    }


def main():
    set_seed(42)

    out_dir = Path(__file__).resolve().parent
    plots_dir = out_dir / "plots"
    results_dir = out_dir / "results"
    plots_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    farm_idx = 4
    generate_plots = False

    data_space = {
        "feature_set": ["full", "wind_only", "simple"],
        "window_size": [3, 6, 12],
        "target_train_ratio": [0.5, 0.6, 0.7],
    }

    model_specs = [
        ("SVR", "flat", False, 10),
        ("ELM", "flat", False, 8),
        ("LSTM", "seq", False, 8),
        ("CNN-LSTM", "seq", False, 8),
        ("TL-SVR", "flat", True, 4),
        ("TL-LSTM", "seq", True, 4),
        ("TL-QLDMR(Median)", "flat", True, 6),
    ]

    search_spaces = {
        "SVR": {
            "C": [1, 10, 100, 300],
            "epsilon": [0.01, 0.05, 0.1, 0.2],
            "gamma": [0.001, 0.01, 0.1, 0.5],
        },
        "ELM": {
            "n_hidden": [200, 400, 800, 1200],
            "activation": ["sigmoid", "tanh", "relu"],
            "reg": [1e-4, 1e-3, 1e-2, 1e-1],
        },
        "LSTM": {
            "hidden_size": [32, 64, 128],
            "num_layers": [1, 2],
            "dropout": [0.0, 0.2, 0.4],
            "lr": [1e-3, 5e-4],
            "epochs": [6, 10, 15],
        },
        "CNN-LSTM": {
            "conv_channels": [16, 32, 64],
            "kernel_size": [3, 5],
            "hidden_size": [32, 64],
            "num_layers": [1, 2],
            "dropout": [0.0, 0.2],
            "lr": [1e-3, 5e-4],
            "epochs": [6, 10, 15],
        },
        "TL-SVR": {
            "C": [1, 10, 100, 300],
            "epsilon": [0.01, 0.05, 0.1, 0.2],
            "gamma": [0.001, 0.01, 0.1, 0.5],
            "source_weight": [0.1, 0.2, 0.5, 1.0],
        },
        "TL-LSTM": {
            "hidden_size": [32, 64, 128],
            "num_layers": [1, 2],
            "dropout": [0.0, 0.2, 0.4],
            "lr": [1e-3, 5e-4],
            "pretrain_epochs": [4, 6, 8],
            "finetune_epochs": [4, 6, 8],
        },
        "TL-QLDMR(Median)": {
            "lambda1": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "lambda2": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            "C_S": [0.1, 0.5, 1.0, 5.0, 10.0],
            "C_T": [1.0, 5.0, 10.0, 20.0, 50.0],
            "kernel_gamma": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
            "nystrom_n_components": [100, 150, 200, 300],
            "nystrom_lr": [0.005, 0.01, 0.02],
            "nystrom_epochs": [8, 12],
            "nystrom_batch_size": [256],
        },
    }

    data_cache = {}
    summary_rows = []

    for model_name, input_type, transfer, n_trials in model_specs:
        print("\n" + "=" * 70)
        print(f"Searching best config for {model_name}")
        print("=" * 70)

        model_file = results_dir / f"{model_name.replace('/', '_').replace(' ', '')}_best.json"
        if model_file.exists():
            payload = json.loads(model_file.read_text(encoding="utf-8"))
            if "max_source_samples" not in payload.get("best_config", {}):
                if model_name in {"SVR", "TL-SVR"}:
                    payload["best_config"]["max_source_samples"] = 10000
                    payload["best_config"]["max_target_train_samples"] = 5000
                else:
                    payload["best_config"]["max_source_samples"] = None
                    payload["best_config"]["max_target_train_samples"] = None
                model_file.write_text(json.dumps(_to_native(payload), indent=2), encoding="utf-8")
            result = {
                "best_metrics": payload["best_metrics"],
                "best_config": payload["best_config"],
            }
        else:
            result = search_model_best(
                model_name=model_name,
                input_type=input_type,
                transfer=transfer,
                search_space=search_spaces[model_name],
                n_trials=n_trials,
                data_space=data_space,
                data_cache=data_cache,
                farm_idx=farm_idx,
                seed=42,
            )

            # Save per-model best file
            payload = {
                "model": model_name,
                "best_metrics": _to_native(result["best_metrics"]),
                "best_config": _to_native(result["best_config"]),
                "data_space": data_space,
                "hyperparam_space": search_spaces[model_name],
                "n_trials": n_trials,
                "seed": 42,
                "selection": "min_rmse_then_mae_then_max_r2",
            }
            model_file.write_text(json.dumps(_to_native(payload), indent=2), encoding="utf-8")

        if generate_plots:
            best_cond = result["best_config"]
            data = _get_data(
                data_cache,
                farm_idx,
                best_cond["feature_set"],
                int(best_cond["window_size"]),
                float(best_cond["target_train_ratio"]),
            )
            scaler_y = data["scaler_y"]

            if input_type == "flat":
                X_T_train = data["X_T_train_flat"]
                X_T_test = data["X_T_test_flat"]
                X_S = data["X_S_flat"]
                input_size = None
            else:
                X_T_train = data["X_T_train_seq"]
                X_T_test = data["X_T_test_seq"]
                X_S = data["X_S_seq"]
                input_size = X_T_train.shape[-1]

            y_T_train = data["y_T_train"]
            y_T_test = data["y_T_test"]
            y_S = data["y_S"]

            if model_name in {"SVR", "TL-SVR"}:
                max_src = 10000
                max_tgt = 5000
                X_S = X_S[:max_src]
                y_S = y_S[:max_src]
                X_T_train = X_T_train[:max_tgt]
                y_T_train = y_T_train[:max_tgt]

            model = _build_model(model_name, best_cond, input_size)
            if transfer:
                model.fit(X_S, y_S, X_T_train, y_T_train)
            else:
                model.fit(X_T_train, y_T_train)

            y_pred_scaled = model.predict(X_T_test)
            y_true = inverse_transform(scaler_y, y_T_test)
            y_pred = inverse_transform(scaler_y, y_pred_scaled)
            y_pred = np.maximum(y_pred, 0)

            safe_name = model_name.replace("/", "_").replace(" ", "").replace("(", "").replace(")", "")
            plot_path = plots_dir / f"farm{farm_idx}_{safe_name}"
            title = f"{model_name} vs True ({data['farm_name']})"
            plot_prediction_with_zooms(y_true, y_pred, title, plot_path)

        summary_rows.append(
            {
                "model": model_name,
                **_to_native(result["best_metrics"]),
                **_to_native(result["best_config"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "best_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Remove any non-final files in results
    keep_files = {summary_path.name}
    for model_name, _, _, _ in model_specs:
        keep_files.add(f"{model_name.replace('/', '_').replace(' ', '')}_best.json")
    for item in results_dir.iterdir():
        if item.is_file() and item.name not in keep_files:
            item.unlink()

    print("\nSaved results to:")
    print(f"- {summary_path}")
    for model_name, _, _, _ in model_specs:
        fname = f"{model_name.replace('/', '_').replace(' ', '')}_best.json"
        print(f"- {results_dir / fname}")
    print(f"Plots: {plots_dir}")


if __name__ == "__main__":
    main()
