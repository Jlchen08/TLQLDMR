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
from experiment2.plots import plot_prediction_segments
from experiment2.models.svr_model import SVRModel, SVRConfig
from experiment2.models.flsvr_model import FLSVRModel, FLSVRConfig
from experiment2.models.ara_svr_model import ARASVRModel, ARASVRConfig
from experiment2.models.clustered_gbr_model import ClusteredGBRModel, ClusteredGBRConfig
from experiment2.models.random_forest_model import RandomForestModel, RandomForestConfig
from experiment2.models.knn_model import KNNModel, KNNConfig
from experiment2.models.gbr_model import GBRModel, GBRConfig
from experiment2.models.stacking_wpf_model import StackingWPFModel, StackingWPFConfig
from experiment2.models.brf_model import BroadRandomForestModel, BRFConfig
from experiment2.models.hho_svr_optimizer import HHOSVRConfig, optimize_hho_svr
from experiment2.models.tl_qldmr_median import TLQLDMRMedianModel, TLQLDMRConfig
from experiment2.models.standard_ldmr_model import StandardLDMRModel, StandardLDMRConfig


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


def _sample_from_space(rng: np.random.Generator, space: dict) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def _split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    n_total = len(X)
    n_val = max(1, int(n_total * val_ratio))
    n_train = max(1, n_total - n_val)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


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
    if model_name == "Standard-SVR":
        return SVRModel(
            SVRConfig(
                C=float(params["C"]),
                epsilon=float(params["epsilon"]),
                gamma=float(params["gamma"]),
                kernel="rbf",
            )
        )
    if model_name == "FLSVR":
        return FLSVRModel(
            FLSVRConfig(
                C=float(params["C"]),
                epsilon=float(params["epsilon"]),
                gamma=float(params["gamma"]),
                max_iter=int(params["max_iter"]),
                tol=float(params["tol"]),
            )
        )
    if model_name == "ARA-SVR":
        return ARASVRModel(
            ARASVRConfig(
                C=float(params["C"]),
                epsilon=float(params["epsilon"]),
                gamma=float(params["gamma"]),
                corr_threshold=float(params["corr_threshold"]),
                min_features=int(params["min_features"]),
            )
        )
    if model_name == "KMeans-GBT":
        return ClusteredGBRModel(
            ClusteredGBRConfig(
                n_clusters=int(params["n_clusters"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] is not None else None,
                max_iter=int(params["max_iter"]),
                learning_rate=float(params["learning_rate"]),
                max_leaf_nodes=int(params["max_leaf_nodes"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                l2_regularization=float(params["l2_regularization"]),
                random_state=int(params.get("random_state", 42)),
                min_cluster_samples=int(params["min_cluster_samples"]),
            )
        )
    if model_name == "RF-WPF":
        return RandomForestModel(
            RandomForestConfig(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] is not None else None,
                min_samples_split=int(params["min_samples_split"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=params["max_features"],
                random_state=int(params.get("random_state", 42)),
            )
        )
    if model_name == "BRF-WPF":
        return BroadRandomForestModel(
            BRFConfig(
                n_feature_groups=int(params["n_feature_groups"]),
                n_feature_nodes=int(params["n_feature_nodes"]),
                n_enhance_nodes=int(params["n_enhance_nodes"]),
                activation=params["activation"],
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]) if params["max_depth"] is not None else None,
                min_samples_leaf=int(params["min_samples_leaf"]),
                max_features=float(params["max_features"]),
                random_state=int(params.get("random_state", 42)),
            )
        )
    if model_name == "TL-QLDMR":
        return TLQLDMRMedianModel(
            TLQLDMRConfig(
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
                variance_mode=str(params.get("variance_mode", "asymmetric")),
                asym_scale=float(params.get("asym_scale", 0.25)),
            )
        )
    if model_name == "Standard-LDMR":
        return StandardLDMRModel(
            StandardLDMRConfig(
                lambda1=float(params["lambda1"]),
                C=float(params["C"]),
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


def search_model_best(
    model_name: str,
    input_type: str,
    transfer: bool,
    search_space: dict,
    n_trials: int,
    data: dict,
    seed: int,
    svr_max_src: int,
    svr_max_tgt: int,
):
    rng = np.random.default_rng(seed)
    best_key = None
    best = None

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

    if model_name in {"SVR", "Standard-SVR", "HHO-SVR", "FLSVR", "ARA-SVR", "KMeans-GBT", "Standard-LDMR"}:
        X_S = X_S[:svr_max_src]
        y_S = y_S[:svr_max_src]
        X_T_train = X_T_train[:svr_max_tgt]
        y_T_train = y_T_train[:svr_max_tgt]

    for trial in range(n_trials):
        set_seed(seed + trial)
        X_T_tr, y_T_tr, X_T_val, y_T_val = _split_train_val(
            X_T_train, y_T_train, val_ratio=0.2, seed=seed + trial
        )
        params = _sample_from_space(rng, search_space)
        if model_name == "HHO-SVR":
            hho_cfg = HHOSVRConfig(
                C_bounds=tuple(params["C_bounds"]),
                epsilon_bounds=tuple(params["epsilon_bounds"]),
                gamma_bounds=tuple(params["gamma_bounds"]),
                population_size=int(params["population_size"]),
                iterations=int(params["iterations"]),
                seed=seed + trial,
            )
            best_params = optimize_hho_svr(X_T_tr, y_T_tr, X_T_val, y_T_val, hho_cfg)
            params = {
                **best_params,
                "population_size": hho_cfg.population_size,
                "iterations": hho_cfg.iterations,
                "C_bounds": list(hho_cfg.C_bounds),
                "epsilon_bounds": list(hho_cfg.epsilon_bounds),
                "gamma_bounds": list(hho_cfg.gamma_bounds),
            }
            model = SVRModel(
                SVRConfig(
                    C=float(best_params["C"]),
                    epsilon=float(best_params["epsilon"]),
                    gamma=float(best_params["gamma"]),
                    kernel="rbf",
                )
            )
            model.fit(X_T_tr, y_T_tr)
        else:
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
        key = (-metrics["r2"], metrics["rmse"], metrics["mae"])
        if best_key is None or key < best_key:
            best_key = key
            best = {
                "params": params,
            }

    # retrain with best params
    set_seed(seed)
    if model_name == "HHO-SVR":
        params = best["params"]
        model = SVRModel(
            SVRConfig(
                C=float(params["C"]),
                epsilon=float(params["epsilon"]),
                gamma=float(params["gamma"]),
                kernel="rbf",
            )
        )
        model.fit(X_T_train, y_T_train)
    else:
        model = _build_model(model_name, best["params"], input_size)
        if transfer:
            model.fit(X_S, y_S, X_T_train, y_T_train)
        else:
            model.fit(X_T_train, y_T_train)

    y_pred_scaled = model.predict(X_T_test)
    y_true = inverse_transform(scaler_y, y_T_test)
    y_pred = inverse_transform(scaler_y, y_pred_scaled)
    y_pred = np.maximum(y_pred, 0)

    metrics_test = compute_metrics(y_true, y_pred)
    return metrics_test, best["params"], y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="experiment2/results_hunt/tlqldmr_hunt_best.json")
    parser.add_argument("--baseline-trials", type=int, default=2)
    parser.add_argument("--svr-max-src", type=int, default=2000)
    parser.add_argument("--svr-max-tgt", type=int, default=1000)
    parser.add_argument(
        "--skip-tlqldmr-train",
        action="store_true",
        help="Use metrics from the TL-QLDMR hunt config and skip retraining/plotting.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip saving prediction plots.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    best_payload = json.loads(config_path.read_text(encoding="utf-8"))
    farm_idx = int(best_payload["farm_idx"])
    split_seed = int(best_payload["split_seed"])
    set_seed(split_seed)

    data = prepare_farm_data(
        farm_idx=farm_idx,
        feature_set=best_payload["feature_set"],
        window_size=int(best_payload["window_size"]),
        target_train_ratio=float(best_payload["target_train_ratio"]),
        extreme_cfg=best_payload["extreme_cfg"],
        split_mode="shuffle",
        split_seed=split_seed,
    )

    out_dir = Path(__file__).resolve().parent
    results_dir = out_dir / "results_selected"
    plots_dir = out_dir / "plots_selected"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("Standard-SVR", "flat", False, args.baseline_trials),
        ("HHO-SVR", "flat", False, args.baseline_trials),
        ("FLSVR", "flat", False, args.baseline_trials),
        ("ARA-SVR", "flat", False, args.baseline_trials),
        ("Standard-LDMR", "flat", False, args.baseline_trials),
        ("KMeans-GBT", "flat", False, args.baseline_trials),
        ("RF-WPF", "flat", False, args.baseline_trials),
        ("BRF-WPF", "flat", False, args.baseline_trials),
        ("TL-QLDMR", "flat", True, 1),
    ]

    search_spaces = {
        "HHO-SVR": {
            "C_bounds": [(0.1, 100.0)],
            "epsilon_bounds": [(0.001, 0.1)],
            "gamma_bounds": [(1e-4, 0.05)],
            "population_size": [6],
            "iterations": [8],
        },
        "Standard-SVR": {
            "C": [8.0, 10.0],
            "epsilon": [0.03, 0.05],
            "gamma": [0.001, 0.002],
        },
        "FLSVR": {
            "C": [10],
            "epsilon": [0.05],
            "gamma": [0.001],
            "max_iter": [25],
            "tol": [1e-4],
        },
        "ARA-SVR": {
            "C": [10],
            "epsilon": [0.05],
            "gamma": [0.001],
            "corr_threshold": [0.4],
            "min_features": [10],
        },
        "Standard-LDMR": {
            "lambda1": [5e-4, 1e-3, 5e-3],
            "C": [80.0, 120.0, 160.0],
            "kernel_gamma": [0.001, 0.002],
            "nystrom_n_components": [800, 1000, 1200],
            "nystrom_lr": [0.01],
            "nystrom_epochs": [48, 64],
            "nystrom_batch_size": [256],
        },
        "KMeans-GBT": {
            "n_clusters": [3],
            "max_depth": [3],
            "max_iter": [80],
            "learning_rate": [0.05],
            "max_leaf_nodes": [15],
            "min_samples_leaf": [40],
            "l2_regularization": [0.1],
            "random_state": [42],
            "min_cluster_samples": [80],
        },
        "RF-WPF": {
            "n_estimators": [80],
            "max_depth": [6],
            "min_samples_split": [6],
            "min_samples_leaf": [4],
            "max_features": [0.6],
            "random_state": [42],
        },
        "BRF-WPF": {
            "n_feature_groups": [3],
            "n_feature_nodes": [12],
            "n_enhance_nodes": [30],
            "activation": ["tanh"],
            "n_estimators": [120],
            "max_depth": [6],
            "min_samples_leaf": [4],
            "max_features": [0.5],
            "random_state": [42],
        },
    }

    summary_rows = []
    for model_name, input_type, transfer, n_trials in model_specs:
        if model_name == "TL-QLDMR" and args.skip_tlqldmr_train and "metrics" in best_payload:
            used_params = best_payload["params"]
            metrics = best_payload["metrics"]
            y_true, y_pred = None, None
        elif model_name == "TL-QLDMR":
            params = best_payload["params"]
            metrics, _, y_true, y_pred = search_model_best(
                model_name,
                input_type,
                transfer,
                {k: [v] for k, v in params.items()},
                1,
                data,
                split_seed,
                args.svr_max_src,
                args.svr_max_tgt,
            )
            used_params = params
        else:
            metrics, used_params, y_true, y_pred = search_model_best(
                model_name,
                input_type,
                transfer,
                search_spaces[model_name],
                n_trials,
                data,
                split_seed,
                args.svr_max_src,
                args.svr_max_tgt,
            )

        payload = {
            "model": model_name,
            "best_metrics": _to_native(metrics),
            "best_config": _to_native(
                {
                    "feature_set": best_payload["feature_set"],
                    "window_size": best_payload["window_size"],
                    "target_train_ratio": best_payload["target_train_ratio"],
                    **used_params,
                }
            ),
            "farm_idx": farm_idx,
            "farm_name": data["farm_name"],
            "split_seed": split_seed,
        }
        (results_dir / f"{model_name}_best.json").write_text(
            json.dumps(_to_native(payload), indent=2), encoding="utf-8"
        )

        if not args.no_plots and y_true is not None and y_pred is not None:
            safe_name = model_name.replace("/", "_").replace(" ", "")
            plot_path = plots_dir / f"farm{farm_idx}_{safe_name}"
            plot_prediction_segments(
                y_true=y_true,
                y_pred=y_pred,
                title=f"{model_name} vs True ({data['farm_name']})",
                out_basepath=plot_path,
            )

        summary_rows.append(
            {
                "model": model_name,
                **_to_native(metrics),
                **_to_native(used_params),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "best_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    keep_files = {summary_path.name}
    for model_name, _, _, _ in model_specs:
        keep_files.add(f"{model_name}_best.json")
    for item in results_dir.iterdir():
        if item.is_file() and item.name not in keep_files:
            item.unlink()

    print("Saved results to:")
    print(f"- {summary_path}")
    print(f"- {results_dir}")
    print(f"- {plots_dir}")


if __name__ == "__main__":
    main()
