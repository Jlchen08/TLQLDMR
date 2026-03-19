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
from experiment2.models.svr_model import SVRModel, SVRConfig
from experiment2.models.blssvr_model import BLSSVRModel, BLSSVRConfig
from experiment2.models.flsvr_model import FLSVRModel, FLSVRConfig
from experiment2.models.ara_svr_model import ARASVRModel, ARASVRConfig
from experiment2.models.clustered_gbr_model import ClusteredGBRModel, ClusteredGBRConfig
from experiment2.models.random_forest_model import RandomForestModel, RandomForestConfig
from experiment2.models.brf_model import BroadRandomForestModel, BRFConfig
from experiment2.models.tl_qldmr_median import TLQLDMRMedianModel, TLQLDMRConfig
from experiment2.models.standard_ldmr_model import StandardLDMRModel, StandardLDMRConfig


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


def _build_model(model_name: str, params: dict):
    if model_name in {"SVR", "HHO-SVR", "Standard-SVR"}:
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
    if model_name == "BLSSVR":
        return BLSSVRModel(
            BLSSVRConfig(
                C=float(params["C"]),
                gamma=float(params["gamma"]),
                n_components=int(params["n_components"]),
                loss_scale=float(params["loss_scale"]),
                max_iter=int(params["max_iter"]),
                tol=float(params["tol"]),
                random_state=int(params.get("random_state", 42)),
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
                tau=float(params.get("tau", 0.5)),
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
                tau=float(params.get("tau", 0.5)),
                kernel_gamma=float(params["kernel_gamma"]),
                solver="fast_nystrom",
                nystrom_n_components=int(params["nystrom_n_components"]),
                nystrom_lr=float(params["nystrom_lr"]),
                nystrom_epochs=int(params["nystrom_epochs"]),
                nystrom_batch_size=int(params["nystrom_batch_size"]),
            )
        )
    raise ValueError(f"Unknown model: {model_name}")


def _load_best_configs(results_dir: Path) -> dict[str, dict]:
    configs = {}
    for path in results_dir.glob("*_best.json"):
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
        default=os.path.join(REPO_ROOT, "experiment2", "results_selected"),
    )
    parser.add_argument("--snr-db", type=str, default="60,40,30,20,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="shuffle",
        choices=["time", "shuffle"],
    )
    parser.add_argument("--svr-max-src", type=int, default=4000)
    parser.add_argument("--svr-max-tgt", type=int, default=1200)
    parser.add_argument(
        "--tl-config",
        type=str,
        default=None,
        help="Optional JSON path to override TL-QLDMR config for noise evaluation.",
    )
    parser.add_argument(
        "--noise-on-train",
        action="store_true",
        default=True,
        help="Add noise to train inputs as well.",
    )
    parser.add_argument(
        "--no-noise-on-train",
        action="store_false",
        dest="noise_on_train",
        help="Disable noise injection on training data.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    with open(args.config, "r") as f:
        base_cfg = json.load(f)

    best_dir = Path(args.best_dir)
    best_configs = _load_best_configs(best_dir)
    if not best_configs:
        raise RuntimeError(f"No *_best.json found in {best_dir}")
    if args.tl_config:
        tl_path = Path(args.tl_config)
        if not tl_path.exists():
            raise FileNotFoundError(f"TL-QLDMR override not found: {tl_path}")
        best_configs["TL-QLDMR"] = json.loads(tl_path.read_text(encoding="utf-8"))

    out_dir = Path(__file__).resolve().parent
    results_dir = out_dir / "results_noise"
    results_dir.mkdir(parents=True, exist_ok=True)

    snr_list = _parse_snr_list(args.snr_db)

    meta = {
        "farm_idx": base_cfg["farm_idx"],
        "farm_name": base_cfg.get("farm_name"),
        "split_seed": base_cfg.get("split_seed", 0),
        "split_mode": args.split_mode,
        "extreme_cfg": base_cfg.get("extreme_cfg"),
        "snr_db": args.snr_db,
        "noise_on_train": args.noise_on_train,
        "svr_max_src": args.svr_max_src,
        "svr_max_tgt": args.svr_max_tgt,
        "tl_override": args.tl_config,
        "models": sorted(best_configs.keys()),
    }
    (results_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    data_cache: dict[tuple, dict] = {}
    rows: list[dict] = []

    model_order = [
        "Standard-SVR",
        "HHO-SVR",
        "FLSVR",
        "BLSSVR",
        "ARA-SVR",
        "Standard-LDMR",
        "KMeans-GBT",
        "RF-WPF",
        "BRF-WPF",
        "TL-QLDMR",
    ]

    for model_name in model_order:
        if model_name not in best_configs:
            continue
        payload = best_configs[model_name]
        cfg = payload.get("best_config", {})

        feature_set = cfg.get("feature_set", base_cfg["feature_set"])
        window_size = int(cfg.get("window_size", base_cfg["window_size"]))
        target_train_ratio = float(cfg.get("target_train_ratio", base_cfg["target_train_ratio"]))

        data_key = (
            feature_set,
            window_size,
            target_train_ratio,
            args.split_mode,
            json.dumps(base_cfg.get("extreme_cfg", {}), sort_keys=True),
        )
        if data_key not in data_cache:
            data_cache[data_key] = prepare_farm_data(
                farm_idx=base_cfg["farm_idx"],
                feature_set=feature_set,
                window_size=window_size,
                target_train_ratio=target_train_ratio,
                extreme_cfg=base_cfg.get("extreme_cfg"),
                split_mode=args.split_mode,
                split_seed=base_cfg.get("split_seed", 0),
            )
        data = data_cache[data_key]
        scaler_y = data["scaler_y"]

        X_S = data["X_S_flat"]
        y_S = data["y_S"]
        X_T_train = data["X_T_train_flat"]
        y_T_train = data["y_T_train"]
        X_T_test = data["X_T_test_flat"]
        y_T_test = data["y_T_test"]

        X_ref = np.vstack([X_S, X_T_train])
        signal_power = np.mean(X_ref ** 2, axis=0)

        transfer = model_name == "TL-QLDMR"
        downsample = model_name in {
            "SVR",
            "Standard-SVR",
            "HHO-SVR",
            "FLSVR",
            "BLSSVR",
            "ARA-SVR",
            "KMeans-GBT",
            "Standard-LDMR",
        }

        for idx, snr_db in enumerate(snr_list):
            set_seed(args.seed + idx)
            rng = np.random.default_rng(args.seed + idx)
            noise_std_vec = _snr_to_noise_std(signal_power, float(snr_db))

            X_S_use = X_S
            X_T_train_use = X_T_train
            X_T_test_use = X_T_test
            if args.noise_on_train:
                X_S_use = _add_noise_with_std(X_S_use, noise_std_vec, rng)
                X_T_train_use = _add_noise_with_std(X_T_train_use, noise_std_vec, rng)
            X_T_test_use = _add_noise_with_std(X_T_test_use, noise_std_vec, rng)

            if downsample:
                X_S_use = X_S_use[: args.svr_max_src]
                y_S_use = y_S[: args.svr_max_src]
                X_T_train_use = X_T_train_use[: args.svr_max_tgt]
                y_T_train_use = y_T_train[: args.svr_max_tgt]
            else:
                y_S_use = y_S
                y_T_train_use = y_T_train

            model = _build_model(model_name, cfg)
            if transfer:
                model.fit(X_S_use, y_S_use, X_T_train_use, y_T_train_use)
            else:
                model.fit(X_T_train_use, y_T_train_use)

            y_pred_scaled = model.predict(X_T_test_use)
            y_true = inverse_transform(scaler_y, y_T_test)
            y_pred = inverse_transform(scaler_y, y_pred_scaled)
            y_pred = np.maximum(y_pred, 0.0)
            metrics = compute_metrics(y_true, y_pred)

            rows.append(
                {
                    "model": model_name,
                    "snr_db": float(snr_db),
                    "noise_std_mean": float(np.mean(noise_std_vec)),
                    **metrics,
                }
            )
            print(
                f"[Noise][{model_name}] SNR={snr_db}dB -> RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "snr_db"])
    df.to_csv(results_dir / "noise_robustness.csv", index=False)

    print("\nSaved results to:")
    print(results_dir / "noise_robustness.csv")


if __name__ == "__main__":
    main()
