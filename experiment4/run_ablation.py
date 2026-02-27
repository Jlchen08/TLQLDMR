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
from experiment4.plots import plot_residual_distribution


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


def _build_model(params: dict) -> TLQLDMRMedianModel:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: TL-QLDMR ablation study.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("experiment2/results_hunt/tlqldmr_hunt_best.json")),
        help="Path to TL-QLDMR best config (experiment2).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("experiment4/results_selected")),
        help="Output directory for metrics and configs.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=str(Path("experiment4/plots")),
        help="Directory for residual distribution plots.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="shuffle",
        choices=["time", "shuffle"],
        help="Target split mode (keep consistent with experiments 1-3).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-seed",
        type=int,
        default=0,
        help="Random seed for model training (Nyström sampling).",
    )
    parser.add_argument(
        "--tl-from-exp2",
        type=str,
        default="",
        help="Optional path to experiment2 TL-QLDMR_best.json. If set, reuse its metrics for TL-QLDMR.",
    )
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

    base_params = cfg["params"]
    base_params = {k: v for k, v in base_params.items()}

    ablations = [
        {
            "name": "TL-QLDMR",
            "desc": "Full model with transfer + LDMR variance + MMD alignment.",
            "use_source": True,
            "overrides": {},
        },
        {
            "name": "w/o Transfer (Target-only QLDMR)",
            "desc": "Remove source data and MMD; train LDMR only on target.",
            "use_source": False,
            "overrides": {"lambda2": 0.0, "C_S": 0.0},
        },
        {
            "name": "w/o LDMR (Variance)",
            "desc": "Remove variance regularizer; keep MMD alignment.",
            "use_source": True,
            "overrides": {"lambda1": 0.0},
        },
        {
            "name": "w/o MMD",
            "desc": "Keep transfer data but remove explicit MMD alignment.",
            "use_source": True,
            "overrides": {"lambda2": 0.0},
        },
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    residuals = {}
    summary_rows = []
    tl_override = None
    tl_override_metrics = None
    if args.tl_from_exp2:
        tl_path = Path(args.tl_from_exp2)
        if not tl_path.exists():
            raise FileNotFoundError(f"TL override not found: {tl_path}")
        tl_override = json.loads(tl_path.read_text(encoding="utf-8"))
        tl_override_metrics = tl_override.get("best_metrics")

    for spec in ablations:
        name = spec["name"]
        params = dict(base_params)
        params.update(spec["overrides"])

        model = _build_model(params)

        set_seed(args.train_seed)
        if spec["use_source"]:
            X_S_fit = X_S
            y_S_fit = y_S
        else:
            X_S_fit = np.empty((0, X_T_train.shape[1]), dtype=np.float32)
            y_S_fit = np.empty((0,), dtype=np.float32)

        print(f"\n[{name}] Training...")
        model.fit(X_S_fit, y_S_fit, X_T_train, y_T_train)

        y_pred = model.predict(X_T_test)
        y_pred_orig = inverse_transform(scaler_y, y_pred)
        y_true_orig = inverse_transform(scaler_y, y_T_test)

        metrics = compute_metrics(y_true_orig, y_pred_orig)
        if name == "TL-QLDMR" and tl_override_metrics is not None:
            metrics = dict(tl_override_metrics)
        summary_rows.append({"model": name, **metrics})

        residuals[name] = y_true_orig - y_pred_orig

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
            "train_samples": int(len(X_T_train)),
            "test_samples": int(len(X_T_test)),
            "params": _to_native(params),
            "metrics": _to_native(metrics),
        }
        (out_dir / f"{name.replace(' ', '_').replace('/', '-')}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)

    plot_dir = Path(args.plot_dir)
    plot_residual_distribution(residuals, plot_dir / "residual_distribution")

    print("\nSaved summary to:", out_dir / "summary_metrics.csv")


if __name__ == "__main__":
    main()
