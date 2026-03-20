import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from experiment2.data_utils import prepare_farm_data, set_seed
from Train_TL_QLDMR import TL_QLDMR


def _resolve_sample_sizes(raw_sizes, n_total_train: int) -> list[int]:
    resolved: list[int] = []
    for item in raw_sizes:
        if isinstance(item, str):
            token = item.strip().lower()
            if token in {"full", "all", "max", "original"}:
                resolved.append(int(n_total_train))
                continue
            resolved.append(int(float(token)))
            continue
        resolved.append(int(item))
    resolved = [max(1, min(int(n_total_train), size)) for size in resolved]
    return sorted(set(resolved))


def _sample_indices(rng: np.random.Generator, n_total: int, size: int) -> np.ndarray:
    replace = size > n_total
    return rng.choice(n_total, size=size, replace=replace)


def _build_subsample_indices(
    rng: np.random.Generator,
    n_source: int,
    n_target: int,
    n_total: int,
    target_ratio: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    n_t = max(1, int(round(n_total * target_ratio)))
    n_s = max(1, n_total - n_t)

    n_t = min(n_t, n_target)
    n_s = min(n_s, n_source)

    # If we still do not reach n_total, try to fill from whichever pool has room
    remaining = n_total - (n_s + n_t)
    if remaining > 0:
        extra_s = min(n_source - n_s, remaining)
        n_s += extra_s
        remaining -= extra_s
    if remaining > 0:
        extra_t = min(n_target - n_t, remaining)
        n_t += extra_t

    if n_s <= 0 or n_t <= 0:
        raise ValueError("Not enough samples to build both source and target subsets.")

    idx_s = _sample_indices(rng, n_source, n_s)
    idx_t = _sample_indices(rng, n_target, n_t)
    return idx_s, idx_t, n_s, n_t


def _build_model(
    solver: str,
    params: dict,
    fast_cfg: dict,
    torch_cfg: dict,
    nystrom_cap: int,
) -> TL_QLDMR:
    kwargs = dict(
        lambda1=float(params["lambda1"]),
        lambda2=float(params["lambda2"]),
        C_S=float(params["C_S"]),
        C_T=float(params["C_T"]),
        tau=float(params["tau"]),
        kernel_gamma=float(params["kernel_gamma"]),
        use_gpu=False,
        solver=solver,
        torch_device="cpu",
    )

    if solver in {"torch_gd", "batch_sgd"}:
        kwargs["torch_lr"] = float(torch_cfg.get("torch_lr", 0.01))
        kwargs["torch_max_iter"] = int(torch_cfg.get("torch_max_iter", 400))

    if solver == "fast_nystrom":
        kwargs["nystrom_n_components"] = int(min(fast_cfg.get("nystrom_n_components", 200), nystrom_cap))
        kwargs["nystrom_lr"] = float(fast_cfg.get("nystrom_lr", 0.01))
        kwargs["nystrom_epochs"] = int(fast_cfg.get("nystrom_epochs", 20))
        kwargs["nystrom_batch_size"] = int(fast_cfg.get("nystrom_batch_size", 256))

    return TL_QLDMR(**kwargs)


def _fit_once(model: TL_QLDMR, X_S, y_S, X_T, y_T, batch_size: int) -> float:
    start = time.perf_counter()
    model.fit(X_S, y_S, X_T, y_T, batch_size=batch_size)
    end = time.perf_counter()
    return end - start


def _plot_time_curve(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "font.size": 10,
            "lines.linewidth": 1.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for solver, group in df.groupby("solver"):
        group = group.sort_values("n_samples")
        ax.plot(
            group["n_samples"],
            group["time_sec"],
            marker="o",
            linestyle="-",
            label=solver,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of training samples (log scale)")
    ax.set_ylabel("Training time (s, log scale)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _fit_scaling_slope(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for solver, group in df.groupby("solver"):
        group = group.sort_values("n_samples")
        x = np.log10(group["n_samples"].values.astype(np.float64))
        y = np.log10(group["time_sec"].values.astype(np.float64))
        if len(x) < 2:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(
            {
                "solver": solver,
                "loglog_slope": float(slope),
                "loglog_intercept": float(intercept),
                "loglog_r2": float(r2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join("experiment5", "config_scaling.json"),
        help="Path to experiment5 config JSON.",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join("experiment5", "results"),
        help="Directory to save CSV results.",
    )
    parser.add_argument(
        "--plotdir",
        default=os.path.join("experiment5", "plots"),
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--solvers",
        default="cvxopt,torch_gd,batch_sgd,fast_nystrom",
        help="Comma-separated solvers to benchmark.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    set_seed(int(cfg.get("split_seed", 42)))

    data = prepare_farm_data(
        farm_idx=int(cfg["farm_idx"]),
        feature_set=cfg.get("feature_set", "full"),
        window_size=int(cfg.get("window_size", 6)),
        target_train_ratio=float(cfg.get("target_train_ratio", 0.6)),
        extreme_cfg=cfg.get("extreme_cfg"),
        split_mode="time",
        split_seed=int(cfg.get("split_seed", 42)),
    )

    X_S = data["X_S_flat"].astype(np.float32)
    y_S = data["y_S"].astype(np.float32)
    X_T = data["X_T_train_flat"].astype(np.float32)
    y_T = data["y_T_train"].astype(np.float32)

    n_total_train = len(X_S) + len(X_T)
    sample_sizes = _resolve_sample_sizes(cfg.get("sample_sizes", []), n_total_train)
    target_ratio = float(cfg.get("target_ratio", 0.3))
    repeats = int(cfg.get("repeat", 1))
    solver_max_n = cfg.get("solver_max_n", {})
    fast_cfg = cfg.get("fast_nystrom", {})
    torch_cfg = cfg.get("batch_sgd", {})
    params = cfg.get("model_params", {})

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]

    rng = np.random.default_rng(int(cfg.get("split_seed", 42)))
    results = []

    for n_total in sample_sizes:
        for rep in range(repeats):
            idx_s, idx_t, n_s, n_t = _build_subsample_indices(
                rng,
                len(X_S),
                len(X_T),
                n_total,
                target_ratio,
            )
            X_S_sub, y_S_sub = X_S[idx_s], y_S[idx_s]
            X_T_sub, y_T_sub = X_T[idx_t], y_T[idx_t]
            nystrom_cap = max(1, n_s + n_t)
            batch_size = max(1, min(256, n_s, n_t))

            for solver in solvers:
                max_n = solver_max_n.get(solver, None)
                if max_n is not None and n_total > int(max_n):
                    results.append(
                        {
                            "solver": solver,
                            "n_samples": n_total,
                            "n_source": n_s,
                            "n_target": n_t,
                            "repeat": rep,
                            "time_sec": np.nan,
                            "status": "skipped",
                            "error": f"n_total>{max_n}",
                        }
                    )
                    continue

                try:
                    model = _build_model(
                        solver, params, fast_cfg, torch_cfg, nystrom_cap
                    )
                    elapsed = _fit_once(
                        model, X_S_sub, y_S_sub, X_T_sub, y_T_sub, batch_size=batch_size
                    )
                    results.append(
                        {
                            "solver": solver,
                            "n_samples": n_total,
                            "n_source": n_s,
                            "n_target": n_t,
                            "repeat": rep,
                            "time_sec": float(elapsed),
                            "status": "ok",
                            "error": "",
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "solver": solver,
                            "n_samples": n_total,
                            "n_source": n_s,
                            "n_target": n_t,
                            "repeat": rep,
                            "time_sec": np.nan,
                            "status": "error",
                            "error": str(e),
                        }
                    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(outdir / "scaling_times.csv", index=False)

    summary_df = (
        results_df[results_df["status"] == "ok"]
        .groupby(["solver", "n_samples"], as_index=False)[["n_source", "n_target", "time_sec"]]
        .mean()
    )
    summary_df = summary_df.sort_values(["solver", "n_samples"]).reset_index(drop=True)
    summary_df.to_csv(outdir / "scaling_summary.csv", index=False)

    fit_df = _fit_scaling_slope(summary_df)
    fit_df.to_csv(outdir / "scaling_fit.csv", index=False)

    plot_df = summary_df.copy()
    if not plot_df.empty:
        plot_path = Path(args.plotdir) / "time_vs_samples"
        _plot_time_curve(plot_df, plot_path)


if __name__ == "__main__":
    main()
