from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _smooth_density(density: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    if kernel_size < 3:
        return density
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3.0
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return np.convolve(density, kernel, mode="same")


def _apply_academic_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-paper")
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "savefig.dpi": 600,
        }
    )


def plot_residual_distribution(
    residuals: dict[str, np.ndarray],
    out_path: Path,
    bins: int = 80,
) -> None:
    if not residuals:
        return

    all_res = np.concatenate([np.asarray(v).ravel() for v in residuals.values()])
    if all_res.size == 0:
        return

    low, high = np.percentile(all_res, [1.0, 99.0])
    if np.isclose(low, high):
        low, high = float(all_res.min()), float(all_res.max())
    pad = 0.05 * (high - low + 1e-6)
    low -= pad
    high += pad
    edges = np.linspace(low, high, bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])

    _apply_academic_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

    palette = [
        "#0f4c81",  # TL-QLDMR deep blue
        "#7a8aa6",
        "#c27b4b",
        "#4f7f6f",
        "#8a4f7d",
    ]
    color_idx = 0

    for name, res in residuals.items():
        hist, _ = np.histogram(res, bins=edges, density=True)
        hist = _smooth_density(hist, kernel_size=7)
        if "TL-QLDMR" in name:
            ax.plot(centers, hist, label=name, linewidth=2.4, color=palette[0])
        else:
            color_idx = min(color_idx + 1, len(palette) - 1)
            ax.plot(centers, hist, label=name, color=palette[color_idx], linestyle="--")

    ax.set_xlabel("Residual (MW)")
    ax.set_ylabel("Density")
    ax.axvline(0.0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(True)
    ax.legend(ncol=2, columnspacing=1.0, handlelength=2.0, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
