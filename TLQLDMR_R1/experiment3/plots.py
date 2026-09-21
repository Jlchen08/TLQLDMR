from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _get_window(idx: int, n: int, radius: int = 50) -> slice:
    start = max(0, idx - radius)
    end = min(n, idx + radius)
    return slice(start, end)


def _select_segment(y: np.ndarray, length: int) -> tuple[int, int]:
    n = len(y)
    if n <= length:
        return 0, n
    step = max(1, length // 5)
    best_start = 0
    best_score = -1.0
    for start in range(0, n - length + 1, step):
        seg = y[start : start + length]
        score = float(np.nanstd(seg))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + length


def plot_interval_segments(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str,
    out_basepath: Path,
    segment_len: int = 600,
    zoom_radius: int = 60,
) -> None:
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    n = len(y_true)

    seg_start, seg_end = _select_segment(y_true, min(segment_len, n))
    seg_slice = slice(seg_start, seg_end)

    peak_idx = int(np.argmax(y_true))
    valley_idx = int(np.argmin(y_true))
    peak_slice = _get_window(peak_idx, n, radius=zoom_radius)
    valley_slice = _get_window(valley_idx, n, radius=zoom_radius)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    idx_main = np.arange(seg_start, seg_end)
    axes[0].fill_between(
        idx_main,
        lower[seg_slice],
        upper[seg_slice],
        color="#1f77b4",
        alpha=0.22,
        label="90% PI",
    )
    axes[0].plot(idx_main, y_true[seg_slice], color="black", label="True")
    axes[0].plot(
        idx_main,
        0.5 * (lower[seg_slice] + upper[seg_slice]),
        color="#1f77b4",
        linestyle="--",
        label="Interval center",
    )
    axes[0].set_title(f"{title}\nSegment [{seg_start}, {seg_end})")
    axes[0].set_xlabel("Time Step (15 min)")
    axes[0].set_ylabel("Power (MW)")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(loc="upper right")

    idx_peak = np.arange(*peak_slice.indices(n))
    axes[1].fill_between(
        idx_peak,
        lower[peak_slice],
        upper[peak_slice],
        color="#1f77b4",
        alpha=0.22,
    )
    axes[1].plot(idx_peak, y_true[peak_slice], color="black", label="True")
    axes[1].set_title("Peak Zoom")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Power (MW)")
    axes[1].grid(alpha=0.25, linestyle="--")

    idx_valley = np.arange(*valley_slice.indices(n))
    axes[2].fill_between(
        idx_valley,
        lower[valley_slice],
        upper[valley_slice],
        color="#1f77b4",
        alpha=0.22,
    )
    axes[2].plot(idx_valley, y_true[valley_slice], color="black", label="True")
    axes[2].set_title("Valley Zoom")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Power (MW)")
    axes[2].grid(alpha=0.25, linestyle="--")

    fig.tight_layout()

    out_basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_basepath.with_suffix(".png"), dpi=300)
    fig.savefig(out_basepath.with_suffix(".pdf"))
    plt.close(fig)
