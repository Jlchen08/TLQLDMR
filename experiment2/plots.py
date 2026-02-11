from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _get_window(idx: int, n: int, radius: int = 50) -> slice:
    start = max(0, idx - radius)
    end = min(n, idx + radius)
    return slice(start, end)


def plot_prediction_with_zooms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_basepath: Path,
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    peak_idx = int(np.argmax(y_true))
    valley_idx = int(np.argmin(y_true))

    peak_slice = _get_window(peak_idx, n, radius=60)
    valley_slice = _get_window(valley_idx, n, radius=60)

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

    # Main plot
    axes[0].plot(y_true, label="True", color="black")
    axes[0].plot(y_pred, label="Prediction", color="#1f77b4", linestyle="--")
    axes[0].set_title(title)
    axes[0].set_xlabel("Time Step (15 min)")
    axes[0].set_ylabel("Power (MW)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    # Peak zoom
    axes[1].plot(
        np.arange(*peak_slice.indices(n)),
        y_true[peak_slice],
        color="black",
        label="True",
    )
    axes[1].plot(
        np.arange(*peak_slice.indices(n)),
        y_pred[peak_slice],
        color="#1f77b4",
        linestyle="--",
        label="Prediction",
    )
    axes[1].set_title("Peak Zoom")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Power (MW)")
    axes[1].grid(alpha=0.3)

    # Valley zoom
    axes[2].plot(
        np.arange(*valley_slice.indices(n)),
        y_true[valley_slice],
        color="black",
        label="True",
    )
    axes[2].plot(
        np.arange(*valley_slice.indices(n)),
        y_pred[valley_slice],
        color="#1f77b4",
        linestyle="--",
        label="Prediction",
    )
    axes[2].set_title("Valley Zoom")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Power (MW)")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()

    out_basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_basepath.with_suffix(".png"), dpi=300)
    fig.savefig(out_basepath.with_suffix(".pdf"))
    plt.close(fig)


def plot_prediction_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "font.size": 10,
            "lines.linewidth": 1.6,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(y_true, color="tab:blue", label="True")
    ax.plot(y_pred, color="tab:orange", linestyle="--", label="Prediction")
    ax.set_title(title)
    ax.set_xlabel("Time Step (15-min)")
    ax.set_ylabel("Power (MW)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


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


def plot_prediction_segments(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_basepath: Path,
    segment_len: int = 600,
    zoom_radius: int = 60,
) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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

    axes[0].plot(
        np.arange(seg_start, seg_end),
        y_true[seg_slice],
        color="black",
        label="True",
    )
    axes[0].plot(
        np.arange(seg_start, seg_end),
        y_pred[seg_slice],
        color="#1f77b4",
        linestyle="--",
        label="Prediction",
    )
    axes[0].set_title(f"{title}\nSegment [{seg_start}, {seg_end})")
    axes[0].set_xlabel("Time Step (15 min)")
    axes[0].set_ylabel("Power (MW)")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(loc="upper right")

    axes[1].plot(
        np.arange(*peak_slice.indices(n)),
        y_true[peak_slice],
        color="black",
        label="True",
    )
    axes[1].plot(
        np.arange(*peak_slice.indices(n)),
        y_pred[peak_slice],
        color="#1f77b4",
        linestyle="--",
        label="Prediction",
    )
    axes[1].set_title("Peak Zoom")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Power (MW)")
    axes[1].grid(alpha=0.25, linestyle="--")

    axes[2].plot(
        np.arange(*valley_slice.indices(n)),
        y_true[valley_slice],
        color="black",
        label="True",
    )
    axes[2].plot(
        np.arange(*valley_slice.indices(n)),
        y_pred[valley_slice],
        color="#1f77b4",
        linestyle="--",
        label="Prediction",
    )
    axes[2].set_title("Valley Zoom")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Power (MW)")
    axes[2].grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    out_basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_basepath.with_suffix(".png"), dpi=300)
    fig.savefig(out_basepath.with_suffix(".pdf"))
    plt.close(fig)
