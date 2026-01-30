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
