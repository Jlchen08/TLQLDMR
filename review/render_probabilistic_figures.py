#!/usr/bin/env python3
"""Render publication figures from the independently verified v2 summaries.

This presentation-only step does not recompute scores or select results.  The
line styles and staggered hollow markers make coincident method curves visible.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "review/probabilistic_runs/probabilistic_analysis_v2/aggregate_results.json"
DEFAULT_OUTPUT = ROOT / "review/probabilistic_runs/probabilistic_presentation_v1"
GRID = np.array([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
METHODS = ["Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR", "TargetSVQR", "HistGBR"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.15,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def method_styles():
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#6A3D9A"]
    markers = ["o", "s", "^", "D", "v", "P"]
    lines = ["-", "-", "--", "--", "-.", ":"]
    offsets = [0, 0, 1, 1, 0, 1]
    labels = {
        "Full": "Full", "NoVariance": "No variance",
        "NoDiscrepancy": "No discrepancy", "WeightedSVQR": "Weighted SVQR",
        "TargetSVQR": "Target SVQR", "HistGBR": "HistGBR",
    }
    return {m: dict(color=c, marker=mk, linestyle=ls, markevery=(off, 2),
                    markerfacecolor="white", markeredgecolor=c,
                    markeredgewidth=0.9, markersize=4.2, label=labels[m])
            for m, c, mk, ls, off in zip(METHODS, colors, markers, lines, offsets)}


def load_summaries(path: Path):
    rows = json.loads(path.read_text())
    by_method = {row["method"]: row for row in rows}
    if set(by_method) != set(METHODS):
        raise ValueError("The verified summary must contain all six methods exactly once")
    return [by_method[m] for m in METHODS]


def render_reliability(out: Path, summaries, styles) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), sharex=True, sharey=True)
    for ax, key, title in zip(
        axes,
        ["raw_empirical_cdf", "rearranged_empirical_cdf"],
        ["(a) Independently fitted quantiles", "(b) Monotone rearrangement"],
    ):
        ax.plot([0, 1], [0, 1], color="0.35", ls=(0, (4, 2)), lw=0.9, label="Ideal", zorder=1)
        for z, summary in enumerate(summaries, start=2):
            ax.plot(GRID, summary[key], **styles[summary["method"]], zorder=z)
        ax.set_title(title)
        ax.set_xlabel("Nominal quantile level")
        ax.grid(True, color="0.88", lw=0.55)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(r"Empirical $P(Y\leq \hat q_\tau)$")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.055))
    fig.subplots_adjust(bottom=0.24, wspace=0.08)
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"probabilistic_reliability.{ext}")
    plt.close(fig)


def render_coverage(out: Path, summaries, styles) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), sharex=True)
    nominal = np.array([0.50, 0.80, 0.90, 0.95])
    for z, summary in enumerate(summaries, start=2):
        coverage = [summary["intervals"][str(v)]["static"]["picp"] for v in nominal]
        width = [summary["intervals"][str(v)]["static"]["mpiw"] for v in nominal]
        axes[0].plot(nominal, coverage, **styles[summary["method"]], zorder=z)
        axes[1].plot(nominal, width, **styles[summary["method"]], zorder=z)
    axes[0].plot([0.47, 0.97], [0.47, 0.97], color="0.35", ls=(0, (4, 2)),
                 lw=0.9, label="Ideal", zorder=1)
    axes[0].set_title("(a) Static Q2-calibrated coverage")
    axes[1].set_title("(b) Static Q2-calibrated sharpness")
    axes[0].set_ylabel("PICP")
    axes[1].set_ylabel("MPIW (MW)")
    for ax in axes:
        ax.set_xlabel("Nominal central coverage")
        ax.grid(True, color="0.88", lw=0.55)
        ax.set_xticks(nominal)
    axes[0].set_ylim(0.45, 1.01)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.055))
    fig.subplots_adjust(bottom=0.24, wspace=0.24)
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"probabilistic_coverage_sharpness.{ext}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_style()
    summaries = load_summaries(args.input)
    styles = method_styles()
    render_reliability(args.output, summaries, styles)
    render_coverage(args.output, summaries, styles)
    manifest = {
        "role": "presentation-only rendering of independently verified aggregate results",
        "input": str(args.input.resolve().relative_to(ROOT)),
        "input_sha256": digest(args.input),
        "source": str(Path(__file__).resolve().relative_to(ROOT)),
        "source_sha256": digest(Path(__file__)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": importlib.metadata.version("matplotlib"),
        "outputs": {
            p.name: digest(p) for p in sorted(args.output.glob("probabilistic_*.*"))
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
