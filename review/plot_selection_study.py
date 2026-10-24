#!/usr/bin/env python3
"""Reusable journal-figure style and scaffold for the selection study.

The selection_v1 adapter requires all 112 final evaluations and verifies the
reported metrics against their saved predictions. It never fabricates data,
silently drops seeds, or falls back to results from a different study.
``--check-style`` tests typography without drawing example observations.
``--v1-only`` redraws the complete, earlier common-setting curves using this
same style, without reading incomplete selection-study results.

Captions belong in the manuscript. The methods paragraph should define the
seed summary, tuning/fitting timing boundary, computing environment, and
selection protocol. Figure interiors contain only axes, legends, panel labels,
and necessary data annotations; audit prose stays in the provenance sidecar.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR", "TargetSVQR", "HistGBR")
METHOD_LABELS = ("Full", "No variance", "No discrepancy", "Weighted SVQR", "Target SVQR", "HistGBR")


# Okabe--Ito hues, with dark gray replacing yellow for visibility on white.
# Line/marker differences preserve series identity without color perception.
SERIES_STYLES = (
    {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    {"color": "#009E73", "marker": "^", "linestyle": "-."},
    {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
    {"color": "#333333", "marker": "v", "linestyle": "-"},
    {"color": "#56B4E9", "marker": "P", "linestyle": "--"},
    {"color": "#E69F00", "marker": "X", "linestyle": "-."},
)

COLUMN_WIDTH_IN = 3.35
FULL_WIDTH_IN = 7.0
STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.formatter.use_mathtext": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.15,
    "lines.markersize": 4,
    "lines.markeredgewidth": 0.6,
    "errorbar.capsize": 2.5,
    "legend.frameon": False,
    "legend.handlelength": 2.4,
    "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.35,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Keep text as vector paths so exported SVGs retain the selected serif face.
    "svg.fonttype": "path",
    "svg.hashsalt": "tlqldmr-selection-study",
}


@contextmanager
def paper_style():
    """Use the paper style locally; do not change other scripts' rcParams."""
    with plt.rc_context(STYLE):
        yield


def new_figure(*, nrows=1, ncols=1, width="single", height=None, sharex=False,
               sharey=False):
    """Create a figure at its intended publication width inside paper_style."""
    if width not in {"single", "double"}:
        raise ValueError("width must be 'single' or 'double'")
    if nrows < 1 or ncols < 1:
        raise ValueError("The number of rows and columns must be positive")
    inches = COLUMN_WIDTH_IN if width == "single" else FULL_WIDTH_IN
    if height is None:
        height = max(2.2, inches / ncols * 0.72) * nrows
    if not math.isfinite(height) or height <= 0:
        raise ValueError("Figure height must be positive and finite")
    fig, axes = plt.subplots(nrows, ncols, figsize=(inches, height), squeeze=False,
                             sharex=sharex, sharey=sharey, layout="constrained")
    return fig, axes


def style_axis(ax, *, xlabel=None, ylabel=None, grid="y"):
    """Apply sparse framing; set grid=None for panels that need no grid."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)
    if grid is not None:
        if grid not in {"x", "y", "both"}:
            raise ValueError("grid must be x, y, both, or None")
        ax.grid(axis=grid, color="0.88", linewidth=0.45, linestyle="-")
    return ax


def panel_label(ax, label: str):
    """Place a short panel label without introducing an extra plot title."""
    return ax.text(0.0, 1.035, label, transform=ax.transAxes, ha="left",
                   va="bottom", fontsize=9, fontweight="bold", clip_on=False)


def series_style(index: int):
    """Return a fresh style mapping; do not silently reuse an exhausted key."""
    if not 0 <= index < len(SERIES_STYLES):
        raise ValueError("A unique visual key is required for every series")
    return dict(SERIES_STYLES[index])


def summarize_seeds(values: Mapping[int, float], expected_seeds: Sequence[int]):
    """Summarize all required seeds; missing/extra seeds are hard errors.

    Input must already identify a single condition, metric, and population.
    The experiment adapter must check those identities before this call.
    Min/max are descriptive seed ranges, not confidence limits.
    """
    required = list(expected_seeds)
    if not required or len(set(required)) != len(required):
        raise ValueError("Expected seeds must be a nonempty unique sequence")
    if set(values) != set(required):
        raise ValueError(f"Seed mismatch: expected {required}, got {list(values)}")
    numbers = np.array([float(values[seed]) for seed in required], dtype=float)
    if not np.all(np.isfinite(numbers)):
        raise ValueError("Nonfinite metric; do not silently omit a seed")
    return {"seeds": required, "values": numbers.tolist(),
            "mean": float(numbers.mean()), "min": float(numbers.min()),
            "max": float(numbers.max())}


def plot_seed_range(ax, x: Sequence[float], summaries: Sequence[dict], *,
                    index=0, label=None):
    """Draw means and ranges prepared by summarize_seeds, without smoothing."""
    points = np.asarray(x, dtype=float)
    if points.ndim != 1 or len(points) != len(summaries) or len(points) == 0:
        raise ValueError("One nonempty summary is required per x coordinate")
    if not np.all(np.isfinite(points)) or len(set(points)) != len(points):
        raise ValueError("Coordinates must be finite and unique")
    if not np.all(np.diff(points) > 0):
        raise ValueError("Explicitly order every requested condition before plotting")
    means = np.array([row["mean"] for row in summaries], dtype=float)
    lower = np.array([row["min"] for row in summaries], dtype=float)
    upper = np.array([row["max"] for row in summaries], dtype=float)
    if not np.all(np.isfinite(np.r_[means, lower, upper])):
        raise ValueError("Summary contains nonfinite values")
    if np.any(lower > means) or np.any(means > upper):
        raise ValueError("Means must be within the displayed seed range")
    keys = {tuple(row["seeds"]) for row in summaries}
    if len(keys) != 1:
        raise ValueError("Every plotted condition must include the same seed list")
    return ax.errorbar(points, means,
                       yerr=np.vstack([means - lower, upper - means]),
                       label=label, elinewidth=0.7, **series_style(index))


def source_hashes(paths: Iterable[Path]):
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths}


def save_figure(fig, stem: Path, *, title: str, description: str, provenance: dict):
    """Save matched PDF/SVG and a provenance sidecar, with no caption in image.

    The schema adapter must establish completeness before export. Requiring
    this flag prevents partial or unvalidated plots from being called final.
    """
    if provenance.get("complete") is not True:
        raise ValueError("Final export requires complete=True after schema validation")
    if not provenance.get("sources"):
        raise ValueError("Record the actual input files and hashes before export")
    stem = Path(stem)
    if stem.suffix:
        raise ValueError("Pass a file stem without .pdf or .svg")
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {fmt: stem.with_suffix('.' + fmt) for fmt in ("pdf", "svg")}
    fig.savefig(outputs["pdf"], bbox_inches="tight", pad_inches=0.035,
                metadata={"Title": title, "Subject": description,
                          "Creator": "Matplotlib; plot_selection_study.py",
                          "CreationDate": None, "ModDate": None})
    fig.savefig(outputs["svg"], bbox_inches="tight", pad_inches=0.035,
                metadata={"Title": title, "Description": description,
                          "Creator": "plot_selection_study.py", "Date": None})
    record = {"title": title, "description": description, "style": STYLE,
              "outputs": {key: str(value) for key, value in outputs.items()},
              **provenance}
    stem.with_suffix(".provenance.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return outputs


def _number(value, name, *, nonnegative=False, probability=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric {name}: {value!r}")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"Invalid {name}: {value!r}")
    if probability and not 0 <= number <= 1:
        raise ValueError(f"Invalid probability {name}: {value!r}")
    return number


def _verify_predictions(path, record, alpha):
    """Recompute every displayed metric from each final prediction CSV."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"window_index", "time", "y_true", "prediction", "raw_q025", "raw_q975",
                    "static_lower", "static_upper", "rolling_lower", "rolling_upper"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Prediction CSV missing fields: {path}")
        rows = list(reader)
    if len(rows) != record["counts"]["n_test"] or not rows:
        raise ValueError(f"Prediction count mismatch: {path}")
    values = {key: np.array([float(row[key]) for row in rows], dtype=float)
              for key in required - {"time", "window_index"}}
    if any(not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError(f"Nonfinite prediction data: {path}")
    times = np.array([row["time"] for row in rows], dtype="datetime64[ns]")
    ids = np.array([int(row["window_index"]) for row in rows], dtype=np.int64)
    if np.any(np.isnat(times)) or not np.all(np.diff(times) > np.timedelta64(0, "ns")):
        raise ValueError(f"Test timestamps are not strictly chronological: {path}")
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Repeated test identifiers: {path}")
    y, prediction = values["y_true"], values["prediction"]
    error = y - prediction
    denominator = np.sum((y - y.mean()) ** 2)
    if denominator <= 0:
        raise ValueError(f"Undefined test R-squared: {path}")
    calculated = {"point": {"n": len(y), "rmse": float(np.sqrt(np.mean(error ** 2))),
                            "mae": float(np.mean(np.abs(error))),
                            "r2": float(1 - np.sum(error ** 2) / denominator)}}
    for name in ("raw", "static", "rolling"):
        if name == "raw":
            low = np.minimum(values["raw_q025"], values["raw_q975"])
            high = np.maximum(values["raw_q025"], values["raw_q975"])
        else:
            low, high = values[f"{name}_lower"], values[f"{name}_upper"]
        if np.any(low > high):
            raise ValueError(f"Unordered {name} intervals: {path}")
        calculated[f"{name}_interval"] = {
            "n": len(y), "picp": float(np.mean((y >= low) & (y <= high))),
            "mpiw": float(np.mean(high - low)),
            "winkler": float(np.mean(high - low + 2 / alpha *
                                      (np.maximum(low - y, 0) + np.maximum(y - high, 0))))}
    for block, metrics in calculated.items():
        for metric, value in metrics.items():
            supplied = record[block][metric]
            if not np.isclose(supplied, value, rtol=1e-9, atol=1e-8):
                raise ValueError(f"Saved metric mismatch {path}: {block}/{metric}: {supplied} vs {value}")
    population = hashlib.sha256(ids.tobytes() + times.tobytes() + y.tobytes()).hexdigest()
    return population


def load_selection(directory: Path):
    """Validate the frozen 7-task, 6-method, 112-evaluation selection schema."""
    directory = Path(directory)
    manifest_path, results_path = directory / "manifest.json", directory / "results.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError("Selection-study manifest is not complete; no final figures are generated")
    protocol_path, lock_path = directory / "protocol.json", directory / "selection_lock.json"
    protocol, lock = (json.loads(p.read_text()) for p in (protocol_path, lock_path))
    if protocol != manifest["protocol"] or protocol.get("version") != "selection_v1":
        raise ValueError("Protocol does not match the frozen selection_v1 schema")
    if lock["protocol_sha256"] != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
        raise ValueError("Model-selection lock refers to a different protocol")
    if tuple(protocol["methods"]) != METHODS or protocol["seeds"] != [0, 1, 2]:
        raise ValueError("Unexpected method order or seed specification")
    tasks = {"within_site6": 6, **{f"cross_site{site}": site for site in range(1, 7)}}
    if {task["name"]: task["site"] for task in protocol["tasks"]} != tasks:
        raise ValueError("Expected the within-Farm6 task and all six cross-farm tasks")
    alpha = _number(protocol["alpha"], "alpha", probability=True)
    if alpha != 0.05 or protocol["taus"] != [0.5, 0.025, 0.975]:
        raise ValueError("Unexpected nominal level or fitted quantiles")
    # Read capacities from the hashed producer, without importing/training it.
    producer = ROOT / "review/run_selection_study.py"
    expected_hash = manifest["source_hashes"]["review/run_selection_study.py"]
    if hashlib.sha256(producer.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Study producer changed; recover its frozen capacity definition before plotting")
    tree = ast.parse(producer.read_text())
    capacity = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Name) and target.id == "CAPACITY" for target in node.targets))
    if set(capacity) != set(range(1, 7)) or any(float(value) <= 0 for value in capacity.values()):
        raise ValueError("Missing or invalid site capacities")
    selected = {(row["task"], row["method"]): row["selected_index"] for row in lock["selections"]}
    if len(selected) != 42 or len(lock["selections"]) != 42:
        raise ValueError("Incomplete or duplicated selection lock")
    entries = json.loads(results_path.read_text())
    expected = {(task, method, seed) for task in tasks for method in METHODS
                for seed in ([0] if method == "HistGBR" else protocol["seeds"])}
    if not isinstance(entries, list) or len(entries) != len(expected) or len(entries) != 112:
        raise ValueError("Final selection figures require all 112 evaluations")
    lookup, populations, common_counts = {}, {}, {}
    paths = [manifest_path, results_path, protocol_path, lock_path, producer]
    for record in entries:
        key = (record["task"], record["method"], record["seed"])
        if key not in expected or key in lookup or record["site"] != tasks[key[0]]:
            raise ValueError(f"Unexpected or duplicate evaluation: {key}")
        if record["selected_index"] != selected[key[:2]]:
            raise ValueError(f"Evaluation does not use the locked model: {key}")
        counts = record["counts"]
        for field in ("n_source", "n_target_fit", "n_validation", "n_calibration", "n_test"):
            number = _number(counts[field], f"{key}/{field}", nonnegative=True)
            if number != int(number) or (field != "n_source" and number == 0):
                raise ValueError(f"Invalid evaluation count: {key}/{field}")
        if (key[1] == "TargetSVQR") != (counts["n_source"] == 0):
            raise ValueError(f"Source-count/method mismatch: {key}")
        shared = tuple(counts[k] for k in ("n_target_fit", "n_validation", "n_calibration", "n_test"))
        if key[0] in common_counts and shared != common_counts[key[0]]:
            raise ValueError(f"Methods use different target populations: {key}")
        common_counts[key[0]] = shared
        if key[0].startswith("cross_") and counts["n_target_fit"] != protocol["cross_target_fit_budget"]:
            raise ValueError(f"Unexpected supervised target fitting budget: {key}")
        for block, metrics in (("point", ("rmse", "mae", "r2")),
                               ("raw_interval", ("picp", "mpiw", "winkler")),
                               ("static_interval", ("picp", "mpiw", "winkler")),
                               ("rolling_interval", ("picp", "mpiw", "winkler"))):
            if record[block]["n"] != counts["n_test"]:
                raise ValueError(f"Metric sample count mismatch: {key}/{block}")
            for metric in metrics:
                _number(record[block][metric], f"{key}/{block}/{metric}",
                        nonnegative=metric != "r2", probability=metric == "picp")
        if record["calibration"]["n"] != counts["n_calibration"]:
            raise ValueError(f"Calibration count mismatch: {key}")
        fits = record["fits"]
        if len(fits) != 3 or sorted(fit["tau"] for fit in fits) != sorted(protocol["taus"]):
            raise ValueError(f"Missing fitted quantile: {key}")
        folder = directory / key[0] / key[1] / f"final_seed{key[2]}"
        result_file, prediction_file = folder / "result.json", folder / "predictions.csv"
        if json.loads(result_file.read_text()) != record:
            raise ValueError(f"Aggregate and individual result records disagree: {key}")
        population = _verify_predictions(prediction_file, record, alpha)
        if key[0] in populations and populations[key[0]] != population:
            raise ValueError(f"Test identifiers or outcomes differ between methods/seeds: {key}")
        populations[key[0]] = population
        lookup[key] = record
        paths.extend([result_file, prediction_file])
    if set(lookup) != expected:
        raise ValueError("Missing final evaluations")
    return {"lookup": lookup, "protocol": protocol, "capacity_mw": capacity,
            "provenance": {"complete": True, "evaluations": len(lookup),
                           "sources": source_hashes(paths), "capacity_mw": capacity,
                           "test_population_sha256": populations,
                           "metric_verification": "all point/raw/static/rolling metrics recomputed from saved CSVs",
                           "seed_summary": "kernel methods: seeds 0,1,2 mean and minimum/maximum; HistGBR: seed 0",
                           "test_status": protocol["test_status"],
                           "online_protocol": protocol["online_protocol"],
                           "method_order": list(METHODS)}}


def selected_summary(report, task, method, block, metric, *, normalized=False):
    seeds = [0] if method == "HistGBR" else report["protocol"]["seeds"]
    values = {}
    for seed in seeds:
        row = report["lookup"][(task, method, seed)]
        factor = 100 / report["capacity_mw"][row["site"]] if normalized else 1
        values[seed] = row[block][metric] * factor
    return summarize_seeds(values, seeds)


def _mean_range(rows):
    mean = np.array([row["mean"] for row in rows])
    error = np.vstack([mean - np.array([row["min"] for row in rows]),
                       np.array([row["max"] for row in rows]) - mean])
    tolerance = 1e-12 * np.maximum(1., np.abs(mean))[None, :]
    if np.any(error < -tolerance):
        raise ValueError("A reported seed mean lies outside its seed range")
    return mean, np.maximum(error, 0.)


def _coverage_axis(ax, alpha):
    ax.axhline(1 - alpha, color="0.45", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylim(0, 1.035)
    ax.set_yticks(np.arange(0, 1.01, 0.2))


def plot_cross_site(report, output):
    """Both panels show all six methods and all six target farms."""
    with paper_style():
        fig, axes = new_figure(width="double", ncols=2, height=3.3)
        offsets = np.linspace(-0.24, 0.24, len(METHODS))
        plotted, handles = {}, []
        for index, method in enumerate(METHODS):
            series = []
            for ax, block, metric, normalized in (
                (axes[0, 0], "point", "rmse", True),
                (axes[0, 1], "rolling_interval", "picp", False)):
                rows = [selected_summary(report, f"cross_site{site}", method, block, metric,
                                         normalized=normalized) for site in range(1, 7)]
                mean, error = _mean_range(rows)
                styling = series_style(index); styling["linestyle"] = "none"
                handle = ax.errorbar(np.arange(1, 7) + offsets[index], mean, yerr=error,
                                     elinewidth=0.75, **styling)
                if block == "point":
                    handles.append(handle)
                series.append({"metric": block + "/" + metric, "normalized_percent_capacity": normalized,
                               "site_summaries": rows})
            plotted[method] = series
        for index, ax in enumerate(axes.flat):
            style_axis(ax, xlabel="Target wind farm")
            ax.set_xticks(range(1, 7)); ax.set_xlim(0.55, 6.45)
            panel_label(ax, f"({chr(97 + index)})")
        axes[0, 0].set_ylabel("NRMSE (% of capacity)")
        axes[0, 0].set_ylim(bottom=0)
        axes[0, 1].set_ylabel("Rolling PICP")
        _coverage_axis(axes[0, 1], report["protocol"]["alpha"])
        fig.legend(handles, METHOD_LABELS, loc="outside upper center", ncols=3)
        result = save_figure(fig, output / "selection_cross_farm", title="Supervised cross-farm comparison",
            description="Capacity-normalized point RMSE and rolling interval coverage for all six methods. "
                        "Points show seed means; bars show seed ranges. HistGBR uses seed 0. The dashed line is nominal 0.95 coverage.",
            provenance={**report["provenance"], "plotted_values": plotted})
        plt.close(fig)
    return result


def plot_calibration(report, output):
    with paper_style():
        fig, axes = new_figure(width="double", ncols=2, height=3.15)
        handles, plotted = [], {}
        for index, calibration in enumerate(("static", "rolling")):
            plotted[calibration] = []
            for ax, metric, normalized in ((axes[0, 0], "picp", False), (axes[0, 1], "mpiw", True)):
                rows = [selected_summary(report, f"cross_site{site}", "Full", calibration + "_interval",
                                         metric, normalized=normalized) for site in range(1, 7)]
                mean, error = _mean_range(rows)
                styling = series_style(index); styling["linestyle"] = "none"
                handle = ax.errorbar(np.arange(1, 7) + (-0.085 if index == 0 else 0.085), mean,
                                     yerr=error, elinewidth=0.75, **styling)
                if metric == "picp":
                    handles.append(handle)
                plotted[calibration].append({"metric": metric, "normalized_percent_capacity": normalized,
                                             "site_summaries": rows})
        for index, ax in enumerate(axes.flat):
            style_axis(ax, xlabel="Target wind farm")
            ax.set_xticks(range(1, 7)); ax.set_xlim(0.6, 6.4)
            panel_label(ax, f"({chr(97 + index)})")
        axes[0, 0].set_ylabel("PICP"); _coverage_axis(axes[0, 0], report["protocol"]["alpha"])
        axes[0, 1].set_ylabel("Mean width (% of capacity)"); axes[0, 1].set_ylim(bottom=0)
        fig.legend(handles, ["Static calibration", "Rolling calibration"], loc="outside upper center", ncols=2)
        result = save_figure(fig, output / "selection_calibration", title="Static and rolling calibration of Full",
            description="Full-model cross-farm coverage and capacity-normalized interval width, using all three seeds. "
                        "Bars show seed ranges; the dashed line marks nominal 0.95 coverage. Rolling calibration uses previously revealed outcomes.",
            provenance={**report["provenance"], "plotted_values": plotted})
        plt.close(fig)
    return result


def plot_within_comparison(report, output):
    with paper_style():
        fig, axes = new_figure(width="double", ncols=2, height=3.5, sharey=True)
        y = np.arange(len(METHODS))
        rows = [selected_summary(report, "within_site6", method, "point", "rmse") for method in METHODS]
        mean, error = _mean_range(rows)
        axes[0, 0].errorbar(mean, y, xerr=error, color="0.25", marker="o", linestyle="none", elinewidth=0.75)
        plotted = {"point_rmse": rows}; handles = []
        for index, calibration in enumerate(("static", "rolling")):
            rows = [selected_summary(report, "within_site6", method, calibration + "_interval", "winkler")
                    for method in METHODS]
            mean, error = _mean_range(rows); styling = series_style(index); styling["linestyle"] = "none"
            handles.append(axes[0, 1].errorbar(mean, y + (-0.14 if index == 0 else 0.14), xerr=error,
                                             elinewidth=0.75, **styling))
            plotted[calibration + "_winkler"] = rows
        for index, ax in enumerate(axes.flat):
            style_axis(ax, grid="x"); ax.set_xlim(left=0)
            ax.set_yticks(y, METHOD_LABELS); panel_label(ax, f"({chr(97 + index)})")
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_xlabel("RMSE (MW)")
        axes[0, 1].set_xlabel("Winkler score (MW)")
        axes[0, 1].legend(handles, ["Static interval", "Rolling interval"], loc="upper left",
                          bbox_to_anchor=(0.0, 1.18), ncols=2)
        result = save_figure(fig, output / "selection_within_farm", title="Validation-selected models on Farm 6",
            description="Within-farm point RMSE and static/rolling interval Winkler scores. Model order follows the protocol. "
                        "Kernel models use all three seeds; HistGBR uses seed 0. Bars show seed ranges.",
            provenance={**report["provenance"], "plotted_values": plotted})
        plt.close(fig)
    return result


def redraw_common_setting_curves(directory, output):
    """Use the existing verified v1 reader; change only the figure styling."""
    sys.path.insert(0, str(ROOT))
    from review.plot_replied_comment_study import collect, errors
    report = collect(Path(directory))
    if report["manifest_status"] != "complete" or report["missing_branches"]:
        raise ValueError("All common-setting curve points must be complete before export")
    if any(row["seeds"] != [0, 1, 2] for group in ("landmarks", "scarcity") for row in report[group]):
        raise ValueError("Common-setting plots require all three preset seeds")
    paths = [Path(directory) / "manifest.json"]
    for entry in report["sources"].values():
        if "result_file" in entry:
            paths.append(Path(entry["result_file"]))
    provenance = {"complete": True, "sources": source_hashes(paths), "study": "chronological_v1",
                  "data_source": "original saved results, unchanged by plotting",
                  "seed_summary": "three-seed mean/range for landmarks and scarcity; seed 0 for source scale",
                  "timing_scope": "solver fitting on shared host; excludes loading, preprocessing, prediction",
                  "plotted_values": {key: report[key] for key in ("landmarks", "source_scale", "scarcity")}}
    with paper_style():
        fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.3), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        left = fig.add_subplot(grid[:, 0]); upper = fig.add_subplot(grid[0, 1]); lower = fig.add_subplot(grid[1, 1])
        rows = sorted(report["landmarks"], key=lambda row: row["requested_m"])
        seconds, time_error = errors(rows, "fit_seconds"); rmse, error = errors(rows, "rmse")
        left.errorbar(seconds, rmse, xerr=time_error, yerr=error, elinewidth=0.75, **series_style(0))
        offsets = [(5, 7), (-5, -17), (6, 9), (5, -19), (5, 8), (-7, -17)]
        for x, y, row, offset in zip(seconds, rmse, rows, offsets):
            left.annotate(str(row["requested_m"]), (x, y), xytext=offset, textcoords="offset points",
                          fontsize=8, ha="right" if offset[0] < 0 else "left")
        left.margins(x=0.17, y=0.15)
        style_axis(left, xlabel="Fitting time (s)", ylabel="Test RMSE (MW)"); panel_label(left, "(a)")
        rows = sorted(report["source_scale"], key=lambda row: row["source_n"])
        source = np.array([row["source_n"] for row in rows])
        for index, (ax, metric, ylabel) in enumerate(((upper, "rmse", "Test RMSE (MW)"),
                                                     (lower, "fit_seconds", "Fitting time (s)"))):
            ax.plot(source, [row[metric]["mean"] for row in rows], **series_style(index))
            ax.set_xticks(source, [f"{value / 1000:.1f}" for value in source])
            style_axis(ax, xlabel=r"Source samples ($10^3$)", ylabel=ylabel)
            ax.margins(x=0.08, y=0.18); panel_label(ax, "(b)" if index == 0 else "(c)")
        timing = save_figure(fig, output / "chronological_accuracy_runtime",
            title="Landmark and source-size sensitivity at common settings",
            description="(a) RMSE versus fitting time; labels give landmark count. Points and bars show three-seed means and ranges. "
                        "(b,c) Source-size accuracy and fitting time at seed 0.", provenance=provenance)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(5.8, 3.0), layout="constrained")
        rows = sorted(report["scarcity"], key=lambda row: row["target_n"])
        budgets = [row["target_n"] for row in rows]; rmse, error = errors(rows, "rmse")
        ax.errorbar(budgets, rmse, yerr=error, elinewidth=0.75, **series_style(0))
        ax.set_xscale("log", base=2); ax.set_xticks(budgets, [str(value) for value in budgets]); ax.minorticks_off()
        ax.margins(x=0.06, y=0.12)
        style_axis(ax, xlabel="Target fitting labels", ylabel="Test RMSE (MW)")
        scarcity = save_figure(fig, output / "scarcity", title="Target-label sensitivity at common settings",
            description="Chronological Farm 6 test RMSE as the target fitting-label budget varies; means and ranges over seeds 0, 1, and 2. "
                        "The source pool, test population, and total domain coefficient masses are fixed.", provenance=provenance)
        plt.close(fig)
    report["captions"] = {
        "accuracy_runtime": "(a) Predictive RMSE versus fitting time across landmark counts. "
                            "(b,c) Source-size predictive error and fitting time. Landmark fits use all three seeds; source-size fits use seed 0.",
        "scarcity": "Target fitting-label sensitivity; points show three-seed means and bars show seed ranges."}
    report["partial_preview"] = False
    report["matplotlib_version"] = matplotlib.__version__
    report["plotting_script"] = str(Path(__file__).resolve())
    report["plotting_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["plot_style"] = STYLE
    report["output_formats"] = ["pdf", "svg"]
    (output / "chronological_plot_provenance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return {"accuracy_runtime": timing, "scarcity": scarcity}


def check_style():
    """Render typography only in memory; no fake observations or final files."""
    with paper_style():
        fig, axes = new_figure(width="double", ncols=2)
        for index, ax in enumerate(axes.flat):
            style_axis(ax, xlabel="Target fitting labels", ylabel="Test RMSE (MW)")
            panel_label(ax, f"({chr(97 + index)})")
            ax.text(0.5, 0.5, r"Typography check: $\lambda_1,\,\tau,\,m$",
                    transform=ax.transAxes, ha="center")
        fig.canvas.draw()
        font = font_manager.findfont(font_manager.FontProperties(family="STIXGeneral"),
                                     fallback_to_default=False)
        plt.close(fig)
    return {"matplotlib": matplotlib.__version__, "serif_font": font,
            "rendering": "passed; no plot files written"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-style", action="store_true",
                        help="Check serif and math rendering in memory")
    parser.add_argument("--input", type=Path, default=ROOT / "review/selection_runs/selection_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TLQLDMR_R1/figures")
    parser.add_argument("--v1-input", type=Path, default=ROOT / "review/replied_comment_runs/chronological_v1")
    parser.add_argument("--v1-only", action="store_true", help="Redraw completed common-setting curves only")
    parser.add_argument("--skip-v1", action="store_true", help="Generate selection-study figures only")
    args = parser.parse_args()
    if args.check_style:
        print(json.dumps(check_style(), indent=2))
        return
    if args.v1_only and args.skip_v1:
        parser.error("--v1-only and --skip-v1 are mutually exclusive")
    try:
        output = {}
        if not args.v1_only:
            report = load_selection(args.input)
            output.update(cross_farm=plot_cross_site(report, args.output_dir),
                          calibration=plot_calibration(report, args.output_dir),
                          within_farm=plot_within_comparison(report, args.output_dir))
        if not args.skip_v1:
            output.update(redraw_common_setting_curves(args.v1_input, args.output_dir))
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.error(str(error))
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
