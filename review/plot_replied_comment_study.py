#!/usr/bin/env python3
"""Plot predeclared landmark/source-size and label-budget comparisons.

Final figures require every specified seed and point. --allow-partial includes
every available requested run and writes conspicuously marked *.partial.pdf
previews only. It never writes or replaces a final figure with partial results.
No training, parameter selection, or favorable-seed filtering is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2166AC", "#D6601D"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Reader:
    def __init__(self, directory):
        self.directory = directory
        self.cache, self.sources, self.missing = {}, {}, []
        self.available = {p.parent.name for p in directory.glob("*/result.json")}
        self.available.update(p.parent.name for p in directory.glob("*/alias.json"))

    def get(self, name, seed, trail=()):
        if name in self.cache:
            record = self.cache[name]
        elif name not in self.available:
            self.missing.append(name)
            return None
        elif name in trail or Path(name).name != name:
            raise ValueError(f"Invalid/cyclic alias: {name}")
        else:
            folder = self.directory / name
            alias = folder / "alias.json"
            if alias.exists():
                reference = json.loads(alias.read_text())["reference"]
                record = self.get(reference, seed, (*trail, name))
                if record is None:
                    self.missing.append(name)
                    return None
                self.sources[name] = {"alias_of": reference, "alias_sha256": digest(alias)}
            else:
                path = folder / "result.json"
                record = json.loads(path.read_text())
                self.sources[name] = {"result_file": str(path.resolve()), "result_sha256": digest(path)}
            self.cache[name] = record
        if record["task"] != "point" or record["seed"] != seed:
            raise ValueError(f"Wrong task or seed in {name}; expected point seed {seed}.")
        if not record["fits"]:
            raise ValueError(f"No model fitting record in {name}.")
        return record


def row(record):
    values = {"rmse": float(record["metrics"]["rmse"]),
              "fit_seconds": float(sum(f["fit_seconds"] for f in record["fits"])),
              "target_n": int(record["target_n"]), "source_n": int(record["source_n"]),
              "test_n": int(record["metrics"]["n"]), "seed": int(record["seed"]),
              "effective_m": int(record["fits"][0]["effective_landmarks"])}
    if not all(math.isfinite(v) for v in values.values()) or values["fit_seconds"] <= 0:
        raise ValueError(f"Invalid numeric plotting values for {record['name']}.")
    return values


def group(records):
    values = [row(r) for r in records]
    if not values:
        return None
    result = {"runs": len(values), "seeds": sorted(v["seed"] for v in values), "individual": values}
    for key in ("rmse", "fit_seconds"):
        x = np.array([v[key] for v in values])
        result[key] = {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max())}
    for key in ("target_n", "source_n", "test_n", "effective_m"):
        distinct = {v[key] for v in values}
        if len(distinct) != 1:
            raise ValueError(f"Seed replicates disagree on {key}: {distinct}")
        result[key] = distinct.pop()
    return result


def collect(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cfg, reader = manifest["protocol"], Reader(directory)
    result = {"manifest_status": manifest.get("status"), "protocol": cfg,
              "manifest_sha256": digest(manifest_path), "landmarks": [], "source_scale": [], "scarcity": []}
    seeds = cfg["seeds"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated seeds in protocol must not be counted as independent runs.")
    for m in cfg["landmark_grid"]:
        records = [r for seed in seeds if (r := reader.get(f"landmarks_m{m}_s{seed}", seed)) is not None]
        summary = group(records)
        if summary:
            if summary["effective_m"] != m:
                raise ValueError(f"Requested m={m}, effective m={summary['effective_m']}; plot labels require review.")
            result["landmarks"].append(dict(requested_m=m, **summary))
    for fraction in cfg["source_fractions"]:
        name = "scale_fraction1" if fraction == 1 else f"scale_fraction{fraction}"
        record = reader.get(name, 0)
        if record:
            result["source_scale"].append(dict(fraction=fraction, **group([record])))
    for budget in cfg["label_budgets"]:
        records = []
        for seed in seeds:
            name = f"components_Full_point_s{seed}" if budget == "all" else f"scarcity_n{budget}_s{seed}"
            record = reader.get(name, seed)
            if record:
                records.append(record)
        summary = group(records)
        if summary:
            if budget != "all" and summary["target_n"] != int(budget):
                raise ValueError(f"Requested target budget {budget}, observed {summary['target_n']}.")
            result["scarcity"].append(dict(requested_budget=budget, **summary))
    test_counts = {r["test_n"] for section in ("landmarks", "source_scale", "scarcity") for r in result[section]}
    if len(test_counts) > 1:
        raise ValueError(f"Plot branches use different test-set sizes: {test_counts}")
    for section in ("landmarks", "scarcity"):
        if len({r["source_n"] for r in result[section]}) > 1:
            raise ValueError(f"Source size changes within {section} comparison.")
    result["missing_branches"] = sorted(set(reader.missing))
    result["sources"] = reader.sources
    return result


def errors(rows, key):
    means = np.array([r[key]["mean"] for r in rows])
    lower = means - np.array([r[key]["min"] for r in rows])
    upper = np.array([r[key]["max"] for r in rows]) - means
    return means, np.vstack([lower, upper])


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "savefig.dpi": 300})


def accuracy_runtime(report, path, partial):
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    left, right = axes
    rows = sorted(report["landmarks"], key=lambda r: r["requested_m"])
    if rows:
        seconds, xerr = errors(rows, "fit_seconds")
        rmse, yerr = errors(rows, "rmse")
        left.errorbar(seconds, rmse, xerr=xerr, yerr=yerr, fmt="o-", color=BLUE,
                      capsize=3, linewidth=1.4, markersize=4.5, elinewidth=.9)
        for i, (x, y, r) in enumerate(zip(seconds, rmse, rows)):
            label = f"m = {r['requested_m']}" + (f" (n={r['runs']})" if partial else "")
            # Keep the second label clear of the neighboring error-bar cap.
            offset = (-7, -26) if i == 1 else (5, 6 if i % 2 == 0 else -13)
            left.annotate(label, (x, y), xytext=offset, textcoords="offset points", fontsize=8,
                          ha="right" if i == 1 else "left")
        left.margins(x=.18, y=.17)
    else:
        left.text(.5, .5, "No completed landmark runs", transform=left.transAxes, ha="center")
    left.set(xlabel="Solver fitting time (s)", ylabel="Test RMSE (MW)", title="(a) Landmark accuracy–runtime trade-off")
    left.grid(alpha=.2)
    rows = sorted(report["source_scale"], key=lambda r: r["source_n"])
    twin = right.twinx()
    twin.spines["right"].set_visible(True)
    if rows:
        source = [r["source_n"] for r in rows]
        rmse = [r["rmse"]["mean"] for r in rows]
        seconds = [r["fit_seconds"]["mean"] for r in rows]
        h1, = right.plot(source, rmse, "o-", color=BLUE, linewidth=1.4, markersize=4.5, label="RMSE")
        h2, = twin.plot(source, seconds, "s--", color=ORANGE, linewidth=1.4, markersize=4, label="Fitting time")
        right.set_xticks(source)
        right.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x / 1000:g}k"))
        right.legend(handles=[h1, h2], loc="best", frameon=False)
        right.margins(x=.1, y=.15)
        twin.margins(y=.15)
    else:
        right.text(.5, .5, "No completed source-size runs", transform=right.transAxes, ha="center")
    right.set(xlabel="Source fitting samples", ylabel="Test RMSE (MW)", title="(b) Source-size sensitivity (seed 0)")
    twin.set_ylabel("Solver fitting time (s)", color=ORANGE)
    twin.tick_params(axis="y", colors=ORANGE)
    right.grid(alpha=.2)
    seed_text = ", ".join(map(str, report["protocol"]["seeds"]))
    caption = (f"Landmark points: means across seeds {seed_text}; error bars are minima–maxima, not confidence intervals.\n"
               "Time covers solver.fit only (features, regularizer, optimization); shared-host timings exclude loading, preprocessing and prediction.")
    if partial:
        figure.suptitle("PARTIAL PREVIEW — requested runs are missing", color="#A50026", fontsize=11)
        caption = "PARTIAL: all available requested seeds are included; n labels give run counts.\n" + caption
    figure.text(.5, .025, caption, ha="center", va="bottom", fontsize=7.5)
    figure.tight_layout(rect=(0, .16 if partial else .12, 1, .94 if partial else 1), w_pad=2.2)
    figure.savefig(path, bbox_inches="tight", metadata={"Title": "Chronological accuracy and fitting-time sensitivity", "Subject": caption})
    plt.close(figure)
    return caption


def scarcity(report, path, partial):
    figure, axis = plt.subplots(figsize=(6.8, 4.3))
    rows = sorted(report["scarcity"], key=lambda r: r["target_n"])
    if rows:
        labels = [r["target_n"] for r in rows]
        rmse, yerr = errors(rows, "rmse")
        axis.errorbar(labels, rmse, yerr=yerr, fmt="o-", color=BLUE, linewidth=1.5,
                      capsize=3, markersize=4.5, elinewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(labels, [str(n) for n in labels])
        axis.minorticks_off()
        axis.margins(x=.07, y=.15)
        if partial:
            for x, y, r in zip(labels, rmse, rows):
                axis.annotate(f"n={r['runs']}", (x, y), xytext=(4, 6), textcoords="offset points", fontsize=8)
    else:
        axis.text(.5, .5, "No completed target-budget runs", transform=axis.transAxes, ha="center")
    axis.set(xlabel="Target fitting labels (log scale)", ylabel="Test RMSE (MW)", title="Target-label scarcity under chronological evaluation")
    axis.grid(alpha=.2)
    seed_text = ", ".join(map(str, report["protocol"]["seeds"]))
    caption = (f"Points are means across seeds {seed_text}; bars show minima–maxima, not confidence intervals.\n"
               "Source data, test set and aggregate domain loss weights are fixed; parameters are not claimed optimal.")
    if partial:
        axis.set_title("PARTIAL PREVIEW — target-label scarcity", color="#A50026")
        caption = "PARTIAL: all available requested seeds are included; n labels give run counts.\n" + caption
    figure.text(.5, .025, caption, ha="center", va="bottom", fontsize=7.5)
    figure.tight_layout(rect=(0, .17 if partial else .13, 1, 1))
    figure.savefig(path, bbox_inches="tight", metadata={"Title": "Chronological target-label scarcity", "Subject": caption})
    plt.close(figure)
    return caption


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "review/replied_comment_runs/chronological_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "TLQLDMR_R1/figures")
    parser.add_argument("--allow-partial", action="store_true", help="Write marked *.partial.pdf previews, never final figure paths.")
    args = parser.parse_args()
    if not (args.input / "manifest.json").is_file():
        parser.error("No study manifest exists; no figures generated.")
    report = collect(args.input)
    if report["missing_branches"] and not args.allow_partial:
        parser.error("Final figures require all predeclared runs. Missing: " + ", ".join(report["missing_branches"]))
    # Even a complete study uses preview filenames if partial mode was requested.
    partial = args.allow_partial
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    suffix = ".partial" if partial else ""
    runtime_path = args.output_dir / f"chronological_accuracy_runtime{suffix}.pdf"
    scarcity_path = args.output_dir / f"scarcity{suffix}.pdf"
    report["captions"] = {"accuracy_runtime": accuracy_runtime(report, runtime_path, partial),
                          "scarcity": scarcity(report, scarcity_path, partial)}
    report["partial_preview"] = partial
    report["matplotlib_version"] = matplotlib.__version__
    provenance_path = args.output_dir / f"chronological_plot_provenance{suffix}.json"
    provenance_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"accuracy_runtime": str(runtime_path), "scarcity": str(scarcity_path),
                      "provenance": str(provenance_path), "partial": partial,
                      "missing_runs": len(report["missing_branches"])}))


if __name__ == "__main__":
    main()
