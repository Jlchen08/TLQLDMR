#!/usr/bin/env python3
"""Summarize completed chronological-study branches without fitting any model.

Metrics are recomputed from saved predictions. Alias branches are resolved rather
than counted as independent fits. Weather/quarter labels are reconstructed with
the manifest's temporal protocol and checked against saved data IDs. Missing or
unfinished branches remain explicit; no missing results are filled in.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.temporal_data import TemporalProtocol, load_temporal_farm


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(frame, task, alpha):
    if not len(frame):
        return {"n": 0}
    y = frame["y_true"].to_numpy(float)
    if task == "point":
        error = y - frame["prediction"].to_numpy(float)
        denominator = float(np.sum((y - y.mean()) ** 2))
        return {"n": len(y), "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mae": float(np.mean(abs(error))),
                "r2": float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else None}
    lo, hi = frame["lower"].to_numpy(float), frame["upper"].to_numpy(float)
    if np.any(hi < lo):
        raise ValueError("Saved interval endpoints are not ordered.")
    width = hi - lo
    span = float(np.ptp(y))
    return {"n": len(y), "picp": float(np.mean((lo <= y) & (y <= hi))),
            "mpiw": float(width.mean()), "pinaw": float(width.mean() / span) if span > 0 else None,
            "winkler": float(np.mean(width + (2 / alpha) *
                                      (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))))}


def aggregate(records, expected_seeds=None):
    """Descriptive seed means/ranges; these ranges are not confidence intervals."""
    names = sorted({key for r in records for key in r})
    result = {"runs": len(records), "seeds": sorted({r["seed"] for r in records if "seed" in r})}
    if expected_seeds is not None:
        result["missing_seeds"] = sorted(set(expected_seeds) - set(result["seeds"]))
    for key in names:
        if key == "seed":
            continue
        values = [float(r[key]) for r in records
                  if isinstance(r.get(key), (int, float, np.integer, np.floating))
                  and np.isfinite(r[key])]
        if not values:
            result[key] = None
        else:
            x = np.asarray(values)
            result[key] = {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
                           "sd": float(x.std(ddof=1)) if len(x) > 1 else None, "valid_runs": len(x)}
    return result


def q99_core(data, ids):
    wind = np.percentile(data.training_weather["wind_hub"], 99)
    low, high = np.percentile(data.training_weather["temperature"], [1, 99])
    hub, temp = data.target_weather["wind_hub"][ids], data.target_weather["temperature"][ids]
    return (hub > wind) | (temp < low) | (temp > high) | (hub > 25)


class Summary:
    def __init__(self, directory):
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.protocol = self.manifest["protocol"]
        self.alpha = float(self.protocol.get("alpha", .05))
        self.records, self.frames, self.datasets, self.errors, self.aliases = {}, {}, {}, [], {}
        self.audit = {}
        self.available_branches = {path.parent.name for path in directory.glob("*/result.json")
                                   if (path.parent / "predictions.csv").is_file()}
        self.available_branches.update(path.parent.name for path in directory.glob("*/alias.json"))

    def load(self, name, trail=()):
        if name in self.records:
            return self.records[name]
        if name not in self.available_branches:
            return None
        if name in trail:
            raise ValueError("Alias cycle: " + " -> ".join((*trail, name)))
        if Path(name).name != name:
            raise ValueError("An alias must reference a branch in the study directory.")
        folder = self.directory / name
        alias = folder / "alias.json"
        if alias.exists():
            reference = json.loads(alias.read_text())["reference"]
            source = self.load(reference, (*trail, name))
            if source is None:
                return None
            self.aliases[name] = reference
            self.records[name] = dict(source, logical_name=name, alias_of=reference)
            return self.records[name]
        path = folder / "result.json"
        csv_path = folder / "predictions.csv"
        if not path.exists() or not csv_path.exists():
            return None
        record = json.loads(path.read_text())
        frame = pd.read_csv(csv_path)
        if frame["window_index"].duplicated().any():
            raise ValueError(f"Duplicate prediction window IDs: {name}")
        required = ["y_true", "prediction"] if record["task"] == "point" else ["y_true", "lower", "upper"]
        if not np.isfinite(frame[required].to_numpy(float)).all():
            raise ValueError(f"Nonfinite saved predictions: {name}")
        recomputed = metrics(frame, record["task"], self.alpha)
        checks = {}
        for key, old in record["metrics"].items():
            new = recomputed.get(key)
            checks[key] = old is None and new is None or (
                old is not None and new is not None and bool(np.isclose(old, new, rtol=1e-9, atol=1e-9)))
        if not all(checks.values()):
            self.errors.append({"branch": name, "problem": "Saved result metrics differ from predictions", "checks": checks})
        canonical = record["name"]
        self.frames[canonical] = frame
        self.audit[canonical] = {"metrics_match_predictions": checks,
                                 "result_sha256": digest(path), "predictions_sha256": digest(csv_path)}
        record = dict(record, logical_name=name, recomputed_metrics=recomputed,
                      fit_seconds=sum(f["fit_seconds"] for f in record["fits"]))
        self.records[name] = record
        return record

    def data(self, site):
        if site in self.datasets:
            return self.datasets[site]
        expected_code = self.manifest.get("source_hashes", {}).get("review/temporal_data.py")
        if expected_code and digest(ROOT / "review/temporal_data.py") != expected_code:
            raise ValueError("Temporal loader changed since fitting; subgroup reconstruction refused.")
        audit_file = self.directory / "data" / f"farm{site}_audit.json"
        audit = json.loads(audit_file.read_text())
        recorded = str(audit["provenance"]["data_file"])
        marker = "<package-root>/"
        path = ROOT / recorded[len(marker):] if recorded.startswith(marker) else Path(recorded)
        if digest(path) != audit["provenance"]["data_sha256"]:
            raise ValueError(f"Farm {site} input hash differs from fitting input.")
        with contextlib.redirect_stdout(io.StringIO()):
            data = load_temporal_farm(path, TemporalProtocol(**self.protocol["temporal"]))
        cached = np.load(self.directory / "data" / f"farm{site}_ids.npz", allow_pickle=False)
        if not np.array_equal(data.target_row_indices, cached["raw_target_row"]) or not np.array_equal(data.timestamps, cached["times"]):
            raise ValueError(f"Farm {site} window IDs or timestamps do not reproduce.")
        self.datasets[site] = (data, float(audit["capacity"]))
        return self.datasets[site]

    def frame(self, record):
        return self.frames[record["name"]]

    def values(self, record, subset=None):
        frame = self.frame(record)
        values = metrics(frame if subset is None else frame.loc[subset], record["task"], self.alpha)
        values.update(seed=record["seed"], source_n=record["source_n"], target_n=record["target_n"],
                      calibration_n=record["calibration"]["n"] if record["calibration"] else 0,
                      fit_seconds=record["fit_seconds"],
                      effective_landmarks=min(f["effective_landmarks"] for f in record["fits"]),
                      epochs=max(f["options"]["epochs"] for f in record["fits"]),
                      adam_updates_per_quantile=max(f["options"]["epochs"] * math.ceil(
                          (record["source_n"] + record["target_n"]) / f["options"]["batch_size"])
                          for f in record["fits"]))
        return values

    def find(self, name):
        try:
            return self.load(name)
        except Exception as error:
            self.errors.append({"branch": name, "problem": f"{type(error).__name__}: {error}"})
            return None

    def grouped(self, names, expected_seeds=None):
        records = [record for name in names if (record := self.find(name)) is not None]
        return {"branches": [r["logical_name"] for r in records],
                "statistics": aggregate([self.values(r) for r in records], expected_seeds)}

    def subgroup_values(self, record, group, selector):
        data, _ = self.data(record["site"])
        frame = self.frame(record)
        ids = frame["window_index"].to_numpy(int)
        if not np.array_equal(frame["raw_target_row"].to_numpy(int), data.target_row_indices[ids]):
            raise ValueError("Prediction raw-row IDs do not match reconstructed data.")
        if not np.allclose(frame["y_true"], data.y_raw[ids], rtol=0, atol=1e-9):
            raise ValueError("Prediction labels do not match reconstructed data.")
        return self.values(record, selector(data, ids))

    def build(self):
        cfg = self.protocol
        seeds = cfg["seeds"]
        expected = []
        output = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "input_directory": str(self.directory.resolve()), "manifest_status": self.manifest.get("status"),
                  "available_branches_at_start": sorted(self.available_branches),
                  "protocol": cfg, "components": [], "scarcity": [], "weighting": [], "thresholds": [],
                  "landmarks": [], "source_scale": [], "cross_farm": [], "subgroups": []}
        for variant in cfg["components"]:
            for task in ("point", "interval"):
                names = [f"components_{variant}_{task}_s{s}" for s in seeds]
                expected.extend(names)
                output["components"].append(dict(variant=variant, task=task, **self.grouped(names, seeds)))
        for budget in cfg["label_budgets"]:
            names = [f"components_Full_point_s{s}" if budget == "all" else f"scarcity_n{budget}_s{s}" for s in seeds]
            expected.extend(names)
            output["scarcity"].append(dict(requested_budget=budget, **self.grouped(names, seeds)))
        for scheme in cfg["weight_schemes"]:
            for task in ("point", "interval"):
                name = f"components_Full_{task}_s0" if scheme == "domain_balanced" else f"weights_{scheme}_{task}"
                expected.append(name)
                group = self.grouped([name], [0])
                record = self.find(name)
                coefficients = None
                if record:
                    opts = record["fits"][0]["options"]
                    ns, nt = record["source_n"], record["target_n"]
                    source_mass, target_mass = ns * opts["C_S"], nt * opts["C_T"]
                    coefficients = {"C_S": opts["C_S"], "C_T": opts["C_T"], "source_mass": source_mass,
                                    "target_mass": target_mass, "target_fraction": target_mass / (source_mass + target_mass)}
                output["weighting"].append(dict(scheme=scheme, task=task, coefficients=coefficients, **group))
        for q in cfg["threshold_quantiles"]:
            for task in ("point", "interval"):
                name = f"threshold_q{q}_{task}"
                expected.append(name)
                record = self.find(name)
                # q=98 is a predeclared alias even while the runner has not reached alias creation.
                if record is None and q == 98:
                    record = self.find(f"components_Full_{task}_s0")
                row = {"q": q, "cutout_ms": 25, "task": task, "branch": record["logical_name"] if record else None,
                       "own_test": self.values(record) if record else None, "common_q99_core": None}
                if record:
                    try:
                        row["common_q99_core"] = self.subgroup_values(record, "common_q99_core", q99_core)
                        data, _ = self.data(record["site"])
                        ids = np.flatnonzero(data.split == "test")
                        row["expected_common_core_n"] = int(q99_core(data, ids).sum())
                        row["common_core_population_complete"] = row["common_q99_core"]["n"] == row["expected_common_core_n"]
                    except Exception as error:
                        self.errors.append({"branch": name, "problem": f"Common-core reconstruction: {error}"})
                output["thresholds"].append(row)
        definition_file = self.directory / "threshold_definitions.json"
        output["threshold_definitions"] = json.loads(definition_file.read_text()) if definition_file.exists() else None
        for definition in output["threshold_definitions"] or []:
            if definition["identical_domain_mask_to_25"]:
                continue
            q, cutoff = definition["q"], definition["cutout_ms"]
            for task in ("point", "interval"):
                name = f"threshold_q{q}_cutout{cutoff}_{task}"
                expected.append(name)
                record = self.find(name)
                row = {"q": q, "cutout_ms": cutoff, "task": task,
                       "branch": record["logical_name"] if record else None,
                       "own_test": self.values(record) if record else None, "common_q99_core": None}
                if record:
                    try:
                        row["common_q99_core"] = self.subgroup_values(record, "common_q99_core", q99_core)
                        data, _ = self.data(record["site"])
                        ids = np.flatnonzero(data.split == "test")
                        row["expected_common_core_n"] = int(q99_core(data, ids).sum())
                        row["common_core_population_complete"] = row["common_q99_core"]["n"] == row["expected_common_core_n"]
                    except Exception as error:
                        self.errors.append({"branch": name, "problem": f"Common-core reconstruction: {error}"})
                output["thresholds"].append(row)
        for landmarks in cfg["landmark_grid"]:
            names = [f"landmarks_m{landmarks}_s{s}" for s in seeds]
            expected.extend(names)
            output["landmarks"].append(dict(requested_landmarks=landmarks, **self.grouped(names, seeds)))
        for fraction in cfg["source_fractions"]:
            name = "scale_fraction1" if fraction == 1 else f"scale_fraction{fraction}"
            expected.append(name)
            output["source_scale"].append(dict(source_fraction=fraction, **self.grouped([name], [0])))
        for site in cfg["cross_farm_sites"]:
            for variant in ("Full", "TargetOnly"):
                for task in ("point", "interval"):
                    name = f"crossfarm_site{site}_{variant}_{task}"
                    expected.append(name)
                    record = self.find(name)
                    row = {"site": site, "variant": variant, "task": task,
                           "values": self.values(record) if record else None,
                           "supervised_target_adaptation": True, "requested_target_fit_labels": cfg["cross_farm_target_labels"]}
                    if record:
                        try:
                            _, capacity = self.data(site)
                            row["capacity_mw"] = capacity
                            if task == "point":
                                row["values"]["rmse_per_capacity"] = row["values"]["rmse"] / capacity
                                row["values"]["mae_per_capacity"] = row["values"]["mae"] / capacity
                            else:
                                row["values"]["mpiw_per_capacity"] = row["values"]["mpiw"] / capacity
                        except Exception as error:
                            self.errors.append({"branch": name, "problem": f"Capacity/data verification: {error}"})
                    output["cross_farm"].append(row)
        try:
            data, _ = self.data(int(cfg.get("within_site", 6)))
            quarter_labels = pd.DatetimeIndex(data.timestamps).to_period("Q").astype(str).to_numpy()
            quarters = sorted(set(quarter_labels[data.split == "test"]))
            definitions = [("quarter", q, lambda d, ids, q=q: pd.DatetimeIndex(d.timestamps[ids]).to_period("Q").astype(str).to_numpy() == q)
                           for q in quarters]
            definitions += [
                ("weather", "high_wind", lambda d, ids: d.target_weather["wind_hub"][ids] > d.preprocessing["domain_thresholds"]["wind_ms"]),
                ("weather", "cold", lambda d, ids: d.target_weather["temperature"][ids] < d.preprocessing["domain_thresholds"]["temperature_low_c"]),
                ("weather", "hot", lambda d, ids: d.target_weather["temperature"][ids] > d.preprocessing["domain_thresholds"]["temperature_high_c"]),
                ("weather", "cutout", lambda d, ids: d.target_weather["wind_hub"][ids] > d.preprocessing["domain_thresholds"]["cutout_ms"])]
            for kind, label, selector in definitions:
                for variant in cfg["components"]:
                    for task in ("point", "interval"):
                        values = []
                        for seed in seeds:
                            record = self.find(f"components_{variant}_{task}_s{seed}")
                            if record:
                                values.append(self.subgroup_values(record, label, selector))
                        output["subgroups"].append({"kind": kind, "group": label, "variant": variant, "task": task,
                                                    "weather_groups_overlap": kind == "weather", "statistics": aggregate(values, seeds)})
        except Exception as error:
            self.errors.append({"problem": f"Quarter/weather grouping: {error}"})
        output["data_audits"] = {path.stem: json.loads(path.read_text()) for path in sorted((self.directory / "data").glob("farm*_audit.json"))}
        output["aliases"] = self.aliases
        output["prediction_verification"] = self.audit
        output["missing_expected_branches"] = sorted({name for name in expected if name not in self.records})
        unique = {r["name"]: r for r in self.records.values()}
        output["completed_unique_branches"] = len(unique)
        output["completed_unique_quantile_fits"] = sum(len(r["fits"]) for r in unique.values())
        output["errors"] = self.errors
        output["interpretation_limits"] = [
            "Means and ranges describe the retained fixed seeds, not confidence intervals or significance tests.",
            "Cross-farm experiments are supervised adaptation with 128 requested fitting labels from the target farm; intervals also use separately counted target calibration labels. They are not unlabeled transfer.",
            "The supplied original workbooks do not prove an absence of upstream processing or historical dataset knowledge.",
            "All numerical configurations are fixed, not claimed optimal or selected by target validation.",
            "Outcome-time observed weather defines test subgroups retrospectively; event-onset prediction is not evaluated.",
            "Weather groups overlap; their counts must not be added. Quarter results cover the actual test quarters, not all meteorological seasons.",
            "Threshold-specific test populations differ; common-q99-core metrics provide a shared-population diagnostic.",
            "fit_seconds covers solver.fit only, including feature/regularizer construction and optimization; it excludes loader/preprocessing, prediction, and summary recomputation. No inference-latency or full-pipeline runtime claim follows."]
        return output


def escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def number(value, decimals=3):
    return "---" if value is None else f"{value:.{decimals}f}"


def stat(row, key, decimals=3, spread=True):
    value = row.get(key)
    if value is None:
        return "---"
    if not isinstance(value, dict):
        return number(value, decimals)
    result = number(value["mean"], decimals)
    return result + " [" + number(value["min"], decimals) + ", " + number(value["max"], decimals) + "]" if spread else result


def table(caption, label, headers, rows):
    if not rows:
        return "% No completed rows for " + label + "\n"
    return ("\\begin{table}[ht]\n\\centering\n\\small\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
            "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{" + "l" * len(headers) + "}\n\\toprule\n" +
            " & ".join(headers) + r" \\" + "\n\\midrule\n" +
            "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) +
            "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n\n")


def latex(report):
    variant_labels = {
        "NoMMD": "No discrepancy",
    }
    group_labels = {
        "cutout": "$>25$\\,m/s",
    }
    parts = ["% Generated from saved predictions. Missing runs are never filled in.\n"
             "% Brackets are across-seed minima/maxima, not confidence intervals.\n"]
    for task in ("point", "interval"):
        metric_keys = ["rmse", "mae", "r2"] if task == "point" else ["picp", "mpiw", "winkler"]
        metric_titles = ["RMSE (MW)", "MAE (MW)", "$R^2$"] if task == "point" else ["PICP", "MPIW (MW)", "Winkler (MW)"]
        component_rows = []
        for r in report["components"]:
            s = r["statistics"]
            if r["task"] == task and s["runs"]:
                component_rows.append([escape(variant_labels.get(r["variant"], r["variant"])), s["runs"], stat(s, "n", 0, False)] + [stat(s, k) for k in metric_keys] + [stat(s, "fit_seconds", 2)])
        parts.append(table("Chronological component evaluation: seed mean [minimum, maximum]. Solver fitting times exclude data loading, preprocessing and prediction.", "tab:chronological_components_" + task,
                           ["Variant", "Runs", "$n_{test}$"] + metric_titles + ["Fit (s)"], component_rows))
        weight_rows = []
        for r in report["weighting"]:
            s = r["statistics"]
            if r["task"] == task and s["runs"]:
                c = r["coefficients"]
                weight_rows.append([escape(r["scheme"]), number(c["C_S"], 5), number(c["C_T"], 5), number(c["target_fraction"], 4)] + [stat(s, k, spread=False) for k in metric_keys])
        parts.append(table("Weighting sensitivity at matched total loss weight; seed 0. Domain normalization and imbalance compensation are the same configuration at this scale.", "tab:chronological_weighting_" + task,
                           ["Scheme", "$C_S$", "$C_T$", "Target mass fraction"] + metric_titles, weight_rows))
        threshold_rows = []
        for r in report["thresholds"]:
            if r["task"] == task and r["own_test"]:
                own, core = r["own_test"], r["common_q99_core"] or {}
                threshold_rows.append([r["q"], r["cutout_ms"], own["n"], core.get("n", "---")] + [number(own.get(k)) for k in metric_keys] + [number(core.get(k)) for k in metric_keys])
        parts.append(table("Threshold sensitivity on each definition's test population and the shared 99th-percentile core; seed 0. Meteorological percentiles are estimated from training observations.", "tab:chronological_thresholds_" + task,
                           ["Percentile", "Nominal high-wind cutoff (m/s)", "$n_{own}$", "$n_{core}$"] + ["Own " + k for k in metric_titles] + ["Core " + k for k in metric_titles], threshold_rows))
        cross_rows = []
        for r in report["cross_farm"]:
            v = r["values"]
            if r["task"] == task and v:
                cross_rows.append([r["site"], escape(r["variant"]), v["source_n"], v["target_n"], v["calibration_n"], v["n"]] + [number(v.get(k)) for k in metric_keys])
        parts.append(table("Cross-farm supervised adaptation with 128 requested target fitting labels; source normals come from the other five farms. Target-only fitting uses the same selected labels. Interval calibration labels are additional and reported separately. Fixed parameters are not claimed optimal.", "tab:chronological_crossfarm_" + task,
                           ["Site", "Variant", "$n_S$", "$n_T$", "$n_{cal}$", "$n_{test}$"] + metric_titles, cross_rows))
        for kind in ("quarter", "weather"):
            rows = []
            for r in report["subgroups"]:
                s = r["statistics"]
                if r["kind"] == kind and r["task"] == task and s["runs"]:
                    rows.append([group_labels.get(r["group"], escape(r["group"])), escape(variant_labels.get(r["variant"], r["variant"])), s["runs"], stat(s, "n", 0, False)] + [stat(s, k) for k in metric_keys])
            parts.append(table("Component results by " + kind + ": seed mean [minimum, maximum]. Weather groups overlap; zero-count groups have no performance estimate. Quarters cover only the evaluation period.",
                               "tab:chronological_" + kind + "_" + task, ["Group", "Variant", "Runs", "$n$"] + metric_titles, rows))
    for section, first, key, label, caption in [
        ("scarcity", "Target-label budget", "requested_budget", "scarcity", "Target-label scarcity with fixed source data and matched domain loss masses; seed mean [minimum, maximum]. Only point fitting is evaluated here."),
        ("landmarks", "Requested landmarks", "requested_landmarks", "landmarks", "Accuracy--runtime trade-off with fixed training and test data; seed mean [minimum, maximum]. Time covers solver fitting only."),
        ("source_scale", "Source fraction", "source_fraction", "source_scale", "Source-size sensitivity with a fixed target training pool and test set; seed 0. Source selection is evenly spaced in training time.")]:
        rows = []
        for r in report[section]:
            s = r["statistics"]
            if s["runs"]:
                rows.append([escape(r[key]), s["runs"], stat(s, "source_n", 0, False), stat(s, "target_n", 0, False), stat(s, "effective_landmarks", 0, False), stat(s, "rmse"), stat(s, "mae"), stat(s, "fit_seconds", 2)])
        parts.append(table(caption, "tab:chronological_" + label,
                           [first, "Runs", "$n_S$", "$n_T$", "Effective $m$", "RMSE (MW)", "MAE (MW)", "Fit (s)"], rows))
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "review/replied_comment_runs/chronological_v1")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--latex-output", type=Path)
    args = parser.parse_args()
    if not (args.input / "manifest.json").is_file():
        parser.error("No study manifest exists yet; no summary or values have been generated.")
    summary = Summary(args.input)
    report = summary.build()
    json_path = args.json_output or args.input / "study_summary.json"
    tex_path = args.latex_output or args.input / "study_tables.tex"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tex_path.write_text(latex(report))
    print(json.dumps({"json": str(json_path), "latex": str(tex_path), "unique_completed_branches": report["completed_unique_branches"],
                      "missing_branches": len(report["missing_expected_branches"]), "verification_errors": len(report["errors"])}))


if __name__ == "__main__":
    main()
