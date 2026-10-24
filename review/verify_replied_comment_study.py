#!/usr/bin/env python3
"""Independently verify saved chronological-study artifacts, including partial runs.

This verifier never imports the study runner or fits a forecasting model. It
rebuilds raw windows with the separately audited loader, reconstructs the actual
fitting sample selection and preprocessing, and evaluates the saved Nyström
models directly. Passing checks validate the recorded artifacts, not a proof of
data exchangeability, optimality, or absence of all upstream measurement issues.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import io
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review.temporal_data import TemporalProtocol, load_temporal_farm


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, name, *, atol=1e-8, rtol=1e-8):
    a, b = np.asarray(actual), np.asarray(expected)
    check(a.shape == b.shape, f"{name}: shape mismatch {a.shape} != {b.shape}")
    check(np.isfinite(a).all() and np.isfinite(b).all(), f"{name}: nonfinite value")
    check(np.allclose(a, b, atol=atol, rtol=rtol),
          f"{name}: mismatch (max absolute difference {float(np.max(np.abs(a-b))) if a.size else 0})")


def same_ids(actual, expected, name):
    check(np.array_equal(np.asarray(actual), np.asarray(expected)), f"{name}: IDs/order differ")


def spaced(ids, cap):
    return ids if len(ids) <= cap else ids[np.linspace(0, len(ids) - 1, cap).astype(int)]


def classify(data, quantile=98, cutout=25):
    wind = np.percentile(data.training_weather["wind_hub"], quantile)
    low, high = np.percentile(data.training_weather["temperature"], [100 - quantile, quantile])
    w, t = data.target_weather["wind_hub"], data.target_weather["temperature"]
    return (w > wind) | (w > cutout) | (t < low) | (t > high)


def planned_branches(cfg):
    jobs, aliases = [], {}
    for seed in cfg["seeds"]:
        for variant in cfg["components"]:
            jobs.extend(f"components_{variant}_{task}_s{seed}" for task in ("point", "interval"))
        jobs.extend(f"scarcity_n{budget}_s{seed}" for budget in cfg["label_budgets"] if budget != "all")
    for scheme in cfg["weight_schemes"]:
        if scheme != "domain_balanced":
            jobs.extend(f"weights_{scheme}_{task}" for task in ("point", "interval"))
    for q in cfg["threshold_quantiles"]:
        for task in ("point", "interval"):
            name = f"threshold_q{q}_{task}"
            if q == 98:
                aliases[name] = f"components_Full_{task}_s0"
            else:
                jobs.append(name)
    for seed in cfg["seeds"]:
        for m in cfg["landmark_grid"]:
            name = f"landmarks_m{m}_s{seed}"
            if m == cfg["model"]["landmarks"]:
                aliases[name] = f"components_Full_point_s{seed}"
            else:
                jobs.append(name)
    for fraction in cfg["source_fractions"]:
        name = f"scale_fraction{fraction}"
        if fraction == 1:
            aliases["scale_fraction1"] = "components_Full_point_s0"
        else:
            jobs.append(name)
    for site in cfg["cross_farm_sites"]:
        for variant in ("Full", "TargetOnly"):
            jobs.extend(f"crossfarm_site{site}_{variant}_{task}" for task in ("point", "interval"))
    return jobs, aliases


class Verifier:
    def __init__(self, run_dir, manifest):
        self.run_dir, self.manifest = run_dir, manifest
        self.cfg = manifest["protocol"]
        self.data, self.capacities, self.data_checks = {}, {}, {}

    def farm(self, site):
        if site in self.data:
            return self.data[site]
        matches = [ROOT / path for path in self.manifest["data_hashes"]
                   if re.search(rf"Wind farm site {site} \(", Path(path).name)]
        check(len(matches) == 1, f"Farm {site}: expected one source workbook")
        path = matches[0]
        with contextlib.redirect_stdout(io.StringIO()):
            data = load_temporal_farm(path, TemporalProtocol(**self.cfg["temporal"]))
        self.capacities[site] = float(re.search(r"(\d+)MW", path.name).group(1))
        audit_path = self.run_dir / "data" / f"farm{site}_audit.json"
        audit = json.loads(audit_path.read_text())
        check(audit["provenance"] == data.provenance, f"Farm {site}: provenance differs")
        check(audit["statistics"] == data.statistics, f"Farm {site}: exclusion statistics differ")
        check(audit["preprocessing"] == data.preprocessing, f"Farm {site}: threshold/protocol metadata differ")
        with np.load(self.run_dir / "data" / f"farm{site}_ids.npz", allow_pickle=False) as cached:
            for key, expected in (("times", data.timestamps), ("split", data.split),
                                  ("raw_target_row", data.target_row_indices), ("input_rows", data.input_row_indices)):
                same_ids(cached[key], expected, f"Farm {site}: cached {key}")
        check(not data.preprocessing["label_scaler_fitted"], f"Farm {site}: loader unexpectedly fitted label scaler")
        self.data[site] = data
        self.data_checks[str(site)] = {"status": "passed", "windows": len(data.y_raw),
                                       "sha256": data.provenance["data_sha256"]}
        return data

    def expected_selection(self, name, result):
        cfg = self.cfg
        site = int(cfg["within_site"])
        q, cutoff, seed, variant, task, scheme = 98, 25, 0, "Full", "point", "domain_balanced"
        m, fraction, budget, cross = cfg["model"]["landmarks"], 1.0, None, False
        if match := re.fullmatch(r"components_(\w+)_(point|interval)_s(\d+)", name):
            variant, task, seed = match[1], match[2], int(match[3])
        elif match := re.fullmatch(r"scarcity_n(\d+)_s(\d+)", name):
            budget, seed = int(match[1]), int(match[2])
        elif match := re.fullmatch(r"weights_(\w+)_(point|interval)", name):
            scheme, task = match[1], match[2]
        elif match := re.fullmatch(r"threshold_q([\d.]+)_cutout([\d.]+)_(point|interval)", name):
            q, cutoff, task = float(match[1]), float(match[2]), match[3]
        elif match := re.fullmatch(r"threshold_q([\d.]+)_(point|interval)", name):
            q, task = float(match[1]), match[2]
        elif match := re.fullmatch(r"landmarks_m(\d+)_s(\d+)", name):
            m, seed = int(match[1]), int(match[2])
        elif match := re.fullmatch(r"scale_fraction([\d.]+)", name):
            fraction = float(match[1])
        elif match := re.fullmatch(r"crossfarm_site(\d+)_(Full|TargetOnly)_(point|interval)", name):
            site, variant, task, cross = int(match[1]), match[2], match[3], True
        else:
            raise AssertionError(f"Unrecognized predeclared branch: {name}")
        for key, expected in (("site", site), ("seed", seed), ("variant", variant), ("task", task), ("scheme", scheme)):
            check(result[key] == expected, f"{name}: result {key} disagrees with branch name")
        d = self.farm(site)
        domain = classify(d, q, cutoff)
        source = np.flatnonzero((d.split == "train") & ~domain)
        target = np.flatnonzero((d.split == "train") & domain)
        test = np.flatnonzero((d.split == "test") & domain)
        calibration = np.flatnonzero((d.split == "calibration") & domain)
        if budget is not None:
            target = np.random.default_rng(seed).permutation(target)[:budget]
        if fraction != 1:
            source = spaced(source, max(1, int(len(source) * fraction)))
        sources = [(site, source)]
        if cross:
            target = np.random.default_rng(0).permutation(target)[:cfg["cross_farm_target_labels"]]
            sources = []
            for other in range(1, 7):
                if other == site:
                    continue
                od = self.farm(other)
                indices = np.flatnonzero((od.split == "train") & ~classify(od))
                sources.append((other, spaced(indices, cfg["cross_farm_source_cap_per_site"])))
        reference_n = sum(len(ids) for _, ids in sources) + len(target)
        if variant == "TargetOnly":
            sources = []
        return d, sources, target, test, calibration, m, reference_n

    def verify_branch(self, name):
        folder = self.run_dir / name
        result = json.loads((folder / "result.json").read_text())
        check(result["name"] == name, "Result branch name differs from directory")
        d, sources, target, test, calibration, landmarks, reference_n = self.expected_selection(name, result)
        site, task, cfg = result["site"], result["task"], self.cfg
        cap = self.capacities[site]
        source_sites = np.concatenate([np.full(len(ids), s) for s, ids in sources]) if sources else np.empty(0, int)
        source_rows = np.concatenate([self.farm(s).target_row_indices[ids] for s, ids in sources]) if sources else np.empty(0, int)
        with np.load(folder / "fit_ids.npz", allow_pickle=False) as saved:
            expected = {"source_site": source_sites, "source_raw_row": source_rows,
                        "target_raw_row": d.target_row_indices[target], "target_window_index": target,
                        "test_window_index": test, "calibration_window_index": calibration}
            for key, values in expected.items():
                same_ids(saved[key], values, key)
        check(len(target) == len(np.unique(target)), "Duplicated target fitting IDs")
        for role, ids, split in (("target_fit", target, "train"), ("test", test, "test"),
                                 ("calibration", calibration, "calibration")):
            check(np.all(d.split[ids] == split), f"{role}: incorrect temporal period")
        check(not np.intersect1d(target, test).size and not np.intersect1d(target, calibration).size
              and not np.intersect1d(test, calibration).size, "Fit/calibration/test IDs overlap")
        for s, ids in sources:
            check(np.all(self.farm(s).split[ids] == "train"), "Source observations outside training period")
            if s == site:
                check(not np.intersect1d(ids, target).size, "Source and target fitting labels overlap")
        xs = [self.farm(s).X[ids] for s, ids in sources]
        ys = [self.farm(s).y_raw[ids] / self.capacities[s] for s, ids in sources]
        fit_x = np.concatenate(xs + [d.X[target]])
        fit_y = np.concatenate(ys + [d.y_raw[target] / cap])
        finite = np.isfinite(fit_x)
        count = finite.sum(axis=0)
        mean = np.divide(np.where(finite, fit_x, 0).sum(axis=0, dtype=np.float64), count,
                         out=np.zeros(fit_x.shape[1]), where=count > 0)
        scale = np.where(finite, fit_x, mean).std(axis=0)
        scale[scale == 0] = 1
        label_mean, label_scale = float(fit_y.mean()), float(fit_y.std()) or 1.0
        state = json.loads((folder / "preprocessing.json").read_text())
        for key, expected in (("mean", mean), ("scale", scale), ("label_mean", label_mean), ("label_scale", label_scale)):
            close(state[key], expected, f"actual-fit preprocessing {key}", atol=1e-12, rtol=1e-12)
        ns, nt = len(source_rows), len(target)
        check((state["fit_source_count"], state["fit_target_count"]) == (ns, nt), "Preprocessing fit counts differ")
        check((result["source_n"], result["target_n"]) == (ns, nt), "Result fit counts differ")
        def transform(raw):
            return ((np.where(np.isfinite(raw), raw, mean) - mean) / scale).astype(np.float32)

        test_x = transform(d.X[test])
        cal_x = transform(d.X[calibration]) if task == "interval" else None
        table = pd.read_csv(folder / "predictions.csv")
        self.check_table_ids(table, d, test, "test")
        taus = [cfg["point_tau"]] if task == "point" else cfg["interval_taus"]
        check(len(result["fits"]) == len(taus), "Saved quantile model count differs")
        predicted, cal_predicted, model_checks = [], [], []
        batches_per_epoch = math.ceil((ns + nt) / cfg["model"]["batch_size"])
        reference_updates = cfg["model"]["epochs"] * math.ceil(reference_n / cfg["model"]["batch_size"])
        expected_epochs = math.ceil(reference_updates / batches_per_epoch) if result["variant"] == "TargetOnly" else cfg["model"]["epochs"]
        for tau, fit in zip(taus, result["fits"]):
            opts = fit["options"]
            check(opts["tau"] == tau and opts["n_components"] == landmarks, "Quantile/landmark options differ")
            expected_opts = {"lambda1": 0.0 if result["variant"] == "NoVariance" else cfg["model"]["lambda1"],
                             "lambda2": 0.0 if result["variant"] in ("NoMMD", "TargetOnly") else cfg["model"]["lambda2"],
                             "kernel_gamma": cfg["model"]["gamma"], "epochs": expected_epochs,
                             "learning_rate": cfg["model"]["learning_rate"], "batch_size": cfg["model"]["batch_size"],
                             "variance_mode": "centered_residual", "asym_scale": 0.0}
            for key, expected in expected_opts.items():
                check(opts[key] == expected, f"Model option {key} differs from frozen protocol")
            check(fit["adam_updates"] == expected_epochs * batches_per_epoch, "Actual Adam update count differs")
            check(fit["reference_full_updates"] == reference_updates, "Full-model reference update count differs")
            if result["variant"] == "TargetOnly":
                check(reference_updates <= fit["adam_updates"] < reference_updates + batches_per_epoch,
                      "Target-only update budget is not the least whole-epoch match")
            total = cfg["model"]["total_loss_weight"]
            if ns == 0:
                cs, ct = 0.0, total / nt
            elif result["scheme"] == "domain_balanced":
                cs, ct = total / (2 * ns), total / (2 * nt)
            elif result["scheme"] == "sample_equal":
                cs = ct = total / (ns + nt)
            else:
                ratio = (3000 if task == "point" else 8000) if result["scheme"] == "historical_ratio" else 10 * ns / nt
                cs, ct = total / (ns + ratio * nt), ratio * total / (ns + ratio * nt)
            close([opts["C_S"], opts["C_T"]], [cs, ct], "Loss coefficients")
            model_path = folder / f"model_tau_{tau}.npz"
            with np.load(model_path, allow_pickle=False) as model:
                w, b = model["w"], float(model["b"])
                anchors, normalization = model["landmarks"], model["normalization"]
                check(anchors.shape == (len(w), d.X.shape[1]), "Landmark shape differs")
                check(normalization.shape == (len(w), len(w)), "Normalization shape differs")
                check(len(w) == fit["effective_landmarks"], "Effective landmark count differs")
                check(len(w) == min(landmarks, 2000, ns if ns else nt), "Effective landmark count differs from fitting pool")
                check(all(np.isfinite(a).all() for a in (w, anchors, normalization)) and np.isfinite(b), "Nonfinite saved model")
                kmm = np.exp(-opts["kernel_gamma"] * cdist(anchors, anchors, "sqeuclidean"))
                identity = normalization @ (kmm + 1e-6 * np.eye(len(w))) @ normalization
                close(identity, np.eye(len(w)), "Stabilized Nyström normalization", atol=2e-5, rtol=2e-5)
                def predict(x):
                    output = []
                    for start in range(0, len(x), 1024):
                        kernel = np.exp(-opts["kernel_gamma"] * cdist(x[start:start+1024], anchors, "sqeuclidean"))
                        scaled = kernel @ (normalization @ w) + b
                        output.append((scaled * label_scale + label_mean) * cap)
                    return np.concatenate(output)
                predicted.append(predict(test_x))
                if task == "interval":
                    cal_predicted.append(predict(cal_x))
                model_checks.append({"tau": tau, "sha256": sha256(model_path),
                                     "normalization_max_identity_error": float(np.max(np.abs(identity - np.eye(len(w)))))})
        y = d.y_raw[test]
        calibration_result = None
        if task == "point":
            close(table["prediction"], predicted[0], "Saved-model point predictions", atol=2e-7)
            error = y - table["prediction"].to_numpy()
            total_variation = np.sum((y - y.mean()) ** 2)
            metrics = {"n": len(y), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
                       "r2": float(1 - np.sum(error ** 2) / total_variation) if total_variation > 0 else None}
            check(result["calibration"] is None, "Point branch unexpectedly has interval calibration")
        else:
            cal_table = pd.read_csv(folder / "calibration.csv")
            self.check_table_ids(cal_table, d, calibration, "calibration")
            for col, pred in zip(("raw_low", "raw_high"), predicted):
                close(table[col], pred, f"Saved-model test {col}", atol=2e-7)
            for col, pred in zip(("raw_low", "raw_high"), cal_predicted):
                close(cal_table[col], pred, f"Saved-model calibration {col}", atol=2e-7)
            raw_lo, raw_hi = np.minimum(*cal_predicted), np.maximum(*cal_predicted)
            cy = d.y_raw[calibration]
            scores = np.maximum(0, np.maximum(raw_lo - cy, cy - raw_hi))
            close(cal_table["score"], scores, "Calibration scores", atol=2e-7)
            rank = math.ceil((len(cy) + 1) * (1 - cfg["alpha"]))
            check(rank <= len(cy), "Invalid finite-sample calibration rank")
            radius = float(np.partition(scores, rank - 1)[rank - 1])
            recorded = result["calibration"]
            check(recorded["n"] == len(cy) and recorded["rank"] == rank, "Calibration count/rank differs")
            close([recorded["q_hat"], recorded["alpha"]], [radius, cfg["alpha"]], "Calibration radius/alpha", atol=2e-7)
            lo, hi = np.minimum(*predicted), np.maximum(*predicted)
            for col, expected in (("ordered_lower", lo), ("ordered_upper", hi), ("lower", lo - radius), ("upper", hi + radius)):
                close(table[col], expected, f"Interval {col}", atol=5e-7)
            lower, upper = table["lower"].to_numpy(), table["upper"].to_numpy()
            width = upper - lower
            check((width >= 0).all(), "Negative interval widths")
            span = np.ptp(y)
            metrics = {"n": len(y), "picp": float(np.mean((y >= lower) & (y <= upper))),
                       "mpiw": float(width.mean()), "pinaw": float(width.mean() / span) if span > 0 else None,
                       "winkler": float(np.mean(width + 2 / cfg["alpha"] * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))))}
            calibration_result = {"n": len(cy), "rank": rank, "q_hat": radius}
        for key, expected in metrics.items():
            if expected is None:
                check(result["metrics"][key] is None, f"Metric {key} should be null")
            else:
                close(result["metrics"][key], expected, f"CSV-recalculated metric {key}", atol=1e-8)
        return {"name": name, "status": "passed", "site": site, "task": task,
                "fit_source_count": ns, "fit_target_count": nt, "metrics_recalculated": metrics,
                "calibration_recalculated": calibration_result, "models": model_checks,
                "prediction_csv_sha256": sha256(folder / "predictions.csv"),
                "checks": ["predeclared sample selection", "temporal roles and ID separation", "actual-fit preprocessing",
                           "saved-model predictions", "source hashes and rebuilt windows", "calibration rank and test-independent radius" if task == "interval" else "point metrics"]}

    @staticmethod
    def check_table_ids(table, data, indices, role):
        same_ids(table["window_index"], indices, f"{role} CSV window IDs")
        same_ids(table["raw_target_row"], data.target_row_indices[indices], f"{role} CSV original row IDs")
        same_ids(pd.to_datetime(table["time"]).to_numpy(dtype="datetime64[ns]"), data.timestamps[indices], f"{role} CSV timestamps")
        close(table["y_true"], data.y_raw[indices], f"{role} labels match original workbook", atol=1e-10)


def verify_run(run_dir, branch_patterns=None, require_complete=False):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir.resolve()),
              "manifest_status_at_start": manifest["status"], "scope": "all declared branches" if not branch_patterns else branch_patterns,
              "global_checks": [], "branches": [], "aliases": [], "pending": []}
    try:
        for category in ("source_hashes", "data_hashes"):
            for relative, expected in manifest[category].items():
                check(sha256(ROOT / relative) == expected, f"Hash mismatch: {relative}")
            report["global_checks"].append({"check": category, "status": "passed", "files": len(manifest[category])})
        # Use the frozen study protocol, never TemporalProtocol's defaults.
        TemporalProtocol(**manifest["protocol"]["temporal"]).boundaries()
        report["temporal_protocol"] = manifest["protocol"]["temporal"]
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        return report
    verifier = Verifier(run_dir, manifest)
    jobs, aliases = planned_branches(manifest["protocol"])
    # Conditional cut-out branches exist only if their event membership changes.
    # Also inspect any unexpected completed directory instead of overlooking it.
    observed = sorted(path.parent.name for path in run_dir.glob("*/result.json"))
    jobs.extend(name for name in observed if name not in jobs)
    selected = lambda name: not branch_patterns or any(fnmatch.fnmatchcase(name, pattern) for pattern in branch_patterns)
    for name in jobs:
        if not selected(name):
            continue
        if not (run_dir / name / "result.json").exists():
            report["pending"].append(name)
            continue
        try:
            report["branches"].append(verifier.verify_branch(name))
        except json.JSONDecodeError as error:
            if manifest["status"] == "running":
                report["pending"].append(name)
            else:
                report["branches"].append({"name": name, "status": "failed", "error": str(error)})
        except Exception as error:
            report["branches"].append({"name": name, "status": "failed", "error": f"{type(error).__name__}: {error}"})
    for name, expected_reference in aliases.items():
        if not selected(name):
            continue
        path = run_dir / name / "alias.json"
        if not path.exists():
            report["pending"].append(name)
            continue
        try:
            alias = json.loads(path.read_text())
            check(alias["reference"] == expected_reference, "Alias reference differs from predeclared identity")
            check((run_dir / expected_reference / "result.json").exists(), "Alias refers to an unfinished branch")
            report["aliases"].append({"name": name, "reference": expected_reference, "status": "passed"})
        except Exception as error:
            report["aliases"].append({"name": name, "status": "failed", "error": f"{type(error).__name__}: {error}"})
    report["rebuilt_farms"] = verifier.data_checks
    if not branch_patterns and (run_dir / "threshold_definitions.json").exists():
        try:
            definitions = json.loads((run_dir / "threshold_definitions.json").read_text())
            cfg = manifest["protocol"]
            data = verifier.farm(cfg["within_site"])
            check(len(definitions) == len(cfg["threshold_quantiles"]) * len(cfg["cutout_sensitivity_ms"]),
                  "Threshold-definition row count differs")
            for row in definitions:
                q, cutout = row["q"], row["cutout_ms"]
                alternative, baseline = classify(data, q, cutout), classify(data, q)
                identical = bool(np.array_equal(alternative, baseline))
                check(row["identical_domain_mask_to_25"] == identical, "Cut-out mask equivalence differs")
                expected = {"n_train_source": np.sum((data.split == "train") & ~alternative),
                            "n_train_target": np.sum((data.split == "train") & alternative),
                            "n_cal": np.sum((data.split == "calibration") & alternative),
                            "n_test": np.sum((data.split == "test") & alternative)}
                for key, count in expected.items():
                    check(row[key] == int(count), f"Cut-out {q}/{cutout} {key} differs")
                if not identical:
                    for task in ("point", "interval"):
                        branch = f"threshold_q{q}_cutout{cutout}_{task}"
                        check((run_dir / branch / "result.json").exists(), f"Missing required changed-cutout branch {branch}")
            report["global_checks"].append({"check": "threshold definitions and conditional cut-out branches", "status": "passed"})
        except Exception as error:
            report["global_checks"].append({"check": "threshold definitions", "status": "failed", "error": str(error)})
    summary_path = run_dir / "summary.csv"
    if summary_path.exists():
        try:
            summary = pd.read_csv(summary_path).set_index("name")
            check(summary.index.is_unique, "Summary contains duplicate branch names")
            checked_rows = 0
            for branch in report["branches"]:
                if branch["status"] != "passed":
                    continue
                name = branch["name"]
                if name not in summary.index and manifest["status"] == "running":
                    continue  # result.json is written just before the next summary snapshot.
                check(name in summary.index, f"Summary missing completed branch {name}")
                for key, expected in branch["metrics_recalculated"].items():
                    actual = summary.loc[name, key]
                    if expected is None:
                        check(pd.isna(actual), f"Summary {name}/{key} should be missing")
                    else:
                        close(actual, expected, f"Summary {name}/{key}")
                checked_rows += 1
            report["global_checks"].append({"check": "summary CSV metrics", "status": "passed", "rows_checked": checked_rows})
        except Exception as error:
            report["global_checks"].append({"check": "summary CSV metrics", "status": "failed", "error": str(error)})
    if manifest["status"] == "complete" and not branch_patterns:
        for artifact in ("summary.csv", "threshold_definitions.json"):
            if not (run_dir / artifact).exists():
                report["global_checks"].append({"check": artifact, "status": "failed", "error": "Completed study is missing required aggregate artifact"})
    report["passed_branches"] = sum(b["status"] == "passed" for b in report["branches"])
    failed = any(b["status"] == "failed" for b in report["branches"] + report["aliases"] + report["global_checks"])
    if require_complete and (report["pending"] or manifest["status"] != "complete"):
        failed = True
        report["completion_error"] = "Required complete run is not complete."
    if not report["branches"] and not report["aliases"]:
        report["status"] = "pending" if not failed else "failed"
    else:
        report["status"] = "failed" if failed else ("passed_partial" if report["pending"] or manifest["status"] != "complete" else "passed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "review/replied_comment_runs/chronological_v1")
    parser.add_argument("--branch", action="append", help="Optional branch name/glob; repeat to verify several subsets.")
    parser.add_argument("--output", type=Path, help="Optional JSON report file; stdout always receives the report.")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = verify_run(args.run_dir, args.branch, args.require_complete)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
