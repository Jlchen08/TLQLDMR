#!/usr/bin/env python3
"""Frozen equal-budget multi-quantile selection on the Farm 6 task.

All six methods receive the same 16 candidate configurations.  Each candidate
is scored on 2020 Q1 by its mean pinball loss over a fixed nine-quantile grid.
Selections are locked before any Q2 calibration or Q3--Q4 test prediction is
made.  The test period has been inspected in earlier studies, so all resulting
comparisons are exploratory follow-up evidence rather than confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_selection_study import METHODS, Study, conformal_quantile, protocol, rolling_expansions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "review/probabilistic_runs/probabilistic_selection_v2"
GRID = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
ALPHAS = [0.50, 0.20, 0.10, 0.05]
INTERVALS = [(0.50, 3, 5), (0.20, 2, 6), (0.10, 1, 7), (0.05, 0, 8)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")


def qname(tau: float) -> str:
    return "q" + str(int(round(1000 * tau))).zfill(3)


def pinball(y, q, tau):
    residual = np.asarray(y) - np.asarray(q)
    return np.maximum(tau * residual, (tau - 1.0) * residual)


def interval_score(y, lo, hi, alpha):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    if np.any(lo > hi):
        raise ValueError("Interval endpoints are not ordered")
    return hi - lo + 2.0 / alpha * (np.maximum(lo-y, 0) +
                                     np.maximum(y-hi, 0))


def interval_metrics(y, lo, hi, alpha):
    score = interval_score(y, lo, hi, alpha)
    return {
        "n": int(len(y)),
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "mpiw": float(np.mean(hi-lo)),
        "winkler": float(np.mean(score)),
    }


def wis(y, quantile_matrix):
    """WIS for four central intervals after row-wise monotone rearrangement."""
    score = 0.5 * np.abs(np.asarray(y) - quantile_matrix[:, 4])
    for alpha, lower_index, upper_index in INTERVALS:
        lo = quantile_matrix[:, lower_index]
        hi = quantile_matrix[:, upper_index]
        score += alpha / 2.0 * interval_score(y, lo, hi, alpha)
    return float(np.mean(score) / (len(ALPHAS) + 0.5))


def study_protocol():
    return {
        "version": "probabilistic_selection_v2",
        "test_status": "exploratory follow-up on previously inspected 2020 Q3-Q4",
        "scope": "within-Farm-6 comparison of all six methods",
        "quantile_grid": GRID,
        "central_interval_nominal_coverages": [1-a for a in ALPHAS],
        "candidate_budget": 16,
        "selection_period": "2020 Q1",
        "selection_rule": (
            "minimum unweighted mean Q1 pinball loss across all nine quantiles; "
            "ties use candidate index; no convergence filtering"
        ),
        "selection_lock": "all six selections locked before Q2 or Q3-Q4 evaluation",
        "refit_rule": "models retain 2019 fitting rows only; Q1 labels are never fitted",
        "calibration": (
            "row-wise monotone rearrangement followed by separate Q2 split-conformal "
            "expansion at 50%, 80%, 90%, and 95% nominal coverage"
        ),
        "rolling_calibration": "most recent 256 revealed scores; update only after forecast",
        "reported_scores": (
            "raw and rearranged nine-quantile mean pinball loss, rearranged WIS, "
            "raw crossing rate, reliability, PICP, MPIW, and Winkler score"
        ),
        "kernel_seeds": [0, 1, 2],
        "tree_seeds": [0],
        "rearrangement": (
            "sort the nine independently fitted quantiles within each forecast row; "
            "retain and report the unrearranged predictions and crossing rate"
        ),
    }


def candidate_folder(out, method, index, tau):
    return out / "within_site6" / method / "candidates" / f"c{index:02d}" / qname(tau)


def tune(study: Study, out: Path, cfg):
    task = cfg["tasks"][0]
    selections = []
    for method in METHODS:
        ctx = study.context(task, method)
        ids = ctx["ids"]["validation"]
        y = ctx["data"].y_raw[ids]
        candidate_rows = []
        for index in range(cfg["candidate_count_per_method"]):
            predictions = []
            diagnostics = []
            for tau in GRID:
                folder = candidate_folder(out, method, index, tau)
                fit = study.fit(task, method, index, 0, tau, folder)
                predictions.append(study.predict(ctx, folder, "validation"))
                diagnostics.append({"tau": tau, "fit": fit["diagnostics"]})
            matrix = np.column_stack(predictions)
            losses = [float(pinball(y, matrix[:, j], tau).mean())
                      for j, tau in enumerate(GRID)]
            row = {
                "candidate_index": index,
                "candidate": study.candidate(task, method, index),
                "validation_mean_pinball_9": float(np.mean(losses)),
                "validation_pinball_by_quantile": {
                    str(t): value for t, value in zip(GRID, losses)
                },
                "validation_crossing_rate": float(
                    np.mean(np.any(np.diff(matrix, axis=1) < 0, axis=1))
                ),
                "diagnostics": diagnostics,
            }
            pd.DataFrame({
                "window_index": ids,
                "y_true": y,
                **{qname(t): matrix[:, j] for j, t in enumerate(GRID)},
            }).to_csv(out / "within_site6" / method / "candidates" /
                      f"c{index:02d}" / "validation_quantiles.csv", index=False)
            candidate_rows.append(row)
            print(json.dumps({"phase": "probability_selection", "method": method,
                              "candidate": index,
                              "q1_pinball": row["validation_mean_pinball_9"]}),
                  flush=True)
        best = min(candidate_rows,
                   key=lambda x: (x["validation_mean_pinball_9"],
                                  x["candidate_index"]))
        selection = {
            "task": task["name"], "site": task["site"], "method": method,
            "selected_index": best["candidate_index"],
            "selected_validation_mean_pinball_9": best["validation_mean_pinball_9"],
            "candidates": candidate_rows,
        }
        path = out / "within_site6" / method / "selection.json"
        save(path, selection)
        selections.append(selection)
        print(json.dumps({"phase": "probability_selection_locked", "method": method,
                          "selected": best["candidate_index"],
                          "q1_pinball": best["validation_mean_pinball_9"]}), flush=True)
        study.context_cache.clear()
        study.feature_cache.clear()

    files = {str((out / "within_site6" / row["method"] /
                  "selection.json").relative_to(out)):
             digest(out / "within_site6" / row["method"] / "selection.json")
             for row in selections}
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest(out / "protocol.json"),
        "selections": selections,
        "selection_files": files,
    }
    save(out / "selection_lock.json", lock)
    return selections


def validate_lock(out: Path):
    lock = json.loads((out / "selection_lock.json").read_text())
    if lock["protocol_sha256"] != digest(out / "protocol.json"):
        raise ValueError("Selection lock protocol mismatch")
    expected = {f"within_site6/{method}/selection.json" for method in METHODS}
    if set(lock["selection_files"]) != expected:
        raise ValueError("Selection lock list is incomplete")
    for rel, expected_hash in lock["selection_files"].items():
        if digest(out / rel) != expected_hash:
            raise ValueError(f"Locked selection changed: {rel}")
    rows = lock["selections"]
    if len(rows) != len(METHODS) or {r["method"] for r in rows} != set(METHODS):
        raise ValueError("Duplicate or missing embedded selections")
    return rows


def evaluate(study: Study, out: Path, cfg, selections):
    task = cfg["tasks"][0]
    selected = {row["method"]: row["selected_index"] for row in selections}
    results = []
    for method in METHODS:
        ctx = study.context(task, method)
        index = selected[method]
        seeds = [0] if method == "HistGBR" else cfg["seeds"]
        for seed in seeds:
            test_predictions = []
            cal_predictions = []
            fit_records = []
            for tau in GRID:
                if seed == 0:
                    folder = candidate_folder(out, method, index, tau)
                else:
                    folder = (out / "within_site6" / method / f"final_seed{seed}" /
                              qname(tau))
                fit = study.fit(task, method, index, seed, tau, folder)
                fit_records.append({"path": str(folder.relative_to(out)), **fit})
                test_predictions.append(study.predict(ctx, folder, "test"))
                cal_predictions.append(study.predict(ctx, folder, "calibration"))

            raw = np.column_stack(test_predictions)
            cal_raw = np.column_stack(cal_predictions)
            rearranged = np.sort(raw, axis=1)
            cal_rearranged = np.sort(cal_raw, axis=1)
            ti = ctx["ids"]["test"]
            ci = ctx["ids"]["calibration"]
            y = ctx["data"].y_raw[ti]
            cy = ctx["data"].y_raw[ci]
            raw_loss = [float(pinball(y, raw[:, j], tau).mean())
                        for j, tau in enumerate(GRID)]
            rearranged_loss = [float(pinball(y, rearranged[:, j], tau).mean())
                               for j, tau in enumerate(GRID)]
            result = {
                "task": task["name"], "site": task["site"], "method": method,
                "seed": seed, "selected_index": index, "n_test": int(len(y)),
                "raw_crossing_rate": float(
                    np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1))
                ),
                "raw_mean_pinball_9": float(np.mean(raw_loss)),
                "rearranged_mean_pinball_9": float(np.mean(rearranged_loss)),
                "rearranged_wis": wis(y, rearranged),
                "raw_quantiles": {}, "rearranged_quantiles": {},
                "intervals": {}, "fits": fit_records,
            }
            for j, tau in enumerate(GRID):
                result["raw_quantiles"][str(tau)] = {
                    "empirical_cdf": float(np.mean(y <= raw[:, j])),
                    "mean_pinball": raw_loss[j],
                }
                result["rearranged_quantiles"][str(tau)] = {
                    "empirical_cdf": float(np.mean(y <= rearranged[:, j])),
                    "mean_pinball": rearranged_loss[j],
                }

            for alpha, lower_index, upper_index in INTERVALS:
                nominal = 1-alpha
                lo = rearranged[:, lower_index]
                hi = rearranged[:, upper_index]
                clo = cal_rearranged[:, lower_index]
                chi = cal_rearranged[:, upper_index]
                cal_scores = np.maximum.reduce([clo-cy, cy-chi, np.zeros(len(cy))])
                test_scores = np.maximum.reduce([lo-y, y-hi, np.zeros(len(y))])
                expansion, rank = conformal_quantile(cal_scores, alpha)
                rolling = rolling_expansions(cal_scores, test_scores, 256, alpha)
                result["intervals"][str(nominal)] = {
                    "alpha": alpha, "calibration_rank": rank,
                    "static_expansion": expansion,
                    "raw_rearranged": interval_metrics(y, lo, hi, alpha),
                    "static": interval_metrics(y, lo-expansion, hi+expansion, alpha),
                    "rolling": interval_metrics(y, lo-rolling, hi+rolling, alpha),
                }

            base = out / "within_site6" / method / f"final_seed{seed}"
            base.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "window_index": ti,
                "time": ctx["data"].timestamps[ti].astype(str),
                "y_true": y,
                **{f"raw_{qname(t)}": raw[:, j] for j, t in enumerate(GRID)},
                **{f"rearranged_{qname(t)}": rearranged[:, j]
                   for j, t in enumerate(GRID)},
            }).to_csv(base / "test_quantiles.csv", index=False)
            pd.DataFrame({
                "window_index": ci,
                "time": ctx["data"].timestamps[ci].astype(str),
                "y_true": cy,
                **{f"raw_{qname(t)}": cal_raw[:, j] for j, t in enumerate(GRID)},
                **{f"rearranged_{qname(t)}": cal_rearranged[:, j]
                   for j, t in enumerate(GRID)},
            }).to_csv(base / "calibration_quantiles.csv", index=False)
            save(base / "result.json", result)
            results.append(result)
            print(json.dumps({"phase": "probability_evaluation", "method": method,
                              "seed": seed, "selected": index,
                              "pinball": result["raw_mean_pinball_9"],
                              "wis": result["rearranged_wis"],
                              "crossing": result["raw_crossing_rate"]}), flush=True)
        study.context_cache.clear()
        study.feature_cache.clear()
    save(out / "results.json", results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = protocol()
    cfg["tasks"] = [cfg["tasks"][0]]
    frozen = study_protocol()
    sources = [Path(__file__), ROOT / "review/run_selection_study.py",
               ROOT / "review/convex_refinement.py",
               ROOT / "review/selection_data.py"]
    manifest = {
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": frozen,
        "base_candidate_protocol_sha256": digest(
            ROOT / "review/selection_runs/selection_v1/protocol.json"),
        "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "python": platform.python_version(), "numpy": np.__version__,
    }

    if args.output.exists():
        old = json.loads((args.output / "manifest.json").read_text())
        if not args.resume:
            raise FileExistsError("Use --resume for the frozen study")
        for key in ["protocol", "base_candidate_protocol_sha256", "source_hashes"]:
            if old[key] != manifest[key]:
                raise ValueError(f"Frozen {key} changed")
        manifest = old
    else:
        args.output.mkdir(parents=True)
        save(args.output / "protocol.json", frozen)
        save(args.output / "manifest.json", manifest)
    if args.freeze_only:
        return

    manifest["status"] = "running"
    save(args.output / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            study = Study(args.output, cfg)
            if (args.output / "selection_lock.json").exists():
                selections = validate_lock(args.output)
            else:
                selections = tune(study, args.output, cfg)
            results = evaluate(study, args.output, cfg, selections)
        manifest["status"] = "complete"
        manifest["selection_records"] = len(selections)
        manifest["probabilistic_records"] = len(results)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
