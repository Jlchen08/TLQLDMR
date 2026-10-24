#!/usr/bin/env python3
"""Frozen follow-up for multi-quantile reliability and paired block inference.

The Q3--Q4 test period was inspected in earlier studies.  This script therefore
produces exploratory follow-up evidence, never independent confirmation.  It
reuses the Q1-locked point-selected configurations without selecting anything
from calibration or test results and retains every prescribed task/method/seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from run_selection_study import (
    METHODS, Study, conformal_quantile, interval_metrics, protocol,
    rolling_expansions,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "review/selection_runs/selection_v1"
DEFAULT_OUT = ROOT / "review/probabilistic_runs/probabilistic_v1"
GRID = [0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]
EXTRA = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]
ALPHAS = [0.50, 0.20, 0.10, 0.05]
COMPARATORS = ["WeightedSVQR", "TargetSVQR", "HistGBR"]


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
    return hi - lo + 2.0 / alpha * (np.maximum(lo-y, 0) +
                                     np.maximum(y-hi, 0))


def study_protocol():
    return {
        "version": "probabilistic_v1",
        "reference": "selection_v1 Q1-locked configurations",
        "test_status": "exploratory follow-up on previously inspected 2020 Q3-Q4",
        "quantile_grid": GRID,
        "central_interval_nominal_coverages": [1-a for a in ALPHAS],
        "within_farm_scope": "all six methods on Farm 6",
        "cross_farm_scope": "full model on target sites 1--6",
        "selection_rule": "no new selection; reuse point-selected settings locked on Q1",
        "calibration": "Q2 split conformal expansion separately at each nominal level",
        "rolling_calibration": "most recent 256 revealed scores; update only after forecast",
        "proper_scores": "pinball at nine quantiles and raw weighted interval score",
        "kernel_seeds": [0, 1, 2],
        "tree_seeds": [0],
        "block_inference": {
            "scope": "full versus WeightedSVQR, TargetSVQR, HistGBR at all six sites",
            "unit": "non-overlapping seven-day calendar blocks",
            "bootstrap_replicates": 10000,
            "sign_flip_replicates": 99999,
            "seed": 20261022,
            "multiplicity": "Holm adjustment over all 18 site-comparator tests",
            "estimand": "seed-averaged squared error; reported effect is Full minus comparator RMSE",
        },
    }


def reference_selection():
    lock = json.loads((REFERENCE / "selection_lock.json").read_text())
    return {(row["task"], row["method"]): row["selected_index"]
            for row in lock["selections"]}


def old_predictions(task, method, seed, split):
    base = REFERENCE / task / method / f"final_seed{seed}"
    file = base / ("predictions.csv" if split == "test" else "calibration.csv")
    frame = pd.read_csv(file)
    if split == "test":
        return frame, {0.025: frame.raw_q025.to_numpy(),
                       0.5: frame.prediction.to_numpy(),
                       0.975: frame.raw_q975.to_numpy()}
    return frame, {0.025: frame.q025.to_numpy(), 0.975: frame.q975.to_numpy()}


def evaluate_quantiles(out, base_cfg):
    study = Study(out, base_cfg)
    chosen = reference_selection()
    scopes = [(base_cfg["tasks"][0], METHODS)] + [
        (task, ["Full"]) for task in base_cfg["tasks"][1:]
    ]
    rows = []
    for task, methods in scopes:
        for method in methods:
            ctx = study.context(task, method)
            index = chosen[(task["name"], method)]
            seeds = [0] if method == "HistGBR" else base_cfg["seeds"]
            for seed in seeds:
                test_frame, test_q = old_predictions(task["name"], method, seed, "test")
                cal_frame, cal_q = old_predictions(task["name"], method, seed, "calibration")
                for tau in EXTRA:
                    folder = out / task["name"] / method / f"seed{seed}" / f"tau{tau}"
                    study.fit(task, method, index, seed, tau, folder)
                    test_q[tau] = study.predict(ctx, folder, "test")
                    cal_q[tau] = study.predict(ctx, folder, "calibration")
                y = test_frame.y_true.to_numpy()
                cy = cal_frame.y_true.to_numpy()
                if not np.array_equal(test_frame.window_index.to_numpy(), ctx["ids"]["test"]):
                    raise ValueError("Test identifiers differ from frozen study")
                matrix = np.column_stack([test_q[t] for t in GRID])
                crossing = np.any(np.diff(matrix, axis=1) < 0, axis=1)
                result = {
                    "task": task["name"], "site": task["site"], "method": method,
                    "seed": seed, "selected_index": index, "n_test": len(y),
                    "quantiles": {}, "crossing_rate": float(crossing.mean()),
                }
                for tau in GRID:
                    result["quantiles"][str(tau)] = {
                        "empirical_cdf": float(np.mean(y <= test_q[tau])),
                        "mean_pinball": float(pinball(y, test_q[tau], tau).mean()),
                    }
                result["mean_pinball_9"] = float(np.mean([
                    pinball(y, test_q[t], t).mean() for t in GRID]))
                weighted_scores = 0.5 * np.abs(y - test_q[0.5])
                result["intervals"] = {}
                for alpha in ALPHAS:
                    lower_tau, upper_tau = alpha/2, 1-alpha/2
                    lo = np.minimum(test_q[lower_tau], test_q[upper_tau])
                    hi = np.maximum(test_q[lower_tau], test_q[upper_tau])
                    clo = np.minimum(cal_q[lower_tau], cal_q[upper_tau])
                    chi = np.maximum(cal_q[lower_tau], cal_q[upper_tau])
                    cal_scores = np.maximum.reduce([clo-cy, cy-chi, np.zeros(len(cy))])
                    test_scores = np.maximum.reduce([lo-y, y-hi, np.zeros(len(y))])
                    expansion, rank = conformal_quantile(cal_scores, alpha)
                    rolling = rolling_expansions(cal_scores, test_scores, 256, alpha)
                    raw = interval_metrics(y, lo, hi, alpha)
                    static = interval_metrics(y, lo-expansion, hi+expansion, alpha)
                    causal = interval_metrics(y, lo-rolling, hi+rolling, alpha)
                    result["intervals"][str(1-alpha)] = {
                        "alpha": alpha, "calibration_rank": rank,
                        "static_expansion": expansion,
                        "raw": raw, "static": static, "rolling": causal,
                    }
                    weighted_scores += alpha/2 * interval_score(y, lo, hi, alpha)
                result["raw_wis"] = float(np.mean(weighted_scores)/(len(ALPHAS)+0.5))
                folder = out / task["name"] / method / f"seed{seed}"
                output = pd.DataFrame({
                    "window_index": test_frame.window_index,
                    "time": test_frame.time,
                    "y_true": y,
                    **{qname(t): test_q[t] for t in GRID},
                })
                output.to_csv(folder / "quantile_predictions.csv", index=False)
                save(folder / "probabilistic_result.json", result)
                rows.append(result)
                print(json.dumps({"phase": "probabilistic", "task": task["name"],
                                  "method": method, "seed": seed,
                                  "wis": result["raw_wis"],
                                  "crossing": result["crossing_rate"]}), flush=True)
            study.context_cache.clear()
            study.feature_cache.clear()
    save(out / "probabilistic_results.json", rows)
    return rows


def aligned_seed_errors(task, method):
    seeds = [0] if method == "HistGBR" else [0, 1, 2]
    frames = [pd.read_csv(REFERENCE/task/method/f"final_seed{s}"/"predictions.csv")
              for s in seeds]
    ids = frames[0].window_index.to_numpy()
    y = frames[0].y_true.to_numpy()
    times = pd.to_datetime(frames[0].time).to_numpy()
    for frame in frames[1:]:
        if not np.array_equal(ids, frame.window_index.to_numpy()):
            raise ValueError("Prediction rows are not aligned")
        if not np.allclose(y, frame.y_true.to_numpy(), rtol=0, atol=0):
            raise ValueError("Test labels are not aligned")
    se = np.mean([(frame.prediction.to_numpy()-y)**2 for frame in frames], axis=0)
    return ids, times, y, se


def holm(pvalues):
    pvalues = np.asarray(pvalues, float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues)-rank)*pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_block_inference(out):
    rng = np.random.default_rng(20261022)
    reports = []
    for site in range(1, 7):
        task = f"cross_site{site}"
        ids, times, y, full = aligned_seed_errors(task, "Full")
        start = times.min().astype("datetime64[D]")
        block = ((times.astype("datetime64[D]")-start).astype(int)//7).astype(int)
        blocks = np.unique(block)
        block_indices = [np.flatnonzero(block == value) for value in blocks]
        for comparator in COMPARATORS:
            ids2, times2, y2, other = aligned_seed_errors(task, comparator)
            if not (np.array_equal(ids, ids2) and np.array_equal(times, times2)
                    and np.allclose(y, y2, rtol=0, atol=0)):
                raise ValueError("Comparator rows are not aligned")
            effect = float(np.sqrt(full.mean()) - np.sqrt(other.mean()))
            boot = np.empty(10000)
            for b in range(len(boot)):
                draw = rng.integers(0, len(block_indices), len(block_indices))
                take = np.concatenate([block_indices[i] for i in draw])
                boot[b] = np.sqrt(full[take].mean()) - np.sqrt(other[take].mean())
            block_loss = np.array([(full[ii]-other[ii]).mean() for ii in block_indices])
            observed = abs(block_loss.mean())
            signs = rng.choice([-1.0, 1.0], size=(99999, len(block_loss)))
            permuted = np.abs((signs*block_loss).mean(axis=1))
            p = float((1+np.sum(permuted >= observed))/(len(permuted)+1))
            reports.append({
                "site": site, "comparator": comparator, "n": len(y),
                "calendar_blocks": len(block_indices),
                "full_minus_comparator_rmse": effect,
                "cluster_bootstrap_95_ci": [float(np.quantile(boot, .025)),
                                             float(np.quantile(boot, .975))],
                "block_sign_flip_p_unadjusted": p,
            })
    adjusted = holm([r["block_sign_flip_p_unadjusted"] for r in reports])
    for row, value in zip(reports, adjusted):
        row["block_sign_flip_p_holm_18"] = float(value)
    save(out / "paired_block_inference.json", {
        "interpretation": (
            "Exploratory inference on a reused test period. Seven-day blocks retain "
            "within-block temporal dependence; they do not prove independent events. "
            "Algorithmic seeds are averaged within observations and are not treated as "
            "independent replicates. Positive RMSE effects favor the comparator."),
        "comparisons": reports,
    })
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = protocol()
    p = study_protocol()
    sources = [Path(__file__), ROOT/"review/run_selection_study.py",
               ROOT/"review/convex_refinement.py", ROOT/"review/selection_data.py"]
    manifest = {
        "status": "frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": p, "base_protocol_sha256": digest(REFERENCE/"protocol.json"),
        "selection_lock_sha256": digest(REFERENCE/"selection_lock.json"),
        "source_hashes": {str(x.relative_to(ROOT)): digest(x) for x in sources},
        "python": platform.python_version(), "numpy": np.__version__,
    }
    if args.output.exists():
        old = json.loads((args.output/"manifest.json").read_text())
        if not args.resume:
            raise FileExistsError("Use --resume for the frozen study")
        for key in ["protocol", "base_protocol_sha256", "selection_lock_sha256",
                    "source_hashes"]:
            if old[key] != manifest[key]:
                raise ValueError(f"Frozen {key} changed")
        manifest = old
    else:
        args.output.mkdir(parents=True)
        save(args.output/"protocol.json", p)
        save(args.output/"manifest.json", manifest)
    if args.freeze_only:
        return
    manifest["status"] = "running"
    save(args.output/"manifest.json", manifest)
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            rows = evaluate_quantiles(args.output, cfg)
            inference = paired_block_inference(args.output)
        manifest.update({"status": "complete", "probabilistic_records": len(rows),
                         "paired_tests": len(inference),
                         "elapsed_seconds": time.perf_counter()-started})
    except Exception as error:
        manifest.update({"status": "failed", "error": repr(error)})
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        save(args.output/"manifest.json", manifest)


if __name__ == "__main__":
    main()
