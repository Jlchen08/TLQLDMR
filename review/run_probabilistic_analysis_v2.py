#!/usr/bin/env python3
"""Pre-specified analysis of the probability-selected Farm 6 experiment.

Freeze this source and protocol before the Q3--Q4 probability evaluation is
available.  The first analysis run records hashes of the completed experiment
outputs, then creates aggregate metrics, paired block inference, manuscript
tables, and vector figures without changing any fitted prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "review/probabilistic_runs/probabilistic_selection_v2"
DEFAULT_OUT = ROOT / "review/probabilistic_runs/probabilistic_analysis_v2"
METHODS = ["Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR",
           "TargetSVQR", "HistGBR"]
GRID = np.array([0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975])
INTERVALS = [(0.50, 3, 5), (0.20, 2, 6), (0.10, 1, 7), (0.05, 0, 8)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")


def protocol():
    return {
        "version": "probabilistic_analysis_v2",
        "frozen_before_test_evaluation": True,
        "aggregation": "kernel mean over seeds 0,1,2; deterministic HistGBR seed 0",
        "reliability": (
            "empirical P(y<=q_tau), mean and maximum absolute deviation from tau, "
            "reported for raw and row-wise rearranged quantiles"
        ),
        "crossing": (
            "row rate with any adjacent decrease; mean positive adjacent decrease "
            "over all row-pairs; mean maximum decrease conditional on a crossed row"
        ),
        "scores": (
            "raw nine-quantile mean pinball; rearranged mean pinball and standard "
            "four-interval WIS; verify WIS=2*mean pinball after rearrangement"
        ),
        "interval_summary": "raw-rearranged, Q2-static, and causal-rolling PICP/MPIW/Winkler",
        "paired_inference": {
            "scope": "Full versus each of five alternatives on Farm 6",
            "estimand": "mean per-forecast rearranged WIS difference; Full minus comparator",
            "seed_handling": "average algorithmic seeds within forecast",
            "unit": "non-overlapping seven-day calendar blocks",
            "bootstrap_replicates": 10000,
            "sign_flip_replicates": 99999,
            "multiplicity": "Holm over five comparisons",
            "seed": 20261022,
        },
        "graphics": (
            "vector PDF and SVG; STIX serif; fixed color/marker mapping; all six methods; "
            "no method or quantile selected using test performance"
        ),
    }


def qname(tau):
    return "q" + str(int(round(1000*float(tau)))).zfill(3)


def pinball(y, q, tau):
    residual = np.asarray(y)-np.asarray(q)
    return np.maximum(tau*residual, (tau-1)*residual)


def interval_score(y, lo, hi, alpha):
    return hi-lo + 2/alpha*(np.maximum(lo-y, 0)+np.maximum(y-hi, 0))


def point_wis(y, ordered):
    score = 0.5*np.abs(y-ordered[:, 4])
    for alpha, lower, upper in INTERVALS:
        score += alpha/2*interval_score(y, ordered[:, lower], ordered[:, upper], alpha)
    return score/(len(INTERVALS)+0.5)


def result_rows():
    return json.loads((EXPERIMENT / "results.json").read_text())


def method_seeds(method):
    return [0] if method == "HistGBR" else [0, 1, 2]


def load_prediction(method, seed):
    path = EXPERIMENT / "within_site6" / method / f"final_seed{seed}" / "test_quantiles.csv"
    frame = pd.read_csv(path)
    raw = frame[[f"raw_{qname(t)}" for t in GRID]].to_numpy(float)
    ordered = frame[[f"rearranged_{qname(t)}" for t in GRID]].to_numpy(float)
    return frame, raw, ordered


def aggregate():
    rows = result_rows()
    summaries = []
    for method in METHODS:
        rr = [row for row in rows if row["method"] == method]
        if len(rr) != len(method_seeds(method)):
            raise ValueError(f"Incomplete results for {method}")
        raw_cdf = np.mean([[row["raw_quantiles"][str(float(t))]["empirical_cdf"]
                            for t in GRID] for row in rr], axis=0)
        rearranged_cdf = np.mean([
            [row["rearranged_quantiles"][str(float(t))]["empirical_cdf"]
             for t in GRID] for row in rr], axis=0)
        violation_all = []
        maximum_crossed = []
        for seed in method_seeds(method):
            _, raw, ordered = load_prediction(method, seed)
            if not np.allclose(ordered, np.sort(raw, axis=1), rtol=0, atol=0):
                raise ValueError("Saved rearrangement is not the row-wise sort")
            violation = np.maximum(raw[:, :-1]-raw[:, 1:], 0)
            violation_all.append(violation.ravel())
            row_max = violation.max(axis=1)
            maximum_crossed += row_max[row_max > 0].tolist()
        intervals = {}
        for nominal in [0.5, 0.8, 0.9, 0.95]:
            key = str(nominal)
            intervals[key] = {}
            for mode in ["raw_rearranged", "static", "rolling"]:
                intervals[key][mode] = {
                    metric: float(np.mean([row["intervals"][key][mode][metric]
                                           for row in rr]))
                    for metric in ["picp", "mpiw", "winkler"]
                }
        summary = {
            "method": method,
            "seeds": method_seeds(method),
            "selected_index": rr[0]["selected_index"],
            "q1_selected_mean_pinball_9": float(json.loads(
                (EXPERIMENT / "within_site6" / method / "selection.json").read_text()
            )["selected_validation_mean_pinball_9"]),
            "raw_mean_pinball_9": float(np.mean([r["raw_mean_pinball_9"] for r in rr])),
            "rearranged_mean_pinball_9": float(np.mean([
                r["rearranged_mean_pinball_9"] for r in rr])),
            "rearranged_wis": float(np.mean([r["rearranged_wis"] for r in rr])),
            "raw_crossing_rate": float(np.mean([r["raw_crossing_rate"] for r in rr])),
            "mean_positive_adjacent_decrease_all_pairs_mw": float(
                np.mean(np.concatenate(violation_all))),
            "mean_maximum_decrease_crossed_rows_mw": (
                float(np.mean(maximum_crossed)) if maximum_crossed else 0.0),
            "raw_empirical_cdf": raw_cdf.tolist(),
            "rearranged_empirical_cdf": rearranged_cdf.tolist(),
            "raw_mace": float(np.mean(np.abs(raw_cdf-GRID))),
            "raw_max_calibration_error": float(np.max(np.abs(raw_cdf-GRID))),
            "rearranged_mace": float(np.mean(np.abs(rearranged_cdf-GRID))),
            "rearranged_max_calibration_error": float(
                np.max(np.abs(rearranged_cdf-GRID))),
            "intervals": intervals,
        }
        if not np.isclose(summary["rearranged_wis"],
                          2*summary["rearranged_mean_pinball_9"],
                          rtol=1e-10, atol=1e-10):
            raise ValueError(f"WIS/pinball identity fails for {method}")
        summaries.append(summary)
    return summaries


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def aligned_wis(method):
    frames = []
    scores = []
    for seed in method_seeds(method):
        frame, _, ordered = load_prediction(method, seed)
        y = frame.y_true.to_numpy(float)
        frames.append(frame)
        scores.append(point_wis(y, ordered))
    reference = frames[0]
    ids = reference.window_index.to_numpy()
    times = pd.to_datetime(reference.time).to_numpy()
    y = reference.y_true.to_numpy(float)
    for frame in frames[1:]:
        if not np.array_equal(ids, frame.window_index.to_numpy()):
            raise ValueError("Seed identifiers differ")
        if not np.array_equal(times, pd.to_datetime(frame.time).to_numpy()):
            raise ValueError("Seed times differ")
        if not np.allclose(y, frame.y_true.to_numpy(float), rtol=0, atol=0):
            raise ValueError("Seed labels differ")
    return ids, times, y, np.mean(scores, axis=0)


def paired_inference():
    rng = np.random.default_rng(20261022)
    ids, times, y, full = aligned_wis("Full")
    first = times.min().astype("datetime64[D]")
    block = ((times.astype("datetime64[D]")-first).astype(int)//7).astype(int)
    indices = [np.flatnonzero(block == value) for value in np.unique(block)]
    reports = []
    for comparator in METHODS[1:]:
        ids2, times2, y2, other = aligned_wis(comparator)
        if not (np.array_equal(ids, ids2) and np.array_equal(times, times2)
                and np.allclose(y, y2, rtol=0, atol=0)):
            raise ValueError("Comparator rows are not aligned")
        difference = full-other
        sums = np.array([difference[ii].sum() for ii in indices])
        counts = np.array([len(ii) for ii in indices])
        effect = float(difference.mean())
        bootstrap = np.empty(10000)
        for iteration in range(len(bootstrap)):
            draw = rng.integers(0, len(indices), len(indices))
            bootstrap[iteration] = sums[draw].sum()/counts[draw].sum()
        observed = abs(effect)
        exceed = 0
        remaining = 99999
        while remaining:
            count = min(10000, remaining)
            signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(indices)))
            exceed += int(np.sum(np.abs((signs @ sums)/counts.sum()) >= observed))
            remaining -= count
        reports.append({
            "comparator": comparator, "n": int(len(y)),
            "calendar_blocks": int(len(indices)),
            "full_minus_comparator_wis_mw": effect,
            "cluster_bootstrap_95_ci_mw": [float(np.quantile(bootstrap, 0.025)),
                                            float(np.quantile(bootstrap, 0.975))],
            "block_sign_flip_p_unadjusted": float((1+exceed)/100000),
        })
    adjusted = holm([r["block_sign_flip_p_unadjusted"] for r in reports])
    for row, value in zip(reports, adjusted):
        row["block_sign_flip_p_holm_5"] = float(value)
    return reports


def tex_tables(out, summaries, inference):
    labels = {"Full":"Full", "NoVariance":"No variance",
              "NoDiscrepancy":"No discrepancy", "WeightedSVQR":"Weighted SVQR",
              "TargetSVQR":"Target SVQR", "HistGBR":"HistGBR"}
    lines = [r'\begin{table}[htbp]', r'\centering', r'\color{blue}', r'\small',
             r'\caption{\textcolor{blue}{Probability-selected Farm~6 evaluation on nine quantiles. Kernel entries are means over three landmark seeds; HistGBR uses its deterministic seed-0 fit. Selection uses Q1 mean pinball loss (MPL); test MPL is computed on unrearranged forecasts. WIS and calibration error use the fixed row-wise monotone rearrangement. Crossing is retained from raw forecasts. Lower scores and calibration error are preferable.}}',
             r'\label{tab:probabilistic_scores}', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Method & Candidate & Q1 MPL & Test MPL & WIS (MW) & MACE & Crossing (\%) \\',
             r'\midrule']
    for s in summaries:
        lines.append(f"{labels[s['method']]} & c{s['selected_index']:02d} & "
                     f"{s['q1_selected_mean_pinball_9']:.3f} & "
                     f"{s['raw_mean_pinball_9']:.3f} & {s['rearranged_wis']:.3f} & "
                     f"{100*s['rearranged_mace']:.2f} & "
                     f"{100*s['raw_crossing_rate']:.2f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (out / "probabilistic_scores.tex").write_text("\n".join(lines)+"\n")

    lines = [r'\begin{table}[htbp]', r'\centering', r'\color{blue}', r'\small',
             r'\caption{\textcolor{blue}{Probability-selected Farm~6 interval calibration. Each cell gives PICP/MPIW, where MPIW is in MW. S uses fixed Q2 calibration and R is the causal 256-score rolling update.}}',
             r'\label{tab:probabilistic_coverage}', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Method & Calibration & 50\% & 80\% & 90\% & 95\% \\', r'\midrule']
    for s in summaries:
        for mode, mode_label in [("static", "S"), ("rolling", "R")]:
            cells=[]
            for nominal in [0.5,0.8,0.9,0.95]:
                x=s['intervals'][str(nominal)][mode]
                cells.append(f"{x['picp']:.3f}/{x['mpiw']:.2f}")
            lines.append(f"{labels[s['method']]} & {mode_label} & " + " & ".join(cells) + r' \\')
        if s is not summaries[-1]: lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (out / "probabilistic_coverage.tex").write_text("\n".join(lines)+"\n")

    lines = [r'\begin{table}[htbp]', r'\centering', r'\color{blue}', r'\small',
             r'\caption{\textcolor{blue}{Exploratory paired seven-day-block inference for probability-selected Farm~6 forecasts. Effects are Full minus comparator WIS in MW; negative values favor Full. $p_{\rm H}$ is Holm-adjusted across five comparisons.}}',
             r'\label{tab:probabilistic_inference}', r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{lrrr}', r'\toprule',
             r'Comparator & WIS difference & 95\% block-bootstrap interval & $p_{\rm H}$ \\', r'\midrule']
    for row in inference:
        lo,hi=row['cluster_bootstrap_95_ci_mw'];p=row['block_sign_flip_p_holm_5']
        ps='$<0.001$' if p<.001 else f'{p:.3f}'
        lines.append(f"{labels[row['comparator']]} & {row['full_minus_comparator_wis_mw']:.3f} & "
                     f"[{lo:.3f}, {hi:.3f}] & {ps} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (out / "probabilistic_inference.tex").write_text("\n".join(lines)+"\n")


def figures(out, summaries):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family":"serif", "font.serif":["STIX Two Text","STIXGeneral","DejaVu Serif"],
        "mathtext.fontset":"stix", "font.size":8.5, "axes.labelsize":9,
        "axes.titlesize":9, "legend.fontsize":7, "xtick.labelsize":8,
        "ytick.labelsize":8, "axes.linewidth":0.7, "lines.linewidth":1.15,
        "savefig.bbox":"tight", "pdf.fonttype":42, "ps.fonttype":42,
    })
    colors=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#6A3D9A"]
    markers=["o","s","^","D","v","P"]
    labels={"Full":"Full", "NoVariance":"No variance",
            "NoDiscrepancy":"No discrepancy", "WeightedSVQR":"Weighted SVQR",
            "TargetSVQR":"Target SVQR", "HistGBR":"HistGBR"}
    for kind,key in [("raw","raw_empirical_cdf"),("rearranged","rearranged_empirical_cdf")]:
        pass
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.05),sharex=True,sharey=True)
    for ax,key,title in zip(axes,["raw_empirical_cdf","rearranged_empirical_cdf"],
                            ["(a) Independently fitted quantiles","(b) Monotone rearrangement"]):
        ax.plot([0,1],[0,1],color="0.35",ls="--",lw=.9,label="Ideal")
        for s,c,m in zip(summaries,colors,markers):
            ax.plot(GRID,s[key],color=c,marker=m,ms=3.4,label=labels[s['method']])
        ax.set_title(title);ax.set_xlabel("Nominal quantile level")
        ax.grid(True,color="0.88",lw=.55);ax.set_xlim(0,1);ax.set_ylim(0,1)
    axes[0].set_ylabel(r"Empirical $P(Y\leq \hat q_\tau)$")
    handles,lab=axes[1].get_legend_handles_labels()
    fig.legend(handles,lab,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.5,-.055))
    fig.subplots_adjust(bottom=.24,wspace=.08)
    for ext in ["pdf","svg"]:fig.savefig(out/f"probabilistic_reliability.{ext}")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.1,3.05),sharex=True)
    nominal=np.array([.5,.8,.9,.95])
    for s,c,m in zip(summaries,colors,markers):
        coverage=[s['intervals'][str(v)]['static']['picp'] for v in nominal]
        width=[s['intervals'][str(v)]['static']['mpiw'] for v in nominal]
        axes[0].plot(nominal,coverage,color=c,marker=m,ms=3.4,label=labels[s['method']])
        axes[1].plot(nominal,width,color=c,marker=m,ms=3.4,label=labels[s['method']])
    axes[0].plot([.47,.97],[.47,.97],color="0.35",ls="--",lw=.9,label="Ideal")
    axes[0].set_title("(a) Static Q2-calibrated coverage")
    axes[1].set_title("(b) Static Q2-calibrated sharpness")
    axes[0].set_ylabel("PICP");axes[1].set_ylabel("MPIW (MW)")
    for ax in axes:
        ax.set_xlabel("Nominal central coverage");ax.grid(True,color="0.88",lw=.55)
        ax.set_xticks(nominal)
    axes[0].set_ylim(.45,1.01)
    handles,lab=axes[1].get_legend_handles_labels()
    fig.legend(handles,lab,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.055))
    fig.subplots_adjust(bottom=.24,wspace=.24)
    for ext in ["pdf","svg"]:fig.savefig(out/f"probabilistic_coverage_sharpness.{ext}")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--freeze-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    p=protocol()
    sources=[Path(__file__)]
    manifest={
        "status":"frozen", "created_utc":datetime.now(timezone.utc).isoformat(),
        "protocol":p,
        "experiment_protocol_sha256":digest(EXPERIMENT/"protocol.json"),
        "experiment_runner_sha256":digest(ROOT/"review/run_probabilistic_selection_study.py"),
        "source_hashes":{str(x.relative_to(ROOT)):digest(x) for x in sources},
        "python":platform.python_version(),"numpy":np.__version__,
    }
    if args.output.exists():
        old=json.loads((args.output/"manifest.json").read_text())
        if not args.resume:raise FileExistsError("Use --resume for the frozen analysis")
        for key in ["protocol","experiment_protocol_sha256","experiment_runner_sha256","source_hashes"]:
            if old[key]!=manifest[key]:raise ValueError(f"Frozen {key} changed")
        manifest=old
    else:
        args.output.mkdir(parents=True)
        save(args.output/"protocol.json",p);save(args.output/"manifest.json",manifest)
    if args.freeze_only:return
    experiment_manifest=json.loads((EXPERIMENT/"manifest.json").read_text())
    if experiment_manifest.get("status")!="complete":raise ValueError("Experiment is not complete")
    inputs=[EXPERIMENT/"manifest.json",EXPERIMENT/"selection_lock.json",EXPERIMENT/"results.json"]
    inputs += sorted(EXPERIMENT.glob("within_site6/*/final_seed*/test_quantiles.csv"))
    input_hashes={str(x.relative_to(ROOT)):digest(x) for x in inputs}
    if "input_hashes" in manifest and manifest["input_hashes"]!=input_hashes:
        raise ValueError("Completed experiment inputs changed")
    manifest["input_hashes"]=input_hashes;manifest["status"]="running"
    save(args.output/"manifest.json",manifest)
    try:
        summaries=aggregate();inference=paired_inference()
        save(args.output/"aggregate_results.json",summaries)
        save(args.output/"paired_probability_inference.json",{
            "interpretation":"Exploratory reused-test inference; positive effects favor the comparator.",
            "comparisons":inference})
        tex_tables(args.output,summaries,inference);figures(args.output,summaries)
        manifest["status"]="complete";manifest["methods"]=len(summaries)
        manifest["paired_comparisons"]=len(inference)
    except Exception as error:
        manifest["status"]="failed";manifest["error"]=repr(error);raise
    finally:
        manifest["updated_utc"]=datetime.now(timezone.utc).isoformat()
        save(args.output/"manifest.json",manifest)


if __name__=="__main__":main()
