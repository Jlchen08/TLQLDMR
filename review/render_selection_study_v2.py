#!/usr/bin/env python3
"""Render the complete, verified selection-v2 analyses for publication.

This module is presentation only.  It reads the completed analysis products in
``selection_v2_analysis`` and ``matched_ablation_v2_analysis``; it never opens
the selection or matched-experiment ``results.json`` files.  Every expected
task and method is validated before any table or figure is written.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "review/selection_runs/selection_v2_analysis"
DEFAULT_MATCHED = ROOT / "review/selection_runs/matched_ablation_v2_analysis"
DEFAULT_OUTPUT = ROOT / "review/selection_runs/selection_v2_render"

METHODS = (
    "Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR",
    "TargetSVQR", "HistGBR",
)
METHOD_LABELS = {
    "Full": "TL-QLDMR",
    "NoVariance": "No variance",
    "NoDiscrepancy": "No discrepancy",
    "WeightedSVQR": "Weighted SVQR",
    "TargetSVQR": "Target SVQR",
    "HistGBR": "HistGBR",
}
COMPARATORS = METHODS[1:]
VARIANTS = METHODS[:4]
ABLATION_COMPARATORS = VARIANTS[1:]
SCOPES = ("within", "cross")
TASKS = {
    "within": tuple(f"within_site{site}" for site in range(1, 7)),
    "cross": tuple(f"cross_site{site}" for site in range(1, 7)),
}
CAPACITY = {1: 99.0, 2: 200.0, 3: 99.0, 4: 66.0, 5: 36.0, 6: 96.0}

SELECTION_FILES = (
    "selection_v2_analysis.json",
    "task_method_seed_mean_rmse.csv",
    "scope_summary.csv",
    "full_pairwise_inference.csv",
)
MATCHED_FILES = (
    "summary.json", "task_metrics.csv", "aggregate_metrics.csv",
    "paired_inference.csv",
)
OUTPUT_FILES = (
    "primary_within_rmse.tex",
    "scope_macro_summary.tex",
    "full_paired_inference.tex",
    "matched_ablation.tex",
    "within_site_normalized_rmse.pdf",
    "within_site_normalized_rmse.png",
    "scope_average_rank.pdf",
    "scope_average_rank.png",
    "render_protocol.json",
)

# Okabe--Ito colors, with dark gray in place of yellow on a white background.
METHOD_STYLES = {
    "Full": dict(color="#0072B2", marker="o", linestyle="-"),
    "NoVariance": dict(color="#D55E00", marker="s", linestyle="--"),
    "NoDiscrepancy": dict(color="#009E73", marker="^", linestyle="-."),
    "WeightedSVQR": dict(color="#CC79A7", marker="D", linestyle=":"),
    "TargetSVQR": dict(color="#333333", marker="v", linestyle="-"),
    "HistGBR": dict(color="#56B4E9", marker="P", linestyle="--"),
}
STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "lines.linewidth": 1.1,
    "lines.markersize": 4.2,
    "lines.markeredgewidth": 0.75,
    "errorbar.capsize": 2.1,
    "legend.frameon": False,
    "savefig.dpi": 400,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks required columns: {sorted(missing)}")


def _finite(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} contains a nonfinite value.")


def _site(task: str) -> int:
    try:
        value = int(str(task).rsplit("site", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Unrecognized task name: {task}") from error
    if value not in CAPACITY:
        raise ValueError(f"Unrecognized site in task: {task}")
    return value


def _verify_hashed_files(root: Path, hashes: dict, names: Iterable[str], label: str) -> None:
    if not isinstance(hashes, dict):
        raise ValueError(f"{label} does not contain an artifact-hash mapping.")
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} artifact: {path}")
        if hashes.get(name) != digest(path):
            raise ValueError(f"{label} hash mismatch: {name}")


def _parse_seed_values(value: object) -> list[float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError("Malformed seed_rmse_mw list.") from error
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("seed_rmse_mw must be a nonempty list.")
    values = [float(item) for item in value]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError("seed_rmse_mw must contain finite nonnegative values.")
    return values


def _assert_unique_design(frame: pd.DataFrame, keys: list[str], expected: set[tuple],
                          name: str) -> None:
    observed = [tuple(row) for row in frame[keys].itertuples(index=False, name=None)]
    if len(observed) != len(set(observed)):
        raise ValueError(f"{name} contains duplicate design rows.")
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise ValueError(
            f"{name} is incomplete or unexpected; missing={missing[:2]}, extra={extra[:2]}"
        )


def _ordered(frame: pd.DataFrame, keys: list[str], orders: dict[str, tuple]) -> pd.DataFrame:
    result = frame.copy()
    helper = []
    for key in keys:
        order = orders[key]
        mapping = {value: index for index, value in enumerate(order)}
        if not set(result[key]).issubset(mapping):
            raise ValueError(f"Cannot order unexpected {key} value.")
        name = f"__order_{key}"
        result[name] = result[key].map(mapping)
        helper.append(name)
    result = result.sort_values(helper, kind="stable").drop(columns=helper)
    return result.reset_index(drop=True)


def _validate_selection(root: Path) -> dict[str, object]:
    root = Path(root)
    manifest_path = root / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed selection-analysis manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("status") != "complete":
        raise ValueError("Selection analysis is not complete.")
    _verify_hashed_files(root, manifest.get("artifact_hashes"), SELECTION_FILES,
                         "selection analysis")

    report = _read_json(root / "selection_v2_analysis.json")
    if not isinstance(report, dict):
        raise ValueError("Selection analysis JSON must contain an object.")
    counts = report.get("reporting_counts", {})
    if counts.get("tasks") != 12 or counts.get("methods") != 6 or \
            counts.get("task_method_rows") != 72 or counts.get("full_comparisons") != 60:
        raise ValueError("Selection analysis JSON does not certify the complete 12 x 6 design.")

    point = pd.read_csv(root / "task_method_seed_mean_rmse.csv")
    _require_columns(point, (
        "scope", "task", "site", "method", "evaluation_seeds", "seed_rmse_mw",
        "seed_mean_rmse_mw", "capacity_mw",
        "seed_mean_capacity_normalized_rmse_percent", "task_rank",
    ), "selection task metrics")
    expected = {
        (scope, task, site, method)
        for scope in SCOPES for task in TASKS[scope]
        for site in [_site(task)] for method in METHODS
    }
    _assert_unique_design(point, ["scope", "task", "site", "method"], expected,
                          "selection task metrics")
    _finite(point, ("site", "evaluation_seeds", "seed_mean_rmse_mw", "capacity_mw",
                    "seed_mean_capacity_normalized_rmse_percent", "task_rank"),
            "selection task metrics")
    seed_values = []
    for row in point.itertuples(index=False):
        values = _parse_seed_values(row.seed_rmse_mw)
        expected_seeds = 1 if row.method == "HistGBR" else 3
        if int(row.evaluation_seeds) != expected_seeds or len(values) != expected_seeds:
            raise ValueError(f"Unexpected seed count for {row.task}/{row.method}.")
        if not math.isclose(float(np.mean(values)), float(row.seed_mean_rmse_mw),
                            rel_tol=2e-9, abs_tol=2e-9):
            raise ValueError(f"Seed mean mismatch for {row.task}/{row.method}.")
        if not math.isclose(float(row.capacity_mw), CAPACITY[int(row.site)],
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Capacity mismatch for {row.task}/{row.method}.")
        expected_nrmse = 100 * float(row.seed_mean_rmse_mw) / float(row.capacity_mw)
        if not math.isclose(expected_nrmse,
                            float(row.seed_mean_capacity_normalized_rmse_percent),
                            rel_tol=2e-9, abs_tol=2e-9):
            raise ValueError(f"Normalized RMSE mismatch for {row.task}/{row.method}.")
        seed_values.append(values)
    point["_seed_values"] = seed_values

    scope = pd.read_csv(root / "scope_summary.csv")
    _require_columns(scope, (
        "scope", "method", "tasks",
        "site_equal_macro_capacity_normalized_rmse_percent",
        "average_task_rank", "task_wins_including_ties",
    ), "selection scope summary")
    _assert_unique_design(
        scope, ["scope", "method"],
        {(scope_name, method) for scope_name in SCOPES for method in METHODS},
        "selection scope summary",
    )
    _finite(scope, ("tasks", "site_equal_macro_capacity_normalized_rmse_percent",
                    "average_task_rank", "task_wins_including_ties"),
            "selection scope summary")
    for row in scope.itertuples(index=False):
        selected = point[(point["scope"] == row.scope) & (point["method"] == row.method)]
        if int(row.tasks) != 6 or len(selected) != 6:
            raise ValueError(f"Scope summary is incomplete for {row.scope}/{row.method}.")
        expected_macro = selected["seed_mean_capacity_normalized_rmse_percent"].mean()
        expected_rank = selected["task_rank"].mean()
        if not np.isclose(expected_macro,
                          row.site_equal_macro_capacity_normalized_rmse_percent,
                          rtol=2e-9, atol=2e-9):
            raise ValueError(f"Macro NRMSE mismatch for {row.scope}/{row.method}.")
        if not np.isclose(expected_rank, row.average_task_rank, rtol=2e-9, atol=2e-9):
            raise ValueError(f"Average-rank mismatch for {row.scope}/{row.method}.")

    inference = pd.read_csv(root / "full_pairwise_inference.csv")
    _require_columns(inference, (
        "scope", "task", "site", "comparator",
        "full_minus_comparator_mse_mw2",
        "time_block_bootstrap_95_ci_low_mw2",
        "time_block_bootstrap_95_ci_high_mw2",
        "block_sign_flip_p_holm_within_scope_30",
    ), "selection Full inference")
    expected_inference = {
        (scope_name, task, _site(task), comparator)
        for scope_name in SCOPES for task in TASKS[scope_name]
        for comparator in COMPARATORS
    }
    _assert_unique_design(
        inference, ["scope", "task", "site", "comparator"], expected_inference,
        "selection Full inference",
    )
    _finite(inference, (
        "site", "full_minus_comparator_mse_mw2",
        "time_block_bootstrap_95_ci_low_mw2",
        "time_block_bootstrap_95_ci_high_mw2",
        "block_sign_flip_p_holm_within_scope_30",
    ), "selection Full inference")
    pvalues = inference["block_sign_flip_p_holm_within_scope_30"].to_numpy(float)
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("Selection adjusted p-values must lie in [0, 1].")
    if np.any(inference["time_block_bootstrap_95_ci_low_mw2"] >
              inference["time_block_bootstrap_95_ci_high_mw2"]):
        raise ValueError("Selection confidence-interval endpoints are reversed.")

    point = _ordered(point, ["scope", "task", "method"], {
        "scope": SCOPES,
        "task": TASKS["within"] + TASKS["cross"],
        "method": METHODS,
    })
    scope = _ordered(scope, ["scope", "method"], {
        "scope": SCOPES, "method": METHODS,
    })
    inference = _ordered(inference, ["scope", "task", "comparator"], {
        "scope": SCOPES,
        "task": TASKS["within"] + TASKS["cross"],
        "comparator": COMPARATORS,
    })
    return {"manifest": manifest, "report": report, "point": point,
            "scope": scope, "inference": inference}


def _validate_matched(root: Path) -> dict[str, object]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed matched-analysis manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("status") != "complete" or \
            manifest.get("heldout_results_read") is not True:
        raise ValueError("Matched-ablation analysis is not complete.")
    _verify_hashed_files(root, manifest.get("output_hashes"), MATCHED_FILES,
                         "matched-ablation analysis")
    summary = _read_json(root / "summary.json")
    if not isinstance(summary, dict) or summary.get("complete_evaluation_keys") != 144 or \
            summary.get("task_metric_rows") != 48 or \
            summary.get("paired_tests") != {"within": 18, "cross": 18}:
        raise ValueError("Matched analysis JSON does not certify the complete 12 x 4 design.")

    task = pd.read_csv(root / "task_metrics.csv")
    _require_columns(task, (
        "scope", "task", "site", "variant", "capacity_mw",
        "seed0_rmse_mw", "seed1_rmse_mw", "seed2_rmse_mw",
        "mean_seed_rmse_mw", "normalized_rmse", "rank",
    ), "matched task metrics")
    expected = {
        (scope_name, task_name, _site(task_name), variant)
        for scope_name in SCOPES for task_name in TASKS[scope_name]
        for variant in VARIANTS
    }
    _assert_unique_design(task, ["scope", "task", "site", "variant"], expected,
                          "matched task metrics")
    _finite(task, (
        "site", "capacity_mw", "seed0_rmse_mw", "seed1_rmse_mw",
        "seed2_rmse_mw", "mean_seed_rmse_mw", "normalized_rmse", "rank",
    ), "matched task metrics")
    seeds = task[["seed0_rmse_mw", "seed1_rmse_mw", "seed2_rmse_mw"]].to_numpy(float)
    if np.any(seeds < 0):
        raise ValueError("Matched RMSE values must be nonnegative.")
    if not np.allclose(seeds.mean(axis=1), task["mean_seed_rmse_mw"],
                       rtol=2e-9, atol=2e-9):
        raise ValueError("Matched mean seed RMSE is inconsistent.")
    if not np.allclose(task["mean_seed_rmse_mw"] / task["capacity_mw"],
                       task["normalized_rmse"], rtol=2e-9, atol=2e-9):
        raise ValueError("Matched normalized RMSE is inconsistent.")

    aggregate = pd.read_csv(root / "aggregate_metrics.csv")
    _require_columns(aggregate, (
        "scope", "variant", "sites", "macro_normalized_rmse", "mean_rank",
    ), "matched aggregate metrics")
    _assert_unique_design(
        aggregate, ["scope", "variant"],
        {(scope_name, variant) for scope_name in SCOPES for variant in VARIANTS},
        "matched aggregate metrics",
    )
    _finite(aggregate, ("sites", "macro_normalized_rmse", "mean_rank"),
            "matched aggregate metrics")
    for row in aggregate.itertuples(index=False):
        selected = task[(task["scope"] == row.scope) & (task["variant"] == row.variant)]
        if int(row.sites) != 6 or len(selected) != 6:
            raise ValueError(f"Matched aggregate is incomplete for {row.scope}/{row.variant}.")
        if not np.isclose(selected["normalized_rmse"].mean(), row.macro_normalized_rmse,
                          rtol=2e-9, atol=2e-9):
            raise ValueError(f"Matched macro NRMSE mismatch for {row.scope}/{row.variant}.")
        if not np.isclose(selected["rank"].mean(), row.mean_rank,
                          rtol=2e-9, atol=2e-9):
            raise ValueError(f"Matched mean-rank mismatch for {row.scope}/{row.variant}.")

    inference = pd.read_csv(root / "paired_inference.csv")
    _require_columns(inference, (
        "scope", "task", "site", "comparison",
        "mean_squared_error_difference_mw2", "ci_lower_mw2", "ci_upper_mw2",
        "p_holm",
    ), "matched paired inference")
    comparisons = tuple(f"Full - {variant}" for variant in ABLATION_COMPARATORS)
    expected_inference = {
        (scope_name, task_name, _site(task_name), comparison)
        for scope_name in SCOPES for task_name in TASKS[scope_name]
        for comparison in comparisons
    }
    _assert_unique_design(
        inference, ["scope", "task", "site", "comparison"], expected_inference,
        "matched paired inference",
    )
    _finite(inference, (
        "site", "mean_squared_error_difference_mw2", "ci_lower_mw2",
        "ci_upper_mw2", "p_holm",
    ), "matched paired inference")
    if np.any((inference["p_holm"] < 0) | (inference["p_holm"] > 1)):
        raise ValueError("Matched adjusted p-values must lie in [0, 1].")
    if np.any(inference["ci_lower_mw2"] > inference["ci_upper_mw2"]):
        raise ValueError("Matched confidence-interval endpoints are reversed.")

    task = _ordered(task, ["scope", "task", "variant"], {
        "scope": SCOPES,
        "task": TASKS["within"] + TASKS["cross"],
        "variant": VARIANTS,
    })
    aggregate = _ordered(aggregate, ["scope", "variant"], {
        "scope": SCOPES, "variant": VARIANTS,
    })
    inference = _ordered(inference, ["scope", "task", "comparison"], {
        "scope": SCOPES,
        "task": TASKS["within"] + TASKS["cross"],
        "comparison": comparisons,
    })
    return {"manifest": manifest, "summary": summary, "task": task,
            "aggregate": aggregate, "inference": inference}


def load_validated_inputs(selection_root: Path, matched_root: Path) -> tuple[dict, dict]:
    """Load only finalized analysis artifacts and validate both complete designs."""
    return _validate_selection(selection_root), _validate_matched(matched_root)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_p(value: float) -> str:
    value = float(value)
    return "$<0.001$" if value < .001 else f"{value:.3f}"


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _primary_table(point: pd.DataFrame) -> str:
    frame = point[point["scope"] == "within"].pivot(
        index="site", columns="method", values="seed_mean_rmse_mw"
    ).reindex(index=range(1, 7), columns=METHODS)
    lines = [
        r"\begin{table*}[pos=t,width=\FullWidth]", r"\centering", r"\color{blue}",
        r"\caption{Q3--Q4 RMSE (MW) on the six primary within-farm normal-to-extreme tasks. Values are means over the predeclared evaluation seeds.}",
        r"\label{tab:selection-v2-primary-within}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        "Farm & " + " & ".join(_latex_escape(METHOD_LABELS[m]) for m in METHODS) + r" \\",
        r"\midrule",
    ]
    for site, row in frame.iterrows():
        lines.append(f"Farm {site} & " + " & ".join(_fmt(row[m]) for m in METHODS) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _scope_table(scope: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[pos=t]", r"\centering", r"\color{blue}",
        r"\caption{Site-equal capacity-normalized RMSE and average task rank in the primary within-farm and secondary cross-farm scopes.}",
        r"\label{tab:selection-v2-scope}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrr}", r"\toprule",
        r"Scope & Method & Macro NRMSE (\%) & Average rank \\", r"\midrule",
    ]
    for scope_name in SCOPES:
        rows = scope[scope["scope"] == scope_name]
        for index, row in enumerate(rows.itertuples(index=False)):
            scope_cell = ("Within" if scope_name == "within" else "Cross") if index == 0 else ""
            lines.append(
                f"{scope_cell} & {_latex_escape(METHOD_LABELS[row.method])} & "
                f"{_fmt(row.site_equal_macro_capacity_normalized_rmse_percent)} & "
                f"{_fmt(row.average_task_rank, 2)} " + r"\\"
            )
        if scope_name == "within":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _inference_table(inference: pd.DataFrame) -> str:
    lines = [
        r"{\color{blue}\small", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{longtable}{lllrrr}",
        r"\caption{Paired seven-day-block inference for TL-QLDMR minus each comparator. Negative squared-error differences favor TL-QLDMR; Holm correction is applied separately within each scope.}\label{tab:selection-v2-full-inference}\\",
        r"\toprule",
        r"Scope & Farm & Comparator & $\Delta$MSE (MW$^2$) & 95\% CI & Holm $p$ \\",
        r"\midrule", r"\endfirsthead", r"\toprule",
        r"Scope & Farm & Comparator & $\Delta$MSE (MW$^2$) & 95\% CI & Holm $p$ \\",
        r"\midrule", r"\endhead", r"\midrule",
        r"\multicolumn{6}{r}{Continued on next page} \\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot",
    ]
    for row in inference.itertuples(index=False):
        lines.append(
            f"{'Within' if row.scope == 'within' else 'Cross'} & Farm {row.site} & "
            f"{_latex_escape(METHOD_LABELS[row.comparator])} & "
            f"{_fmt(row.full_minus_comparator_mse_mw2)} & "
            f"[{_fmt(row.time_block_bootstrap_95_ci_low_mw2)}, "
            f"{_fmt(row.time_block_bootstrap_95_ci_high_mw2)}] & "
            f"{_fmt_p(row.block_sign_flip_p_holm_within_scope_30)} " + r"\\"
        )
    lines.extend([r"\end{longtable}", r"}", ""])
    return "\n".join(lines)


def _matched_table(task: pd.DataFrame) -> str:
    frame = task.pivot(index=["scope", "site"], columns="variant",
                       values="mean_seed_rmse_mw").reindex(columns=VARIANTS)
    lines = [
        r"\begin{table}[pos=t]", r"\centering", r"\color{blue}",
        r"\caption{Matched component ablation on all twelve tasks. Values are three-seed mean RMSE (MW); every removal uses the corresponding task's locked TL-QLDMR configuration and artifacts.}",
        r"\label{tab:selection-v2-matched-ablation}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        "Scope & Farm & " + " & ".join(_latex_escape(METHOD_LABELS[m]) for m in VARIANTS) + r" \\",
        r"\midrule",
    ]
    for scope_index, scope_name in enumerate(SCOPES):
        for site in range(1, 7):
            row = frame.loc[(scope_name, site)]
            scope_cell = ("Within" if scope_name == "within" else "Cross") if site == 1 else ""
            lines.append(
                f"{scope_cell} & Farm {site} & " +
                " & ".join(_fmt(row[variant]) for variant in VARIANTS) + r" \\"
            )
        if scope_index == 0:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _save_figure(fig: mpl.figure.Figure, output: Path, stem: str) -> None:
    metadata = {
        "Title": stem.replace("_", " "),
        "Creator": "render_selection_study_v2.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=.035,
                metadata=metadata)
    fig.savefig(output / f"{stem}.png", bbox_inches="tight", pad_inches=.035,
                dpi=400, metadata={"Software": "render_selection_study_v2.py"})
    plt.close(fig)


def _render_within(point: pd.DataFrame, output: Path) -> None:
    within = point[point["scope"] == "within"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 3.45), layout="constrained")
        centers = np.arange(1, 7, dtype=float)
        offsets = np.linspace(-.25, .25, len(METHODS))
        for method_index, (method, offset) in enumerate(zip(METHODS, offsets)):
            selected = within[within["method"] == method].sort_values("site")
            means, lower, upper = [], [], []
            for _, row in selected.iterrows():
                values = 100 * np.asarray(row["_seed_values"], dtype=float) / float(row["capacity_mw"])
                means.append(values.mean())
                lower.append(values.min())
                upper.append(values.max())
            means = np.asarray(means)
            lower = np.asarray(lower)
            upper = np.asarray(upper)
            style = dict(METHOD_STYLES[method])
            is_full = method == "Full"
            ax.errorbar(
                centers + offset, means,
                yerr=np.vstack([means - lower, upper - means]),
                label=METHOD_LABELS[method], color=style["color"],
                marker=style["marker"], linestyle="none",
                markersize=5.2 if is_full else 4.2,
                markerfacecolor=style["color"] if is_full else "white",
                markeredgecolor=style["color"], markeredgewidth=1.1 if is_full else .75,
                elinewidth=1.05 if is_full else .7, capsize=2.2, zorder=4 if is_full else 3,
            )
        ax.set_xticks(centers, [f"Farm {site}" for site in range(1, 7)])
        ax.set_ylabel("Capacity-normalized RMSE (%)")
        ax.set_xlabel("Target farm")
        ax.grid(axis="y", color="0.88", linewidth=.5)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.17), ncol=3,
                  columnspacing=1.1, handletextpad=.45)
        _save_figure(fig, output, "within_site_normalized_rmse")


def _render_ranks(scope: pd.DataFrame, output: Path) -> None:
    pivot = scope.pivot(index="method", columns="scope", values="average_task_rank").reindex(
        index=METHODS, columns=SCOPES)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 3.55), layout="constrained")
        y = np.arange(len(METHODS))
        for index, method in enumerate(METHODS):
            values = pivot.loc[method]
            full = method == "Full"
            ax.plot(values.to_numpy(float), [index - .075, index + .075], color="0.78",
                    linewidth=1.5 if full else .9, zorder=1)
            for scope_name, marker, color, label, y_offset in (
                ("within", "o", "#0072B2", "Within-farm (primary)", -.075),
                ("cross", "s", "#D55E00", "Cross-farm (stress)", .075),
            ):
                ax.plot(values[scope_name], index + y_offset, marker=marker, linestyle="none",
                        color=color, markerfacecolor=color if full else "white",
                        markeredgewidth=1.15 if full else .85,
                        markersize=6.0 if full else 4.8,
                        label=label if index == 0 else None, zorder=3)
        ax.set_yticks(y, [METHOD_LABELS[method] for method in METHODS])
        ax.invert_yaxis()
        ax.set_xlim(.75, 6.25)
        ax.set_xticks(np.arange(1, 7))
        ax.set_xlabel("Average task rank (lower is better)")
        ax.grid(axis="x", color="0.88", linewidth=.5)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.12), ncol=2)
        ax.get_yticklabels()[0].set_fontweight("bold")
        _save_figure(fig, output, "scope_average_rank")


def render(selection_root: Path, matched_root: Path, output: Path) -> dict:
    selection_root = Path(selection_root).resolve()
    matched_root = Path(matched_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Render output already exists: {output}")
    selection, matched = load_validated_inputs(selection_root, matched_root)

    staging = output.with_name(output.name + f".tmp.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        _atomic_text(staging / "primary_within_rmse.tex",
                     _primary_table(selection["point"]))
        _atomic_text(staging / "scope_macro_summary.tex",
                     _scope_table(selection["scope"]))
        _atomic_text(staging / "full_paired_inference.tex",
                     _inference_table(selection["inference"]))
        _atomic_text(staging / "matched_ablation.tex",
                     _matched_table(matched["task"]))
        _render_within(selection["point"], staging)
        _render_ranks(selection["scope"], staging)

        protocol = {
            "version": "selection_v2_render_1",
            "role": "presentation-only rendering of completed, independently verified analyses",
            "selection_design_required": "12 tasks x 6 methods = 72 task-method rows",
            "matched_design_required": "12 tasks x 4 variants = 48 task-variant rows",
            "task_policy": "retain every predeclared task, method, variant, and seed; no outcome-based filtering",
            "method_order": list(METHODS),
            "variant_order": list(VARIANTS),
            "scope_order": list(SCOPES),
            "primary_figure": (
                "six within-farm capacity-normalized RMSE values; descriptive min-max "
                "seed ranges are shown when repeated algorithmic seeds exist"
            ),
            "rank_figure": "site-equal average task ranks in within and cross scopes",
            "latex_style": "all tables use LaTeX small size and fixed tabular spacing; no scaling box",
        }
        _atomic_text(staging / "render_protocol.json",
                     json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")

        manifest = {
            "version": "selection_v2_render_manifest_1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(Path(__file__).resolve().relative_to(ROOT)),
            "source_sha256": digest(Path(__file__)),
            "inputs": {
                "selection_analysis_root": str(selection_root),
                "matched_analysis_root": str(matched_root),
                "selection_analysis_manifest_sha256": digest(
                    selection_root / "analysis_manifest.json"),
                "matched_analysis_manifest_sha256": digest(matched_root / "manifest.json"),
                "files": {
                    str(path): digest(path)
                    for path in [
                        *(selection_root / name for name in SELECTION_FILES),
                        *(matched_root / name for name in MATCHED_FILES),
                    ]
                },
            },
            "validated_counts": {
                "selection_task_method_rows": 72,
                "selection_scope_rows": 12,
                "selection_full_inference_rows": 60,
                "matched_task_variant_rows": 48,
                "matched_aggregate_rows": 8,
                "matched_inference_rows": 36,
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": importlib.metadata.version("matplotlib"),
            },
            "outputs": {name: digest(staging / name) for name in OUTPUT_FILES},
        }
        _atomic_text(staging / "manifest.json",
                     json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        staging.replace(output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-analysis", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--matched-analysis", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.selection_analysis, args.matched_analysis, args.output)


if __name__ == "__main__":
    main()
