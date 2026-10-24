#!/usr/bin/env python3
"""Create a complete revision-evidence digest from finalized v2 analyses.

The command is deliberately downstream of the two frozen analyses.  It calls
the publication renderer's audited input validator, reads analysis artifacts
only, and never opens an experiment ``results.json`` or prediction file.
Every predeclared scope, task, method, comparison, and matched ablation is
retained in the JSON and Markdown outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import review.render_selection_study_v2 as renderer


DEFAULT_SELECTION = ROOT / "review/selection_runs/selection_v2_analysis"
DEFAULT_MATCHED = ROOT / "review/selection_runs/matched_ablation_v2_analysis"
OUTPUT_FILES = ("revision_evidence.json", "revision_evidence.md")
SELECTION_EXTRA_FILES = (
    "first_tier_all_pair_tests.csv",
    "conservative_first_tier.csv",
)
FIRST_TIER_CAVEAT = (
    "Membership records non-rejection of inferiority to the empirical endpoint "
    "winner under the predeclared block test. It does not establish equal "
    "performance and is not a formal model-confidence set."
)
MATCHED_CAVEAT = (
    "These component effects are conditional on each task's locked Full "
    "configuration, preprocessing, landmarks, and feature map. The removal "
    "variants were not independently retuned, and failure to reject does not "
    "establish equal performance."
)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} lacks required columns: {sorted(missing)}")


def _finite(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}.{column} contains a nonfinite value.")


def _close(a: object, b: object) -> bool:
    return math.isclose(float(a), float(b), rel_tol=2e-9, abs_tol=2e-9)


def _direction(value: float) -> str:
    if value < 0:
        return "Full_lower_loss"
    if value > 0:
        return "Full_higher_loss"
    return "equal_observed_loss"


def _macro_direction(value: float) -> str:
    if value < 0:
        return "Full_lower_macro_NRMSE"
    if value > 0:
        return "Full_higher_macro_NRMSE"
    return "equal_observed_macro_NRMSE"


def _bool(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must contain a boolean value.")


def _verify_extra_selection(selection_root: Path, selection: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = selection["manifest"]
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("Selection analysis lacks its artifact-hash mapping.")
    snapshots = {}
    for name in SELECTION_EXTRA_FILES:
        path = selection_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing selection first-tier artifact: {path}")
        payload = path.read_bytes()
        if hashes.get(name) != hashlib.sha256(payload).hexdigest():
            raise ValueError(f"Selection first-tier hash mismatch: {name}")
        snapshots[name] = payload
    report = selection["report"]
    if report.get("version") != "selection_v2_analysis_2":
        raise ValueError("Unexpected selection analysis version.")

    pairs = pd.read_csv(io.BytesIO(snapshots["first_tier_all_pair_tests.csv"]))
    _require_columns(pairs, (
        "scope", "holm_family", "method_a", "method_b",
        "macro_nrmse_a_minus_b", "block_swap_p_unadjusted_two_sided",
        "block_swap_p_holm_all_15",
    ), "first-tier all-pair tests")
    expected_pairs = {
        (scope, method_a, method_b)
        for scope in renderer.SCOPES
        for index, method_a in enumerate(renderer.METHODS)
        for method_b in renderer.METHODS[index + 1:]
    }
    observed_pairs = list(pairs[["scope", "method_a", "method_b"]]
                          .itertuples(index=False, name=None))
    if len(observed_pairs) != 30 or len(set(observed_pairs)) != 30 or \
            set(observed_pairs) != expected_pairs:
        raise ValueError("First-tier tests do not contain both complete 15-test families.")
    _finite(pairs, (
        "macro_nrmse_a_minus_b", "block_swap_p_unadjusted_two_sided",
        "block_swap_p_holm_all_15",
    ), "first-tier all-pair tests")
    for column in ("block_swap_p_unadjusted_two_sided", "block_swap_p_holm_all_15"):
        if np.any((pairs[column] < 0) | (pairs[column] > 1)):
            raise ValueError(f"{column} must lie in [0, 1].")
    scope_frame = selection["scope"]
    for row in pairs.itertuples(index=False):
        scope_rows = scope_frame[scope_frame["scope"] == row.scope].set_index("method")
        expected = (
            float(scope_rows.loc[row.method_a,
                                 "site_equal_macro_capacity_normalized_rmse_percent"]) -
            float(scope_rows.loc[row.method_b,
                                 "site_equal_macro_capacity_normalized_rmse_percent"])
        ) / 100
        if row.holm_family != f"{row.scope}_15_all_method_pairs" or \
                not _close(row.macro_nrmse_a_minus_b, expected):
            raise ValueError(f"First-tier effect mismatch for {row.scope}/{row.method_a}/{row.method_b}.")

    membership = pd.read_csv(io.BytesIO(snapshots["conservative_first_tier.csv"]))
    _require_columns(membership, (
        "scope", "method", "empirical_endpoint_winner",
        "macro_capacity_normalized_rmse", "difference_from_winner",
        "holm_p_vs_winner", "in_conservative_nonrejection_set", "interpretation",
    ), "first-tier membership")
    expected_membership = {
        (scope, method) for scope in renderer.SCOPES for method in renderer.METHODS
    }
    observed_membership = list(membership[["scope", "method"]]
                               .itertuples(index=False, name=None))
    if len(observed_membership) != 12 or len(set(observed_membership)) != 12 or \
            set(observed_membership) != expected_membership:
        raise ValueError("First-tier membership does not contain the complete 2 x 6 design.")
    _finite(membership, (
        "macro_capacity_normalized_rmse", "difference_from_winner", "holm_p_vs_winner",
    ), "first-tier membership")
    if np.any((membership["holm_p_vs_winner"] < 0) |
              (membership["holm_p_vs_winner"] > 1)):
        raise ValueError("First-tier membership p-values must lie in [0, 1].")

    for row in membership.itertuples(index=False):
        scope_rows = scope_frame[scope_frame["scope"] == row.scope]
        winner_row = scope_rows.sort_values(
            ["site_equal_macro_capacity_normalized_rmse_percent", "method"],
            kind="stable",
        ).iloc[0]
        # The analyzer breaks exact endpoint ties by its fixed METHODS order.
        minimum = scope_rows["site_equal_macro_capacity_normalized_rmse_percent"].min()
        tied = scope_rows[np.isclose(
            scope_rows["site_equal_macro_capacity_normalized_rmse_percent"], minimum,
            rtol=0, atol=1e-14,
        )]["method"].tolist()
        winner = min(tied, key=renderer.METHODS.index)
        if row.empirical_endpoint_winner != winner:
            raise ValueError(f"First-tier winner mismatch in {row.scope}.")
        method_row = scope_rows[scope_rows["method"] == row.method].iloc[0]
        method_macro = float(method_row.site_equal_macro_capacity_normalized_rmse_percent) / 100
        winner_macro = float(winner_row.site_equal_macro_capacity_normalized_rmse_percent) / 100
        if not _close(row.macro_capacity_normalized_rmse, method_macro) or \
                not _close(row.difference_from_winner, method_macro - winner_macro):
            raise ValueError(f"First-tier endpoint mismatch for {row.scope}/{row.method}.")
        included = _bool(row.in_conservative_nonrejection_set,
                         name="in_conservative_nonrejection_set")
        if row.method == winner:
            expected_p = 1.0
        else:
            match = pairs[(pairs["scope"] == row.scope) & (
                ((pairs["method_a"] == row.method) & (pairs["method_b"] == winner)) |
                ((pairs["method_a"] == winner) & (pairs["method_b"] == row.method))
            )]
            if len(match) != 1:
                raise ValueError("Missing membership-to-pair comparison.")
            expected_p = float(match.iloc[0].block_swap_p_holm_all_15)
        if not _close(row.holm_p_vs_winner, expected_p):
            raise ValueError(f"First-tier adjusted-p mismatch for {row.scope}/{row.method}.")
        expected_included = not (method_macro > winner_macro and expected_p < .05)
        if included != expected_included:
            raise ValueError(f"First-tier inclusion mismatch for {row.scope}/{row.method}.")

    json_pairs = report.get("first_tier_all_pair_tests")
    json_membership = report.get("conservative_first_tier")
    if not isinstance(json_pairs, list) or len(json_pairs) != 30 or \
            not isinstance(json_membership, list) or len(json_membership) != 12:
        raise ValueError("Selection JSON omits complete first-tier results.")
    pair_map = {(row.get("scope"), row.get("method_a"), row.get("method_b")): row
                for row in json_pairs}
    if len(pair_map) != 30:
        raise ValueError("Selection JSON contains duplicate first-tier tests.")
    for row in pairs.itertuples(index=False):
        item = pair_map.get((row.scope, row.method_a, row.method_b), {})
        for name in ("macro_nrmse_a_minus_b", "block_swap_p_unadjusted_two_sided",
                     "block_swap_p_holm_all_15"):
            if name not in item or not _close(item[name], getattr(row, name)):
                raise ValueError(f"Selection JSON/CSV first-tier mismatch for {row.scope}/{row.method_a}/{row.method_b}.")
    member_map = {(row.get("scope"), row.get("method")): row
                  for row in json_membership}
    if len(member_map) != 12:
        raise ValueError("Selection JSON contains duplicate first-tier memberships.")
    for row in membership.itertuples(index=False):
        item = member_map.get((row.scope, row.method), {})
        for name in ("macro_capacity_normalized_rmse", "difference_from_winner",
                     "holm_p_vs_winner"):
            if name not in item or not _close(item[name], getattr(row, name)):
                raise ValueError(f"Selection JSON/CSV membership mismatch for {row.scope}/{row.method}.")
        if item.get("empirical_endpoint_winner") != row.empirical_endpoint_winner or \
                _bool(item.get("in_conservative_nonrejection_set"),
                      name="JSON non-rejection membership") != _bool(
                          row.in_conservative_nonrejection_set,
                          name="CSV non-rejection membership"):
            raise ValueError(f"Selection JSON/CSV membership identity mismatch for {row.scope}/{row.method}.")
    if report.get("first_tier_caveat") != manifest.get("protocol", {}).get(
            "first_tier", {}).get("interpretation"):
        raise ValueError("First-tier caveat differs between frozen protocol and report.")

    return pairs, membership


def _validate_report_crosschecks(selection: dict, matched: dict) -> None:
    report = selection["report"]
    counts = report.get("reporting_counts", {})
    expected_seed_map = {method: [0, 1, 2] for method in renderer.METHODS[:-1]}
    expected_seed_map["HistGBR"] = [0]
    if counts.get("evaluation_seeds_by_method") != expected_seed_map or \
            counts.get("locked_result_records") != 192:
        raise ValueError("Selection analysis has unexpected evaluation seed or record counts.")

    point = selection["point"]
    for scope_name in renderer.SCOPES:
        for task_name in renderer.TASKS[scope_name]:
            rows = point[(point["scope"] == scope_name) & (point["task"] == task_name)]
            expected_ranks = rows["seed_mean_rmse_mw"].rank(method="average")
            if not np.allclose(expected_ranks, rows["task_rank"], rtol=0, atol=1e-12):
                raise ValueError(f"Selection task ranks are inconsistent for {task_name}.")

    for key, frame in (("primary_within_summary", selection["scope"]),
                       ("secondary_cross_summary", selection["scope"])):
        scope = "within" if key.startswith("primary") else "cross"
        rows = report.get(key)
        if not isinstance(rows, list) or len(rows) != 6:
            raise ValueError(f"Selection JSON lacks complete {key}.")
        by_method = {row.get("method"): row for row in rows}
        for csv_row in frame[frame["scope"] == scope].itertuples(index=False):
            json_row = by_method.get(csv_row.method, {})
            for name in ("site_equal_macro_capacity_normalized_rmse_percent",
                         "average_task_rank", "task_wins_including_ties"):
                if name not in json_row or not _close(json_row[name], getattr(csv_row, name)):
                    raise ValueError(f"Selection JSON/CSV mismatch for {scope}/{csv_row.method}/{name}.")
            task_rows = point[(point["scope"] == scope) &
                              (point["method"] == csv_row.method)]
            expected_wins = int((task_rows["task_rank"] == 1).sum())
            if int(csv_row.task_wins_including_ties) != expected_wins:
                raise ValueError(f"Selection win count mismatch for {scope}/{csv_row.method}.")

    json_inference = report.get("full_pairwise_inference")
    if not isinstance(json_inference, list) or len(json_inference) != 60:
        raise ValueError("Selection JSON omits complete Full comparisons.")
    json_map = {(row.get("scope"), row.get("task"), row.get("comparator")): row
                for row in json_inference}
    if len(json_map) != 60:
        raise ValueError("Selection JSON contains duplicate Full comparisons.")
    for row in selection["inference"].itertuples(index=False):
        item = json_map.get((row.scope, row.task, row.comparator), {})
        interval = item.get("time_block_bootstrap_95_ci_mw2")
        checks = (
            (item.get("full_minus_comparator_mse_mw2"), row.full_minus_comparator_mse_mw2),
            (item.get("block_sign_flip_p_holm_within_scope_30"),
             row.block_sign_flip_p_holm_within_scope_30),
        )
        if not isinstance(interval, list) or len(interval) != 2 or \
                any(a is None or not _close(a, b) for a, b in checks) or \
                not _close(interval[0], row.time_block_bootstrap_95_ci_low_mw2) or \
                not _close(interval[1], row.time_block_bootstrap_95_ci_high_mw2):
            raise ValueError(f"Selection JSON/CSV inference mismatch for {row.task}/{row.comparator}.")

    _require_columns(selection["inference"], (
        "role", "holm_family", "n_test",
        "full_evaluation_seeds_averaged_within_timestamp",
        "comparator_evaluation_seeds_averaged_within_timestamp",
        "calendar_blocks", "block_size_min", "block_size_max",
        "full_minus_comparator_normalized_mse", "block_sign_flip_p_unadjusted",
    ), "complete selection Full inference")
    _finite(selection["inference"], (
        "n_test", "full_evaluation_seeds_averaged_within_timestamp",
        "comparator_evaluation_seeds_averaged_within_timestamp", "calendar_blocks",
        "block_size_min", "block_size_max", "full_minus_comparator_normalized_mse",
        "block_sign_flip_p_unadjusted",
    ), "complete selection Full inference")
    for row in selection["inference"].itertuples(index=False):
        expected_role = "primary" if row.scope == "within" else "secondary_stress"
        expected_seeds = 1 if row.comparator == "HistGBR" else 3
        expected_normalized = (float(row.full_minus_comparator_mse_mw2) /
                               renderer.CAPACITY[int(row.site)] ** 2)
        if row.role != expected_role or row.holm_family != f"{row.scope}_30_full_comparisons" or \
                int(row.full_evaluation_seeds_averaged_within_timestamp) != 3 or \
                int(row.comparator_evaluation_seeds_averaged_within_timestamp) != expected_seeds or \
                int(row.n_test) <= 0 or int(row.calendar_blocks) <= 0 or \
                int(row.block_size_min) <= 0 or int(row.block_size_max) < int(row.block_size_min) or \
                not _close(row.full_minus_comparator_normalized_mse, expected_normalized):
            raise ValueError(f"Complete selection inference fields are inconsistent for {row.task}/{row.comparator}.")
        if not 0 <= float(row.block_sign_flip_p_unadjusted) <= 1:
            raise ValueError("Selection unadjusted p-values must lie in [0, 1].")

    summary = matched["summary"]
    if summary.get("version") != "matched_ablation_v2_analysis_results_1":
        raise ValueError("Unexpected matched-ablation analysis version.")
    json_aggregate = summary.get("aggregate_metrics")
    if not isinstance(json_aggregate, list) or len(json_aggregate) != 8:
        raise ValueError("Matched JSON omits complete aggregate metrics.")
    aggregate_map = {(row.get("scope"), row.get("variant")): row
                     for row in json_aggregate}
    for row in matched["aggregate"].itertuples(index=False):
        item = aggregate_map.get((row.scope, row.variant), {})
        if any(name not in item or not _close(item[name], getattr(row, name))
               for name in ("sites", "macro_normalized_rmse", "mean_rank")):
            raise ValueError(f"Matched JSON/CSV aggregate mismatch for {row.scope}/{row.variant}.")
    for scope_name in renderer.SCOPES:
        for task_name in renderer.TASKS[scope_name]:
            rows = matched["task"][(matched["task"]["scope"] == scope_name) &
                                   (matched["task"]["task"] == task_name)]
            expected_ranks = rows["mean_seed_rmse_mw"].rank(method="average")
            if not np.allclose(expected_ranks, rows["rank"], rtol=0, atol=1e-12):
                raise ValueError(f"Matched task ranks are inconsistent for {task_name}.")
    _require_columns(matched["inference"], (
        "n", "blocks", "p_unadjusted", "reject_holm_0_05",
    ), "complete matched inference")
    _finite(matched["inference"], ("n", "blocks", "p_unadjusted"),
            "complete matched inference")
    for row in matched["inference"].itertuples(index=False):
        recorded = _bool(row.reject_holm_0_05, name="reject_holm_0_05")
        if int(row.n) <= 0 or int(row.blocks) <= 0 or \
                not 0 <= float(row.p_unadjusted) <= 1:
            raise ValueError(f"Matched inference counts or p-value are invalid for {row.task}/{row.comparison}.")
        if recorded != (float(row.p_holm) <= .05):
            raise ValueError(f"Matched reject flag mismatch for {row.task}/{row.comparison}.")
    # Check the required semantics directly because punctuation is frozen upstream.
    interpretation = str(summary.get("interpretation", ""))
    required = ("conditional on each task's locked Full configuration",
                "not independently retuned", "failure to reject")
    if not all(fragment in interpretation for fragment in required):
        raise ValueError("Matched summary omits the frozen conditional interpretation.")


def _scope_summaries(selection: dict, membership: pd.DataFrame) -> list[dict]:
    result = []
    for scope in renderer.SCOPES:
        rows = selection["scope"][selection["scope"]["scope"] == scope].copy()
        rows = rows.sort_values(
            ["site_equal_macro_capacity_normalized_rmse_percent", "average_task_rank", "method"],
            kind="stable",
        )
        member_map = {
            row.method: _bool(row.in_conservative_nonrejection_set,
                              name="in_conservative_nonrejection_set")
            for row in membership[membership["scope"] == scope].itertuples(index=False)
        }
        methods = []
        for position, row in enumerate(rows.itertuples(index=False), start=1):
            methods.append({
                "macro_nrmse_position": position,
                "method": row.method,
                "method_label": renderer.METHOD_LABELS[row.method],
                "macro_capacity_normalized_rmse_percent":
                    float(row.site_equal_macro_capacity_normalized_rmse_percent),
                "average_task_rank": float(row.average_task_rank),
                "task_wins_including_ties": int(row.task_wins_including_ties),
                "in_conservative_nonrejection_set": member_map[row.method],
            })
        result.append({
            "scope": scope,
            "role": "primary_within_farm" if scope == "within" else "secondary_cross_farm_stress",
            "tasks": 6,
            "methods_sorted_by_macro_nrmse": methods,
            "membership_caveat": FIRST_TIER_CAVEAT,
        })
    return result


def _full_relations(selection: dict, pairs: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    scope_relations, task_relations = [], []
    scope_frame = selection["scope"]
    inference = selection["inference"]
    for scope in renderer.SCOPES:
        rows = scope_frame[scope_frame["scope"] == scope].set_index("method")
        full = rows.loc["Full"]
        for comparator in renderer.COMPARATORS:
            other = rows.loc[comparator]
            delta = (float(full.site_equal_macro_capacity_normalized_rmse_percent) -
                     float(other.site_equal_macro_capacity_normalized_rmse_percent))
            task_rows = inference[(inference["scope"] == scope) &
                                  (inference["comparator"] == comparator)]
            scope_test = pairs[(pairs["scope"] == scope) & (
                ((pairs["method_a"] == "Full") & (pairs["method_b"] == comparator)) |
                ((pairs["method_a"] == comparator) & (pairs["method_b"] == "Full"))
            )]
            if len(scope_test) != 1:
                raise ValueError(f"Missing scope-level Full comparison for {scope}/{comparator}.")
            scope_p = float(scope_test.iloc[0].block_swap_p_holm_all_15)
            directions = task_rows["full_minus_comparator_mse_mw2"].map(_direction)
            significant = task_rows["block_sign_flip_p_holm_within_scope_30"] <= .05
            scope_relations.append({
                "scope": scope,
                "comparator": comparator,
                "comparator_label": renderer.METHOD_LABELS[comparator],
                "full_macro_nrmse_percent":
                    float(full.site_equal_macro_capacity_normalized_rmse_percent),
                "comparator_macro_nrmse_percent":
                    float(other.site_equal_macro_capacity_normalized_rmse_percent),
                "full_minus_comparator_macro_nrmse_percentage_points": delta,
                "observed_direction": _macro_direction(delta),
                "scope_all_pair_holm_adjusted_p": scope_p,
                "scope_difference_significant_at_0_05": scope_p <= .05,
                "task_direction_counts": {
                    "Full_lower_loss": int((directions == "Full_lower_loss").sum()),
                    "Full_higher_loss": int((directions == "Full_higher_loss").sum()),
                    "equal_observed_loss": int((directions == "equal_observed_loss").sum()),
                },
                "holm_task_counts": {
                    "significant": int(significant.sum()),
                    "not_significant": int((~significant).sum()),
                },
            })
    for row in inference.itertuples(index=False):
        effect = float(row.full_minus_comparator_mse_mw2)
        pvalue = float(row.block_sign_flip_p_holm_within_scope_30)
        task_relations.append({
            "scope": row.scope,
            "task": row.task,
            "site": int(row.site),
            "comparator": row.comparator,
            "comparator_label": renderer.METHOD_LABELS[row.comparator],
            "role": row.role,
            "holm_family": row.holm_family,
            "n_test": int(row.n_test),
            "full_evaluation_seeds_averaged_within_timestamp":
                int(row.full_evaluation_seeds_averaged_within_timestamp),
            "comparator_evaluation_seeds_averaged_within_timestamp":
                int(row.comparator_evaluation_seeds_averaged_within_timestamp),
            "calendar_blocks": int(row.calendar_blocks),
            "block_size_min": int(row.block_size_min),
            "block_size_max": int(row.block_size_max),
            "full_minus_comparator_mse_mw2": effect,
            "full_minus_comparator_normalized_mse":
                float(row.full_minus_comparator_normalized_mse),
            "bootstrap_95_ci_mw2": [
                float(row.time_block_bootstrap_95_ci_low_mw2),
                float(row.time_block_bootstrap_95_ci_high_mw2),
            ],
            "unadjusted_p": float(row.block_sign_flip_p_unadjusted),
            "holm_adjusted_p": pvalue,
            "significant_at_0_05": pvalue <= .05,
            "observed_direction": _direction(effect),
        })
    return scope_relations, task_relations


def _matched_results(matched: dict) -> tuple[list[dict], list[dict]]:
    aggregates, comparisons = [], []
    for row in matched["aggregate"].itertuples(index=False):
        aggregates.append({
            "scope": row.scope,
            "variant": row.variant,
            "variant_label": renderer.METHOD_LABELS[row.variant],
            "sites": int(row.sites),
            "macro_capacity_normalized_rmse_percent": 100 * float(row.macro_normalized_rmse),
            "mean_rank": float(row.mean_rank),
        })
    for row in matched["inference"].itertuples(index=False):
        effect = float(row.mean_squared_error_difference_mw2)
        pvalue = float(row.p_holm)
        comparisons.append({
            "scope": row.scope,
            "task": row.task,
            "site": int(row.site),
            "comparison": row.comparison,
            "n_test": int(row.n),
            "calendar_blocks": int(row.blocks),
            "full_minus_removal_mse_mw2": effect,
            "bootstrap_95_ci_mw2": [float(row.ci_lower_mw2), float(row.ci_upper_mw2)],
            "unadjusted_p": float(row.p_unadjusted),
            "holm_adjusted_p": pvalue,
            "significant_at_0_05": pvalue <= .05,
            "observed_direction": _direction(effect),
        })
    return aggregates, comparisons


def _sentences(scopes: list[dict], scope_relations: list[dict],
               matched_aggregates: list[dict]) -> list[str]:
    lines = []
    for item in scopes:
        full = next(row for row in item["methods_sorted_by_macro_nrmse"]
                    if row["method"] == "Full")
        scope_label = "primary within-farm" if item["scope"] == "within" \
            else "secondary cross-farm stress"
        membership = "was" if full["in_conservative_nonrejection_set"] else "was not"
        lines.append(
            f"Across the six {scope_label} tasks, TL-QLDMR recorded a site-equal "
            f"macro capacity-normalized RMSE of "
            f"{full['macro_capacity_normalized_rmse_percent']:.3f}%, an average task "
            f"rank of {full['average_task_rank']:.2f}, and "
            f"{full['task_wins_including_ties']} task wins (including ties); it placed "
            f"{full['macro_nrmse_position']} of six by the observed macro endpoint and "
            f"{membership} in the conservative non-rejection set."
        )
    for item in scope_relations:
        delta = item["full_minus_comparator_macro_nrmse_percentage_points"]
        relation = "lower" if delta < 0 else "higher" if delta > 0 else "equal to"
        magnitude = abs(delta)
        scope_label = "within-farm" if item["scope"] == "within" else "cross-farm stress"
        if relation == "equal to":
            phrase = "equal to the observed value for"
        else:
            phrase = f"{magnitude:.3f} percentage points {relation} than"
        lines.append(
            f"In the {scope_label} scope, TL-QLDMR's macro capacity-normalized RMSE "
            f"was {phrase} {item['comparator_label']}; among the six task-level paired "
            f"tests, {item['holm_task_counts']['significant']} were significant after "
            f"Holm adjustment and {item['holm_task_counts']['not_significant']} were not."
        )
    for scope in renderer.SCOPES:
        rows = [row for row in matched_aggregates if row["scope"] == scope]
        full = next(row for row in rows if row["variant"] == "Full")
        removals = ", ".join(
            f"{row['variant_label']} {row['macro_capacity_normalized_rmse_percent']:.3f}%"
            for row in rows if row["variant"] != "Full"
        )
        label = "within-farm" if scope == "within" else "cross-farm stress"
        lines.append(
            f"Under the matched, conditional component-removal protocol in the {label} "
            f"scope, Full recorded macro capacity-normalized RMSE "
            f"{full['macro_capacity_normalized_rmse_percent']:.3f}% and mean rank "
            f"{full['mean_rank']:.2f}, compared with {removals}."
        )
    return lines


def build_evidence(selection_root: Path, matched_root: Path) -> tuple[dict, dict[str, str]]:
    selection_root, matched_root = Path(selection_root).resolve(), Path(matched_root).resolve()
    selection, matched = renderer.load_validated_inputs(selection_root, matched_root)
    pairs, membership = _verify_extra_selection(selection_root, selection)
    _validate_report_crosschecks(selection, matched)

    scopes = _scope_summaries(selection, membership)
    scope_relations, task_relations = _full_relations(selection, pairs)
    matched_aggregates, matched_inference = _matched_results(matched)
    sentences = _sentences(scopes, scope_relations, matched_aggregates)

    seed_map = selection["report"]["reporting_counts"]["evaluation_seeds_by_method"]
    kernel_methods = len(renderer.METHODS) - 1
    selection_queries = 12 * 6 * 54
    q1_fit_records = 12 * (kernel_methods * 54 * 3 + 54)
    counts = {
        "scopes": 2,
        "tasks_total": 12,
        "tasks_by_scope": {"within": 6, "cross": 6},
        "methods": 6,
        "task_method_rows": 72,
        "configuration_queries_per_task_method": 54,
        "configuration_queries_total": selection_queries,
        "q1_selection_fit_records_total": q1_fit_records,
        "evaluation_seed_labels_by_method": seed_map,
        "locked_main_evaluation_records": 192,
        "main_evaluation_records_by_method": {
            method: 12 * len(seeds) for method, seeds in seed_map.items()
        },
        "full_task_comparisons": 60,
        "first_tier_all_pair_tests": int(len(pairs)),
        "first_tier_membership_rows": int(len(membership)),
        "matched_variants": 4,
        "matched_task_variant_rows": 48,
        "matched_evaluation_seeds_per_variant": 3,
        "matched_evaluation_records": 144,
        "matched_task_comparisons": 36,
        "heldout_evaluation_records_total": 336,
    }
    evidence = {
        "version": "revision_evidence_v2_1",
        "scope_policy": (
            "Complete reporting of all six primary within-farm tasks and all six "
            "secondary cross-farm stress tasks; no task, method, seed, comparison, "
            "or effect direction is filtered."
        ),
        "counts": counts,
        "scope_method_results": scopes,
        "first_tier_caveat": FIRST_TIER_CAVEAT,
        "full_scope_relations": scope_relations,
        "full_task_comparisons": task_relations,
        "matched_ablation": {
            "conditional_caveat": MATCHED_CAVEAT,
            "aggregate_results": matched_aggregates,
            "all_task_comparisons": matched_inference,
        },
        "manuscript_ready_numeric_sentences": sentences,
    }
    input_files = [
        selection_root / "analysis_manifest.json",
        *(selection_root / name for name in renderer.SELECTION_FILES),
        *(selection_root / name for name in SELECTION_EXTRA_FILES),
        matched_root / "manifest.json",
        *(matched_root / name for name in renderer.MATCHED_FILES),
    ]
    hashes = {str(path): digest(path) for path in input_files}
    return evidence, hashes


def _markdown(evidence: dict) -> str:
    lines = [
        "# Complete v2 revision evidence",
        "",
        evidence["scope_policy"],
        "",
        "## Exact design counts",
        "",
        "```json",
        json.dumps(evidence["counts"], ensure_ascii=False, indent=2),
        "```",
    ]
    for scope in evidence["scope_method_results"]:
        label = "Primary within-farm" if scope["scope"] == "within" \
            else "Secondary cross-farm stress"
        lines.extend([
            "", f"## {label}: all six methods", "",
            "| Position | Method | Macro NRMSE (%) | Average rank | Wins | Non-rejection member |",
            "|---:|---|---:|---:|---:|:---:|",
        ])
        for row in scope["methods_sorted_by_macro_nrmse"]:
            lines.append(
                f"| {row['macro_nrmse_position']} | {row['method_label']} | "
                f"{row['macro_capacity_normalized_rmse_percent']:.6f} | "
                f"{row['average_task_rank']:.6f} | {row['task_wins_including_ties']} | "
                f"{'yes' if row['in_conservative_nonrejection_set'] else 'no'} |"
            )
        lines.extend(["", scope["membership_caveat"]])

    lines.extend(["", "## Full relations at scope level", "",
                  "| Scope | Comparator | Full (%) | Comparator (%) | Difference (pp) | Direction | Scope Holm p | Scope significant | Significant tasks | Non-significant tasks |",
                  "|---|---|---:|---:|---:|---|---:|:---:|---:|---:|"])
    for row in evidence["full_scope_relations"]:
        lines.append(
            f"| {row['scope']} | {row['comparator_label']} | "
            f"{row['full_macro_nrmse_percent']:.6f} | "
            f"{row['comparator_macro_nrmse_percent']:.6f} | "
            f"{row['full_minus_comparator_macro_nrmse_percentage_points']:.6f} | "
            f"{row['observed_direction']} | {row['scope_all_pair_holm_adjusted_p']:.9g} | "
            f"{'yes' if row['scope_difference_significant_at_0_05'] else 'no'} | "
            f"{row['holm_task_counts']['significant']} | "
            f"{row['holm_task_counts']['not_significant']} |"
        )

    lines.extend(["", "## All 60 Full task comparisons", "",
                  "| Scope | Task | Comparator | n | Blocks | MSE difference (MW²) | 95% CI | Raw p | Holm p | Significant | Direction |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---|"])
    for row in evidence["full_task_comparisons"]:
        interval = row["bootstrap_95_ci_mw2"]
        lines.append(
            f"| {row['scope']} | {row['task']} | {row['comparator_label']} | "
            f"{row['n_test']} | {row['calendar_blocks']} | "
            f"{row['full_minus_comparator_mse_mw2']:.9g} | "
            f"[{interval[0]:.9g}, {interval[1]:.9g}] | {row['unadjusted_p']:.9g} | "
            f"{row['holm_adjusted_p']:.9g} | "
            f"{'yes' if row['significant_at_0_05'] else 'no'} | {row['observed_direction']} |"
        )

    lines.extend(["", "## Matched component removal", "", evidence["matched_ablation"]["conditional_caveat"], "",
                  "| Scope | Variant | Macro NRMSE (%) | Mean rank |",
                  "|---|---|---:|---:|"])
    for row in evidence["matched_ablation"]["aggregate_results"]:
        lines.append(
            f"| {row['scope']} | {row['variant_label']} | "
            f"{row['macro_capacity_normalized_rmse_percent']:.6f} | {row['mean_rank']:.6f} |"
        )
    lines.extend(["", "### All 36 matched task comparisons", "",
                  "| Scope | Task | Comparison | n | Blocks | MSE difference (MW²) | 95% CI | Raw p | Holm p | Significant | Direction |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|:---:|---|"])
    for row in evidence["matched_ablation"]["all_task_comparisons"]:
        interval = row["bootstrap_95_ci_mw2"]
        lines.append(
            f"| {row['scope']} | {row['task']} | {row['comparison']} | "
            f"{row['n_test']} | {row['calendar_blocks']} | "
            f"{row['full_minus_removal_mse_mw2']:.9g} | "
            f"[{interval[0]:.9g}, {interval[1]:.9g}] | {row['unadjusted_p']:.9g} | "
            f"{row['holm_adjusted_p']:.9g} | "
            f"{'yes' if row['significant_at_0_05'] else 'no'} | {row['observed_direction']} |"
        )
    lines.extend(["", "## Manuscript-ready numeric sentences", ""])
    lines.extend(f"- {sentence}" for sentence in evidence["manuscript_ready_numeric_sentences"])
    lines.append("")
    return "\n".join(lines)


def summarize(selection_root: Path, matched_root: Path, output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory must not already exist: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence, input_hashes = build_evidence(selection_root, matched_root)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp.", dir=output.parent))
    try:
        (staging / "revision_evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "revision_evidence.md").write_text(_markdown(evidence), encoding="utf-8")
        manifest = {
            "version": "revision_evidence_manifest_v2_1",
            "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(Path(__file__).resolve().relative_to(ROOT)),
            "source_sha256": digest(Path(__file__)),
            "input_policy": "completed analysis artifacts only; no held-out result or prediction files read",
            "inputs": input_hashes,
            "outputs": {name: digest(staging / name) for name in OUTPUT_FILES},
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name, expected_hash in input_hashes.items():
            path = Path(name)
            if not path.is_file() or digest(path) != expected_hash:
                raise ValueError(f"Validated analysis input changed before publication: {path}")
        if output.exists():
            raise FileExistsError(f"Output directory appeared during publication: {output}")
        staging.rename(output)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-analysis", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--matched-analysis", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; it must not already exist.")
    args = parser.parse_args()
    summarize(args.selection_analysis, args.matched_analysis, args.output)


if __name__ == "__main__":
    main()
