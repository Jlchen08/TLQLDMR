#!/usr/bin/env python3
"""Integrate verified v2 prose into the revised manuscript.

The tool is deliberately downstream-only.  It reads one completed prose JSON,
one selection-study template, and four manuscript source files.  It never
opens run directories, prediction files, or per-fit records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


VERSION = "integrate_v2_manuscript_prose_1"
RESULT_VERSION = "v2_result_prose_1"
RESULT_BLOCKS = (
    "RESULTS_POINT",
    "RESULTS_TASK_INFERENCE",
    "RESULTS_FIRST_TIER",
    "RESULTS_MATCHED_ABLATION",
    "RESULTS_OPTIMIZATION",
)
TARGET_NAMES = (
    "selection_study.tex",
    "TLQLDMR_energy.tex",
    "Highlights.tex",
    "selection_appendix.tex",
)
SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent

# First-run guards for the authoritative revised sources that existed when this
# integrator was written.  Once a generated marker is present, its payload is
# updated idempotently and the historical hash is no longer consulted.
EXPECTED_OLD_ABSTRACT_SHA256 = (
    "7560313f7677bdf7fa77008c24ff843731abc3a64bac923c0e7f9ad685ae8136"
)
EXPECTED_OLD_CONTRIBUTION_SHA256 = (
    "882122d6a2023ca380610525a36b9398572a042bd0779ec570ff94e4186e0c8c"
)
EXPECTED_OLD_CONCLUSION_SHA256 = (
    "e13e026526a11daf1f7b3c344ee91eeee0987494667dbc79605257a8c1c8ef28"
)
EXPECTED_OLD_HIGHLIGHTS_SHA256 = (
    "a6fc3b331f61a785d7e082cbd2f7d4c374bb915c3207c2655a29b6709e26c188"
)

GENERATED_BEGIN = "% BEGIN V2 GENERATED {name}"
GENERATED_END = "% END V2 GENERATED {name}"


class IntegrationError(RuntimeError):
    """A fail-closed input or integration error."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise IntegrationError(f"Cannot read required input {path}: {error}") from error


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise IntegrationError(f"{label} must be a JSON object.")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise IntegrationError(f"{label} must be a JSON array.")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationError(f"{label} must be a nonempty string.")
    return value.strip()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise IntegrationError(f"{label} must be numeric.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise IntegrationError(f"{label} must be numeric.") from error
    if not math.isfinite(result):
        raise IntegrationError(f"{label} must be finite.")
    return result


def _integer(value: Any, label: str, low: int, high: int) -> int:
    number = _number(value, label)
    result = int(number)
    if number != result or not low <= result <= high:
        raise IntegrationError(f"{label} must be an integer in [{low}, {high}].")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise IntegrationError(f"{label} must be boolean.")
    return value


def _unique_index(items: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in items:
        value = _string(item.get(key), f"{label}.{key}")
        if value in output:
            raise IntegrationError(f"Duplicate {label} key: {value}")
        output[value] = item
    return output


def _method_name(method: str, supplied: str | None = None) -> str:
    names = {
        "Full": "TL-QLDMR",
        "NoVariance": "the no-variance variant",
        "NoDiscrepancy": "the no-discrepancy variant",
        "WeightedSVQR": "pooled weighted support vector quantile regression",
        "TargetSVQR": "target-only support vector quantile regression",
        "HistGBR": "histogram gradient boosting",
    }
    if method not in names:
        raise IntegrationError(f"Unknown method identifier: {method}")
    # The fixed map prevents an upstream display label from silently introducing
    # an undefined abbreviation into front matter.
    return names[method]


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _tex_percent(value: float) -> str:
    return f"{value:.3f}\\%"


def _has_unescaped_percent(text: str) -> bool:
    """Return True when TeX would interpret a percent sign as a comment."""
    for index, character in enumerate(text):
        if character != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return True
    return False


def _load_results(path: Path) -> tuple[
    dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], str
]:
    source_text = _read_text(path)
    try:
        payload = json.loads(source_text)
    except json.JSONDecodeError as error:
        raise IntegrationError(f"Invalid result JSON {path}: {error}") from error
    payload = _object(payload, "result payload")
    if payload.get("version") != RESULT_VERSION:
        raise IntegrationError(
            f"Expected result version {RESULT_VERSION!r}, got {payload.get('version')!r}."
        )
    input_policy = _string(payload.get("input_policy"), "input_policy")
    if "completed hash-verified" not in input_policy.lower():
        raise IntegrationError(
            "Result JSON does not identify itself as derived from completed hash-verified inputs."
        )
    source_hashes = _object(payload.get("source_sha256"), "source_sha256")
    if not source_hashes or any(
        not isinstance(name, str) or
        not isinstance(value, str) or
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for name, value in source_hashes.items()
    ):
        raise IntegrationError("source_sha256 must contain nonempty audited SHA-256 entries.")

    blocks = _object(payload.get("manuscript_blocks"), "manuscript_blocks")
    if set(blocks) != set(RESULT_BLOCKS):
        raise IntegrationError(
            "manuscript_blocks must contain exactly: " + ", ".join(RESULT_BLOCKS)
        )
    for name in RESULT_BLOCKS:
        text = _string(blocks[name], f"manuscript_blocks.{name}")
        if "@@RESULTS_" in text:
            raise IntegrationError(f"Result block {name} contains a result placeholder.")
        if _has_unescaped_percent(text):
            raise IntegrationError(
                f"Result block {name} contains an unescaped TeX percent sign."
            )
        if re.search(r"(?<![A-Za-z])Full(?![A-Za-z])", text):
            raise IntegrationError(
                f"Result block {name} exposes the internal method identifier 'Full'; "
                "use TL-QLDMR in manuscript prose."
            )

    scopes = _unique_index(
        [_object(item, "scope_results item") for item in _list(payload.get("scope_results"), "scope_results")],
        "scope", "scope_results",
    )
    if set(scopes) != {"within", "cross"}:
        raise IntegrationError("scope_results must contain exactly within and cross.")
    for scope, row in scopes.items():
        expected_role = (
            "primary_within_farm"
            if scope == "within"
            else "secondary_cross_farm_stress"
        )
        if row.get("role") != expected_role:
            raise IntegrationError(f"{scope} must have role {expected_role!r}.")
        winner = _object(row.get("empirical_winner"), f"scope_results.{scope}.empirical_winner")
        full = _object(row.get("full"), f"scope_results.{scope}.full")
        _method_name(_string(winner.get("method"), f"{scope}.winner.method"))
        _number(winner.get("macro_capacity_normalized_rmse_percent"), f"{scope}.winner.macro")
        macro = _number(full.get("macro_capacity_normalized_rmse_percent"), f"{scope}.full.macro")
        if macro < 0:
            raise IntegrationError(f"{scope}.full.macro must be nonnegative.")
        average_rank = _number(full.get("average_task_rank"), f"{scope}.full.rank")
        if not 1 <= average_rank <= 6:
            raise IntegrationError(f"{scope}.full.rank must lie in [1, 6].")
        _integer(full.get("task_wins_including_ties"), f"{scope}.full.wins", 0, 6)
        _integer(full.get("macro_nrmse_position"), f"{scope}.full.position", 1, 6)
        _boolean(full.get("in_conservative_nonrejection_set"), f"{scope}.full.first_tier")
        members = _list(row.get("first_tier_members"), f"{scope}.first_tier_members")
        member_methods = {
            _string(_object(item, f"{scope}.member").get("method"), f"{scope}.member.method")
            for item in members
        }
        if not member_methods:
            raise IntegrationError(f"{scope} first tier cannot be empty.")
        if winner["method"] not in member_methods:
            raise IntegrationError(f"{scope} empirical winner must belong to the first tier.")
        if ("Full" in member_methods) != full["in_conservative_nonrejection_set"]:
            raise IntegrationError(f"{scope} TL-QLDMR first-tier flag disagrees with membership.")

    matched = _object(payload.get("matched_ablation"), "matched_ablation")
    matched_scopes = _unique_index(
        [_object(item, "matched scope item") for item in _list(matched.get("scope_results"), "matched.scope_results")],
        "scope", "matched.scope_results",
    )
    if set(matched_scopes) != {"within", "cross"}:
        raise IntegrationError("matched scope_results must contain exactly within and cross.")
    for scope, row in matched_scopes.items():
        full = _object(row.get("full"), f"matched.{scope}.full")
        _number(full.get("macro_capacity_normalized_rmse_percent"), f"matched.{scope}.full.macro")
        mean_rank = _number(full.get("mean_rank"), f"matched.{scope}.full.rank")
        if not 1 <= mean_rank <= 4:
            raise IntegrationError(f"matched.{scope}.full.rank must lie in [1, 4].")
        _integer(full.get("macro_nrmse_position"), f"matched.{scope}.full.position", 1, 4)
        variants = _list(row.get("variants_sorted_by_macro_nrmse"), f"matched.{scope}.variants")
        if len(variants) != 4:
            raise IntegrationError(f"matched.{scope} must retain all four variants.")
        identifiers = {
            _string(_object(item, "matched variant").get("variant"), "matched variant identifier")
            for item in variants
        }
        if identifiers != {"Full", "NoVariance", "NoDiscrepancy", "WeightedSVQR"}:
            raise IntegrationError(f"matched.{scope} has an incomplete variant set.")

    caveats = _object(payload.get("interpretation_rules"), "interpretation_rules")
    tier_caveat = _string(caveats.get("first_tier"), "interpretation_rules.first_tier").lower()
    if "does not establish" not in tier_caveat or "equivalence" not in tier_caveat:
        raise IntegrationError("First-tier caveat must explicitly reject an equivalence interpretation.")
    matched_caveat = _string(caveats.get("matched"), "interpretation_rules.matched").lower()
    if "conditional" not in matched_caveat or "not independently retuned" not in matched_caveat:
        raise IntegrationError("Matched caveat must state conditionality and lack of independent retuning.")

    inference = _object(payload.get("full_task_inference"), "full_task_inference")
    for name in ("significant_favorable_to_full", "significant_adverse_to_full"):
        _list(inference.get(name), f"full_task_inference.{name}")
    for name in (
        "significant_negative_full_minus_removal",
        "significant_positive_full_minus_removal",
    ):
        _list(matched.get(name), f"matched_ablation.{name}")

    return payload, scopes, matched_scopes, _sha256_text(source_text)


def _tier_status(scope: dict[str, Any]) -> str:
    winner = _object(scope["empirical_winner"], "empirical winner")
    full = _object(scope["full"], "full scope result")
    if winner["method"] == "Full":
        return "was the empirical endpoint winner and therefore entered the conservative first tier"
    if full["in_conservative_nonrejection_set"]:
        return "was retained in the conservative first tier"
    return "was excluded from the conservative first tier"


def _scope_sentence(scope_name: str, row: dict[str, Any]) -> str:
    label = "primary within-site" if scope_name == "within" else "secondary cross-site stress"
    winner = _object(row["empirical_winner"], f"{scope_name} winner")
    full = _object(row["full"], f"{scope_name} full")
    wins = _integer(full["task_wins_including_ties"], "full task wins", 0, 6)
    win_label = "task win" if wins == 1 else "task wins"
    return (
        f"In the {label} scope, {_method_name(winner['method'])} had the lowest observed "
        f"site-equal capacity-normalized root mean squared error at "
        f"{_tex_percent(_number(winner['macro_capacity_normalized_rmse_percent'], 'winner macro'))}. "
        f"TL-QLDMR recorded {_tex_percent(_number(full['macro_capacity_normalized_rmse_percent'], 'full macro'))}, "
        f"an average task rank of {_number(full['average_task_rank'], 'full rank'):.2f}, and "
        f"{wins} {win_label} including ties; "
        f"it ranked {_ordinal(_integer(full['macro_nrmse_position'], 'full position', 1, 6))} of six "
        f"and {_tier_status(row)}."
    )


def _direction_summary(payload: dict[str, Any]) -> str:
    inference = _object(payload["full_task_inference"], "full task inference")
    favorable = len(_list(inference["significant_favorable_to_full"], "favorable comparisons"))
    adverse = len(_list(inference["significant_adverse_to_full"], "adverse comparisons"))
    finding_label = "finding" if favorable == 1 else "findings"
    return (
        f"Among 60 predeclared task-level comparisons, {favorable} Holm-adjusted {finding_label} "
        f"favored TL-QLDMR and {adverse} favored a comparator; non-rejection for the remaining "
        f"comparisons does not establish equal performance."
    )


def _matched_sentence(scope_name: str, row: dict[str, Any]) -> str:
    label = "primary within-site" if scope_name == "within" else "secondary cross-site stress"
    full = _object(row["full"], f"matched {scope_name} full")
    variants = [_object(item, "matched variant") for item in row["variants_sorted_by_macro_nrmse"]]
    winner = variants[0]
    full_rank = _number(full["mean_rank"], "matched full rank")
    mean_rank_position = 1 + sum(
        _number(item["mean_rank"], "matched variant mean rank") < full_rank
        for item in variants
    )
    mean_rank_text = (
        f"the lowest mean task rank ({full_rank:.2f})"
        if mean_rank_position == 1
        else f"mean task rank {full_rank:.2f} ({_ordinal(mean_rank_position)} of four)"
    )
    return (
        f"In the conditional matched-removal analysis for the {label} scope, "
        f"TL-QLDMR recorded {_tex_percent(_number(full['macro_capacity_normalized_rmse_percent'], 'matched full macro'))} "
        f"and ranked {_ordinal(_integer(full['macro_nrmse_position'], 'matched full position', 1, 4))} "
        f"of four by macro RMSE, with {mean_rank_text}; "
        f"the lowest observed matched endpoint was {_method_name(_string(winner['variant'], 'matched winner'))} "
        f"at {_tex_percent(_number(winner['macro_capacity_normalized_rmse_percent'], 'matched winner macro'))}."
    )


def _matched_direction_summary(payload: dict[str, Any]) -> str:
    matched = _object(payload["matched_ablation"], "matched ablation")
    favorable = len(_list(matched["significant_negative_full_minus_removal"], "matched favorable"))
    adverse = len(_list(matched["significant_positive_full_minus_removal"], "matched adverse"))
    return (
        f"Across 36 matched task-level contrasts, {favorable} Holm-adjusted effects favored "
        f"TL-QLDMR and {adverse} favored a removal variant. These effects are conditional on "
        f"the TL-QLDMR-selected settings; the variants were not independently retuned, and "
        f"non-rejection does not establish equivalence."
    )


def _brief_tier_status(scope: dict[str, Any]) -> str:
    winner = _object(scope["empirical_winner"], "empirical winner")
    full = _object(scope["full"], "full scope result")
    if winner["method"] == "Full":
        return "was the empirical winner"
    if full["in_conservative_nonrejection_set"]:
        return "remained in the conservative first tier"
    return "was outside the conservative first tier"


def _brief_scope_sentence(scope_name: str, row: dict[str, Any]) -> str:
    label = "primary within-site" if scope_name == "within" else "secondary cross-site"
    winner = _object(row["empirical_winner"], f"{scope_name} winner")
    full = _object(row["full"], f"{scope_name} full")
    return (
        f"In the {label} scope, {_method_name(winner['method'])} was the empirical winner for "
        f"site-equal capacity-normalized root mean squared error at "
        f"{_tex_percent(_number(winner['macro_capacity_normalized_rmse_percent'], 'winner macro'))}; "
        f"TL-QLDMR achieved {_tex_percent(_number(full['macro_capacity_normalized_rmse_percent'], 'full macro'))}, "
        f"ranked {_ordinal(_integer(full['macro_nrmse_position'], 'full position', 1, 6))} of six, "
        f"and {_brief_tier_status(row)}."
    )


def _frontmatter(payload: dict[str, Any], scopes: dict[str, dict[str, Any]],
                 matched_scopes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = _scope_sentence("within", scopes["within"])
    secondary = _scope_sentence("cross", scopes["cross"])
    directions = _direction_summary(payload)
    matched_primary = _matched_sentence("within", matched_scopes["within"])
    matched_secondary = _matched_sentence("cross", matched_scopes["cross"])
    matched_directions = _matched_direction_summary(payload)

    inference = _object(payload["full_task_inference"], "full task inference")
    task_favorable = len(_list(inference["significant_favorable_to_full"], "favorable comparisons"))
    task_adverse = len(_list(inference["significant_adverse_to_full"], "adverse comparisons"))
    matched = _object(payload["matched_ablation"], "matched ablation")
    matched_favorable = len(_list(matched["significant_negative_full_minus_removal"], "matched favorable"))
    matched_adverse = len(_list(matched["significant_positive_full_minus_removal"], "matched adverse"))
    matched_primary_full = _object(matched_scopes["within"]["full"], "matched primary full")
    matched_secondary_full = _object(matched_scopes["cross"]["full"], "matched secondary full")
    matched_primary_rows = [_object(item, "matched primary variant") for item in
                            matched_scopes["within"]["variants_sorted_by_macro_nrmse"]]
    matched_secondary_rows = [_object(item, "matched secondary variant") for item in
                              matched_scopes["cross"]["variants_sorted_by_macro_nrmse"]]
    matched_primary_mean_position = 1 + sum(
        _number(item["mean_rank"], "matched primary variant mean rank") <
        _number(matched_primary_full["mean_rank"], "matched primary full mean rank")
        for item in matched_primary_rows
    )
    matched_secondary_mean_position = 1 + sum(
        _number(item["mean_rank"], "matched secondary variant mean rank") <
        _number(matched_secondary_full["mean_rank"], "matched secondary full mean rank")
        for item in matched_secondary_rows
    )
    if matched_primary_mean_position == matched_secondary_mean_position == 1:
        matched_mean_rank_summary = (
            "TL-QLDMR had the lowest mean task rank in both matched scopes."
        )
    else:
        matched_mean_rank_summary = (
            f"Its matched mean-rank positions were {_ordinal(matched_primary_mean_position)} "
            f"and {_ordinal(matched_secondary_mean_position)}, respectively."
        )

    abstract = " ".join((
        "Wind-power forecasting during meteorological extremes combines limited target labels with source--target distribution differences.",
        "We formulate the Transfer Learning Quantile Large-margin Distribution Machine for Regression (TL-QLDMR) using symmetric centered source-residual variance, a projected predictive-mean discrepancy, and weighted pinball loss.",
        "A frozen chronological comparison uses 128 target fitting labels and separates six primary within-site tasks from six secondary leave-one-farm-out stress tasks.",
        _brief_scope_sentence("within", scopes["within"]),
        _brief_scope_sentence("cross", scopes["cross"]),
        "First-tier membership is non-rejection of inferiority rather than equivalence or superiority.",
        f"Of 60 Holm-adjusted task comparisons, {task_favorable} favored TL-QLDMR and {task_adverse} favored a comparator.",
        f"With its settings held fixed, TL-QLDMR ranked "
        f"{_ordinal(_integer(matched_primary_full['macro_nrmse_position'], 'matched primary position', 1, 4))} "
        f"of four in the primary matched scope and "
        f"{_ordinal(_integer(matched_secondary_full['macro_nrmse_position'], 'matched secondary position', 1, 4))} "
        f"of four in the secondary matched scope; {matched_favorable} Holm-adjusted removal effects "
        f"favored TL-QLDMR and {matched_adverse} favored a removal variant. "
        f"{matched_mean_rank_summary}",
        "The matched effects are conditional because the variants were not independently retuned.",
        "The public Q3--Q4 period had been examined previously, so these results are test-locked evidence on a reused period rather than independent external confirmation and do not establish universal model superiority.",
    ))

    primary_full = _object(scopes["within"]["full"], "primary full")
    secondary_full = _object(scopes["cross"]["full"], "secondary full")
    contribution = (
        "A frozen chronological comparison retains six primary within-site tasks, six secondary "
        "leave-one-farm-out stress tasks, all prescribed methods and seeds, and 54 distinct Q1 "
        "configuration queries per task--method pair. TL-QLDMR records site-equal "
        f"capacity-normalized root mean squared errors of {_tex_percent(_number(primary_full['macro_capacity_normalized_rmse_percent'], 'primary macro'))} "
        f"(rank {_ordinal(_integer(primary_full['macro_nrmse_position'], 'primary position', 1, 6))} of six) in the primary scope and "
        f"{_tex_percent(_number(secondary_full['macro_capacity_normalized_rmse_percent'], 'secondary macro'))} "
        f"(rank {_ordinal(_integer(secondary_full['macro_nrmse_position'], 'secondary position', 1, 6))} of six) in the secondary scope; "
        f"it {_tier_status(scopes['within'])} in the primary scope and {_tier_status(scopes['cross'])} "
        "in the secondary scope. First-tier membership is a non-rejection result rather than "
        "equivalence. The matched primary analysis places TL-QLDMR "
        f"{_ordinal(_integer(matched_primary_full['macro_nrmse_position'], 'matched primary position', 1, 4))} of four under its selected "
        f"configuration by macro RMSE and {_ordinal(matched_primary_mean_position)} of four by "
        "mean task rank, so the component-removal evidence is conditional rather than an "
        "independently retuned comparison. The formal test period is reused and therefore does "
        "not provide independent external confirmation."
    )

    conclusion = " ".join((
        "The implementation passes deterministic objective, gradient, and primal--dual consistency checks.",
        primary,
        secondary,
        "The first-tier statements are non-rejections of inferiority to each empirical endpoint winner and do not establish equivalence, equal performance, or superiority.",
        directions,
        matched_primary,
        matched_secondary,
        matched_directions,
        "The separate Farm~6 nine-quantile and earlier calibration studies remain legacy exploratory sensitivity analyses and are not part of the formal evidence chain.",
        "Taken together, the formal results provide a scope- and task-dependent assessment of TL-QLDMR rather than evidence of universal superiority.",
        "Because the public Q3--Q4 test period had been examined before the frozen protocol was designed, the comparison is test-locked evidence on a reused period and does not provide independent external confirmation.",
    ))

    primary_position = _integer(primary_full["macro_nrmse_position"], "primary position", 1, 6)
    secondary_position = _integer(secondary_full["macro_nrmse_position"], "secondary position", 1, 6)
    matched_position = _integer(matched_primary_full["macro_nrmse_position"], "matched primary position", 1, 4)

    def rank_highlight(label: str, scope: dict[str, Any], position: int) -> str:
        if scope["empirical_winner"]["method"] == "Full":
            suffix = "it is the empirical endpoint winner"
        elif scope["full"]["in_conservative_nonrejection_set"]:
            suffix = "first-tier status is not equivalence"
        else:
            suffix = "it is outside the conservative first tier"
        return f"{label} TL-QLDMR rank: {_ordinal(position)} of six; {suffix}."

    highlights = [
        "TL-QLDMR: Transfer Learning Quantile Large-margin Distribution Machine for Regression",
        rank_highlight("Primary", scopes["within"], primary_position),
        rank_highlight("Secondary", scopes["cross"], secondary_position),
        ("Matched mean rank: lowest in both scopes; conditional on settings and reused test."
         if matched_primary_mean_position == matched_secondary_mean_position == 1 else
         f"Matched mean rank: {_ordinal(matched_primary_mean_position)} of four; conditional on "
         "settings and a reused test period."),
    ]
    for index, text in enumerate(highlights, 1):
        if len(text) > 85:
            raise IntegrationError(
                f"Generated highlight {index} has {len(text)} visible characters (>85): {text}"
            )
        acronyms = set(re.findall(r"(?<![A-Za-z])[A-Z][A-Z0-9-]{1,}(?![A-Za-z])", text))
        if acronyms - {"TL-QLDMR"}:
            raise IntegrationError(
                f"Generated highlight {index} contains undefined abbreviations: {sorted(acronyms)}"
            )
    if "Transfer Learning Quantile Large-margin Distribution Machine for Regression" not in highlights[0]:
        raise IntegrationError("The first highlight must define TL-QLDMR.")

    return {
        "abstract": abstract,
        "contribution": contribution,
        "conclusion": conclusion,
        "highlights": highlights,
    }


def _render_selection(template: str, blocks: dict[str, Any]) -> tuple[str, int]:
    lines = template.splitlines(keepends=True)
    marker_re = re.compile(r"^% @@(?P<name>RESULTS_[A-Z_]+)@@\s*$")
    seen: list[str] = []
    output: list[str] = []
    index = 0
    while index < len(lines):
        plain = lines[index].rstrip("\r\n")
        match = marker_re.match(plain)
        if not match:
            output.append(lines[index])
            index += 1
            continue
        name = match.group("name")
        if name not in RESULT_BLOCKS:
            raise IntegrationError(f"Unexpected result placeholder in template: {name}")
        if name in seen:
            raise IntegrationError(f"Duplicate result placeholder in template: {name}")
        seen.append(name)
        result_text = _string(blocks[name], f"manuscript_blocks.{name}")
        output.append("{\\color{blue}" + result_text + "}\n")
        index += 1
        # Remove only the human instruction comments belonging to this token.
        # Canonical BEGIN/END comments and ordinary prose are never consumed.
        while index < len(lines):
            candidate = lines[index].rstrip("\r\n")
            if not candidate.startswith("% "):
                break
            if re.match(r"^% (?:BEGIN|END) [A-Z]+ BLOCK ", candidate):
                break
            index += 1
    if set(seen) != set(RESULT_BLOCKS) or len(seen) != len(RESULT_BLOCKS):
        missing = sorted(set(RESULT_BLOCKS) - set(seen))
        raise IntegrationError(
            f"Template must contain each of the five result placeholders exactly once; missing={missing}."
        )
    rendered = "".join(output)
    if "@@RESULTS_" in rendered:
        raise IntegrationError("A result placeholder remains after template rendering.")
    return rendered, len(seen)


def _marked_payload(name: str, body: str, indent: str = "") -> str:
    begin = indent + GENERATED_BEGIN.format(name=name)
    end = indent + GENERATED_END.format(name=name)
    return f"{begin}\n{body}\n{end}"


def _replace_existing_marker(text: str, name: str, replacement: str) -> str | None:
    begin = re.escape(GENERATED_BEGIN.format(name=name))
    end = re.escape(GENERATED_END.format(name=name))
    # Restrict trailing whitespace to the current line.  ``\s`` would also
    # consume following blank lines and make a second run non-idempotent.
    pattern = re.compile(rf"(?ms)^[ \t]*{begin}[ \t]*$.*?^[ \t]*{end}[ \t]*$")
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    if len(matches) != 1:
        raise IntegrationError(f"Expected one generated marker pair for {name}, found {len(matches)}.")
    return text[:matches[0].start()] + replacement + text[matches[0].end():]


def _replace_abstract(text: str, body: str) -> str:
    rendered = _marked_payload("abstract", "{\\color{blue}" + body + "}")
    updated = _replace_existing_marker(text, "abstract", rendered)
    if updated is not None:
        return updated
    if text.count("\\begin{abstract}") != 1 or text.count("\\end{abstract}") != 1:
        raise IntegrationError("Expected exactly one abstract environment.")
    start = text.index("\\begin{abstract}") + len("\\begin{abstract}")
    end = text.index("\\end{abstract}", start)
    old = text[start:end]
    if _sha256_text(old) != EXPECTED_OLD_ABSTRACT_SHA256:
        raise IntegrationError(
            "Unmarked abstract does not match the audited old paragraph hash; refusing broad replacement."
        )
    return text[:start] + "\n" + rendered + "\n" + text[end:]


def _replace_contribution(text: str, body: str) -> str:
    rendered = _marked_payload(
        "contribution_result", "    \\item {\\color{blue}" + body + "}", indent="    "
    )
    updated = _replace_existing_marker(text, "contribution_result", rendered)
    if updated is not None:
        return updated
    pattern = re.compile(
        r"(?m)^\s*\\item \{\\color\{blue\}A frozen chronological evaluation[^\n]*\}\s*$"
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise IntegrationError(
            f"Expected one unmarked frozen-evaluation contribution item, found {len(matches)}."
        )
    old = matches[0].group(0)
    if _sha256_text(old) != EXPECTED_OLD_CONTRIBUTION_SHA256:
        raise IntegrationError(
            "Unmarked contribution item does not match the audited old hash; refusing replacement."
        )
    return text[:matches[0].start()] + rendered + text[matches[0].end():]


def _replace_conclusion(text: str, body: str) -> str:
    rendered = _marked_payload("conclusion_evidence", "{\\color{blue}" + body + "}")
    updated = _replace_existing_marker(text, "conclusion_evidence", rendered)
    if updated is not None:
        return updated
    prefix = "{\\color{blue}The implementation passes deterministic"
    suffix = "\n\n{\\color{blue}The formal comparison uses a public six-farm dataset"
    if text.count(prefix) != 1 or text.count(suffix) != 1:
        raise IntegrationError("Could not identify the unique audited conclusion-evidence boundaries.")
    start = text.index(prefix)
    end = text.index(suffix, start)
    old = text[start:end]
    if _sha256_text(old) != EXPECTED_OLD_CONCLUSION_SHA256:
        raise IntegrationError(
            "Unmarked conclusion evidence does not match the audited old paragraph hash; refusing replacement."
        )
    return text[:start] + rendered + text[end:]


def _replace_highlights(text: str, bullets: list[str]) -> str:
    body = "\n".join(f"    \\item {bullet}" for bullet in bullets)
    rendered = _marked_payload("highlights", body, indent="    ")
    updated = _replace_existing_marker(text, "highlights", rendered)
    if updated is None:
        if text.count("\\begin{itemize}") != 1 or text.count("\\end{itemize}") != 1:
            raise IntegrationError("Expected exactly one Highlights itemize environment.")
        start = text.index("\\begin{itemize}") + len("\\begin{itemize}")
        end = text.index("\\end{itemize}", start)
        old = text[start:end]
        if _sha256_text(old) != EXPECTED_OLD_HIGHLIGHTS_SHA256:
            raise IntegrationError(
                "Unmarked Highlights list does not match the audited old hash; refusing replacement."
            )
        updated = text[:start] + "\n" + rendered + "\n" + text[end:]
    length_pattern = re.compile(
        r"(?m)^% Visible bullet lengths, including spaces and punctuation: [0-9, ]+\.$"
    )
    length_matches = list(length_pattern.finditer(updated))
    if len(length_matches) != 1:
        raise IntegrationError(
            f"Expected one visible-length audit comment in Highlights, found {len(length_matches)}."
        )
    length_line = (
        "% Visible bullet lengths, including spaces and punctuation: "
        + ", ".join(str(len(item)) for item in bullets) + "."
    )
    match = length_matches[0]
    return updated[:match.start()] + length_line + updated[match.end():]


def _render_appendix() -> str:
    return r"""\section{Validation-selected comparison details}\label{app:selection_details}
{\color{blue}The formal frozen comparison records and the legacy probability sensitivity are grouped separately below. The formal tables retain all primary and secondary tasks, prescribed methods and seeds, paired comparisons, selected configurations, and optimization-accounting records. The legacy table reports only the earlier exploratory Farm~6 multi-quantile configuration search; it is not part of the formal point-comparison evidence.}

\subsection{Formal held-out results and paired inference}
\input{selection_appendix_metrics}
\FloatBarrier
\input{selection_inference}

\subsection{Frozen selection and optimization audit}
\input{selection_configs}
\input{selection_optimization_summary}
\FloatBarrier

\subsection{Legacy exploratory probability configuration}
{\color{blue}The following configuration table belongs to the reused-test exploratory nine-quantile sensitivity analysis and is reported for auditability rather than independent confirmation.}
\input{probabilistic_configs}
\FloatBarrier
"""


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def integrate(results_path: Path, template_path: Path, root: Path,
              mode: str) -> tuple[dict[str, Any], int]:
    root = root.resolve()
    results_path = results_path.resolve()
    template_path = template_path.resolve()
    payload, scopes, matched_scopes, results_sha256 = _load_results(results_path)
    template = _read_text(template_path)
    blocks = _object(payload["manuscript_blocks"], "manuscript_blocks")
    selection_study, replaced = _render_selection(template, blocks)
    generated = _frontmatter(payload, scopes, matched_scopes)

    main_path = root / "TLQLDMR_energy.tex"
    highlights_path = root / "Highlights.tex"
    appendix_path = root / "selection_appendix.tex"
    selection_path = root / "selection_study.tex"
    for required in (main_path, highlights_path, appendix_path):
        if not required.is_file():
            raise IntegrationError(f"Missing manuscript target: {required}")

    target_snapshots = {
        path: path.read_bytes() if path.exists() else b""
        for path in (selection_path, main_path, highlights_path, appendix_path)
    }
    try:
        main = target_snapshots[main_path].decode("utf-8")
        highlights_source = target_snapshots[highlights_path].decode("utf-8")
    except UnicodeDecodeError as error:
        raise IntegrationError(f"A manuscript target is not valid UTF-8: {error}") from error
    main = _replace_abstract(main, generated["abstract"])
    main = _replace_contribution(main, generated["contribution"])
    main = _replace_conclusion(main, generated["conclusion"])
    highlights = _replace_highlights(highlights_source, generated["highlights"])
    appendix = _render_appendix()
    desired = {
        selection_path: selection_study,
        main_path: main,
        highlights_path: highlights,
        appendix_path: appendix,
    }

    target_report = []
    changed_paths: list[Path] = []
    for path, text in desired.items():
        before = target_snapshots[path]
        after = text.encode("utf-8")
        changed = before != after
        if changed:
            changed_paths.append(path)
        target_report.append({
            "path": str(path),
            "before_sha256": _sha256_bytes(before),
            "after_sha256": _sha256_bytes(after),
            "changed": changed,
            "after_bytes": len(after),
        })

    if mode == "write":
        # All generation and validation completes before the first replacement.
        for path, text in desired.items():
            if path in changed_paths:
                current = path.read_bytes() if path.exists() else b""
                if current != target_snapshots[path]:
                    raise IntegrationError(
                        f"Target changed after its validated snapshot; refusing to overwrite: {path}"
                    )
                _atomic_write(path, text.encode("utf-8"))

    status = (
        "written" if mode == "write" and changed_paths else
        "clean" if not changed_paths else
        "out_of_date" if mode == "check" else
        "would_change"
    )
    report = {
        "version": VERSION,
        "status": status,
        "mode": mode,
        "input_policy": (
            "Only the supplied completed prose JSON, the supplied template, and the four "
            "manuscript target files were read; no raw or held-out result artifact was opened."
        ),
        "inputs": {
            "results": {"path": str(results_path), "sha256": results_sha256},
            "template": {"path": str(template_path), "sha256": _sha256_text(template)},
            "manuscript_root": str(root),
        },
        "validation": {
            "result_version": payload["version"],
            "result_placeholders_replaced": replaced,
            "remaining_result_placeholders": 0,
            "scope_order_kept_separate": ["primary_within", "secondary_cross"],
            "first_tier_non_equivalence_caveat_present": True,
            "matched_conditional_caveat_present": True,
            "reused_test_limitation_present": True,
            "appendix_groups": ["formal_heldout", "formal_audit", "legacy_probability"],
            "highlight_visible_characters": [len(item) for item in generated["highlights"]],
        },
        "generated_frontmatter": generated,
        "targets": target_report,
        "changed_count": len(changed_paths),
    }
    return report, 1 if mode == "check" and changed_paths else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=REPOSITORY_ROOT
        / "review"
        / "selection_runs"
        / "downstream_v2"
        / "v2_result_prose.json",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=SCRIPT_DIR / "final_selection_study_template.tex",
    )
    parser.add_argument("--manuscript-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Validate and report without writing (default).")
    modes.add_argument("--write", action="store_true", help="Atomically replace changed target files.")
    modes.add_argument("--check", action="store_true", help="Exit 1 if any target differs from generated output.")
    parser.add_argument("--report", type=Path, help="Optionally atomically save the JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mode = "write" if args.write else "check" if args.check else "dry-run"
    try:
        report, code = integrate(args.results, args.template, args.manuscript_root, mode)
    except (IntegrationError, OSError) as error:
        report = {
            "version": VERSION,
            "status": "error",
            "mode": mode,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        code = 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        try:
            _atomic_write(args.report.resolve(), rendered.encode("utf-8"))
        except OSError as error:
            print(json.dumps({
                "version": VERSION,
                "status": "error",
                "error_type": type(error).__name__,
                "error": f"Cannot write report: {error}",
            }, ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
    print(rendered, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
