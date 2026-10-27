#!/usr/bin/env python3
"""Read-only validation of response-letter Original/Revised change blocks.

The validator deliberately does not edit TeX files and does not inspect any
experimental result directory.  Structural checks and repeated-marker checks
are exact after whitespace normalization.  Source checks have two levels:

* ``strict_match`` compares the response excerpt with the mapped manuscript
  source while preserving LaTeX commands such as ``\\ref`` and environments.
* ``visible_match`` is a conservative diagnostic which removes known layout
  shells, resolves references from the manuscript AUX file when available,
  and compares the remaining text.  It is useful for distinguishing a source
  transcription problem from a layout/reference-only rewrite, but it does not
  replace TeX expansion and is never treated as strict compliance.

Explicit declarations that the original manuscript had no corresponding
passage are classified as additions and are exempt from source matching.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSE = ROOT / "TLQLDMR_response_1st" / "main.tex"
DEFAULT_SOURCE = ROOT / "TLQLDMR_manuscript" / "TLQLDMR_energy.tex"
DEFAULT_AUX = ROOT / "TLQLDMR_manuscript" / "build" / "TLQLDMR_energy.aux"
DEFAULT_FRAMEWORK = ROOT / "TLQLDMR_manuscript" / "framework.tex"

BEGIN_RE = re.compile(r"^\s*%\s*BEGIN VERIFIED CHANGE\s+(\S+)\s*$")
END_RE = re.compile(r"^\s*%\s*END VERIFIED CHANGE\s+(\S+)\s*$")
ORIGINAL_LABEL = "Original text:"
REVISED_LABEL = "Revised text:"
DELETION_LABEL = "Deletion (no replacement text):"


@dataclass(frozen=True)
class VerifiedChange:
    """One parsed VERIFIED CHANGE occurrence."""

    marker: str
    begin_line: int
    end_line: int
    original_label_line: int | None
    revised_label_line: int | None
    original: str
    revised: str


@dataclass
class ParseResult:
    blocks: list[VerifiedChange] = field(default_factory=list)
    errors: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class SourceSpec:
    """Location of the authoritative source for one response marker.

    Line spans are one-based and inclusive.  ``literal`` is used for text in
    the diagram source where a whole TikZ line contains non-visible node code.
    """

    source: str
    start_line: int | None = None
    end_line: int | None = None
    literal: str | None = None

    def __post_init__(self) -> None:
        line_span = self.start_line is not None or self.end_line is not None
        if line_span == (self.literal is not None):
            raise ValueError("SourceSpec must define exactly one of line span or literal")
        if line_span and (
            self.start_line is None
            or self.end_line is None
            or self.start_line < 1
            or self.end_line < self.start_line
        ):
            raise ValueError("Invalid one-based source line span")


# These are the 62 source transcriptions identified by the audit.  The original
# manuscript is intentionally immutable, so line spans are also a useful alarm:
# if it moves, extraction fails visibly rather than silently changing provenance.
CANONICAL_SOURCE_SPECS: dict[str, SourceSpec] = {
    "related_variance": SourceSpec("manuscript", 316, 325),
    "variance_roles": SourceSpec("manuscript", 378, 382),
    "domain_roles": SourceSpec("manuscript", 383, 383),
    "weighting": SourceSpec("manuscript", 384, 384),
    "experiment_scope": SourceSpec("manuscript", 609, 609),
    "related_mmd_scope": SourceSpec("manuscript", 331, 335),
    "structural_risk_scope": SourceSpec("manuscript", 377, 377),
    "screening_ranking_scope": SourceSpec("manuscript", 697, 699),
    "screening_interpretation": SourceSpec("manuscript", 695, 695),
    "original_experiment_1_setup_removed": SourceSpec("manuscript", 674, 674),
    "convergence": SourceSpec("manuscript", 603, 603),
    "feature_map": SourceSpec("manuscript", 507, 512),
    "complexity_intro": SourceSpec("manuscript", 550, 550),
    "original_experiment_5_scales_removed": SourceSpec("manuscript", 1021, 1022),
    "original_experiment_2_noise_setup_removed": SourceSpec("manuscript", 838, 839),
    "original_experiment_3_noise_setup_removed": SourceSpec("manuscript", 901, 902),
    "roadmap": SourceSpec("manuscript", 309, 309),
    "legacy_point_discussion": SourceSpec("manuscript", 836, 836),
    "interval_nominal_coverage": SourceSpec("manuscript", 873, 873),
    "dataset_direction_schema": SourceSpec("manuscript", 618, 618),
    "label_supervision_scope": SourceSpec("manuscript", 386, 386),
    "architecture": SourceSpec("manuscript", 353, 353),
    "kernel_constant": SourceSpec("manuscript", 453, 453),
    "bias_recovery": SourceSpec("manuscript", 501, 501),
    "matrix_projected_term_name": SourceSpec("manuscript", 407, 407),
    "auxiliary_projected_term_name": SourceSpec("manuscript", 415, 415),
    "kernel_projected_term_name": SourceSpec("manuscript", 437, 437),
    "weighting_bullet": SourceSpec("manuscript", 387, 387),
    "objective": SourceSpec("manuscript", 365, 374),
    "variance_matrix": SourceSpec("manuscript", 417, 421),
    "beta_recovery": SourceSpec("manuscript", 479, 483),
    "dual_matrices": SourceSpec("manuscript", 496, 499),
    "nystrom_R": SourceSpec("manuscript", 522, 525),
    "nystrom_c": SourceSpec("manuscript", 527, 530),
    "normalization": SourceSpec("manuscript", 408, 413),
    "mmd_matrix": SourceSpec("manuscript", 422, 426),
    "variance_expansion": SourceSpec("manuscript", 440, 441),
    "pinball_subgradient": SourceSpec("manuscript", 534, 542),
    "batch_gradients": SourceSpec("manuscript", 544, 548),
    "algorithm": SourceSpec("manuscript", 552, 583),
    "original_experiment_1_table_1_removed": SourceSpec("manuscript", 676, 693),
    "original_experiment_1_figure_removed": SourceSpec("manuscript", 716, 716),
    "ablation": SourceSpec("manuscript", 967, 1012),
    "original_experiment_2_table_1_removed": SourceSpec("manuscript", 733, 754),
    "complexity_table": SourceSpec("manuscript", 586, 601),
    "archive_caption_15": SourceSpec("manuscript", 1026, 1026),
    "original_experiment_5_table_1_removed": SourceSpec("manuscript", 1024, 1044),
    "archive_caption_12": SourceSpec("manuscript", 843, 843),
    "archive_caption_14": SourceSpec("manuscript", 908, 908),
    "original_experiment_2_table_2_removed": SourceSpec("manuscript", 841, 862),
    "original_experiment_3_table_2_removed": SourceSpec("manuscript", 906, 961),
    "original_experiment_3_table_1_removed": SourceSpec("manuscript", 875, 897),
    "archive_caption_6": SourceSpec("manuscript", 735, 735),
    "archive_caption_7": SourceSpec("manuscript", 768, 768),
    "archive_caption_8": SourceSpec("manuscript", 784, 784),
    "archive_caption_9": SourceSpec("manuscript", 800, 800),
    "archive_caption_10": SourceSpec("manuscript", 816, 816),
    "archive_caption_11": SourceSpec("manuscript", 832, 832),
    "archive_caption_13": SourceSpec("manuscript", 877, 877),
    "framework_0": SourceSpec(
        "framework", literal=r"Feature-space alignment\\MMD in RKHS"
    ),
    "framework_3": SourceSpec(
        "framework",
        literal=r"Complexity reduction\\$O(N^3)\rightarrow O(T \cdot N \cdot m)$",
    ),
    "framework_4": SourceSpec(
        "framework",
        literal=r"Windowed covariates\\$W=24$, meteorology,\\power history",
    ),
}

FRAMEWORK_MARKERS = frozenset(
    {*(f"framework_{index}" for index in range(5)), "framework_target_domain"}
)


# This is deliberately an exact allowlist, rather than a substring heuristic.
# Otherwise a sentence beginning "No corresponding passage ..." could conceal
# arbitrary text and bypass provenance validation.  New declaration wording must
# be reviewed and registered here explicitly.
APPROVED_ABSENCE_DECLARATIONS = frozenset(
    {
        "No corresponding passage appears in the original manuscript. The following text is added in the chronological evaluation subsection of the revised manuscript.",
        "The original manuscript did not report the per-rule counts, overlaps, episode durations, seasonal composition, or an objective representative-episode rule.",
        "This validation-selected study and its corresponding protocol were not present in the original manuscript.",
        "The original manuscript contained no paired temporal-block uncertainty analysis. The following subsection has been added.",
        "The original manuscript contained no table of paired temporal-block effects, confidence intervals, and multiplicity-adjusted tests. The following appendix table has been added.",
        "The original manuscript contains no corresponding fixed-feature ADMM accuracy analysis.",
        "The original manuscript has no corresponding paragraph reporting these deterministic implementation checks; the following paragraph has been added.",
        "The original manuscript contained no probability-specific equal-budget selection over nine quantiles, multi-level reliability analysis, monotone-rearrangement diagnostics, or paired probability-score inference. The following subsection has been added.",
        "The original manuscript contained no nine-quantile probability-selection table. The following table has been added.",
        "The original manuscript contained no multi-level calibrated coverage and width table. The following table has been added.",
        "The original manuscript contained no paired temporal-block inference table for probability scores. The following table has been added.",
        "The original manuscript contained no table of configurations selected by the probability-specific Q1 criterion. The following table has been added.",
        "These validation-selected main-result tables were not present in the original manuscript.",
        "This reproducibility appendix was not present in the original manuscript.",
        "These locked selected-configuration tables were not present in the original manuscript.",
        "These complete cross-farm metrics were not present in the original manuscript.",
        "The original Introduction did not discuss these recent hybrid point, probabilistic, wind-speed, and ramping-demand studies. The following text has been added.",
        "The original Related Work section had no dedicated critical review of recent wind forecasting under variability and uncertainty. The following subsection has been added.",
        "The original manuscript did not provide this submission-level data and code availability statement or the accompanying reproducibility package.",
        "The original separate Highlights file is not present in the available submission sources; no original wording is reconstructed. The following is a newly prepared Highlights file.",
        "The original manuscript provided a qualitative residual-density discussion without these numerical diagnostics.",
        "The original manuscript had no notation table.",
        "The original notation subsection did not distinguish sample-level tail-rule membership from independently named or tracked meteorological events.",
    }
)


def _line_error(kind: str, line: int, message: str, **extra: object) -> dict[str, object]:
    result: dict[str, object] = {"kind": kind, "line": line, "message": message}
    result.update(extra)
    return result


def _extract_body(lines: Sequence[str]) -> tuple[str, str, int | None, int | None, list[str]]:
    errors: list[str] = []
    original_indices = [i for i, line in enumerate(lines) if ORIGINAL_LABEL in line]
    revised_indices = [
        i for i, line in enumerate(lines)
        if REVISED_LABEL in line or DELETION_LABEL in line
    ]
    if len(original_indices) != 1:
        errors.append(f"expected one Original label, found {len(original_indices)}")
    if len(revised_indices) != 1:
        errors.append(f"expected one Revised label, found {len(revised_indices)}")
    if len(original_indices) != 1 or len(revised_indices) != 1:
        return "", "", (original_indices or [None])[0], (revised_indices or [None])[0], errors
    original_index, revised_index = original_indices[0], revised_indices[0]
    if original_index >= revised_index:
        errors.append("Original label must precede Revised label")
        return "", "", original_index, revised_index, errors
    original = "\n".join(lines[original_index + 1 : revised_index])
    revised = "\n".join(lines[revised_index + 1 :])
    return original, revised, original_index, revised_index, errors


def parse_verified_changes(text: str) -> ParseResult:
    """Parse paired VERIFIED CHANGE comments without modifying the input."""

    result = ParseResult()
    lines = text.splitlines()
    open_marker: str | None = None
    open_line = 0
    body_start = 0
    for index, line in enumerate(lines, start=1):
        begin = BEGIN_RE.match(line)
        end = END_RE.match(line)
        if begin:
            marker = begin.group(1)
            if open_marker is not None:
                result.errors.append(
                    _line_error(
                        "nested_begin",
                        index,
                        f"BEGIN {marker} occurs before END {open_marker}",
                        marker=marker,
                        open_marker=open_marker,
                        open_line=open_line,
                    )
                )
                continue
            open_marker, open_line, body_start = marker, index, index
            continue
        if not end:
            continue
        marker = end.group(1)
        if open_marker is None:
            result.errors.append(
                _line_error("orphan_end", index, f"END {marker} has no matching BEGIN", marker=marker)
            )
            continue
        if marker != open_marker:
            result.errors.append(
                _line_error(
                    "marker_mismatch",
                    index,
                    f"BEGIN {open_marker} is closed by END {marker}",
                    marker=marker,
                    open_marker=open_marker,
                    open_line=open_line,
                )
            )
        inner = lines[body_start:index - 1]
        original, revised, original_offset, revised_offset, body_errors = _extract_body(inner)
        for message in body_errors:
            result.errors.append(
                _line_error(
                    "block_labels",
                    open_line,
                    f"{open_marker}: {message}",
                    marker=open_marker,
                )
            )
        result.blocks.append(
            VerifiedChange(
                marker=open_marker,
                begin_line=open_line,
                end_line=index,
                original_label_line=(
                    body_start + original_offset + 1
                    if original_offset is not None
                    else None
                ),
                revised_label_line=(
                    body_start + revised_offset + 1
                    if revised_offset is not None
                    else None
                ),
                original=original,
                revised=revised,
            )
        )
        open_marker = None
    if open_marker is not None:
        result.errors.append(
            _line_error(
                "missing_end",
                open_line,
                f"BEGIN {open_marker} has no matching END",
                marker=open_marker,
            )
        )
    return result


def strip_response_layout(text: str) -> str:
    """Remove the response-only quote, size, and numeric-color wrappers.

    Manuscript constructs inside the excerpt are preserved for strict matching.
    """

    value = text.strip()
    value = re.sub(r"^\\begin\{quote\}\s*", "", value)
    value = re.sub(r"^\\small\b\s*", "", value)
    value = re.sub(r"\s*\\end\{quote\}\s*$", "", value)
    # The response colors literal cross-reference numbers without changing the
    # copied number. Only this numeric presentation wrapper is source-neutral.
    value = re.sub(
        r"\\ResponseReferenceNumber\{([0-9]+(?:\.[0-9]+)*)\}", r"\1", value
    )
    return value.strip()


def normalize_source_text(text: str) -> str:
    """Normalize insignificant whitespace while preserving LaTeX source tokens."""

    return re.sub(r"\s+", " ", strip_response_layout(text)).strip()


def is_absence_declaration(text: str) -> bool:
    return normalize_source_text(text) in APPROVED_ABSENCE_DECLARATIONS


def parse_aux_labels(text: str) -> dict[str, str]:
    """Read first-field values from ordinary LaTeX ``\\newlabel`` records."""

    labels: dict[str, str] = {}
    pattern = re.compile(r"\\newlabel\{([^{}]+)\}\{\{([^{}]*)\}")
    for match in pattern.finditer(text):
        labels[match.group(1)] = match.group(2)
    return labels


def _balanced_group(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{" and (index == 0 or text[index - 1] != "\\"):
            depth += 1
        elif char == "}" and (index == 0 or text[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


def _rewrite_command(text: str, command: str, arguments: int, keep: int | None) -> str:
    """Drop or unwrap a command with balanced mandatory arguments."""

    token = "\\" + command
    cursor = 0
    output: list[str] = []
    while True:
        found = text.find(token, cursor)
        if found < 0:
            output.append(text[cursor:])
            break
        boundary = found + len(token)
        if boundary < len(text) and text[boundary].isalpha():
            output.append(text[cursor:boundary])
            cursor = boundary
            continue
        output.append(text[cursor:found])
        position = boundary
        while position < len(text) and text[position].isspace():
            position += 1
        if position < len(text) and text[position] == "[":
            close = text.find("]", position + 1)
            if close < 0:
                output.append(text[found:boundary])
                cursor = boundary
                continue
            position = close + 1
            while position < len(text) and text[position].isspace():
                position += 1
        values: list[str] = []
        valid = True
        for argument_index in range(arguments):
            group = _balanced_group(text, position)
            if group is None:
                valid = False
                break
            value, position = group
            values.append(value)
            if argument_index + 1 < arguments:
                while position < len(text) and text[position].isspace():
                    position += 1
        if not valid:
            output.append(text[found:boundary])
            cursor = boundary
            continue
        if keep is not None:
            output.append(values[keep])
        cursor = position
    return "".join(output)


def _remove_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def visible_tex_text(text: str, labels: Mapping[str, str] | None = None) -> str:
    """Return a conservative visible-text projection for diagnostics.

    This intentionally understands only constructs used by the mapped excerpts;
    it is not a general TeX renderer.  A visible match is therefore informative,
    while a visible mismatch should be inspected rather than blindly rewritten.
    """

    labels = labels or {}
    value = _remove_comments(strip_response_layout(text))

    def eqref(match: re.Match[str]) -> str:
        key = match.group(1)
        return f"({labels.get(key, '<REF:' + key + '>')})"

    def ref(match: re.Match[str]) -> str:
        key = match.group(1)
        return labels.get(key, "<REF:" + key + ">")

    value = re.sub(r"\\eqref\{([^{}]+)\}", eqref, value)
    value = re.sub(r"\\ref\{([^{}]+)\}", ref, value)

    # Remove response-only presentation wrappers before unwrapping text commands;
    # doing this later can join ``\\noindent`` to the first word of its payload.
    value = re.sub(
        r"\\(?:begingroup|endgroup|centering|small|noindent|par|hfill|FloatBarrier)\b",
        " ",
        value,
    )

    for command, arguments in (
        ("label", 1),
        ("includegraphics", 1),
        ("setlength", 2),
        ("captionsetup", 1),
        ("vspace", 1),
        ("hspace", 1),
        ("color", 1),
        ("arrayrulecolor", 1),
    ):
        value = _rewrite_command(value, command, arguments, None)
    for command in ("textbf", "textit", "emph", "underline", "caption"):
        value = _rewrite_command(value, command, 1, 0)
    value = _rewrite_command(value, "textcolor", 2, 1)
    for command in ("section", "subsection", "subsubsection"):
        value = _rewrite_command(value, command, 1, 0)
    value = _rewrite_command(value, "captionof", 2, 1)

    # Environments and their placement options are layout, not source prose.
    value = re.sub(
        r"\\begin\{(?:adjustbox|minipage)\}(?:\[[^\]]*\])?\{[^{}]*\}", " ", value
    )
    value = re.sub(r"\\(?:begin|end)\{[^{}]+\}(?:\[[^\]]*\])?", " ", value)
    value = re.sub(
        r"\\(?:begingroup|endgroup|centering|small|noindent|par|hfill|FloatBarrier)\b",
        " ",
        value,
    )
    value = re.sub(r"\\item\b", " ", value)
    value = re.sub(r"\\newline\b", " ", value)
    value = value.replace(r"\\", " ")
    value = value.replace("~", " ")
    value = value.replace("{", " ").replace("}", " ")
    value = re.sub(r"\\%", "%", value)
    value = re.sub(r"\s*;\s*$", "", value.strip())
    return re.sub(r"\s+", " ", value).strip()


def _source_fragment(spec: SourceSpec, sources: Mapping[str, str]) -> str:
    if spec.source not in sources:
        raise KeyError(f"source {spec.source!r} is unavailable")
    source = sources[spec.source]
    if spec.literal is not None:
        count = source.count(spec.literal)
        if count != 1:
            raise ValueError(
                f"literal source fragment occurs {count} times in {spec.source}, expected once"
            )
        return spec.literal
    lines = source.splitlines()
    assert spec.start_line is not None and spec.end_line is not None
    if spec.end_line > len(lines):
        raise ValueError(
            f"mapped lines {spec.start_line}-{spec.end_line} exceed "
            f"{spec.source} length {len(lines)}"
        )
    return "\n".join(lines[spec.start_line - 1 : spec.end_line])


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_tex_bundle(path: Path) -> tuple[str, list[Path]]:
    """Read a TeX source plus local ``\\input`` dependencies, without expansion.

    Concatenation is sufficient because source validation only searches for
    marker-scoped literal fragments.  It also keeps the tracked ``framework.tex``
    wrapper as the declared authority when its diagram body is split into a
    local ``framework.tikz`` input in the working tree.
    """

    seen: set[Path] = set()
    files: list[Path] = []

    def visit(current: Path) -> list[str]:
        resolved = current.resolve()
        if resolved in seen:
            return []
        seen.add(resolved)
        value = current.read_text(encoding="utf-8")
        files.append(resolved)
        pieces = [value]
        for match in re.finditer(r"\\input\{([^{}]+)\}", value):
            dependency = current.parent / match.group(1)
            if not dependency.suffix:
                dependency = dependency.with_suffix(".tex")
            if dependency.exists():
                pieces.extend(visit(dependency))
        return pieces

    return "\n".join(visit(path)), files


def _difference_preview(left: str, right: str, context: int = 7) -> dict[str, object] | None:
    left_tokens, right_tokens = left.split(), right.split()
    limit = min(len(left_tokens), len(right_tokens))
    index = next((i for i in range(limit) if left_tokens[i] != right_tokens[i]), limit)
    if index == len(left_tokens) == len(right_tokens):
        return None
    start = max(0, index - context)
    return {
        "token_index": index,
        "response": " ".join(left_tokens[start : index + context]),
        "source": " ".join(right_tokens[start : index + context]),
    }


def _group_consistency(blocks: Sequence[VerifiedChange], attribute: str) -> dict[str, object]:
    grouped: dict[str, list[tuple[VerifiedChange, str]]] = {}
    for block in blocks:
        normalized = normalize_source_text(getattr(block, attribute))
        grouped.setdefault(block.marker, []).append((block, normalized))
    records: dict[str, object] = {}
    inconsistent = 0
    for marker, entries in sorted(grouped.items()):
        variants: dict[str, list[int]] = {}
        for block, normalized in entries:
            variants.setdefault(normalized, []).append(block.begin_line)
        allowed_absence = (
            attribute == "original"
            and len(variants) > 1
            and all(is_absence_declaration(value) for value in variants)
        )
        consistent = len(variants) == 1 or allowed_absence
        if not consistent:
            inconsistent += 1
        records[marker] = {
            "consistent": consistent,
            "allowed_distinct_absence_declarations": allowed_absence,
            "occurrences": len(entries),
            "variants": [
                {
                    "sha256": _digest_text(value),
                    "begin_lines": lines,
                    "preview": value[:240],
                }
                for value, lines in variants.items()
            ],
        }
    return {"inconsistent_markers": inconsistent, "markers": records}


def validate_response_text(
    response_text: str,
    sources: Mapping[str, str],
    *,
    labels: Mapping[str, str] | None = None,
    source_specs: Mapping[str, SourceSpec] = CANONICAL_SOURCE_SPECS,
) -> dict[str, object]:
    """Validate parsed response text against supplied, already-read sources."""

    parsed = parse_verified_changes(response_text)
    original_consistency = _group_consistency(parsed.blocks, "original")
    revised_consistency = _group_consistency(parsed.blocks, "revised")
    grouped: dict[str, list[VerifiedChange]] = {}
    for block in parsed.blocks:
        grouped.setdefault(block.marker, []).append(block)

    normalized_sources = {
        name: re.sub(r"\s+", " ", text).strip() for name, text in sources.items()
    }
    source_records: dict[str, object] = {}
    counts = {
        "strict_match": 0,
        "declared_addition": 0,
        "visible_only": 0,
        "unmatched": 0,
        "mapping_error": 0,
    }
    mapped_occurrences = 0
    for marker, entries in sorted(grouped.items()):
        variants = {normalize_source_text(block.original) for block in entries}
        begin_lines = [block.begin_line for block in entries]
        spec = source_specs.get(marker)
        if spec is None and all(is_absence_declaration(value) for value in variants):
            status = "declared_addition"
            counts[status] += len(entries)
            source_records[marker] = {
                "status": status,
                "occurrences": len(entries),
                "begin_lines": begin_lines,
                "mapped": marker in source_specs,
            }
            continue
        if len(variants) != 1:
            status = "unmatched"
            counts[status] += len(entries)
            source_records[marker] = {
                "status": status,
                "occurrences": len(entries),
                "begin_lines": begin_lines,
                "reason": "Original variants differ; source comparison is ambiguous",
            }
            continue
        response_original = next(iter(variants))
        if spec is None:
            permitted_sources = ["manuscript"]
            if marker in FRAMEWORK_MARKERS:
                permitted_sources.append("framework")
            matches = [
                name
                for name in permitted_sources
                if normalized_sources.get(name)
                and response_original in normalized_sources[name]
            ]
            status = "strict_match" if matches else "unmatched"
            counts[status] += len(entries)
            source_records[marker] = {
                "status": status,
                "occurrences": len(entries),
                "begin_lines": begin_lines,
                "mapped": False,
                "source_matches": matches,
            }
            continue
        mapped_occurrences += len(entries)
        try:
            source_fragment = _source_fragment(spec, sources)
        except (KeyError, ValueError) as error:
            counts["mapping_error"] += len(entries)
            source_records[marker] = {
                "status": "mapping_error",
                "occurrences": len(entries),
                "begin_lines": begin_lines,
                "mapped": True,
                "source": spec.source,
                "error": str(error),
            }
            continue
        strict_source = normalize_source_text(source_fragment)
        strict = response_original == strict_source
        # Preserve original line boundaries here: a TeX ``%`` comments only to
        # end-of-line.  Running the projection on whitespace-normalized text
        # would incorrectly discard everything after the first such comment.
        response_visible = visible_tex_text(entries[0].original, labels)
        source_visible = visible_tex_text(source_fragment, labels)
        if response_visible == source_visible:
            visible_relation = "equal"
        elif response_visible and response_visible in source_visible:
            # Some response entries quote one complete sentence/equation from a
            # mapped manuscript line that also contains adjacent prose.
            visible_relation = "response_excerpt"
        else:
            visible_relation = None
        visible = visible_relation is not None
        status = "strict_match" if strict else "visible_only" if visible else "unmatched"
        counts[status] += len(entries)
        record: dict[str, object] = {
            "status": status,
            "occurrences": len(entries),
            "begin_lines": begin_lines,
            "mapped": True,
            "source": spec.source,
            "source_lines": (
                [spec.start_line, spec.end_line] if spec.literal is None else None
            ),
            "strict_match": strict,
            "visible_match": visible,
            "visible_relation": visible_relation,
            "strict_similarity": round(
                difflib.SequenceMatcher(None, response_original, strict_source).ratio(), 6
            ),
            "visible_similarity": round(
                difflib.SequenceMatcher(None, response_visible, source_visible).ratio(), 6
            ),
            "response_sha256": _digest_text(response_original),
            "source_sha256": _digest_text(strict_source),
        }
        if not strict:
            record["strict_difference"] = _difference_preview(response_original, strict_source)
        if not visible:
            record["visible_difference"] = _difference_preview(response_visible, source_visible)
        source_records[marker] = record

    expected_mapped = set(source_specs)
    present_mapped = expected_mapped & set(grouped)
    missing_mapped = sorted(expected_mapped - set(grouped))
    structural_failures = len(parsed.errors)
    consistency_failures = (
        int(original_consistency["inconsistent_markers"])
        + int(revised_consistency["inconsistent_markers"])
    )
    source_failures = (
        counts["visible_only"]
        + counts["unmatched"]
        + counts["mapping_error"]
        + len(missing_mapped)
    )
    status = "passed" if not (structural_failures + consistency_failures + source_failures) else "failed"
    return {
        "schema_version": "response_source_validation_1",
        "status": status,
        "scope": {
            "read_only": True,
            "heldout_results_read": False,
            "strict_definition": "quote/small shell removed; whitespace normalized; LaTeX retained",
            "visible_definition": (
                "known layout commands removed and AUX references resolved; diagnostic only"
            ),
        },
        "summary": {
            "blocks": len(parsed.blocks),
            "unique_markers": len(grouped),
            "parse_errors": structural_failures,
            "original_inconsistent_markers": original_consistency["inconsistent_markers"],
            "revised_inconsistent_markers": revised_consistency["inconsistent_markers"],
            "canonical_mapped_markers_expected": len(expected_mapped),
            "canonical_mapped_markers_present": len(present_mapped),
            "canonical_mapped_occurrences": mapped_occurrences,
            "canonical_mapped_markers_missing": missing_mapped,
            "source_occurrence_counts": counts,
        },
        "parse_errors": parsed.errors,
        "original_consistency": original_consistency,
        "revised_consistency": revised_consistency,
        "source_checks": source_records,
    }


def validate_response(
    response_path: Path = DEFAULT_RESPONSE,
    source_path: Path = DEFAULT_SOURCE,
    *,
    aux_path: Path | None = DEFAULT_AUX,
    framework_path: Path | None = DEFAULT_FRAMEWORK,
    source_specs: Mapping[str, SourceSpec] = CANONICAL_SOURCE_SPECS,
) -> dict[str, object]:
    """Read only declared TeX/AUX files and return a JSON-serializable report."""

    sources = {"manuscript": source_path.read_text(encoding="utf-8")}
    framework_files: list[Path] = []
    if framework_path is not None and framework_path.exists():
        sources["framework"], framework_files = _read_tex_bundle(framework_path)
    labels = (
        parse_aux_labels(aux_path.read_text(encoding="utf-8"))
        if aux_path is not None and aux_path.exists()
        else {}
    )
    report = validate_response_text(
        response_path.read_text(encoding="utf-8"),
        sources,
        labels=labels,
        source_specs=source_specs,
    )
    report["paths"] = {
        "response": str(response_path.resolve()),
        "authoritative_manuscript": str(source_path.resolve()),
        "aux_labels": str(aux_path.resolve()) if aux_path and aux_path.exists() else None,
        "included_framework": (
            str(framework_path.resolve())
            if framework_path is not None and framework_path.exists()
            else None
        ),
        "included_framework_files": [str(path) for path in framework_files],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--aux", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--framework", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = validate_response(
        args.response,
        args.source,
        aux_path=args.aux,
        framework_path=args.framework,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_output is not None:
        args.json_output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
