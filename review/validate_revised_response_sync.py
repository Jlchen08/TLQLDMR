#!/usr/bin/env python3
"""Read-only synchronization audit for response-letter Revised text.

The audit compares ``Revised text`` in ``VERIFIED CHANGE`` blocks with the
active revised manuscript and the local files reached by its ``\\input`` or
``\\include`` commands.  It never reads experiment-run directories and never
changes a manuscript, response, result, or manifest.

Four outcomes are distinguished:

``strict``
    The whitespace-normalized response excerpt occurs verbatim in its
    canonical TeX source.  LaTeX commands and references are retained.
``visible``
    Every visible response unit occurs in order in the canonical source after
    removing only known layout shells and resolving ``\\ref``/``\\eqref`` from
    the revised-manuscript AUX file.  This permits response-safe presentation
    of sections, floats, labels, resize boxes, and captions while preserving
    prose, equations, numbers, table cells, and figure captions.
``deletion``
    The marker is on the explicit deletion allowlist, its complete Revised text
    is one of the marker's reviewed deletion declarations, every occurrence's
    Original text has a strict source match in ``validate_response_sources``,
    and the quoted original is absent from the active revised manuscript.
``mismatch``
    At least one visible unit cannot be found in order.  The first unmatched
    unit is reported for inspection.

Same-name response markers are also compared with one another.  Exact
duplicate consistency is required independently of manuscript matching.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

if __package__:
    from review.validate_response_sources import (
        DEFAULT_AUX as DEFAULT_ORIGINAL_AUX,
        DEFAULT_FRAMEWORK as DEFAULT_ORIGINAL_FRAMEWORK,
        DEFAULT_SOURCE as DEFAULT_ORIGINAL_SOURCE,
        VerifiedChange,
        _rewrite_command,
        normalize_source_text,
        parse_aux_labels,
        parse_verified_changes,
        validate_response as validate_original_response,
        visible_tex_text,
    )
else:  # Support ``python review/validate_revised_response_sync.py``.
    from validate_response_sources import (  # type: ignore[no-redef]
        DEFAULT_AUX as DEFAULT_ORIGINAL_AUX,
        DEFAULT_FRAMEWORK as DEFAULT_ORIGINAL_FRAMEWORK,
        DEFAULT_SOURCE as DEFAULT_ORIGINAL_SOURCE,
        VerifiedChange,
        _rewrite_command,
        normalize_source_text,
        parse_aux_labels,
        parse_verified_changes,
        validate_response as validate_original_response,
        visible_tex_text,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSE = ROOT / "TLQLDMR_response_1st" / "main.tex"
DEFAULT_MANUSCRIPT = ROOT / "TLQLDMR_R1" / "TLQLDMR_energy.tex"
DEFAULT_AUX = ROOT / "TLQLDMR_R1" / "build" / "TLQLDMR_energy.aux"
DEFAULT_HIGHLIGHTS = ROOT / "TLQLDMR_R1" / "Highlights.tex"

CANONICAL_BEGIN_RE = re.compile(
    r"^\s*%\s*BEGIN\s+([A-Za-z0-9_-]+)\s+BLOCK\s+(\S+)\s*$"
)
CANONICAL_END_RE = re.compile(
    r"^\s*%\s*END\s+([A-Za-z0-9_-]+)\s+BLOCK\s+(\S+)\s*$"
)
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^{}]+)\}")


# These are expected to remain represented while the v2 experiment revision is
# being synchronized.  Additional selection_* markers are selected
# automatically, so a newly rendered v2 table is audited without editing this
# list first.
CORE_EXPERIMENT_MARKERS = frozenset(
    {
        "abstract",
        "contributions",
        "experiment_scope",
        "limitations",
        "chronological_protocol",
        "chronological_settings",
        "chronological_crossfarm",
        "experimental_setup",
        "solver_accuracy",
        "feature_map",
        "ablation",
        "conclusion_method",
        "conclusion_evidence",
        "highlights_file",
        "data_code_availability",
    }
)

EXPERIMENT_MARKERS = frozenset(
    {
        *CORE_EXPERIMENT_MARKERS,
        "fewshot_evidence",
        "data_partition",
        "metric_definitions",
        "sequential_calibration",
        "extreme_composition",
        "event_scope",
        "event_subgroups",
        "training_tail_rules",
        "threshold_definition",
    }
)

EXPERIMENT_PREFIXES = (
    "chronological_",
    "selection_",
    "probabilistic_",
    "conclusion_",
)


# Files whose complete visible payload is the canonical content for a response
# marker.  They are all ordinary manuscript inputs except Highlights, which is
# an independently submitted source file.
MARKER_FILES: dict[str, str] = {
    "selection_tables": "selection_tables.tex",
    "selection_configs": "selection_configs.tex",
    "selection_point_inference_table": "selection_inference.tex",
    "selection_appendix_metrics": "selection_appendix_metrics.tex",
    "selection_appendix": "selection_appendix.tex",
    "probabilistic_scores_table": "probabilistic_scores.tex",
    "probabilistic_coverage_table": "probabilistic_coverage.tex",
    "probabilistic_inference_table": "probabilistic_inference.tex",
    "probabilistic_configs_table": "probabilistic_configs.tex",
    "highlights_file": "Highlights.tex",
}


# A response marker may quote the content of a differently named canonical
# block.  The alias does not weaken the comparison; it merely identifies the
# manuscript passage to inspect.
MARKER_BLOCK_ALIASES: dict[str, tuple[str, ...]] = {
    "experimental_setup": ("selection_protocol",),
    "ablation": ("ablation", "matched_ablation", "chronological_components"),
}


# Deletions have no active-manuscript replacement to match.  They are admitted
# only through this code-owned allowlist and marker-specific, exact declaration
# text.  A ``*_removed`` suffix, the word "removed", or a failed manuscript
# match is never sufficient on its own.
_GENERIC_DELETION_DECLARATION = (
    "These earlier implementation results are excluded from the submitted "
    "manuscript because they are not comparable evidence for the corrected "
    "model under the current chronological protocols. Historical material "
    "remains in the repository research archive only; that archive is not part "
    "of the submitted manuscript."
)
_ABLATION_DELETION_DECLARATION = (
    _GENERIC_DELETION_DECLARATION
    + " This deletion includes the original ablation table, its numerical "
    "values, the residual-density figure, and the associated interpretation; "
    "the former SymVar comparison is not retained."
)
_TABLE_DELETION_DECLARATION = (
    "The original table caption and all numerical values reproduced above have "
    "been removed. " + _GENERIC_DELETION_DECLARATION
)
_FIGURE_DELETION_DECLARATION = (
    "The six-panel historical domain-shift figure identified by this exact "
    "caption has been removed, together with its screening interpretation. "
    + _GENERIC_DELETION_DECLARATION
)
_SHIFT_DELETION_DECLARATION = (
    "The historical screening description is removed because the submitted "
    "chronological experiments do not report this historical Wasserstein/KS "
    "comparison. " + _GENERIC_DELETION_DECLARATION
)

_GENERIC_DELETION_MARKERS = frozenset(
    {
        "archive_caption_6",
        "archive_caption_7",
        "archive_caption_8",
        "archive_caption_9",
        "archive_caption_10",
        "archive_caption_11",
        "archive_caption_12",
        "archive_caption_13",
        "archive_caption_14",
        "archive_caption_15",
        "data_partition",
        "fewshot_evidence",
        "interval_nominal_coverage",
        "legacy_interval_discussion",
        "legacy_interval_intro",
        "legacy_interval_noise",
        "legacy_point_discussion",
        "legacy_point_intro",
        "legacy_point_noise",
        "legacy_scaling_discussion",
        "legacy_scaling_intro",
        "original_experiment_1_setup_removed",
        "original_experiment_2_baselines_removed",
        "original_experiment_2_noise_setup_removed",
        "original_experiment_3_baselines_removed",
        "original_experiment_3_noise_setup_removed",
        "original_experiment_5_qp_claim_removed",
        "original_experiment_5_scales_removed",
        "screened_counts_scope",
        "screening_interpretation",
        "screening_ranking_scope",
    }
)
_TABLE_DELETION_MARKERS = frozenset(
    {
        "original_experiment_1_table_1_removed",
        "original_experiment_2_table_1_removed",
        "original_experiment_2_table_2_removed",
        "original_experiment_3_table_1_removed",
        "original_experiment_3_table_2_removed",
        "original_experiment_5_table_1_removed",
    }
)

APPROVED_DELETION_DECLARATIONS: dict[str, frozenset[str]] = {
    **{
        marker: frozenset({_GENERIC_DELETION_DECLARATION})
        for marker in _GENERIC_DELETION_MARKERS
    },
    **{
        marker: frozenset({_TABLE_DELETION_DECLARATION})
        for marker in _TABLE_DELETION_MARKERS
    },
    "ablation": frozenset({_ABLATION_DELETION_DECLARATION}),
    "historical_shift_metrics_removed": frozenset({_SHIFT_DELETION_DECLARATION}),
    "original_experiment_1_figure_removed": frozenset(
        {_FIGURE_DELETION_DECLARATION}
    ),
}


@dataclass(frozen=True)
class CanonicalBlock:
    marker: str
    kind: str
    path: Path
    begin_line: int
    end_line: int
    text: str


@dataclass
class CanonicalBundle:
    root: Path
    expanded_text: str
    files: dict[Path, str] = field(default_factory=dict)
    blocks: dict[str, list[CanonicalBlock]] = field(default_factory=dict)
    errors: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class CanonicalCandidate:
    name: str
    text: str
    paths: tuple[str, ...]
    kind: str


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_comment_position(text: str, position: int) -> bool:
    """Return whether ``position`` follows an unescaped percent on its line."""

    line_start = text.rfind("\n", 0, position) + 1
    prefix = text[line_start:position]
    for match in re.finditer(r"%", prefix):
        index = match.start()
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and prefix[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return True
    return False


def _resolve_input(parent: Path, name: str, root: Path) -> Path:
    dependency = parent / name.strip()
    if not dependency.suffix:
        dependency = dependency.with_suffix(".tex")
    resolved = dependency.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"input escapes manuscript directory: {name!r}") from error
    return resolved


def _canonical_error(kind: str, path: Path, line: int, message: str) -> dict[str, object]:
    return {
        "kind": kind,
        "path": str(path),
        "line": line,
        "message": message,
    }


def _parse_canonical_blocks(path: Path, text: str) -> tuple[list[CanonicalBlock], list[dict[str, object]]]:
    blocks: list[CanonicalBlock] = []
    errors: list[dict[str, object]] = []
    lines = text.splitlines()
    open_kind: str | None = None
    open_marker: str | None = None
    open_line = 0
    body_start = 0
    for line_number, line in enumerate(lines, start=1):
        begin = CANONICAL_BEGIN_RE.match(line)
        end = CANONICAL_END_RE.match(line)
        if begin:
            kind, marker = begin.groups()
            if open_marker is not None:
                errors.append(
                    _canonical_error(
                        "nested_canonical_begin",
                        path,
                        line_number,
                        f"BEGIN {marker} occurs before END {open_marker}",
                    )
                )
                continue
            open_kind, open_marker = kind, marker
            open_line, body_start = line_number, line_number
            continue
        if not end:
            continue
        kind, marker = end.groups()
        if open_marker is None:
            errors.append(
                _canonical_error(
                    "orphan_canonical_end",
                    path,
                    line_number,
                    f"END {marker} has no matching BEGIN",
                )
            )
            continue
        if marker != open_marker or kind != open_kind:
            errors.append(
                _canonical_error(
                    "canonical_marker_mismatch",
                    path,
                    line_number,
                    f"BEGIN {open_kind} BLOCK {open_marker} is closed by "
                    f"END {kind} BLOCK {marker}",
                )
            )
        blocks.append(
            CanonicalBlock(
                marker=open_marker,
                kind=open_kind or "UNKNOWN",
                path=path,
                begin_line=open_line,
                end_line=line_number,
                text="\n".join(lines[body_start : line_number - 1]),
            )
        )
        open_kind = open_marker = None
    if open_marker is not None:
        errors.append(
            _canonical_error(
                "missing_canonical_end",
                path,
                open_line,
                f"BEGIN {open_kind} BLOCK {open_marker} has no matching END",
            )
        )
    return blocks, errors


def read_canonical_bundle(main_path: Path) -> CanonicalBundle:
    """Read and expand only the active TeX dependency graph.

    Dependencies must remain inside the directory containing ``main_path``.
    Missing inputs and cycles are reported instead of being silently ignored.
    """

    main_path = main_path.resolve()
    root = main_path.parent.resolve()
    bundle = CanonicalBundle(root=root, expanded_text="")
    active: list[Path] = []

    def expand(path: Path) -> str:
        resolved = path.resolve()
        if resolved in active:
            cycle = " -> ".join(str(item) for item in [*active, resolved])
            bundle.errors.append(
                _canonical_error("input_cycle", resolved, 1, f"input cycle: {cycle}")
            )
            return ""
        if resolved in bundle.files:
            # Repeated inputs are expanded at every call site, as TeX does.
            value = bundle.files[resolved]
        else:
            try:
                value = resolved.read_text(encoding="utf-8")
            except OSError as error:
                bundle.errors.append(
                    _canonical_error("input_read_error", resolved, 1, str(error))
                )
                return ""
            bundle.files[resolved] = value
            blocks, errors = _parse_canonical_blocks(resolved, value)
            bundle.errors.extend(errors)
            for block in blocks:
                bundle.blocks.setdefault(block.marker, []).append(block)

        active.append(resolved)
        output: list[str] = []
        cursor = 0
        for match in INPUT_RE.finditer(value):
            if _is_comment_position(value, match.start()):
                continue
            output.append(value[cursor : match.start()])
            try:
                dependency = _resolve_input(resolved.parent, match.group(1), root)
            except ValueError as error:
                line = value.count("\n", 0, match.start()) + 1
                bundle.errors.append(
                    _canonical_error("unsafe_input", resolved, line, str(error))
                )
                output.append(match.group(0))
            else:
                if not dependency.exists():
                    line = value.count("\n", 0, match.start()) + 1
                    bundle.errors.append(
                        _canonical_error(
                            "missing_input",
                            resolved,
                            line,
                            f"missing input: {dependency}",
                        )
                    )
                    output.append(match.group(0))
                else:
                    output.append(expand(dependency))
            cursor = match.end()
        output.append(value[cursor:])
        active.pop()
        return "".join(output)

    bundle.expanded_text = expand(main_path)
    return bundle


def _expand_fragment(fragment: str, source_path: Path, bundle: CanonicalBundle) -> str:
    """Expand local inputs in a canonical block using the already read bundle."""

    output: list[str] = []
    cursor = 0
    for match in INPUT_RE.finditer(fragment):
        if _is_comment_position(fragment, match.start()):
            continue
        output.append(fragment[cursor : match.start()])
        try:
            dependency = _resolve_input(source_path.parent, match.group(1), bundle.root)
        except ValueError:
            output.append(match.group(0))
        else:
            output.append(bundle.files.get(dependency, match.group(0)))
        cursor = match.end()
    output.append(fragment[cursor:])
    return "".join(output)


def _prepare_visible_tex(text: str) -> str:
    """Normalize response-safe layout constructs before visible projection."""

    value = text
    # Source tables commonly use resizebox while response quotations use an
    # adjustbox environment.  Both display the third/inner argument unchanged.
    value = _rewrite_command(value, "resizebox", 3, 2)
    value = _rewrite_command(value, "scalebox", 2, 1)
    value = _rewrite_command(value, "raisebox", 2, 1)
    value = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline\{[^{}]*\})\b", " ", value)
    value = re.sub(r"\\(?:input|include)\s*\{[^{}]+\}", " ", value)
    return value


def project_visible(text: str, labels: Mapping[str, str] | None = None) -> str:
    """Project TeX to visible content while preserving semantic commands/data."""

    value = visible_tex_text(_prepare_visible_tex(text), labels)
    value = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def visible_units(text: str, labels: Mapping[str, str] | None = None) -> list[str]:
    """Return ordered visible prose/equation/table/caption units.

    Unit matching permits canonical material omitted from a response excerpt,
    such as labels or an intervening included table, but never permits visible
    response material to be omitted or reordered.
    """

    value = _prepare_visible_tex(text)
    # Paragraph commands, blank lines, and table row terminators are semantic
    # boundaries.  A single source newline is only TeX whitespace: treating it
    # as a boundary would let a response omit words merely by inserting a line
    # break in the middle of a sentence.
    boundary = "\x00VISIBLE_UNIT\x00"
    value = re.sub(r"\\par\b", boundary, value)
    value = value.replace(r"\\", boundary)
    value = re.sub(r"\n\s*\n+", boundary, value)
    value = re.sub(r"[\t ]*\n[\t ]*", " ", value)
    raw_units = value.split(boundary)
    units: list[str] = []
    for raw in raw_units:
        projected = project_visible(raw, labels)
        # Known pure layout residue must not become an audit requirement.
        projected = re.sub(
            r"^(?:\\(?:linewidth|textwidth)|!|\[?[htbpH!]+\]?)+$", "", projected
        ).strip()
        if projected:
            units.append(projected)
    return units


def _find_units_in_order(
    response: str,
    canonical: str,
    labels: Mapping[str, str] | None,
) -> tuple[bool, dict[str, object] | None]:
    source_visible = project_visible(canonical, labels)
    units = visible_units(response, labels)
    cursor = 0
    matched = 0
    for index, unit in enumerate(units):
        found = source_visible.find(unit, cursor)
        if found < 0:
            # Give a local fuzzy diagnostic without accepting similarity as a
            # match.  Values and table cells therefore remain exact.
            source_tokens = source_visible[cursor:].split()
            unit_tokens = unit.split()
            width = max(len(unit_tokens) + 12, 24)
            windows = [
                " ".join(source_tokens[start : start + width])
                for start in range(0, max(1, len(source_tokens) - width + 1), max(1, width // 3))
            ]
            closest = max(
                windows or [""],
                key=lambda candidate: difflib.SequenceMatcher(None, unit, candidate).ratio(),
            )
            return False, {
                "unit_index": index,
                "matched_units_before_failure": matched,
                "response_unit": unit[:1200],
                "closest_canonical_window": closest[:1200],
                "similarity": round(
                    difflib.SequenceMatcher(None, unit, closest).ratio(), 6
                ),
            }
        cursor = found + len(unit)
        matched += 1
    if not units:
        return False, {
            "unit_index": 0,
            "matched_units_before_failure": 0,
            "response_unit": "",
            "reason": "Revised text has no visible semantic content",
        }
    return True, None


def _extract_environment(text: str, environment: str) -> str | None:
    match = re.search(
        rf"\\begin\{{{re.escape(environment)}\}}(.*?)\\end\{{{re.escape(environment)}\}}",
        text,
        flags=re.DOTALL,
    )
    return match.group(0) if match else None


def _marker_candidates(
    marker: str,
    bundle: CanonicalBundle,
    highlights_path: Path | None,
) -> list[CanonicalCandidate]:
    candidates: list[CanonicalCandidate] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, text: str, paths: Iterable[Path], kind: str) -> None:
        key = (kind, _sha256(text))
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            CanonicalCandidate(
                name=name,
                text=text,
                paths=tuple(str(path.resolve()) for path in paths),
                kind=kind,
            )
        )

    block_names = (marker, *MARKER_BLOCK_ALIASES.get(marker, ()))
    for block_name in block_names:
        for block in bundle.blocks.get(block_name, []):
            add(
                f"{block.kind} BLOCK {block_name}",
                _expand_fragment(block.text, block.path, bundle),
                (block.path,),
                "canonical_block",
            )

    explicit_name = MARKER_FILES.get(marker)
    if explicit_name is not None:
        explicit_path = bundle.root / explicit_name
        if marker == "highlights_file" and highlights_path is not None:
            explicit_path = highlights_path.resolve()
        text = bundle.files.get(explicit_path.resolve())
        if text is None and explicit_path.exists():
            text = explicit_path.read_text(encoding="utf-8")
        if text is not None:
            add(explicit_name, text, (explicit_path,), "canonical_file")

    if marker == "abstract":
        abstract = _extract_environment(bundle.expanded_text, "abstract")
        if abstract is not None:
            add("abstract environment", abstract, (bundle.root,), "canonical_environment")

    # Markers without a dedicated source anchor are checked against the active
    # expanded manuscript.  For anchored markers this fallback also permits a
    # response excerpt to quote a caption located just outside the prose block.
    add(
        "active revised manuscript (expanded)",
        bundle.expanded_text,
        bundle.files.keys(),
        "expanded_manuscript",
    )
    return candidates


def _is_experiment_marker(marker: str) -> bool:
    return marker in EXPERIMENT_MARKERS or marker.startswith(EXPERIMENT_PREFIXES)


def _duplicate_consistency(
    blocks: Sequence[VerifiedChange], labels: Mapping[str, str] | None
) -> dict[str, object]:
    exact: dict[str, list[int]] = {}
    visible: dict[str, list[int]] = {}
    for block in blocks:
        exact.setdefault(normalize_source_text(block.revised), []).append(block.begin_line)
        visible.setdefault(project_visible(block.revised, labels), []).append(block.begin_line)
    return {
        "consistent": len(exact) == 1,
        "visible_consistent": len(visible) == 1,
        "occurrences": len(blocks),
        "exact_variants": [
            {
                "sha256": _sha256(value),
                "begin_lines": lines,
                "preview": value[:320],
            }
            for value, lines in exact.items()
        ],
        "visible_variant_count": len(visible),
    }


def _check_occurrence(
    block: VerifiedChange,
    candidates: Sequence[CanonicalCandidate],
    labels: Mapping[str, str] | None,
) -> dict[str, object]:
    response_strict = normalize_source_text(block.revised)
    for candidate in candidates:
        canonical_strict = re.sub(r"\s+", " ", candidate.text).strip()
        if response_strict and response_strict in canonical_strict:
            return {
                "status": "strict",
                "begin_line": block.begin_line,
                "revised_label_line": block.revised_label_line,
                "canonical": candidate.name,
                "canonical_kind": candidate.kind,
                "canonical_paths": candidate.paths,
                "response_sha256": _sha256(response_strict),
            }

    failures: list[dict[str, object]] = []
    for candidate in candidates:
        matched, difference = _find_units_in_order(block.revised, candidate.text, labels)
        if matched:
            return {
                "status": "visible",
                "begin_line": block.begin_line,
                "revised_label_line": block.revised_label_line,
                "canonical": candidate.name,
                "canonical_kind": candidate.kind,
                "canonical_paths": candidate.paths,
                "response_sha256": _sha256(response_strict),
                "visible_units": len(visible_units(block.revised, labels)),
            }
        failures.append(
            {
                "canonical": candidate.name,
                "canonical_kind": candidate.kind,
                "difference": difference,
            }
        )
    return {
        "status": "mismatch",
        "begin_line": block.begin_line,
        "revised_label_line": block.revised_label_line,
        "response_sha256": _sha256(response_strict),
        "candidate_failures": failures,
    }


def _deletion_occurrence_checks(
    marker: str,
    blocks: Sequence[VerifiedChange],
    labels: Mapping[str, str] | None,
    source_validation_report: Mapping[str, object] | None,
    canonical_strict: str,
    canonical_visible: str,
) -> list[dict[str, object]]:
    """Validate a declared deletion without treating mismatch as evidence.

    The separate source validator is authoritative for the Original side.  We
    bind its marker record back to this parse by occurrence count and exact
    response line numbers, then independently require the reviewed deletion
    declaration and absence of the quoted original from the revised source.
    """

    common_rejections: list[str] = []
    source_check: Mapping[str, object] = {}
    source_consistency: Mapping[str, object] = {}
    if source_validation_report is None:
        common_rejections.append("original-source validation report is unavailable")
    else:
        if source_validation_report.get("schema_version") != (
            "response_source_validation_1"
        ):
            common_rejections.append("unexpected original-source validation schema")
        if source_validation_report.get("status") != "passed":
            common_rejections.append("original-source validation did not pass")
        raw_checks = source_validation_report.get("source_checks")
        if isinstance(raw_checks, Mapping):
            raw_check = raw_checks.get(marker)
            if isinstance(raw_check, Mapping):
                source_check = raw_check
        if source_check.get("status") != "strict_match":
            common_rejections.append(
                "Original text is not a strict match in validate_response_sources"
            )
        expected_lines = sorted(block.begin_line for block in blocks)
        source_lines = source_check.get("begin_lines")
        if not isinstance(source_lines, list) or source_lines != expected_lines:
            common_rejections.append(
                "original-source occurrence lines do not match this response parse"
            )
        if source_check.get("occurrences") != len(blocks):
            common_rejections.append(
                "original-source occurrence count does not match this response parse"
            )
        raw_consistency = source_validation_report.get("original_consistency")
        if isinstance(raw_consistency, Mapping):
            raw_markers = raw_consistency.get("markers")
            if isinstance(raw_markers, Mapping):
                raw_marker = raw_markers.get(marker)
                if isinstance(raw_marker, Mapping):
                    source_consistency = raw_marker
        if source_consistency.get("consistent") is not True:
            common_rejections.append(
                "Original text occurrences are not source-validator consistent"
            )

    approved = APPROVED_DELETION_DECLARATIONS[marker]
    records: list[dict[str, object]] = []
    for block in blocks:
        rejections = list(common_rejections)
        revised = normalize_source_text(block.revised)
        if revised not in approved:
            rejections.append(
                "Revised text is not the marker's exact approved deletion declaration"
            )

        original = normalize_source_text(block.original)
        if not original:
            rejections.append("Original text is empty")
        elif original in canonical_strict:
            rejections.append(
                "quoted Original text still occurs in the active revised manuscript"
            )
        original_visible = project_visible(block.original, labels)
        if original_visible and original_visible in canonical_visible:
            rejections.append(
                "visible quoted Original text still occurs in the active revised manuscript"
            )

        record: dict[str, object] = {
            "status": "mismatch" if rejections else "deletion",
            "begin_line": block.begin_line,
            "revised_label_line": block.revised_label_line,
            "response_sha256": _sha256(revised),
            "deletion_marker_allowlisted": True,
            "deletion_declaration_exact": revised in approved,
            "original_source_status": source_check.get("status"),
        }
        if rejections:
            record["deletion_rejections"] = rejections
        records.append(record)
    return records


def validate_revised_sync_text(
    response_text: str,
    bundle: CanonicalBundle,
    *,
    labels: Mapping[str, str] | None = None,
    highlights_path: Path | None = None,
    scope: str = "experiments",
    selected_markers: Sequence[str] | None = None,
    source_validation_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate already-read response and canonical source data."""

    if scope not in {"experiments", "all"}:
        raise ValueError("scope must be 'experiments' or 'all'")
    parsed = parse_verified_changes(response_text)
    grouped: dict[str, list[VerifiedChange]] = {}
    for block in parsed.blocks:
        grouped.setdefault(block.marker, []).append(block)

    requested = set(selected_markers or ())
    if requested:
        markers = sorted(requested & set(grouped))
        missing_requested = sorted(requested - set(grouped))
    elif scope == "all":
        markers = sorted(grouped)
        missing_requested = []
    else:
        markers = sorted(marker for marker in grouped if _is_experiment_marker(marker))
        missing_requested = sorted(CORE_EXPERIMENT_MARKERS - set(grouped))

    marker_checks: dict[str, object] = {}
    counts = {"strict": 0, "visible": 0, "deletion": 0, "mismatch": 0}
    inconsistent_duplicates = 0
    deletion_canonical_strict: str | None = None
    deletion_canonical_visible: str | None = None
    for marker in markers:
        entries = grouped[marker]
        consistency = _duplicate_consistency(entries, labels)
        if not consistency["consistent"]:
            inconsistent_duplicates += 1
        deletion_allowlisted = marker in APPROVED_DELETION_DECLARATIONS
        deletion_declaration_exact = deletion_allowlisted and all(
            normalize_source_text(block.revised)
            in APPROVED_DELETION_DECLARATIONS[marker]
            for block in entries
        )
        if deletion_declaration_exact:
            if deletion_canonical_strict is None:
                deletion_canonical_strict = re.sub(
                    r"\s+", " ", bundle.expanded_text
                ).strip()
                deletion_canonical_visible = project_visible(
                    bundle.expanded_text, labels
                )
            assert deletion_canonical_visible is not None
            candidates: list[CanonicalCandidate] = []
            occurrence_checks = _deletion_occurrence_checks(
                marker,
                entries,
                labels,
                source_validation_report,
                deletion_canonical_strict,
                deletion_canonical_visible,
            )
        else:
            candidates = _marker_candidates(marker, bundle, highlights_path)
            occurrence_checks = [
                _check_occurrence(block, candidates, labels) for block in entries
            ]
        for record in occurrence_checks:
            counts[str(record["status"])] += 1
        statuses = {str(record["status"]) for record in occurrence_checks}
        aggregate = (
            "mismatch"
            if "mismatch" in statuses
            else "visible"
            if "visible" in statuses
            else "deletion"
            if "deletion" in statuses
            else "strict"
        )
        marker_checks[marker] = {
            "status": aggregate,
            "duplicate_consistency": consistency,
            "deletion_marker_allowlisted": deletion_allowlisted,
            "deletion_declaration_exact": deletion_declaration_exact,
            "candidate_names": [candidate.name for candidate in candidates],
            "occurrences": occurrence_checks,
        }

    failures = (
        len(parsed.errors)
        + len(bundle.errors)
        + len(missing_requested)
        + counts["mismatch"]
        + inconsistent_duplicates
    )
    return {
        "schema_version": "revised_response_sync_2",
        "status": "passed" if failures == 0 else "failed",
        "scope": {
            "selection": scope if not requested else "explicit_markers",
            "read_only": True,
            "heldout_results_read": False,
            "strict_definition": (
                "whitespace-normalized response Revised text is a literal TeX "
                "fragment of the canonical source"
            ),
            "visible_definition": (
                "all visible response units occur in order after only known "
                "layout-shell removal and AUX reference expansion"
            ),
            "visible_preserves": (
                "prose, equations, numbers, table cells, and figure/table captions"
            ),
            "deletion_definition": (
                "explicit marker allowlist plus exact marker-specific declaration, "
                "strict Original-source validation bound by occurrence lines/count, "
                "and absence of the quoted Original from the active manuscript"
            ),
        },
        "summary": {
            "parsed_verified_change_occurrences": len(parsed.blocks),
            "parsed_unique_markers": len(grouped),
            "checked_unique_markers": len(markers),
            "checked_occurrences": sum(counts.values()),
            "occurrence_counts": counts,
            "inconsistent_duplicate_markers": inconsistent_duplicates,
            "missing_required_or_requested_markers": missing_requested,
            "response_parse_errors": len(parsed.errors),
            "canonical_errors": len(bundle.errors),
            "canonical_files_read": len(bundle.files),
        },
        "response_parse_errors": parsed.errors,
        "canonical_errors": bundle.errors,
        "original_source_validation": {
            "provided": source_validation_report is not None,
            "schema_version": (
                source_validation_report.get("schema_version")
                if source_validation_report is not None
                else None
            ),
            "status": (
                source_validation_report.get("status")
                if source_validation_report is not None
                else None
            ),
        },
        "marker_checks": marker_checks,
    }


def validate_revised_sync(
    response_path: Path = DEFAULT_RESPONSE,
    manuscript_path: Path = DEFAULT_MANUSCRIPT,
    *,
    aux_path: Path | None = DEFAULT_AUX,
    highlights_path: Path | None = DEFAULT_HIGHLIGHTS,
    original_source_path: Path = DEFAULT_ORIGINAL_SOURCE,
    original_aux_path: Path | None = DEFAULT_ORIGINAL_AUX,
    original_framework_path: Path | None = DEFAULT_ORIGINAL_FRAMEWORK,
    scope: str = "experiments",
    selected_markers: Sequence[str] | None = None,
) -> dict[str, object]:
    """Read only declared TeX/AUX inputs and return a serializable report."""

    bundle = read_canonical_bundle(manuscript_path)
    source_validation_report = validate_original_response(
        response_path,
        original_source_path,
        aux_path=original_aux_path,
        framework_path=original_framework_path,
    )
    labels = (
        parse_aux_labels(aux_path.read_text(encoding="utf-8"))
        if aux_path is not None and aux_path.exists()
        else {}
    )
    resolved_highlights = (
        highlights_path.resolve()
        if highlights_path is not None and highlights_path.exists()
        else None
    )
    report = validate_revised_sync_text(
        response_path.read_text(encoding="utf-8"),
        bundle,
        labels=labels,
        highlights_path=resolved_highlights,
        scope=scope,
        selected_markers=selected_markers,
        source_validation_report=source_validation_report,
    )
    report["paths"] = {
        "response": str(response_path.resolve()),
        "active_manuscript": str(manuscript_path.resolve()),
        "active_inputs": sorted(str(path) for path in bundle.files),
        "aux_labels": str(aux_path.resolve()) if aux_path and aux_path.exists() else None,
        "highlights": str(resolved_highlights) if resolved_highlights else None,
        "authoritative_original_manuscript": str(original_source_path.resolve()),
        "original_aux_labels": (
            str(original_aux_path.resolve())
            if original_aux_path is not None and original_aux_path.exists()
            else None
        ),
        "original_framework": (
            str(original_framework_path.resolve())
            if original_framework_path is not None
            and original_framework_path.exists()
            else None
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--aux", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--highlights", type=Path, default=DEFAULT_HIGHLIGHTS)
    parser.add_argument("--original-source", type=Path, default=DEFAULT_ORIGINAL_SOURCE)
    parser.add_argument("--original-aux", type=Path, default=DEFAULT_ORIGINAL_AUX)
    parser.add_argument(
        "--original-framework", type=Path, default=DEFAULT_ORIGINAL_FRAMEWORK
    )
    parser.add_argument("--scope", choices=("experiments", "all"), default="experiments")
    parser.add_argument(
        "--marker",
        action="append",
        dest="markers",
        help="check only this marker (repeatable)",
    )
    args = parser.parse_args()
    report = validate_revised_sync(
        args.response,
        args.manuscript,
        aux_path=args.aux,
        highlights_path=args.highlights,
        original_source_path=args.original_source,
        original_aux_path=args.original_aux,
        original_framework_path=args.original_framework,
        scope=args.scope,
        selected_markers=args.markers,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
