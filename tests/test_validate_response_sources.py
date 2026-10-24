import json
from pathlib import Path

import pytest

from review import validate_response_sources as validator


def _block(marker: str, original: str, revised: str = "Revised passage.") -> str:
    return "\n".join(
        [
            f"% BEGIN VERIFIED CHANGE {marker}",
            r"\noindent\textbf{Original text:}",
            r"\begin{quote}",
            r"\small",
            original,
            r"\end{quote}",
            r"\noindent\textbf{Revised text:}",
            r"\begin{quote}",
            r"\small",
            revised,
            r"\end{quote}",
            f"% END VERIFIED CHANGE {marker}",
        ]
    )


def test_parser_preserves_repeated_markers_and_reports_true_label_lines() -> None:
    response = "preamble\n" + _block("repeated", "Old passage.")
    response += "\n" + _block("repeated", "Old passage.")

    parsed = validator.parse_verified_changes(response)

    assert not parsed.errors
    assert [block.marker for block in parsed.blocks] == ["repeated", "repeated"]
    assert parsed.blocks[0].begin_line == 2
    assert parsed.blocks[0].original_label_line == 3
    assert parsed.blocks[0].revised_label_line == 8


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        ("% END VERIFIED CHANGE orphan", "orphan_end"),
        ("% BEGIN VERIFIED CHANGE open", "missing_end"),
        (
            _block("left", "Old.").replace(
                "% END VERIFIED CHANGE left", "% END VERIFIED CHANGE right"
            ),
            "marker_mismatch",
        ),
        (
            _block("labels", "Old.").replace("Original text:", "Prior wording:"),
            "block_labels",
        ),
    ],
)
def test_parser_reports_structural_errors(response: str, kind: str) -> None:
    parsed = validator.parse_verified_changes(response)
    assert kind in {error["kind"] for error in parsed.errors}


def test_strict_normalization_removes_only_response_shell() -> None:
    wrapped = r"""\begin{quote}
\small
Alpha    Table~\ref{tab:x}.
\begin{equation*}a=b\end{equation*}
\end{quote}"""

    assert validator.normalize_source_text(wrapped) == (
        r"Alpha Table~\ref{tab:x}. \begin{equation*}a=b\end{equation*}"
    )
    assert validator.normalize_source_text(wrapped) != "Alpha Table 7."


def test_visible_projection_removes_formatting_but_preserves_semantics() -> None:
    formatted = r"{\color{blue}\textbf{Table}}~\ref{tab:x}: $C_T=2$."
    plain = r"Table 7: $C_T=2$."
    labels = {"tab:x": "7"}

    assert validator.visible_tex_text(formatted, labels) == validator.visible_tex_text(
        plain, labels
    )
    changed = formatted.replace("C_T", "C_S")
    assert validator.visible_tex_text(changed, labels) != validator.visible_tex_text(
        plain, labels
    )


def test_validation_distinguishes_strict_visible_addition_and_mismatch() -> None:
    source = "\n".join(
        [
            r"Exact source paragraph with Table~\ref{tab:x}.",
            r"Visible source paragraph with Table~\ref{tab:x}.",
            r"Coefficient is $C_T=2$.",
        ]
    )
    declaration = (
        "No corresponding passage appears in the original manuscript. "
        "The following text is added in the chronological evaluation subsection "
        "of the revised manuscript."
    )
    response = "\n".join(
        [
            _block("exact", r"Exact source paragraph with Table~\ref{tab:x}."),
            _block("visible", "Visible source paragraph with Table 7."),
            _block("addition", declaration),
            _block("mismatch", r"Coefficient is $C_T=3$."),
        ]
    )
    specs = {
        "exact": validator.SourceSpec("manuscript", 1, 1),
        "visible": validator.SourceSpec("manuscript", 2, 2),
        "mismatch": validator.SourceSpec("manuscript", 3, 3),
    }

    report = validator.validate_response_text(
        response,
        {"manuscript": source},
        labels={"tab:x": "7"},
        source_specs=specs,
    )

    assert report["summary"]["source_occurrence_counts"] == {
        "strict_match": 1,
        "declared_addition": 1,
        "visible_only": 1,
        "unmatched": 1,
        "mapping_error": 0,
    }
    assert report["status"] == "failed"
    assert report["source_checks"]["visible"]["visible_match"] is True
    assert report["source_checks"]["visible"]["strict_match"] is False


def test_visible_validation_preserves_tex_comment_line_boundaries() -> None:
    source = "\n".join(
        [
            r"\begin{table}",
            r"\caption{Caption}",
            r"\begin{tabular}{l}",
            r"Value \\",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    response_table = "\n".join(
        [
            r"\begingroup",
            r"\textit{Caption}",
            r"\begin{adjustbox}{max width=\linewidth}%",
            r"\begin{tabular}{l}",
            r"Value \\",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"\endgroup",
        ]
    )

    report = validator.validate_response_text(
        _block("table", response_table),
        {"manuscript": source},
        source_specs={"table": validator.SourceSpec("manuscript", 1, 6)},
    )

    assert report["source_checks"]["table"]["status"] == "visible_only"


def test_visible_validation_accepts_a_mapped_complete_excerpt() -> None:
    source = "First complete sentence. Adjacent source sentence."
    report = validator.validate_response_text(
        _block("excerpt", "First complete sentence."),
        {"manuscript": source},
        source_specs={"excerpt": validator.SourceSpec("manuscript", 1, 1)},
    )

    check = report["source_checks"]["excerpt"]
    assert check["status"] == "visible_only"
    assert check["visible_relation"] == "response_excerpt"


def test_only_registered_addition_declarations_receive_exemption() -> None:
    first, second = sorted(validator.APPROVED_ABSENCE_DECLARATIONS)[:2]
    response = "\n".join(
        [
            _block("addition", first, "First revised wording."),
            _block("addition", second, "Second revised wording."),
        ]
    )

    report = validator.validate_response_text(
        response, {"manuscript": ""}, source_specs={}
    )

    assert report["original_consistency"]["markers"]["addition"]["consistent"]
    assert report["original_consistency"]["markers"]["addition"][
        "allowed_distinct_absence_declarations"
    ]
    assert not report["revised_consistency"]["markers"]["addition"]["consistent"]
    assert validator.is_absence_declaration(first)
    assert not validator.is_absence_declaration(first + " Fabricated old claim follows.")

    mapped_claim = validator.validate_response_text(
        _block("mapped", first),
        {"manuscript": "Existing source passage."},
        source_specs={"mapped": validator.SourceSpec("manuscript", 1, 1)},
    )
    assert mapped_claim["source_checks"]["mapped"]["status"] == "unmatched"


def test_default_authority_and_framework_marker_scope() -> None:
    assert len(validator.CANONICAL_SOURCE_SPECS) == 62
    assert validator.DEFAULT_SOURCE == (
        validator.ROOT / "TLQLDMR_manuscript" / "TLQLDMR_energy.tex"
    )
    assert validator.DEFAULT_FRAMEWORK == (
        validator.ROOT / "TLQLDMR_manuscript" / "framework.tex"
    )
    framework_text = r"Feature-space alignment\\MMD in RKHS"
    response = _block("unmapped", framework_text)

    report = validator.validate_response_text(
        response,
        {"manuscript": "No such text.", "framework": framework_text},
        source_specs={},
    )

    assert report["source_checks"]["unmapped"]["status"] == "unmatched"

    scoped = validator.validate_response_text(
        _block("framework_1", framework_text),
        {"manuscript": "No such text.", "framework": framework_text},
        source_specs={},
    )
    assert scoped["source_checks"]["framework_1"]["status"] == "strict_match"

    named = validator.validate_response_text(
        _block("framework_target_domain", framework_text),
        {"manuscript": "No such text.", "framework": framework_text},
        source_specs={},
    )
    assert named["source_checks"]["framework_target_domain"]["status"] == "strict_match"


def test_report_is_json_serializable() -> None:
    report = validator.validate_response_text(
        _block("exact", "Exact."),
        {"manuscript": "Exact."},
        source_specs={"exact": validator.SourceSpec("manuscript", 1, 1)},
    )
    assert json.loads(json.dumps(report))["status"] == "passed"
    assert Path(validator.DEFAULT_SOURCE).name == "TLQLDMR_energy.tex"
