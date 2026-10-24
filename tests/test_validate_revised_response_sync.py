from pathlib import Path

from review import validate_revised_response_sync as validator


def _response_block(marker: str, revised: str) -> str:
    return "\n".join(
        [
            f"% BEGIN VERIFIED CHANGE {marker}",
            r"\noindent\textbf{Original text:}",
            r"\begin{quote}",
            r"\small",
            "Original passage.",
            r"\end{quote}",
            r"\noindent\textbf{Revised text:}",
            r"\begin{quote}",
            r"\small",
            revised,
            r"\end{quote}",
            f"% END VERIFIED CHANGE {marker}",
        ]
    )


def _bundle(tmp_path: Path, body: str, inputs: dict[str, str] | None = None):
    main = tmp_path / "main.tex"
    main.write_text(body, encoding="utf-8")
    for name, value in (inputs or {}).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    return validator.read_canonical_bundle(main)


def _strict_source_report(response: str, marker: str) -> dict[str, object]:
    blocks = [
        block
        for block in validator.parse_verified_changes(response).blocks
        if block.marker == marker
    ]
    return {
        "schema_version": "response_source_validation_1",
        "status": "passed",
        "source_checks": {
            marker: {
                "status": "strict_match",
                "occurrences": len(blocks),
                "begin_lines": [block.begin_line for block in blocks],
            }
        },
        "original_consistency": {
            "markers": {marker: {"consistent": True}}
        },
    }


def test_strict_match_uses_same_name_canonical_block(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        "\n".join(
            [
                "% BEGIN SELECTION BLOCK selection_results",
                r"{\color{blue}Exact result: RMSE is 4.125\,MW.}",
                "% END SELECTION BLOCK selection_results",
            ]
        ),
    )
    response = _response_block(
        "selection_results", r"{\color{blue}Exact result: RMSE is 4.125\,MW.}"
    )

    report = validator.validate_revised_sync_text(
        response, bundle, selected_markers=["selection_results"]
    )

    check = report["marker_checks"]["selection_results"]
    assert report["status"] == "passed"
    assert report["summary"]["occurrence_counts"] == {
        "strict": 1,
        "visible": 0,
        "deletion": 0,
        "mismatch": 0,
    }
    assert check["occurrences"][0]["canonical"] == (
        "SELECTION BLOCK selection_results"
    )


def test_visible_match_allows_layout_shells_and_expanded_refs(tmp_path: Path) -> None:
    canonical = "\n".join(
        [
            "% BEGIN SELECTION BLOCK selection_tables",
            r"\subsection{Results}\label{sec:results}",
            r"Table~\ref{tab:value} reports the locked value.",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Locked validation result.}",
            r"\label{tab:value}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lr}",
            "Method & RMSE \\\\",
            "Full & 4.125 \\\\",
            r"\end{tabular}}",
            r"\end{table}",
            "% END SELECTION BLOCK selection_tables",
        ]
    )
    response_value = "\n".join(
        [
            r"\textbf{Results}",
            "Table 7 reports the locked value.",
            r"\begingroup\centering",
            r"\textit{Locked validation result.}",
            r"\begin{adjustbox}{max width=\linewidth}",
            r"\begin{tabular}{lr}",
            "Method & RMSE \\\\",
            "Full & 4.125 \\\\",
            r"\end{tabular}",
            r"\end{adjustbox}\endgroup",
        ]
    )
    bundle = _bundle(tmp_path, canonical)

    report = validator.validate_revised_sync_text(
        _response_block("selection_tables", response_value),
        bundle,
        labels={"tab:value": "7"},
        selected_markers=["selection_tables"],
    )

    assert report["status"] == "passed"
    assert report["marker_checks"]["selection_tables"]["status"] == "visible"


def test_visible_match_rejects_changed_number(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        "\n".join(
            [
                "% BEGIN SELECTION BLOCK selection_results",
                "Locked RMSE is 4.125 MW.",
                "% END SELECTION BLOCK selection_results",
            ]
        ),
    )

    report = validator.validate_revised_sync_text(
        _response_block("selection_results", "Locked RMSE is 4.126 MW."),
        bundle,
        selected_markers=["selection_results"],
    )

    occurrence = report["marker_checks"]["selection_results"]["occurrences"][0]
    assert report["status"] == "failed"
    assert occurrence["status"] == "mismatch"
    assert occurrence["candidate_failures"][0]["difference"]["response_unit"] == (
        "Locked RMSE is 4.126 MW."
    )


def test_single_line_break_cannot_hide_omitted_sentence_words(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "Alpha required middle omega.")

    report = validator.validate_revised_sync_text(
        _response_block("selection_results", "Alpha\nomega."),
        bundle,
        selected_markers=["selection_results"],
    )

    assert report["status"] == "failed"
    assert report["marker_checks"]["selection_results"]["status"] == "mismatch"


def test_visible_units_may_skip_intervening_canonical_float(tmp_path: Path) -> None:
    bundle = _bundle(
        tmp_path,
        "\n".join(
            [
                "% BEGIN SELECTION BLOCK selection_results",
                "First reported sentence.",
                r"\begin{table}\caption{Detailed table.}A & 1 \\\end{table}",
                "Final reported sentence.",
                "% END SELECTION BLOCK selection_results",
            ]
        ),
    )
    revised = "First reported sentence.\n\nFinal reported sentence."

    report = validator.validate_revised_sync_text(
        _response_block("selection_results", revised),
        bundle,
        selected_markers=["selection_results"],
    )

    assert report["status"] == "passed"
    assert report["marker_checks"]["selection_results"]["status"] == "visible"


def test_duplicate_marker_consistency_is_reported_separately(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, "Canonical sentence.")
    response = "\n".join(
        [
            _response_block("selection_results", "Canonical sentence."),
            _response_block(
                "selection_results", r"\textbf{Canonical sentence.}"
            ),
        ]
    )

    report = validator.validate_revised_sync_text(
        response, bundle, selected_markers=["selection_results"]
    )

    consistency = report["marker_checks"]["selection_results"][
        "duplicate_consistency"
    ]
    assert report["status"] == "failed"
    assert not consistency["consistent"]
    assert consistency["visible_consistent"]
    assert report["summary"]["inconsistent_duplicate_markers"] == 1


def test_bundle_reads_only_local_reachable_inputs_and_rejects_escape(
    tmp_path: Path,
) -> None:
    manuscript = tmp_path / "manuscript"
    manuscript.mkdir()
    outside = tmp_path / "heldout_secret.tex"
    outside.write_text("must not be read", encoding="utf-8")
    main = manuscript / "main.tex"
    main.write_text(
        "\\input{included}\n\\input{../heldout_secret}", encoding="utf-8"
    )
    (manuscript / "included.tex").write_text(
        "% BEGIN STUDY BLOCK chronological_protocol\n"
        "Reachable passage.\n"
        "% END STUDY BLOCK chronological_protocol\n",
        encoding="utf-8",
    )

    bundle = validator.read_canonical_bundle(main)

    assert set(bundle.files) == {main.resolve(), (manuscript / "included.tex").resolve()}
    assert "must not be read" not in bundle.expanded_text
    assert {error["kind"] for error in bundle.errors} == {"unsafe_input"}
    assert bundle.blocks["chronological_protocol"][0].text == "Reachable passage."


def test_allowlisted_exact_deletion_requires_and_uses_original_source_evidence(
    tmp_path: Path,
) -> None:
    marker = "original_experiment_1_setup_removed"
    response = _response_block(
        marker, validator._GENERIC_DELETION_DECLARATION
    )
    bundle = _bundle(tmp_path, "Current revised material only.")

    report = validator.validate_revised_sync_text(
        response,
        bundle,
        selected_markers=[marker],
        source_validation_report=_strict_source_report(response, marker),
    )

    occurrence = report["marker_checks"][marker]["occurrences"][0]
    assert report["status"] == "passed"
    assert report["summary"]["occurrence_counts"]["deletion"] == 1
    assert occurrence["status"] == "deletion"
    assert occurrence["deletion_marker_allowlisted"] is True
    assert occurrence["deletion_declaration_exact"] is True
    assert occurrence["original_source_status"] == "strict_match"


def test_removed_suffix_and_approved_words_do_not_classify_arbitrary_marker(
    tmp_path: Path,
) -> None:
    marker = "arbitrary_claim_removed"
    response = _response_block(marker, validator._GENERIC_DELETION_DECLARATION)
    bundle = _bundle(tmp_path, "Current revised material only.")

    report = validator.validate_revised_sync_text(
        response,
        bundle,
        selected_markers=[marker],
        source_validation_report=_strict_source_report(response, marker),
    )

    occurrence = report["marker_checks"][marker]["occurrences"][0]
    assert report["status"] == "failed"
    assert occurrence["status"] == "mismatch"
    assert "deletion_marker_allowlisted" not in occurrence


def test_allowlisted_deletion_rejects_nonexact_declaration(tmp_path: Path) -> None:
    marker = "original_experiment_1_setup_removed"
    response = _response_block(
        marker,
        validator._GENERIC_DELETION_DECLARATION
        + " A replacement may appear elsewhere.",
    )
    bundle = _bundle(tmp_path, "Current revised material only.")

    report = validator.validate_revised_sync_text(
        response,
        bundle,
        selected_markers=[marker],
        source_validation_report=_strict_source_report(response, marker),
    )

    occurrence = report["marker_checks"][marker]["occurrences"][0]
    assert report["status"] == "failed"
    assert occurrence["status"] == "mismatch"
    assert report["marker_checks"][marker]["deletion_marker_allowlisted"] is True
    assert report["marker_checks"][marker]["deletion_declaration_exact"] is False
    assert "deletion_rejections" not in occurrence


def test_allowlisted_marker_may_become_active_when_text_matches(tmp_path: Path) -> None:
    marker = "ablation"
    revised = "The replacement ablation is active manuscript content."
    response = _response_block(marker, revised)
    bundle = _bundle(tmp_path, revised)

    report = validator.validate_revised_sync_text(
        response,
        bundle,
        selected_markers=[marker],
    )

    check = report["marker_checks"][marker]
    assert report["status"] == "passed"
    assert check["status"] == "strict"
    assert check["deletion_marker_allowlisted"] is True
    assert check["deletion_declaration_exact"] is False


def test_allowlisted_deletion_rejects_failed_original_source_validation(
    tmp_path: Path,
) -> None:
    marker = "original_experiment_1_setup_removed"
    response = _response_block(marker, validator._GENERIC_DELETION_DECLARATION)
    source_report = _strict_source_report(response, marker)
    source_report["status"] = "failed"
    source_report["source_checks"][marker]["status"] = "unmatched"
    bundle = _bundle(tmp_path, "Current revised material only.")

    report = validator.validate_revised_sync_text(
        response,
        bundle,
        selected_markers=[marker],
        source_validation_report=source_report,
    )

    occurrence = report["marker_checks"][marker]["occurrences"][0]
    assert report["status"] == "failed"
    assert occurrence["status"] == "mismatch"
    assert any(
        "not a strict match" in reason
        for reason in occurrence["deletion_rejections"]
    )


def test_allowlisted_deletion_binds_source_evidence_to_occurrence_lines(
    tmp_path: Path,
) -> None:
    marker = "original_experiment_1_setup_removed"
    response = _response_block(marker, validator._GENERIC_DELETION_DECLARATION)
    source_report = _strict_source_report(response, marker)
    source_report["source_checks"][marker]["begin_lines"] = [999]
    bundle = _bundle(tmp_path, "Current revised material only.")

    report = validator.validate_revised_sync_text(
        response,
        bundle,
        selected_markers=[marker],
        source_validation_report=source_report,
    )

    occurrence = report["marker_checks"][marker]["occurrences"][0]
    assert report["status"] == "failed"
    assert any(
        "occurrence lines" in reason
        for reason in occurrence["deletion_rejections"]
    )


def test_allowlisted_deletion_rejects_original_still_in_revised_manuscript(
    tmp_path: Path,
) -> None:
    marker = "original_experiment_1_setup_removed"
    response = _response_block(marker, validator._GENERIC_DELETION_DECLARATION)
    bundle = _bundle(tmp_path, "Original passage.")

    report = validator.validate_revised_sync_text(
        response,
        bundle,
        selected_markers=[marker],
        source_validation_report=_strict_source_report(response, marker),
    )

    occurrence = report["marker_checks"][marker]["occurrences"][0]
    assert report["status"] == "failed"
    assert occurrence["status"] == "mismatch"
    assert any(
        "still occurs" in reason for reason in occurrence["deletion_rejections"]
    )
