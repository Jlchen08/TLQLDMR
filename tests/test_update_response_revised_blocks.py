import json
import os
import stat
from pathlib import Path

import pytest

from review import update_response_revised_blocks as updater


ROOT = Path(__file__).resolve().parents[1]


def _payload(text: str, newline: str = "\n") -> str:
    return newline.join([r"\begin{quote}", r"\small", text, r"\end{quote}", ""])


def _block(
    marker: str,
    original: str = "Original passage.",
    revised: str = "Old revised passage.",
    *,
    newline: str = "\n",
    grouped: bool = False,
) -> str:
    revised_title = (
        r"\endgroup\noindent$\longrightarrow$\enspace\textbf{Revised text:}\par"
        if grouped
        else r"\noindent$\longrightarrow$\enspace\textbf{Revised text:}\par"
    )
    return newline.join(
        [
            f"% BEGIN VERIFIED CHANGE {marker}",
            r"\noindent\textbf{Location near the experiment.}\par",
            r"\noindent$\longrightarrow$\enspace\textbf{Original text:}\par",
            r"\begin{quote}",
            r"\small",
            original,
            r"\end{quote}",
            "",
            revised_title,
            r"\begin{quote}",
            r"\small",
            revised,
            r"\end{quote}",
            f"% END VERIFIED CHANGE {marker}",
            "",
        ]
    )


def _write_mapping(path: Path, mapping: dict[str, str]) -> None:
    path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")


def test_actual_response_structure_is_supported_without_writing() -> None:
    response = ROOT / "TLQLDMR_response_1st" / "main.tex"
    before = response.read_bytes()

    document = updater.parse_document(before)

    assert len(document.blocks) == before.count(b"% BEGIN VERIFIED CHANGE ")
    assert len({block.marker for block in document.blocks}) == 145
    assert response.read_bytes() == before


def test_repeated_marker_update_preserves_every_byte_outside_payloads(
    tmp_path: Path,
) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    data = (
        "PREAMBLE\n"
        + _block("repeat", "Exact original A.", grouped=True)
        + "BETWEEN\n"
        + _block("untouched", "Exact original B.")
        + _block("repeat", "Exact original A.", revised="Different old payload.")
        + "POSTAMBLE\n"
    ).encode()
    response.write_bytes(data)
    replacement = _payload("New {revised} payload.") + _payload("Second quote.")
    _write_mapping(mapping_file, {"repeat": replacement})
    before = updater.parse_document(data)

    report, code = updater.update_file(response, mapping_file, mode="write")

    assert code == 0
    after = updater.parse_document(response.read_bytes())
    assert updater._outside_chunks(before, {"repeat"}) == updater._outside_chunks(
        after, {"repeat"}
    )
    repeated = [
        updater._payload(after, block)
        for block in after.blocks
        if block.marker == "repeat"
    ]
    assert repeated == [replacement.encode(), replacement.encode()]
    assert report["markers"]["repeat"]["occurrences"] == 2
    assert report["markers"]["repeat"]["payload_sha256"] == updater._sha256(
        replacement.encode()
    )
    assert report["markers"]["untouched"]["mapped"] is False
    assert all(report["verified_invariants"].values())


def test_original_location_title_and_end_marker_are_byte_exact(tmp_path: Path) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    response.write_text(_block("one", "Original with ü and $x$.", grouped=True))
    _write_mapping(mapping_file, {"one": _payload("Replacement.")})
    before = updater.parse_document(response.read_bytes())
    old = before.blocks[0]
    immutable_header = before.data[old.begin_start : old.payload_start]
    immutable_end = before.data[old.payload_end : old.end_end]

    updater.update_file(response, mapping_file, mode="write")

    after = updater.parse_document(response.read_bytes())
    new = after.blocks[0]
    assert after.data[new.begin_start : new.payload_start] == immutable_header
    assert after.data[new.payload_end : new.end_end] == immutable_end
    assert updater._original(before, old) == updater._original(after, new)


def test_dry_run_and_check_never_write(tmp_path: Path) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    response.write_text(_block("one"))
    before = response.read_bytes()
    _write_mapping(mapping_file, {"one": _payload("Replacement.")})

    dry_report, dry_code = updater.update_file(response, mapping_file, mode="dry-run")
    check_report, check_code = updater.update_file(response, mapping_file, mode="check")

    assert dry_code == 0
    assert check_code == 1
    assert dry_report["status"] == check_report["status"] == "would_change"
    assert response.read_bytes() == before


def test_check_is_zero_when_payload_is_current(tmp_path: Path) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    current = _payload("Already current.")
    response.write_text(_block("one", revised="Already current."))
    _write_mapping(mapping_file, {"one": current})

    report, code = updater.update_file(response, mapping_file, mode="check")

    assert code == 0
    assert report["status"] == "up_to_date"


def test_write_uses_replace_and_preserves_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    response.write_text(_block("one"))
    response.chmod(0o640)
    _write_mapping(mapping_file, {"one": _payload("Replacement.")})
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def recording_replace(source, target):
        calls.append((os.fspath(source), os.fspath(target)))
        real_replace(source, target)

    monkeypatch.setattr(updater.os, "replace", recording_replace)
    updater.update_file(response, mapping_file, mode="write")

    assert len(calls) == 1
    assert calls[0][1] == os.fspath(response)
    assert stat.S_IMODE(response.stat().st_mode) == 0o640
    assert not list(tmp_path.glob(".main.tex.tmp-*"))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("% END VERIFIED CHANGE x\n", "orphan/duplicate END"),
        ("% BEGIN VERIFIED CHANGE x\n", "missing END"),
        (
            "% BEGIN VERIFIED CHANGE x\n% BEGIN VERIFIED CHANGE y\n",
            "nested/duplicate BEGIN",
        ),
        (
            _block("x").replace(
                "% END VERIFIED CHANGE x", "% END VERIFIED CHANGE y"
            ),
            "does not match",
        ),
        (
            _block("x").replace("Revised text:", "Replacement wording:"),
            "0 Revised text titles",
        ),
        (
            _block("x").replace(
                "\\begin{quote}\n\\small\nOld revised passage.",
                "\\textbf{Revised text:}\\par\n"
                "\\begin{quote}\n\\small\nOld revised passage.",
            ),
            "2 Revised text titles",
        ),
        (
            _block("x").replace("Original text:", "Prior wording:"),
            "0 Original text titles",
        ),
        ("% BEGIN VERIFIED CHANGE\n", "malformed BEGIN"),
        ("% END VERIFIED CHANGE\n", "malformed END"),
    ],
)
def test_structural_failures(text: str, message: str) -> None:
    with pytest.raises(updater.UpdateError, match=message):
        updater.parse_document(text.encode())


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "empty"),
        ("   \n", "empty"),
        ("plain text\n", "at least one quote"),
        (r"\begin{quote}x\n", "end with"),
        ("outside\n" + _payload("inside"), "outside a quote"),
        (_payload("inside") + "outside\n", "outside a quote"),
        ("\\end{quote}\n", "unmatched quote end"),
        ("\\begin{quote}\nx\n", "unclosed quote"),
        (
            "\\begin{quote}\n\\begin{quote}\nx\n\\end{quote}\n\\end{quote}\n",
            "nests quote",
        ),
        (_payload("% BEGIN VERIFIED CHANGE evil"), "contains a VERIFIED"),
    ],
)
def test_payload_failures(value: str, message: str) -> None:
    with pytest.raises(updater.UpdateError, match=message):
        updater.validate_payload("x", value, b"\n")


def test_mapping_rejects_duplicate_json_keys_and_non_string_payload(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x": "a", "x": "b"}')
    with pytest.raises(updater.UpdateError, match="duplicate JSON key"):
        updater.load_mapping(duplicate, b"\n")

    non_string = tmp_path / "non-string.json"
    non_string.write_text('{"x": 4}')
    with pytest.raises(updater.UpdateError, match="must be a JSON string"):
        updater.load_mapping(non_string, b"\n")


def test_unknown_mapping_marker_requires_explicit_allow(tmp_path: Path) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    response.write_text(_block("known"))
    _write_mapping(
        mapping_file,
        {"known": _payload("New."), "absent": _payload("Ignored only if allowed.")},
    )
    with pytest.raises(updater.UpdateError, match="absent from the response"):
        updater.update_file(response, mapping_file, mode="dry-run")

    report, code = updater.update_file(
        response,
        mapping_file,
        mode="dry-run",
        allow_unknown_markers=True,
    )
    assert code == 0
    assert report["ignored_unknown_markers"] == ["absent"]


def test_allow_unknown_permits_a_noop_mapping_with_no_known_marker(tmp_path: Path) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    response.write_text(_block("known"))
    _write_mapping(mapping_file, {"absent": _payload("Ignored.")})
    before = response.read_bytes()
    report, code = updater.update_file(
        response,
        mapping_file,
        mode="dry-run",
        allow_unknown_markers=True,
    )
    assert code == 0
    assert report["status"] == "up_to_date"
    assert report["mapped_markers"] == []
    assert report["ignored_unknown_markers"] == ["absent"]
    assert response.read_bytes() == before


def test_unmapped_repeated_marker_inconsistency_is_rejected(tmp_path: Path) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    response.write_text(
        _block("mapped")
        + _block("inconsistent", revised="First.")
        + _block("inconsistent", revised="Second.")
    )
    _write_mapping(mapping_file, {"mapped": _payload("New.")})
    with pytest.raises(updater.UpdateError, match="inconsistent Revised payload"):
        updater.update_file(response, mapping_file, mode="dry-run")


def test_crlf_payload_is_inserted_verbatim(tmp_path: Path) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    response.write_bytes(_block("one", newline="\r\n").encode())
    replacement = _payload("CRLF payload.", newline="\r\n")
    _write_mapping(mapping_file, {"one": replacement})

    updater.update_file(response, mapping_file, mode="write")

    data = response.read_bytes()
    assert b"\r\n" in data
    assert not updater.re.search(rb"(?<!\r)\n", data)
    document = updater.parse_document(data)
    assert updater._payload(document, document.blocks[0]) == replacement.encode()


def test_cli_check_exit_codes_and_json_report(tmp_path: Path, capsys) -> None:
    response = tmp_path / "main.tex"
    mapping_file = tmp_path / "mapping.json"
    response.write_text(_block("one"))
    _write_mapping(mapping_file, {"one": _payload("New.")})

    code = updater.main(
        ["--response", str(response), "--mapping", str(mapping_file), "--check"]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 1
    assert output["mode"] == "check"
    assert output["markers"]["one"]["occurrences"] == 1
