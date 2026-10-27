#!/usr/bin/env python3
"""Safely replace Revised-text payloads in VERIFIED CHANGE blocks.

The mapping file is a JSON object whose keys are marker names and whose values
are complete LaTeX payloads.  A payload must consist of one or more complete
``quote`` environments and must end with a newline.  The bytes of the payload
are inserted verbatim after the existing ``Revised text:`` title line.

This utility intentionally knows nothing about experiment outputs.  It edits
only the response letter and verifies that marker headings, location text,
Original-text sections, and every byte outside the requested payload spans are
unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESPONSE = ROOT / "TLQLDMR_response_1st" / "main.tex"

_BEGIN_RE = re.compile(
    rb"^[ \t]*%[ \t]*BEGIN[ \t]+VERIFIED[ \t]+CHANGE[ \t]+"
    rb"([A-Za-z0-9_.:-]+)[ \t]*$"
)
_END_RE = re.compile(
    rb"^[ \t]*%[ \t]*END[ \t]+VERIFIED[ \t]+CHANGE[ \t]+"
    rb"([A-Za-z0-9_.:-]+)[ \t]*$"
)
_BEGIN_PREFIX_RE = re.compile(
    rb"^[ \t]*%[ \t]*BEGIN[ \t]+VERIFIED[ \t]+CHANGE(?:[ \t]|$)"
)
_END_PREFIX_RE = re.compile(
    rb"^[ \t]*%[ \t]*END[ \t]+VERIFIED[ \t]+CHANGE(?:[ \t]|$)"
)
_ORIGINAL_TITLE = rb"\textbf{Original text:}\par"
_REVISED_TITLE = rb"\textbf{Revised text:}\par"
_REPLACEMENT_TITLES = (
    _REVISED_TITLE,
    rb"\textbf{Deletion (no replacement text):}\par",
)
_QUOTE_TOKEN_RE = re.compile(rb"\\(begin|end)\{quote\}")
_FORBIDDEN_PAYLOAD_MARKERS = (
    b"BEGIN VERIFIED CHANGE",
    b"END VERIFIED CHANGE",
)


class UpdateError(RuntimeError):
    """Raised when the input cannot be updated without violating invariants."""


@dataclass(frozen=True)
class _Line:
    number: int
    start: int
    end: int
    body: bytes


@dataclass(frozen=True)
class Block:
    """Byte spans for one well-formed VERIFIED CHANGE occurrence."""

    marker: str
    occurrence: int
    begin_line: int
    end_line: int
    begin_start: int
    begin_end: int
    end_start: int
    end_end: int
    original_start: int
    original_end: int
    revised_title_start: int
    revised_title_end: int
    payload_start: int
    payload_end: int


@dataclass(frozen=True)
class Document:
    data: bytes
    blocks: tuple[Block, ...]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _line_body(raw: bytes) -> bytes:
    if raw.endswith(b"\r\n"):
        return raw[:-2]
    if raw.endswith(b"\n") or raw.endswith(b"\r"):
        return raw[:-1]
    return raw


def _lines(data: bytes) -> list[_Line]:
    result: list[_Line] = []
    offset = 0
    for number, raw in enumerate(data.splitlines(keepends=True), start=1):
        result.append(_Line(number, offset, offset + len(raw), _line_body(raw)))
        offset += len(raw)
    return result


def _single_title_line(
    lines: Sequence[_Line], token: bytes | tuple[bytes, ...], title: str, marker: str, begin_line: int
) -> _Line:
    tokens = (token,) if isinstance(token, bytes) else token
    matches = [line for line in lines if any(item in line.body for item in tokens)]
    if len(matches) != 1:
        raise UpdateError(
            f"marker {marker!r} beginning on line {begin_line} has "
            f"{len(matches)} {title} titles; exactly one is required"
        )
    line = matches[0]
    if not line.body.rstrip().endswith(tokens):
        raise UpdateError(
            f"marker {marker!r} line {line.number} has text after its {title} title"
        )
    return line


def parse_document(data: bytes) -> Document:
    """Parse and strictly validate every VERIFIED CHANGE block."""

    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UpdateError(f"response is not valid UTF-8: {exc}") from exc

    lines = _lines(data)
    open_index: int | None = None
    open_marker: str | None = None
    occurrences: defaultdict[str, int] = defaultdict(int)
    blocks: list[Block] = []

    for index, line in enumerate(lines):
        begin_match = _BEGIN_RE.fullmatch(line.body)
        end_match = _END_RE.fullmatch(line.body)

        if begin_match:
            marker = begin_match.group(1).decode("ascii")
            if open_index is not None:
                assert open_marker is not None
                raise UpdateError(
                    f"nested/duplicate BEGIN for {marker!r} on line {line.number}; "
                    f"block {open_marker!r} from line {lines[open_index].number} is open"
                )
            open_index = index
            open_marker = marker
            continue
        if _BEGIN_PREFIX_RE.match(line.body):
            raise UpdateError(f"malformed BEGIN marker on line {line.number}")

        if end_match:
            marker = end_match.group(1).decode("ascii")
            if open_index is None or open_marker is None:
                raise UpdateError(
                    f"orphan/duplicate END for {marker!r} on line {line.number}"
                )
            if marker != open_marker:
                raise UpdateError(
                    f"END marker {marker!r} on line {line.number} does not match "
                    f"BEGIN marker {open_marker!r} on line {lines[open_index].number}"
                )

            interior = lines[open_index + 1 : index]
            original = _single_title_line(
                interior,
                _ORIGINAL_TITLE,
                "Original text",
                marker,
                lines[open_index].number,
            )
            revised = _single_title_line(
                interior,
                _REPLACEMENT_TITLES,
                "Revised text",
                marker,
                lines[open_index].number,
            )
            if original.start >= revised.start:
                raise UpdateError(
                    f"marker {marker!r} on line {lines[open_index].number} places "
                    "Original text after Revised text"
                )
            if revised.end == revised.start + len(revised.body):
                raise UpdateError(
                    f"Revised text title for marker {marker!r} on line "
                    f"{revised.number} has no terminating newline"
                )

            occurrences[marker] += 1
            blocks.append(
                Block(
                    marker=marker,
                    occurrence=occurrences[marker],
                    begin_line=lines[open_index].number,
                    end_line=line.number,
                    begin_start=lines[open_index].start,
                    begin_end=lines[open_index].end,
                    end_start=line.start,
                    end_end=line.end,
                    original_start=original.start,
                    original_end=revised.start,
                    revised_title_start=revised.start,
                    revised_title_end=revised.end,
                    payload_start=revised.end,
                    payload_end=line.start,
                )
            )
            open_index = None
            open_marker = None
            continue
        if _END_PREFIX_RE.match(line.body):
            raise UpdateError(f"malformed END marker on line {line.number}")

    if open_index is not None and open_marker is not None:
        raise UpdateError(
            f"missing END for marker {open_marker!r} beginning on line "
            f"{lines[open_index].number}"
        )
    if not blocks:
        raise UpdateError("response contains no VERIFIED CHANGE blocks")
    return Document(data=data, blocks=tuple(blocks))


def _no_duplicate_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise UpdateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def validate_payload(marker: str, value: str, newline: bytes) -> bytes:
    if not isinstance(value, str):
        raise UpdateError(f"payload for marker {marker!r} must be a JSON string")
    if not value.strip():
        raise UpdateError(f"payload for marker {marker!r} is empty")
    try:
        payload = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UpdateError(f"payload for marker {marker!r} is not UTF-8: {exc}") from exc
    if b"\x00" in payload:
        raise UpdateError(f"payload for marker {marker!r} contains a NUL byte")
    if any(token in payload for token in _FORBIDDEN_PAYLOAD_MARKERS):
        raise UpdateError(
            f"payload for marker {marker!r} contains a VERIFIED CHANGE marker"
        )
    if not payload.endswith(newline):
        ending = "CRLF" if newline == b"\r\n" else "LF"
        raise UpdateError(
            f"payload for marker {marker!r} must end with the response's {ending} newline"
        )
    if newline == b"\n" and b"\r\n" in payload:
        raise UpdateError(f"payload for marker {marker!r} mixes CRLF into an LF file")
    if newline == b"\r\n" and re.search(rb"(?<!\r)\n", payload):
        raise UpdateError(f"payload for marker {marker!r} mixes LF into a CRLF file")

    position = 0
    open_quote = False
    quotes = 0
    for match in _QUOTE_TOKEN_RE.finditer(payload):
        between = payload[position : match.start()]
        kind = match.group(1)
        if not open_quote:
            if between.strip():
                raise UpdateError(
                    f"payload for marker {marker!r} has content outside a quote environment"
                )
            if kind != b"begin":
                raise UpdateError(
                    f"payload for marker {marker!r} has an unmatched quote end"
                )
            open_quote = True
            quotes += 1
        else:
            if kind != b"end":
                raise UpdateError(
                    f"payload for marker {marker!r} nests quote environments"
                )
            open_quote = False
        position = match.end()
    if open_quote:
        raise UpdateError(f"payload for marker {marker!r} has an unclosed quote")
    if quotes == 0:
        raise UpdateError(
            f"payload for marker {marker!r} must contain at least one quote environment"
        )
    if payload[position:].strip():
        raise UpdateError(
            f"payload for marker {marker!r} has content outside a quote environment"
        )
    return payload


def load_mapping(path: Path, newline: bytes) -> dict[str, bytes]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise UpdateError(f"cannot read mapping {path}: {exc}") from exc
    try:
        value = json.loads(text, object_pairs_hook=_no_duplicate_object)
    except UpdateError:
        raise
    except json.JSONDecodeError as exc:
        raise UpdateError(f"invalid JSON mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise UpdateError("mapping root must be a JSON object")
    mapping: dict[str, bytes] = {}
    for marker, payload in value.items():
        if not isinstance(marker, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", marker):
            raise UpdateError(f"invalid mapping marker name {marker!r}")
        mapping[marker] = validate_payload(marker, payload, newline)
    if not mapping:
        raise UpdateError("mapping is empty")
    return mapping


def _detect_newline(data: bytes) -> bytes:
    first_lf = data.find(b"\n")
    if first_lf < 0:
        raise UpdateError("response has no newline")
    newline = b"\r\n" if first_lf > 0 and data[first_lf - 1] == 13 else b"\n"
    if newline == b"\n" and b"\r\n" in data:
        raise UpdateError("response mixes LF and CRLF newlines")
    if newline == b"\r\n" and re.search(rb"(?<!\r)\n", data):
        raise UpdateError("response mixes CRLF and LF newlines")
    return newline


def _payload(document: Document, block: Block) -> bytes:
    return document.data[block.payload_start : block.payload_end]


def _original(document: Document, block: Block) -> bytes:
    return document.data[block.original_start : block.original_end]


def apply_mapping(document: Document, mapping: Mapping[str, bytes]) -> bytes:
    chunks: list[bytes] = []
    cursor = 0
    for block in document.blocks:
        chunks.append(document.data[cursor : block.payload_start])
        chunks.append(mapping.get(block.marker, _payload(document, block)))
        cursor = block.payload_end
    chunks.append(document.data[cursor:])
    return b"".join(chunks)


def _outside_chunks(document: Document, target_markers: set[str]) -> list[bytes]:
    result: list[bytes] = []
    cursor = 0
    for block in document.blocks:
        if block.marker in target_markers:
            result.append(document.data[cursor : block.payload_start])
            cursor = block.payload_end
    result.append(document.data[cursor:])
    return result


def verify_transformation(
    before: Document, after_data: bytes, mapping: Mapping[str, bytes]
) -> Document:
    """Verify all preservation and repeated-payload invariants."""

    after = parse_document(after_data)
    if len(before.blocks) != len(after.blocks):
        raise UpdateError("block count changed during transformation")
    for old, new in zip(before.blocks, after.blocks):
        # Payloads may legitimately contain a different number of lines, so
        # physical line numbers after the first replacement are not stable.
        # Marker order and per-marker occurrence number are the stable identity.
        identity = (old.marker, old.occurrence)
        new_identity = (new.marker, new.occurrence)
        if identity != new_identity:
            raise UpdateError(
                "block identity changed during transformation: "
                f"{identity!r} != {new_identity!r}"
            )
        if _original(before, old) != _original(after, new):
            raise UpdateError(
                f"Original-text bytes changed for marker {old.marker!r} "
                f"occurrence {old.occurrence}"
            )

    targets = set(mapping)
    if _outside_chunks(before, targets) != _outside_chunks(after, targets):
        raise UpdateError("bytes outside requested Revised-text payloads changed")

    by_marker: defaultdict[str, list[bytes]] = defaultdict(list)
    for block in after.blocks:
        by_marker[block.marker].append(_payload(after, block))
    for marker, payloads in by_marker.items():
        if len(set(payloads)) != 1:
            raise UpdateError(
                f"same-name marker {marker!r} has inconsistent Revised payload bytes"
            )
        if marker in mapping and payloads[0] != mapping[marker]:
            raise UpdateError(f"payload for marker {marker!r} was not inserted verbatim")
    return after


def build_report(
    before: Document,
    after: Document,
    mapping: Mapping[str, bytes],
    *,
    mode: str,
    changed: bool,
    ignored_unknown: Sequence[str],
) -> dict[str, object]:
    grouped: defaultdict[str, list[Block]] = defaultdict(list)
    for block in after.blocks:
        grouped[block.marker].append(block)
    markers: dict[str, object] = {}
    for marker in sorted(grouped):
        blocks = grouped[marker]
        payload_hashes = [_sha256(_payload(after, block)) for block in blocks]
        original_hashes = [_sha256(_original(after, block)) for block in blocks]
        markers[marker] = {
            "occurrences": len(blocks),
            "mapped": marker in mapping,
            "payload_sha256": payload_hashes[0],
            "original_sha256": (
                original_hashes[0]
                if len(set(original_hashes)) == 1
                else original_hashes
            ),
        }
    status = (
        "updated"
        if mode == "write" and changed
        else "would_change"
        if changed
        else "up_to_date"
    )
    return {
        "status": status,
        "mode": mode,
        "changed": changed,
        "file_sha256_before": _sha256(before.data),
        "file_sha256_after": _sha256(after.data),
        "block_count": len(after.blocks),
        "marker_count": len(grouped),
        "mapped_markers": sorted(mapping),
        "ignored_unknown_markers": list(ignored_unknown),
        "verified_invariants": {
            "original_sections_unchanged": True,
            "outside_target_payloads_unchanged": True,
            "same_marker_payloads_identical": True,
            "mapped_payloads_inserted_verbatim": True,
        },
        "markers": markers,
    }


@contextlib.contextmanager
def _locked_response(path: Path):
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise UpdateError(f"cannot open response {path}: {exc}") from exc
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _atomic_write_locked(path: Path, handle, before: bytes, expected: bytes) -> None:
    info = os.fstat(handle.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise UpdateError(f"response is not a regular file: {path}")
    if path.is_symlink():
        raise UpdateError(f"refusing to replace symlink response: {path}")

    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.tmp-", dir=path.parent
        )
        with os.fdopen(descriptor, "wb") as temporary:
            temporary.write(expected)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, stat.S_IMODE(info.st_mode))
        if Path(temporary_name).read_bytes() != expected:
            raise UpdateError("staged response does not match verified output")
        try:
            current = path.read_bytes()
        except OSError as exc:
            raise UpdateError(f"cannot re-read response before publish: {exc}") from exc
        if current != before:
            raise UpdateError("response changed concurrently; refusing to overwrite it")
        os.replace(temporary_name, path)
        temporary_name = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name)


def update_file(
    response: Path,
    mapping_path: Path,
    *,
    mode: str,
    allow_unknown_markers: bool = False,
) -> tuple[dict[str, object], int]:
    """Run one checked update and return ``(report, exit_code)``."""

    if mode not in {"write", "check", "dry-run"}:
        raise ValueError(f"invalid mode {mode!r}")
    with _locked_response(response) as handle:
        before_data = handle.read()
        before = parse_document(before_data)
        newline = _detect_newline(before_data)
        requested = load_mapping(mapping_path, newline)
        known = {block.marker for block in before.blocks}
        unknown = sorted(set(requested) - known)
        if unknown and not allow_unknown_markers:
            raise UpdateError(
                "mapping contains marker(s) absent from the response: " + ", ".join(unknown)
            )
        mapping = {key: value for key, value in requested.items() if key in known}

        expected_data = apply_mapping(before, mapping)
        expected = verify_transformation(before, expected_data, mapping)
        changed = expected_data != before_data
        report = build_report(
            before,
            expected,
            mapping,
            mode=mode,
            changed=changed,
            ignored_unknown=unknown,
        )

        if mode == "write" and changed:
            _atomic_write_locked(response, handle, before_data, expected_data)
            written_data = response.read_bytes()
            if written_data != expected_data:
                raise UpdateError("post-write response differs from verified output")
            written = verify_transformation(before, written_data, mapping)
            report = build_report(
                before,
                written,
                mapping,
                mode=mode,
                changed=changed,
                ignored_unknown=unknown,
            )
        return report, 1 if mode == "check" and changed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--check", action="store_true", help="exit 1 if an update would be required"
    )
    modes.add_argument(
        "--dry-run", action="store_true", help="validate and preview without writing"
    )
    parser.add_argument(
        "--allow-unknown-markers",
        action="store_true",
        help="ignore mapping keys that are absent from the response",
    )
    args = parser.parse_args(argv)
    mode = "check" if args.check else "dry-run" if args.dry_run else "write"
    try:
        report, code = update_file(
            args.response,
            args.mapping,
            mode=mode,
            allow_unknown_markers=args.allow_unknown_markers,
        )
    except UpdateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
