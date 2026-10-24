"""Focused safety tests for the selection-v2 S1 builder and verifier."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

import review.build_reproducibility_supplement as builder
import review.verify_s1_package as verifier


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest:
            lines.append(f"{_sha(path)}  {path.stat().st_size:12d}  {path.relative_to(root)}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_incomplete_v2_refuses_before_touching_existing_package(tmp_path):
    source = tmp_path / "source"
    run = source / builder.SELECTION_REL
    run.mkdir(parents=True)
    (run / "manifest.json").write_text('{"status":"frozen"}\n', encoding="utf-8")
    # A malformed held-out file proves the completion gate returns before it is read.
    (run / "results.json").write_text("not json", encoding="utf-8")

    destination = tmp_path / "published" / "Supplementary_Reproducibility_S1"
    destination.mkdir(parents=True)
    sentinel = destination / "sentinel.bin"
    sentinel.write_bytes(b"old-directory")
    archive = destination.parent / "Supplementary_Reproducibility_S1.zip"
    archive.write_bytes(b"old-zip")

    with pytest.raises(builder.PackageNotReady, match="not complete"):
        builder.build(source, destination, archive)
    assert sentinel.read_bytes() == b"old-directory"
    assert archive.read_bytes() == b"old-zip"
    assert not destination.with_name(destination.name + ".previous").exists()
    assert not archive.with_name(archive.name + ".previous").exists()


def test_exact_copy_does_not_redact_text_or_change_binary_bytes(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b'{"path":"/mnt/private/run"}\n\x00\xff')
    builder.copy_exact(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert _sha(destination) == _sha(source)


def test_copy_relative_set_rejects_validate_to_copy_mutation(tmp_path):
    source = tmp_path / "source"
    artifact = source / "records/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"frozen bytes")
    snapshot = {"records/result.json": _sha(artifact)}
    artifact.write_bytes(b"changed after validation")
    with pytest.raises(ValueError, match="Source changed after validation"):
        builder.copy_relative_set(
            source, tmp_path / "staging", {Path("records/result.json")},
            expected_hashes=snapshot)


def test_failed_staging_self_verification_never_publishes(monkeypatch, tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "published" / "Supplementary_Reproducibility_S1"
    destination.mkdir(parents=True)
    sentinel = destination / "sentinel.bin"
    sentinel.write_bytes(b"old-directory")
    archive = destination.parent / "Supplementary_Reproducibility_S1.zip"
    archive.write_bytes(b"old-zip")
    published = []

    monkeypatch.setattr(builder, "validate_completed_v2", lambda unused: {})
    monkeypatch.setattr(builder, "_build_staging", lambda unused, staging, state: 0)
    monkeypatch.setattr(verifier, "verify", lambda unused: {
        "status": "failed", "errors": ["synthetic staging failure"]})
    monkeypatch.setattr(builder, "_publish",
                        lambda *args, **kwargs: published.append((args, kwargs)))

    with pytest.raises(ValueError, match="Staged S1 failed self-verification"):
        builder.build(source, destination, archive)
    assert published == []
    assert sentinel.read_bytes() == b"old-directory"
    assert archive.read_bytes() == b"old-zip"
    assert not destination.with_name(destination.name + ".previous").exists()
    assert not archive.with_name(archive.name + ".previous").exists()


def test_publish_failure_restores_both_previous_artifacts(monkeypatch, tmp_path):
    destination = tmp_path / "Supplementary_Reproducibility_S1"
    destination.mkdir()
    (destination / "sentinel.bin").write_bytes(b"old-directory")
    archive = tmp_path / "Supplementary_Reproducibility_S1.zip"
    archive.write_bytes(b"old-zip")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.bin").write_bytes(b"new-directory")
    temporary_zip = tmp_path / "new.zip"
    temporary_zip.write_bytes(b"new-zip")
    original_replace = Path.replace

    def fail_new_zip(path: Path, target: Path):
        if path == temporary_zip:
            raise OSError("injected ZIP publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_new_zip)
    with pytest.raises(OSError, match="injected ZIP publication failure"):
        builder._publish(staging, temporary_zip, destination, archive)
    assert (destination / "sentinel.bin").read_bytes() == b"old-directory"
    assert archive.read_bytes() == b"old-zip"
    assert not destination.with_name(destination.name + ".previous").exists()
    assert not archive.with_name(archive.name + ".previous").exists()


def test_publish_post_rename_exception_restores_previous_pair(monkeypatch, tmp_path):
    destination = tmp_path / "Supplementary_Reproducibility_S1"
    destination.mkdir()
    (destination / "sentinel.bin").write_bytes(b"old-directory")
    archive = tmp_path / "Supplementary_Reproducibility_S1.zip"
    archive.write_bytes(b"old-zip")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.bin").write_bytes(b"new-directory")
    temporary_zip = tmp_path / "new.zip"
    temporary_zip.write_bytes(b"new-zip")
    original_replace = Path.replace

    def replace_then_fail(path: Path, target: Path):
        result = original_replace(path, target)
        if path == temporary_zip:
            raise OSError("injected failure after ZIP rename")
        return result

    monkeypatch.setattr(Path, "replace", replace_then_fail)
    with pytest.raises(OSError, match="after ZIP rename"):
        builder._publish(staging, temporary_zip, destination, archive)
    assert (destination / "sentinel.bin").read_bytes() == b"old-directory"
    assert archive.read_bytes() == b"old-zip"
    assert not destination.with_name(destination.name + ".previous").exists()
    assert not archive.with_name(archive.name + ".previous").exists()


def test_zip_must_be_byte_exact_canonical_view_of_staging(tmp_path):
    staging = tmp_path / "staging"
    (staging / "nested").mkdir(parents=True)
    (staging / "nested" / "record.bin").write_bytes(b"frozen")
    archive = tmp_path / "package.zip"
    builder._write_zip(staging, "S1", archive)
    builder._verify_zip_matches_staging(staging, "S1", archive)

    (staging / "nested" / "record.bin").write_bytes(b"changed-after-zip")
    with pytest.raises(ValueError, match="byte-count mismatch|content mismatch"):
        builder._verify_zip_matches_staging(staging, "S1", archive)

    (staging / "nested" / "record.bin").write_bytes(b"frozen")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive, "a") as handle:
            handle.writestr("S1/nested/record.bin", b"duplicate")
    with pytest.raises(ValueError, match="Duplicate S1 ZIP member"):
        builder._verify_zip_matches_staging(staging, "S1", archive)


def test_zip_rejects_symlink_members_and_staging_symlinks(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "record.bin").write_bytes(b"frozen")
    archive = tmp_path / "package.zip"
    info = zipfile.ZipInfo("S1/link")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "record.bin")
    with pytest.raises(ValueError, match="Unsupported S1 ZIP member type"):
        builder._verify_zip_matches_staging(staging, "S1", archive)

    archive.unlink()
    (staging / "link").symlink_to(staging / "record.bin")
    builder._write_zip(staging, "S1", archive)
    with pytest.raises(ValueError, match="Symbolic links"):
        builder._verify_zip_matches_staging(staging, "S1", archive)


def test_dynamic_hash_inventory_rejects_missing_tampered_and_unsafe(tmp_path):
    artifact = tmp_path / "task" / "pooled_data" / "map.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"locked-map")
    mapping = {"task/pooled_data/map.npz": _sha(artifact)}
    assert builder._require_hashes(tmp_path, mapping, label="fit-only") == {
        Path("task/pooled_data/map.npz")
    }
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        builder._require_hashes(tmp_path, mapping, label="fit-only")
    with pytest.raises(ValueError, match="Unsafe"):
        builder._require_hashes(tmp_path, {"../escape.npz": "0" * 64}, label="fit-only")
    with pytest.raises(ValueError, match="Unsafe"):
        builder._require_hashes(
            tmp_path, {"task//pooled_data/map.npz": "0" * 64}, label="fit-only")


def test_verifier_reports_non_utf8_manifest_without_crashing(tmp_path):
    (tmp_path / "MANIFEST.sha256").write_bytes(b"\xff")
    report = verifier.verify(tmp_path, require_v2=False)
    assert report["status"] == "failed"
    assert any("Cannot read MANIFEST.sha256" in error for error in report["errors"])


def test_verifier_allows_only_probability_protocol_from_selection_v1(tmp_path):
    allowed = tmp_path / "review/selection_runs/selection_v1/protocol.json"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("{}\n", encoding="utf-8")
    _refresh_manifest(tmp_path)
    assert verifier.verify(tmp_path, require_v2=False)["status"] == "passed"

    forbidden = allowed.parent / "within_site1/test_predictions.csv"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("prediction\n", encoding="utf-8")
    _refresh_manifest(tmp_path)
    report = verifier.verify(tmp_path, require_v2=False)
    assert report["status"] == "failed"
    assert any("selection-v1 point artifact" in error for error in report["errors"])


@pytest.mark.parametrize("payload", [
    "--enforce-tl-" + "dominance",
    "post_" + "adjustment",
    "Best TL-QLDMR config on " + "test",
])
def test_verifier_rejects_forbidden_content_after_outer_manifest_refresh(tmp_path, payload):
    record = tmp_path / "record.txt"
    record.write_text(payload, encoding="utf-8")
    _refresh_manifest(tmp_path)
    report = verifier.verify(tmp_path, require_v2=False)
    assert report["status"] == "failed"
    assert any("Forbidden legacy content" in error for error in report["errors"])


@pytest.mark.parametrize("relative", [
    "review/inference_runs/paired_point_v3/results.json",
    "review/inference_runs/paired_point_v3_backup/results.json",
    "review/residual_diagnostics.json",
    "review/residual_diagnostics_backup/results.json",
    "review/selection_runs/selection_v1_backup/results.json",
    "experiment2/results_selected.json",
    "experiment2_backup/results.json",
])
def test_verifier_rejects_forbidden_legacy_paths(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    _refresh_manifest(tmp_path)
    report = verifier.verify(tmp_path, require_v2=False)
    assert report["status"] == "failed"
    assert any("Forbidden" in error for error in report["errors"])
