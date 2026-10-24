#!/usr/bin/env python3
"""Resume locked v2 evaluations under a performance-blind solver amendment.

The selection and analysis locks remain unchanged.  This runner adds only a
fixed, conditional warm restart when a locked kernel fit exhausts
the two original ADMM attempts without meeting the unchanged duality-gap
certificate.  The amendment is created after an optimizer-only failure and is
applied identically to every subsequent main or matched-ablation certificate
failure.  It does not change data, features, configurations, seeds, losses,
gap tolerances, prediction/calibration procedures, or analysis rules.  The
certified weights and the predictions subsequently computed from them may of
course differ from the previously uncertified iterate.

Run ``--phase freeze`` once before any warm restart.  The resulting lock also
commits to every completion artifact that already existed when the amendment
was frozen.  ``--phase main`` and ``--phase matched`` then resume the two
locked evaluations through the amended Study class.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from review import run_matched_ablation_v2 as matched
from review import run_selection_study_v2 as selection
from review.convex_refinement import refine_pinball
from review.selection_data import digest


AMENDMENT_VERSION = "solver_certificate_amendment_v2_2"
EXTENDED_ATTEMPTS = (
    {"max_iter": 96000, "gap_rtol": 0.001, "gap_atol": 1e-7, "check_every": 25},
)
DEFAULT_SELECTION_OUTPUT = ROOT / "review/selection_runs/selection_v2"
DEFAULT_MATCHED_OUTPUT = ROOT / "review/selection_runs/matched_ablation_v2"
DEFAULT_AMENDMENT_LOCK = ROOT / "review/selection_runs/solver_certificate_amendment_v2.json"
SUPERSESSION_RECORD = ROOT / "review/selection_runs/solver_certificate_amendment_v2_supersession.json"
SELECTION_ANALYSIS_OUTPUT = ROOT / "review/selection_runs/selection_v2_analysis"
MATCHED_ANALYSIS_OUTPUT = ROOT / "review/selection_runs/matched_ablation_v2_analysis"


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError as error:
        raise ValueError(f"Amended artifacts must remain inside the project root: {path}") from error


def _completion_hashes(selection_output: Path, matched_output: Path) -> dict[str, str]:
    """Hash every completed result and all artifacts on which it depends."""
    paths: list[Path] = []
    for output in (selection_output, matched_output):
        if not output.exists():
            continue
        for result_path in output.rglob("result.json"):
            paths.extend(path for path in result_path.parent.rglob("*") if path.is_file())
    return {_relative(path): digest(path) for path in sorted(set(paths))}


def _analysis_freeze_hashes() -> dict[str, str]:
    """Commit both analysis plans as they existed before solver recovery."""
    required = {
        SELECTION_ANALYSIS_OUTPUT / "analysis_protocol.json",
        SELECTION_ANALYSIS_OUTPUT / "analysis_manifest.json",
        MATCHED_ANALYSIS_OUTPUT / "analysis_protocol.json",
        MATCHED_ANALYSIS_OUTPUT / "analysis_lock.json",
        MATCHED_ANALYSIS_OUTPUT / "manifest.json",
    }
    missing = sorted(path for path in required if not path.is_file())
    if missing:
        raise RuntimeError(f"Incomplete pre-evaluation analysis freeze: {missing[0]}")
    return {_relative(path): digest(path) for path in sorted(required)}


def _failed_locked_fits(selection_output: Path, matched_output: Path) -> list[Path]:
    failures = []
    for output in (selection_output, matched_output):
        if not output.exists():
            continue
        for path in output.rglob("fit.json"):
            if "evaluation" not in path.parts:
                continue
            record = _read_json(path)
            converged = record.get("diagnostics", {}).get("converged")
            if converged is False:
                failures.append(path)
    return sorted(failures)


def _validate_trigger(path: Path, selection_output: Path) -> dict:
    record = _read_json(path)
    diagnostics = record.get("diagnostics", {})
    if diagnostics.get("converged") is not False:
        raise ValueError("The amendment trigger must be an exhausted, uncertified kernel fit.")
    protocol = _read_json(selection_output / "protocol.json")
    attempts = diagnostics.get("attempt_summaries", [])
    if diagnostics.get("attempt_count") != len(protocol["solver_attempts"]):
        raise ValueError("The trigger did not exhaust the original attempt schedule.")
    if len(attempts) != len(protocol["solver_attempts"]):
        raise ValueError("The trigger lacks the original attempt summaries.")
    for observed, expected in zip(attempts, protocol["solver_attempts"]):
        for key, value in expected.items():
            if observed.get(key) != value and key in ("gap_rtol", "gap_atol"):
                raise ValueError("The trigger used a different original gap tolerance.")
        if observed.get("iterations") != expected["max_iter"]:
            raise ValueError("The trigger did not exhaust an original iteration cap.")
    return {
        "fit_record": _relative(path),
        "fit_record_sha256": digest(path),
        "model_sha256": digest(path.with_name("model.npz")),
        "identity": {key: record[key] for key in ("task", "method", "seed", "tau")},
        "optimizer_diagnostics": {
            key: diagnostics.get(key) for key in
            ("primal", "dual_bound", "gap", "relative_gap", "gap_atol", "gap_rtol",
             "iterations", "attempt_count", "stop_reason")
        },
    }


def freeze_amendment(lock_path: Path, selection_output: Path,
                     matched_output: Path) -> dict:
    """Freeze the uniform warm-restart rule without reading predictive metrics."""
    lock_path = Path(lock_path).resolve()
    selection_output = Path(selection_output).resolve()
    matched_output = Path(matched_output).resolve()
    if lock_path.exists():
        return validate_amendment(lock_path, selection_output, matched_output)

    protocol = selection.protocol()
    manifest = selection._initialize(selection_output, protocol, True)
    selection_lock_path = selection_output / "selection_lock.json"
    selection_lock = _read_json(selection_lock_path)
    matched_state = matched.verify_matched_lock(matched_output, selection_output)
    failures = _failed_locked_fits(selection_output, matched_output)
    if not failures:
        raise RuntimeError("A performance-blind amendment requires an observed solver-certificate failure.")
    trigger = _validate_trigger(failures[0], selection_output)
    original_attempts = protocol["solver_attempts"]
    if original_attempts != [
            {"max_iter": 3000, "gap_rtol": 0.001, "gap_atol": 1e-7, "check_every": 25},
            {"max_iter": 12000, "gap_rtol": 0.001, "gap_atol": 1e-7, "check_every": 25}]:
        raise ValueError("The frozen study no longer has the audited original attempt schedule.")

    value = {
        "version": AMENDMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "A locked kernel fit exhausted the predeclared 3000- and 12000-iteration "
            "ADMM attempts and narrowly missed the unchanged optimizer certificate. "
            "The amendment is based only on solver diagnostics, not predictive performance."
        ),
        "scope": (
            "Every locked main-study or matched-ablation kernel fit that remains "
            "uncertified after the original attempts; already certified fits are unchanged."
        ),
        "rule": {
            "original_attempts": original_attempts,
            "conditional_warm_restart": list(EXTENDED_ATTEMPTS),
            "restart": (
                "The additional run starts from the preceding best primal weights and "
                "intercept with a fresh ADMM split/dual state, matching the original retry "
                "mechanism; it is a warm restart rather than an exact state continuation."
            ),
            "stop": (
                "Stop at the first unchanged certificate; fail closed after the final cap."
            ),
            "unchanged": [
                "gap tolerances", "objective", "data and split IDs", "feature maps",
                "selected configurations", "seeds", "prediction and calibration procedures",
                "analysis estimands and inferential rules",
            ],
        },
        "trigger": trigger,
        "preexisting_completion_artifacts": _completion_hashes(
            selection_output, matched_output),
        "preexisting_analysis_freezes": _analysis_freeze_hashes(),
        "partial_evaluation_disclosure": {
            "main_result_records_already_present": len(list(selection_output.rglob("result.json"))),
            "matched_result_records_already_present": len(list(matched_output.rglob("result.json"))),
            "statement": (
                "These artifacts are hash-committed without parsing their metrics.  The test "
                "period was already declared reused; no result is removed or used to alter "
                "a configuration, comparison, estimand, or inferential rule."
            ),
        },
        "selection": {
            "output": _relative(selection_output),
            "selection_lock_sha256": digest(selection_lock_path),
            "protocol_sha256": digest(selection_output / "protocol.json"),
            "source_hashes": selection_lock["source_hashes"],
            "data_hashes": selection_lock["data_hashes"],
            "manifest_last_stable_status": manifest.get("last_stable_status"),
        },
        "matched": {
            "output": _relative(matched_output),
            "matched_lock_sha256": matched_state["matched_lock_sha256"],
            "source_hashes": matched_state["lock"]["source_hashes"],
        },
        "amendment_runner": _relative(Path(__file__)),
        "amendment_runner_sha256": digest(Path(__file__)),
        "superseded_preexecution_draft": {
            "record": _relative(SUPERSESSION_RECORD),
            "record_sha256": digest(SUPERSESSION_RECORD),
        },
    }
    selection.save_json(lock_path, value)
    return validate_amendment(lock_path, selection_output, matched_output)


def validate_amendment(lock_path: Path, selection_output: Path,
                       matched_output: Path) -> dict:
    lock_path = Path(lock_path).resolve()
    selection_output = Path(selection_output).resolve()
    matched_output = Path(matched_output).resolve()
    value = _read_json(lock_path)
    if value.get("version") != AMENDMENT_VERSION:
        raise ValueError("Unsupported solver-certificate amendment version.")
    if value.get("amendment_runner") != _relative(Path(__file__)) or \
            value.get("amendment_runner_sha256") != digest(Path(__file__)):
        raise ValueError("The frozen amendment runner changed.")
    if value.get("rule", {}).get("conditional_warm_restart") != list(EXTENDED_ATTEMPTS):
        raise ValueError("The frozen extended-attempt rule changed.")
    if value.get("selection", {}).get("output") != _relative(selection_output):
        raise ValueError("The amendment points to a different selection output.")
    if value.get("matched", {}).get("output") != _relative(matched_output):
        raise ValueError("The amendment points to a different matched output.")
    if value["selection"]["selection_lock_sha256"] != digest(
            selection_output / "selection_lock.json"):
        raise ValueError("The selection lock changed after the solver amendment.")
    if value["selection"]["protocol_sha256"] != digest(selection_output / "protocol.json"):
        raise ValueError("The selection protocol changed after the solver amendment.")
    selection_lock = _read_json(selection_output / "selection_lock.json")
    if value["selection"]["source_hashes"] != selection_lock["source_hashes"] or \
            value["selection"]["data_hashes"] != selection_lock["data_hashes"]:
        raise ValueError("The selection source or data identity changed.")
    if value["matched"]["matched_lock_sha256"] != digest(matched_output / "matched_lock.json"):
        raise ValueError("The matched-ablation lock changed after the solver amendment.")
    matched_lock = _read_json(matched_output / "matched_lock.json")
    if value["matched"]["source_hashes"] != matched_lock["source_hashes"]:
        raise ValueError("The matched-ablation source identity changed.")
    for relative, expected in value["preexisting_completion_artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"A pre-amendment completion artifact changed: {relative}")
    if value.get("preexisting_analysis_freezes") != _analysis_freeze_hashes():
        raise ValueError("A pre-amendment analysis freeze changed.")
    supersession = value.get("superseded_preexecution_draft", {})
    supersession_path = ROOT / supersession.get("record", "")
    if (not supersession_path.is_file() or
            digest(supersession_path) != supersession.get("record_sha256")):
        raise ValueError("The pre-execution draft supersession record changed.")
    supersession_record = _read_json(supersession_path)
    archived_draft = ROOT / supersession_record.get("superseded_lock", "")
    if (supersession_record.get("executed_fit_count") != 0 or
            not archived_draft.is_file() or
            digest(archived_draft) != supersession_record.get("superseded_lock_sha256")):
        raise ValueError("The unexecuted superseded amendment draft changed.")
    trigger_path = ROOT / value["trigger"]["fit_record"]
    archived = trigger_path.with_name("fit.pre_amendment.json")
    trigger_evidence = archived if archived.exists() else trigger_path
    if digest(trigger_evidence) != value["trigger"]["fit_record_sha256"]:
        raise ValueError("The optimizer-only amendment trigger changed.")
    archived_model = trigger_path.with_name("model.pre_amendment.npz")
    model_evidence = archived_model if archived_model.exists() else trigger_path.with_name("model.npz")
    if digest(model_evidence) != value["trigger"]["model_sha256"]:
        raise ValueError("The optimizer-only trigger model changed.")
    return value


def _copy_once(source: Path, destination: Path, expected_hash: str) -> None:
    if destination.exists():
        if digest(destination) != expected_hash:
            raise ValueError(f"Existing pre-amendment archive changed: {destination}")
        return
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    if digest(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError("An amendment archive copy changed in transit.")
    temporary.replace(destination)


def _restore_archive(source: Path, destination: Path, expected_hash: str) -> None:
    """Atomically restore an interrupted amendment to its committed input."""
    if digest(source) != expected_hash:
        raise ValueError(f"Pre-amendment archive changed: {source}")
    temporary = destination.with_name(destination.name + f".restore.{os.getpid()}")
    shutil.copyfile(source, temporary)
    if digest(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise ValueError("A restored amendment input changed in transit.")
    temporary.replace(destination)


def _validate_completed_amendment(folder: Path, amendment_path: Path,
                                  amendment: dict, amendment_sha: str) -> dict:
    """Validate the full sidecar/archive chain for one completed warm restart."""
    fit_path, model_path = folder / "fit.json", folder / "model.npz"
    completion_path = folder / "solver_amendment.json"
    intent_path = folder / "solver_amendment_intent.json"
    fit_archive = folder / "fit.pre_amendment.json"
    model_archive = folder / "model.pre_amendment.npz"
    required = (fit_path, model_path, completion_path, intent_path,
                fit_archive, model_archive)
    if not all(path.is_file() for path in required):
        raise ValueError(f"An amended fit lacks its audit chain: {folder}")
    record = _read_json(fit_path)
    completion = _read_json(completion_path)
    intent = _read_json(intent_path)
    relative_folder = _relative(folder)
    for sidecar, label in ((intent, "intent"), (completion, "completion")):
        if (sidecar.get("version") != AMENDMENT_VERSION or
                sidecar.get("fit_folder") != relative_folder or
                sidecar.get("amendment_lock_sha256") != amendment_sha):
            raise ValueError(f"Amended-fit {label} identity changed: {folder}")
    for key in ("original_fit_sha256", "original_model_sha256"):
        if intent.get(key) != completion.get(key):
            raise ValueError(f"Amended-fit original hash accounting changed: {folder}/{key}")
    if (digest(fit_archive) != intent["original_fit_sha256"] or
            digest(model_archive) != intent["original_model_sha256"]):
        raise ValueError(f"A pre-amendment archive changed: {folder}")
    if (digest(fit_path) != completion.get("amended_fit_sha256") or
            digest(model_path) != completion.get("amended_model_sha256")):
        raise ValueError(f"An amended fit artifact changed: {folder}")
    diagnostics = record.get("diagnostics", {})
    provenance = diagnostics.get("solver_certificate_amendment", {})
    expected_provenance = {
        "version": AMENDMENT_VERSION,
        "amendment_lock": _relative(amendment_path),
        "amendment_lock_sha256": amendment_sha,
        "original_fit_sha256": intent["original_fit_sha256"],
        "original_model_sha256": intent["original_model_sha256"],
        "extension_attempts_used": 1,
        "gap_tolerances_changed": False,
    }
    if provenance != expected_provenance:
        raise ValueError(f"Amended-fit diagnostics provenance changed: {folder}")
    if completion.get("converged") is not bool(diagnostics.get("converged")):
        raise ValueError(f"Amended-fit certificate status changed: {folder}")
    reports = completion.get("extension_attempts")
    if not isinstance(reports, list) or len(reports) != 1:
        raise ValueError(f"Amended-fit restart count changed: {folder}")
    report, rule = reports[0], amendment["rule"]["conditional_warm_restart"][0]
    for key, expected in rule.items():
        if report.get(key) != expected:
            raise ValueError(f"Amended-fit restart rule changed: {folder}/{key}")
    if (not isinstance(report.get("iterations"), int) or
            not 0 <= report["iterations"] <= rule["max_iter"] or
            report.get("converged") is not completion["converged"]):
        raise ValueError(f"Amended-fit restart diagnostics changed: {folder}")
    if any(diagnostics.get(key) != value for key, value in report.items()
           if key not in ("max_iter", "check_every")):
        raise ValueError(f"Amended-fit final diagnostics changed: {folder}")
    original_count = len(amendment["rule"]["original_attempts"])
    summaries = diagnostics.get("attempt_summaries", [])
    if (diagnostics.get("attempt_count") != original_count + 1 or
            len(summaries) != original_count + 1 or summaries[-1] != report):
        raise ValueError(f"Amended-fit attempt accounting changed: {folder}")
    original_record = _read_json(fit_archive)
    for field in ("task", "method", "config", "seed", "tau", "candidate_id", "feature_map"):
        if original_record.get(field) != record.get(field):
            raise ValueError(f"Amended fit changed locked identity: {folder}/{field}")
    return record


def extend_fit_certificate(study: selection.Study, task: dict, method: str,
                           config: dict, seed: int, tau: float, folder: Path,
                           record: dict, amendment_path: Path,
                           candidate_id: str) -> dict:
    """Apply the frozen warm restart to one uncertified locked fit."""
    if method == "HistGBR":
        return record
    folder = Path(folder).resolve()
    fit_path, model_path = folder / "fit.json", folder / "model.npz"
    completion_path = folder / "solver_amendment.json"
    intent_path = folder / "solver_amendment_intent.json"
    diagnostics_at_entry = record.get("diagnostics", {})
    if (diagnostics_at_entry.get("converged") is not False and
            "solver_certificate_amendment" not in diagnostics_at_entry and
            not completion_path.exists() and not intent_path.exists()):
        return record
    frozen = _read_json(amendment_path)
    selection_output = (ROOT / frozen["selection"]["output"]).resolve()
    matched_output = (ROOT / frozen["matched"]["output"]).resolve()
    amendment = validate_amendment(
        amendment_path, selection_output, matched_output)
    amendment_sha = digest(amendment_path)
    identity = {
        "task": task["name"], "method": method, "config": config,
        "seed": seed, "tau": tau, "candidate_id": candidate_id,
    }
    for field, expected in identity.items():
        if record.get(field) != expected:
            raise ValueError(f"Uncertified locked-fit identity changed: {folder}/{field}")
    if candidate_id == "locked_final":
        expected_folder = (selection_output / task["name"] / method / "evaluation" /
                           f"seed{seed}" / ("tau_" + format(tau, ".17g")))
        result_path = expected_folder.parent / "result.json"
        selection_lock = _read_json(selection_output / "selection_lock.json")
        rows = [row for row in selection_lock["final_selections"]
                if (row["task"], row["method"]) == (task["name"], method)]
        if len(rows) != 1 or rows[0]["selected_config"] != config:
            raise ValueError(f"Main amendment configuration is not selection-locked: {folder}")
    elif candidate_id == "matched_ablation_v2_locked":
        if tau != matched.TAU:
            raise ValueError(f"Matched amendment has an unexpected quantile: {folder}")
        expected_folder = (matched_output / "evaluation" / task["name"] / method /
                           f"seed{seed}")
        result_path = expected_folder / "result.json"
        configurations = _read_json(matched_output / "configurations.json")
        rows = [row for row in configurations["tasks"] if row["task"] == task["name"]]
        if len(rows) != 1 or rows[0]["variants"].get(method) != config:
            raise ValueError(f"Matched amendment configuration is not locked: {folder}")
    else:
        raise ValueError(f"Solver amendment is restricted to locked evaluations: {candidate_id}")
    if folder != expected_folder.resolve():
        raise ValueError(f"Solver amendment path is not the locked evaluation path: {folder}")
    if result_path.exists():
        raise ValueError(f"A completed result must never be amended: {result_path}")
    if completion_path.exists():
        return _validate_completed_amendment(
            folder, amendment_path, amendment, amendment_sha)
    fit_archive = folder / "fit.pre_amendment.json"
    model_archive = folder / "model.pre_amendment.npz"
    if intent_path.exists():
        intent = _read_json(intent_path)
        if (intent.get("version") != AMENDMENT_VERSION or
                intent.get("fit_folder") != _relative(folder) or
                intent.get("amendment_lock_sha256") != amendment_sha):
            raise ValueError(f"Interrupted fit used a different solver amendment: {folder}")
        original_fit_sha = intent["original_fit_sha256"]
        original_model_sha = intent["original_model_sha256"]
        _restore_archive(fit_archive, fit_path, original_fit_sha)
        _restore_archive(model_archive, model_path, original_model_sha)
        record = _read_json(fit_path)
    else:
        if record.get("diagnostics", {}).get("converged") is not False:
            raise ValueError(f"An amended fit is missing its completion record: {folder}")
        original_fit_sha = digest(fit_path)
        original_model_sha = digest(model_path)
        _copy_once(fit_path, fit_archive, original_fit_sha)
        _copy_once(model_path, model_archive, original_model_sha)
        selection.save_json(intent_path, {
            "version": AMENDMENT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "fit_folder": _relative(folder),
            "amendment_lock_sha256": amendment_sha,
            "original_fit_sha256": original_fit_sha,
            "original_model_sha256": original_model_sha,
        })
    if record.get("diagnostics", {}).get("converged") is not False:
        raise ValueError(f"The pre-amendment archive is not an uncertified fit: {folder}")

    context = study.context(task, method)
    feature = study.features(context, seed, config["gamma"], config["rank"])
    expected_feature = str(feature["path"].relative_to(study.output))
    if record.get("feature_map") != expected_feature:
        raise ValueError(f"The uncertified fit does not use its locked feature map: {folder}")
    arrays = {}
    with np.load(model_path) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    required = {"w", "b", "dual", "R", "c", "C", "y"}
    if not required.issubset(arrays):
        raise ValueError(f"The uncertified model lacks optimization state: {folder}")
    z = feature["z"]
    if arrays["y"].shape != context["y"].shape or not np.array_equal(arrays["y"], context["y"]):
        raise ValueError(f"The uncertified fit labels changed: {folder}")
    if z.shape != (len(arrays["y"]), len(arrays["w"])):
        raise ValueError(f"The uncertified feature design changed: {folder}")

    diagnostics = dict(record["diagnostics"])
    attempt_summaries = list(diagnostics.get("attempt_summaries", []))
    original_attempts = amendment["rule"]["original_attempts"]
    if (diagnostics.get("attempt_count") != len(attempt_summaries) or
            len(attempt_summaries) != len(original_attempts)):
        raise ValueError(f"The uncertified attempt accounting changed: {folder}")
    for summary, expected in zip(attempt_summaries, original_attempts):
        if (summary.get("iterations") != expected["max_iter"] or
                summary.get("gap_rtol") != expected["gap_rtol"] or
                summary.get("gap_atol") != expected["gap_atol"]):
            raise ValueError(f"The uncertified original attempt schedule changed: {folder}")

    w, b = arrays["w"], float(arrays["b"])
    extension_reports = []
    extension_start = time.perf_counter()
    solution = None
    for options in EXTENDED_ATTEMPTS:
        solution = refine_pinball(
            z, arrays["y"], arrays["R"], arrays["c"], arrays["C"], tau,
            w0=w, b0=b, **options)
        report = solution.report()
        summary = {key: value for key, value in report.items() if key != "history"}
        summary.update(options)
        extension_reports.append(summary)
        attempt_summaries.append(summary)
        w, b = solution.w, solution.b
        if solution.converged:
            break
    assert solution is not None

    arrays.update(w=solution.w, b=np.asarray(solution.b), dual=solution.dual_variable)
    selection.save_npz(model_path, **arrays)
    latest = solution.report()
    latest.update({
        "solver": diagnostics.get("solver", "ADMM"),
        "fit_seconds_including_feature_access": (
            float(diagnostics.get("fit_seconds_including_feature_access", 0.0)) +
            time.perf_counter() - extension_start),
        "requested_rank": diagnostics.get("requested_rank", config["rank"]),
        "effective_rank": diagnostics.get("effective_rank", z.shape[1]),
        "attempt_count": len(attempt_summaries),
        "attempt_summaries": attempt_summaries,
        "regularizer_components": diagnostics.get("regularizer_components"),
        "solver_certificate_amendment": {
            "version": AMENDMENT_VERSION,
            "amendment_lock": _relative(amendment_path),
            "amendment_lock_sha256": amendment_sha,
            "original_fit_sha256": original_fit_sha,
            "original_model_sha256": original_model_sha,
            "extension_attempts_used": len(extension_reports),
            "gap_tolerances_changed": False,
        },
    })
    amended = dict(record)
    amended["diagnostics"] = latest
    selection.save_json(fit_path, amended)
    completion = {
        "version": AMENDMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit_folder": _relative(folder),
        "amendment_lock_sha256": amendment_sha,
        "original_fit_sha256": original_fit_sha,
        "original_model_sha256": original_model_sha,
        "amended_fit_sha256": digest(fit_path),
        "amended_model_sha256": digest(model_path),
        "converged": bool(solution.converged),
        "extension_attempts": extension_reports,
    }
    selection.save_json(completion_path, completion)
    return _validate_completed_amendment(
        folder, amendment_path, amendment, amendment_sha)


def amended_study_class(amendment_path: Path):
    """Create a Study subclass whose only change is a conditional warm restart."""
    original_class = selection.Study

    class AmendedStudy(original_class):
        def fit(self, task, method, config, seed, tau, folder, candidate_id):
            record = super().fit(task, method, config, seed, tau, folder, candidate_id)
            diagnostics = record.get("diagnostics", {})
            if (method == "HistGBR" or
                    (diagnostics.get("converged") is not False and
                     "solver_certificate_amendment" not in diagnostics)):
                return record
            return extend_fit_certificate(
                self, task, method, config, seed, tau, folder, record,
                amendment_path, candidate_id)

    AmendedStudy.__name__ = "SolverCertificateAmendedStudy"
    return AmendedStudy


def run_main(amendment_path: Path, selection_output: Path,
             matched_output: Path) -> list[dict]:
    validate_amendment(amendment_path, selection_output, matched_output)
    config = selection.protocol()
    manifest = selection._initialize(selection_output, config, True)
    manifest.update({
        "status": "running", "active_phase": "evaluate",
        "solver_certificate_amendment": _relative(amendment_path),
        "solver_certificate_amendment_sha256": digest(amendment_path),
    })
    manifest.pop("error", None)
    selection.save_json(selection_output / "manifest.json", manifest)
    try:
        with threadpool_limits(limits=config["threads"]):
            study_class = amended_study_class(amendment_path)
            study = study_class(selection_output, config, write_global_provenance=False)
            results = study.evaluate(manifest)
        manifest.update({"status": "complete", "last_stable_status": "complete"})
        return results
    except Exception as error:
        manifest.update({"status": "failed", "error": repr(error)})
        raise
    finally:
        manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        selection.save_json(selection_output / "manifest.json", manifest)


def run_matched(amendment_path: Path, selection_output: Path,
                matched_output: Path) -> list[dict]:
    validate_amendment(amendment_path, selection_output, matched_output)
    manifest_path = matched_output / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest.update({
        "solver_certificate_amendment": _relative(amendment_path),
        "solver_certificate_amendment_sha256": digest(amendment_path),
    })
    selection.save_json(manifest_path, manifest)
    original_class = selection.Study
    selection.Study = amended_study_class(amendment_path)
    try:
        with threadpool_limits(limits=1):
            return matched.evaluate(matched_output, selection_output, resume=True)
    finally:
        selection.Study = original_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "main", "matched"), required=True)
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_SELECTION_OUTPUT)
    parser.add_argument("--matched-output", type=Path, default=DEFAULT_MATCHED_OUTPUT)
    parser.add_argument("--amendment-lock", type=Path, default=DEFAULT_AMENDMENT_LOCK)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze_amendment(args.amendment_lock, args.selection_output, args.matched_output)
    elif args.phase == "main":
        run_main(args.amendment_lock, args.selection_output, args.matched_output)
    else:
        run_matched(args.amendment_lock, args.selection_output, args.matched_output)


if __name__ == "__main__":
    main()
