#!/usr/bin/env python3
"""Create a post-completion hash inventory for the verified probability study.

This is an integrity aid, not a preregistration record.  It inventories every
selection, calibration, and test CSV used by the probability-specific study.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "review/probabilistic_runs/probabilistic_selection_v2"
ANALYSIS = ROOT / "review/probabilistic_runs/probabilistic_analysis_v2"
OUTPUT = ANALYSIS / "posthoc_integrity_inventory.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    validation = sorted(EXPERIMENT.glob("within_site6/*/candidates/c*/validation_quantiles.csv"))
    calibration = sorted(EXPERIMENT.glob("within_site6/*/final_seed*/calibration_quantiles.csv"))
    test = sorted(EXPERIMENT.glob("within_site6/*/final_seed*/test_quantiles.csv"))
    if (len(validation), len(calibration), len(test)) != (96, 16, 16):
        raise ValueError("Expected 96 validation, 16 calibration, and 16 test CSV files")
    control = [
        EXPERIMENT / "protocol.json",
        EXPERIMENT / "manifest.json",
        EXPERIMENT / "selection_lock.json",
        EXPERIMENT / "results.json",
        ANALYSIS / "protocol.json",
        ANALYSIS / "manifest.json",
        ANALYSIS / "aggregate_results.json",
        ANALYSIS / "paired_probability_inference.json",
        ROOT / "review/run_probabilistic_selection_study.py",
        ROOT / "review/run_probabilistic_analysis_v2.py",
        ROOT / "review/render_probabilistic_figures.py",
        Path(__file__),
    ]
    files = control + validation + calibration + test
    inventory = {
        "role": "post-completion integrity inventory; not preregistration",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "candidate_validation_csv": len(validation),
            "selected_calibration_csv": len(calibration),
            "selected_test_csv": len(test),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "matplotlib": package_version("matplotlib"),
        },
        "sha256": {
            str(path.relative_to(ROOT)): digest(path) for path in files
        },
    }
    OUTPUT.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(files)} hashes")


if __name__ == "__main__":
    main()
