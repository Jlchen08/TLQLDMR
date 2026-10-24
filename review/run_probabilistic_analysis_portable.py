#!/usr/bin/env python3
"""Run the frozen probability analysis against an explicitly chosen experiment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review import run_probabilistic_analysis_v2 as analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    analysis.EXPERIMENT = args.experiment.resolve()
    forwarded = [str(Path(analysis.__file__).resolve()), "--output", str(args.output)]
    if args.freeze_only:
        forwarded.append("--freeze-only")
    if args.resume:
        forwarded.append("--resume")
    sys.argv = forwarded
    analysis.main()


if __name__ == "__main__":
    main()
