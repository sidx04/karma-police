#!/usr/bin/env python3
"""Evaluate the distilled student against teacher metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks import evaluate_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default="optimise/artifacts/student_xgb.json", help="Path to the trained student booster")
    parser.add_argument("--dataset", default="optimise/data/distilled_dataset.npz", help="Distilled dataset")
    parser.add_argument("--models-path", default="/models", help="Teacher artefacts for class ordering")
    parser.add_argument("--report", default="optimise/reports/student_evaluation.json", help="Where to store evaluation summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_student(
        student_path=args.student,
        dataset=args.dataset,
        models_path=args.models_path,
        report_path=args.report,
    )
    print(report.read_text())


if __name__ == "__main__":
    main()
