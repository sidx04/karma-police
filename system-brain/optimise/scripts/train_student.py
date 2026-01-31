#!/usr/bin/env python3
"""Train a distilled XGBoost student from the cached dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks import train_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="optimise/data/distilled_dataset.npz", help="Path to the distilled dataset")
    parser.add_argument("--config", default="optimise/configs/student_xgb.json", help="Distillation hyperparameters")
    parser.add_argument("--models-path", default="optimise/artifacts", help="Directory to store student artefacts")
    parser.add_argument("--prefix", default="student_xgb", help="Filename prefix for saved model")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = train_student(
        dataset=args.dataset,
        config_path=args.config,
        models_dir=args.models_path,
        prefix=args.prefix,
        val_split=args.val_split,
    )
    print(f"[OK] Student model saved to {path}")


if __name__ == "__main__":
    main()
