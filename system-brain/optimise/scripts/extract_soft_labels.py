#!/usr/bin/env python3
"""Generate a distilled dataset of features + teacher probabilities."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks import extract_soft_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-path", default="/models", help="Directory containing teacher artefacts")
    parser.add_argument("--data-path", default="/data/training_data.pkl", help="Telemetry dataset produced by training pipeline")
    parser.add_argument("--output", default="optimise/data/distilled_dataset.npz", help="Where to store features and soft labels")
    parser.add_argument("--metadata", default="optimise/data/distilled_metadata.json", help="Path for metadata exports")
    parser.add_argument("--batch-size", type=int, default=64, help="Feature extraction batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = extract_soft_labels(
        models_path=args.models_path,
        data_path=args.data_path,
        output=args.output,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
    )
    print(f"[OK] Distilled dataset saved to {path}")


if __name__ == "__main__":
    main()
