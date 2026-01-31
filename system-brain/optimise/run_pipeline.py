#!/usr/bin/env python3
"""Orchestrate the optimisation workflow (distillation pipeline)."""
from __future__ import annotations

import argparse
from pathlib import Path

from tasks import extract_soft_labels, train_student, evaluate_student


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["extract", "train", "evaluate", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument("--models-path", default="/models", help="Teacher artefacts directory")
    parser.add_argument("--data-path", default="/data/training_data.pkl", help="Telemetry dataset path")
    parser.add_argument("--dataset", default="optimise/data/distilled_dataset.npz", help="Cached distilled dataset")
    parser.add_argument("--metadata", default="optimise/data/distilled_metadata.json", help="Where to store distilled metadata")
    parser.add_argument("--config", default="optimise/configs/student_xgb.json", help="Student training config")
    parser.add_argument("--artifacts", default="optimise/artifacts", help="Directory for student artifacts")
    parser.add_argument("--reports", default="optimise/reports/student_evaluation.json", help="Evaluation report path")
    parser.add_argument("--prefix", default="student_xgb", help="Filename prefix for student models")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation fraction during training")
    parser.add_argument("--batch-size", type=int, default=64, help="Feature extraction batch size")
    return parser.parse_args()


def step_extract(args: argparse.Namespace) -> Path:
    print("[STEP] Extracting soft labels from teacher...")
    dataset_path = extract_soft_labels(
        models_path=args.models_path,
        data_path=args.data_path,
        output=args.dataset,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
    )
    print(f"[OK] Distilled dataset created at {dataset_path}")
    return dataset_path


def step_train(args: argparse.Namespace) -> Path:
    print("[STEP] Training distilled student model...")
    model_path = train_student(
        dataset=args.dataset,
        config_path=args.config,
        models_dir=args.artifacts,
        prefix=args.prefix,
        val_split=args.val_split,
    )
    print(f"[OK] Student model saved to {model_path}")
    return model_path


def step_evaluate(args: argparse.Namespace) -> Path:
    student_file = Path(args.artifacts) / f"{args.prefix}.json"
    print(f"[STEP] Evaluating student model at {student_file}...")
    report_path = evaluate_student(
        student_path=student_file,
        dataset=args.dataset,
        models_path=args.models_path,
        report_path=args.reports,
    )
    print(report_path.read_text())
    return report_path


def main() -> None:
    args = parse_args()

    if args.step in ("extract", "all"):
        step_extract(args)
    if args.step in ("train", "all"):
        step_train(args)
    if args.step in ("evaluate", "all"):
        step_evaluate(args)


if __name__ == "__main__":
    main()
