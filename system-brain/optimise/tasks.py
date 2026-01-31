"""Reusable optimisation tasks for distillation workflows."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from .utils.teacher import TeacherModel
from .utils.feature_pipeline import load_telemetry_dataset, extract_feature_matrix
from .utils.distillation import DistillationConfig, train_xgboost_student, save_booster


def extract_soft_labels(
    models_path: Path | str = Path("/models"),
    data_path: Path | str = Path("/data/training_data.pkl"),
    output: Path | str = Path("optimise/data/distilled_dataset.npz"),
    metadata_path: Path | str = Path("optimise/data/distilled_metadata.json"),
    batch_size: int = 64,
) -> Path:
    """Generate a compressed dataset with features and teacher probabilities."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    teacher = TeacherModel(models_path=models_path)
    telemetry, labels = load_telemetry_dataset(data_path)

    features = extract_feature_matrix(
        telemetry,
        teacher.feature_selector,
        teacher.scaler,
        batch_size=batch_size,
    )
    probs = np.vstack([teacher.predict_proba(sample) for sample in telemetry])

    np.savez_compressed(output_path, features=features, teacher_probs=probs, labels=np.array(labels))

    metadata = {
        "models_path": str(models_path),
        "data_path": str(data_path),
        "num_samples": int(features.shape[0]),
        "num_features": int(features.shape[1]),
        "num_classes": int(probs.shape[1]),
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2))

    return output_path


def train_student(
    dataset: Path | str = Path("optimise/data/distilled_dataset.npz"),
    config_path: Path | str = Path("optimise/configs/student_xgb.json"),
    models_dir: Path | str = Path("optimise/artifacts"),
    prefix: str = "student_xgb",
    val_split: float = 0.1,
) -> Path:
    """Train a distilled XGBoost booster and save artefacts."""
    dataset = Path(dataset)
    if not dataset.exists():
        raise FileNotFoundError(f"Distilled dataset missing: {dataset}")

    with np.load(dataset) as data:
        features = data["features"]
        probs = data["teacher_probs"]

    config_dict = json.loads(Path(config_path).read_text())
    config = DistillationConfig(**config_dict)

    n_samples = features.shape[0]
    split_idx = int(n_samples * (1 - val_split))
    train_features, eval_features = features[:split_idx], features[split_idx:]
    train_probs, eval_probs = probs[:split_idx], probs[split_idx:]

    booster = train_xgboost_student(train_features, train_probs, config, eval_features, eval_probs)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{prefix}.json"
    save_booster(booster, model_path)

    manifest = {
        "model_path": str(model_path),
        "config": config_dict,
        "num_samples": n_samples,
        "train_samples": int(train_features.shape[0]),
        "val_samples": int(eval_features.shape[0]),
    }
    (models_dir / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2))

    return model_path


def evaluate_student(
    student_path: Path | str = Path("optimise/artifacts/student_xgb.json"),
    dataset: Path | str = Path("optimise/data/distilled_dataset.npz"),
    models_path: Path | str = Path("/models"),
    report_path: Path | str = Path("optimise/reports/student_evaluation.json"),
) -> Path:
    """Compare student predictions with teacher outputs and record metrics."""
    dataset = Path(dataset)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    with np.load(dataset) as data:
        features = data["features"]
        teacher_probs = data["teacher_probs"]
        labels = data["labels"]

    dmatrix = xgb.DMatrix(features)
    booster = xgb.Booster()
    booster.load_model(str(student_path))
    teacher = TeacherModel(models_path=models_path)
    classes = teacher.encoder.classes_
    num_class = len(classes)

    num_class_attr = booster.attr("num_class")
    if num_class_attr is not None:
        num_class = int(num_class_attr)

    raw_preds = booster.predict(dmatrix, output_margin=True)
    raw_preds = raw_preds.reshape(features.shape[0], num_class)

    def softmax_rows(matrix: np.ndarray) -> np.ndarray:
        matrix = matrix - matrix.max(axis=1, keepdims=True)
        exp = np.exp(matrix)
        return exp / exp.sum(axis=1, keepdims=True)

    student_probs = softmax_rows(raw_preds)
    def class_accuracy(probs: np.ndarray, labels_np: np.ndarray) -> float:
        preds = classes[probs.argmax(axis=1)]
        return float((preds == labels_np).mean())

    def cross_entropy(student_np: np.ndarray, teacher_np: np.ndarray) -> float:
        eps = 1e-9
        return float(-(teacher_np * np.log(student_np + eps)).sum(axis=1).mean())

    def kl_divergence(student_np: np.ndarray, teacher_np: np.ndarray) -> float:
        eps = 1e-9
        return float((teacher_np * (np.log(teacher_np + eps) - np.log(student_np + eps))).sum(axis=1).mean())

    report = {
        "dataset": str(dataset),
        "student": str(student_path),
        "num_samples": int(features.shape[0]),
        "accuracy": class_accuracy(student_probs, labels),
        "teacher_accuracy": class_accuracy(teacher_probs, labels),
        "cross_entropy_vs_teacher": cross_entropy(student_probs, teacher_probs),
        "kl_divergence_vs_teacher": kl_divergence(student_probs, teacher_probs),
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report_path
