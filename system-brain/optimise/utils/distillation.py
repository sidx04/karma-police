"""Utilities to train distilled student models."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xgboost as xgb


_SOFT_LABEL_REGISTRY: Dict[int, np.ndarray] = {}


@dataclass
class DistillationConfig:
    num_boost_round: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 1.0
    temperature: float = 1.0
    early_stopping_rounds: Optional[int] = 20


def soften_targets(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to teacher probabilities."""
    if temperature == 1.0:
        return logits
    scaled = np.log(logits + 1e-9) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _softmax(margins: np.ndarray) -> np.ndarray:
    margins = margins - margins.max(axis=1, keepdims=True)
    exp = np.exp(margins)
    return exp / exp.sum(axis=1, keepdims=True)


def _register_soft_labels(dmat: xgb.DMatrix, soft_targets: np.ndarray) -> None:
    _SOFT_LABEL_REGISTRY[id(dmat)] = soft_targets


def _get_soft_labels(dmat: xgb.DMatrix) -> np.ndarray:
    try:
        return _SOFT_LABEL_REGISTRY[id(dmat)]
    except KeyError as exc:
        raise KeyError("Soft labels not registered for provided DMatrix") from exc


def _create_dmatrix(features: np.ndarray, soft_targets: np.ndarray) -> xgb.DMatrix:
    dmat = xgb.DMatrix(features, label=np.zeros(features.shape[0]))
    _register_soft_labels(dmat, soft_targets)
    return dmat


def _soft_target_objective(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    teacher = _get_soft_labels(dtrain)
    num_rows, num_class = teacher.shape
    preds = preds.reshape(num_rows, num_class)
    y_hat = _softmax(preds)
    grad = (y_hat - teacher).reshape(-1)
    hess = (y_hat * (1 - y_hat)).reshape(-1)
    return grad, hess


def _soft_cross_entropy_metric(preds: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    teacher = _get_soft_labels(dtrain)
    num_rows, num_class = teacher.shape
    preds = preds.reshape(num_rows, num_class)
    student = _softmax(preds)
    loss = -(teacher * np.log(student + 1e-9)).sum(axis=1).mean()
    return "soft_ce", float(loss)


def train_xgboost_student(
    features: np.ndarray,
    teacher_probs: np.ndarray,
    config: DistillationConfig,
    eval_features: Optional[np.ndarray] = None,
    eval_probs: Optional[np.ndarray] = None,
) -> xgb.Booster:
    """Train a single XGBoost booster to mimic the teacher ensemble."""
    num_classes = teacher_probs.shape[1]
    softened = soften_targets(teacher_probs, config.temperature)
    dtrain = _create_dmatrix(features, softened)

    params = {
        "num_class": num_classes,
        "max_depth": config.max_depth,
        "eta": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "min_child_weight": config.min_child_weight,
        "tree_method": "hist",
        "disable_default_eval_metric": 1,
    }

    evals = []
    if eval_features is not None and eval_probs is not None and eval_features.size:
        softened_eval = soften_targets(eval_probs, config.temperature)
        dval = _create_dmatrix(eval_features, softened_eval)
        evals.append((dval, "eval"))

    early_stopping = config.early_stopping_rounds if evals else None

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=config.num_boost_round,
        evals=evals,
        obj=_soft_target_objective,
        feval=_soft_cross_entropy_metric,
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    booster.set_attr(num_class=str(num_classes))
    for dmat in [dtrain] + [mat for mat, _ in evals]:
        _SOFT_LABEL_REGISTRY.pop(id(dmat), None)
    return booster


def save_booster(booster: xgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
