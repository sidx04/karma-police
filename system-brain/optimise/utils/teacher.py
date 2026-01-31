"""Helpers for loading and querying the teacher ensemble."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Re-use production modules by extending sys.path dynamically.
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from demo_runner import MLClassifierService  # type: ignore
from feature_extractor import extract_features  # type: ignore


class TeacherModel:
    """Thin wrapper around the production MLClassifierService."""

    def __init__(self, models_path: Path | str = Path("/models")) -> None:
        self._service = MLClassifierService(models_path=str(models_path))
        if self._service.model is None:
            raise RuntimeError("Failed to load teacher model stack from %s" % models_path)

    @property
    def encoder(self):
        return self._service.encoder

    @property
    def scaler(self):
        return self._service.scaler

    @property
    def feature_selector(self):
        return self._service.feature_selector

    def predict_proba(self, telemetry_sample: Dict[str, Any]) -> np.ndarray:
        """Return the teacher ensemble probabilities for the telemetry sample."""
        result = self._service.classify_workload(telemetry_sample)
        return np.array([result["probabilities"][cls] for cls in self.encoder.classes_])

    def to_feature_vector(self, telemetry_sample: Dict[str, Any]) -> np.ndarray:
        """Convert telemetry dict into the exact feature vector used in production."""
        features = extract_features(telemetry_sample).reshape(1, -1)
        if self.feature_selector is not None:
            features = self.feature_selector.transform(features)
        features = self.scaler.transform(features)
        return features.squeeze(0)

    def dump_metadata(self) -> Dict[str, Any]:
        model = self._service.model
        info: Dict[str, Any] = {}
        if hasattr(model, "named_estimators_"):
            info["estimators"] = {
                name: {
                    "type": type(estimator).__name__,
                    **({"n_estimators": getattr(estimator, "n_estimators", None)} if hasattr(estimator, "n_estimators") else {}),
                }
                for name, estimator in model.named_estimators_.items()
            }
        return info


def export_metadata(path: Path, teacher: TeacherModel) -> None:
    path.write_text(json.dumps(teacher.dump_metadata(), indent=2))
