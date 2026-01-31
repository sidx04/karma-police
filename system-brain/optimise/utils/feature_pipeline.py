"""Feature extraction helpers that stay in sync with the production pipeline."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_extractor import extract_features  # type: ignore


def load_telemetry_dataset(data_path: Path | str = Path("/data/training_data.pkl")) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load raw telemetry windows and labels produced by the trainer."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry dataset not found: {path}")
    with path.open("rb") as handle:
        data, labels = pickle.load(handle)
    return data, labels


def batches(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Yield chunks to control memory usage during feature extraction."""
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_feature_matrix(
    telemetry: Iterable[Dict[str, Any]],
    selector,
    scaler,
    batch_size: int = 64,
) -> np.ndarray:
    """Convert telemetry samples into the production feature space."""
    features: List[np.ndarray] = []
    for chunk in batches(telemetry, size=batch_size):
        matrix = np.array([extract_features(sample) for sample in chunk])
        if selector is not None:
            matrix = selector.transform(matrix)
        matrix = scaler.transform(matrix)
        features.append(matrix)
    return np.vstack(features)
