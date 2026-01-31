#!/usr/bin/env python3
"""
Live Workload Classification Service
Polls System's Eye API, classifies workload, serves via HTTP endpoint
"""

import json
import pickle
import time
import threading
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import numpy as np

from feature_extractor import extract_features


class ClassificationResult:
    """Stores latest classification result"""

    def __init__(self):
        self.workload_type = "unknown"
        self.confidence = 0.0
        self.probabilities = {}
        self.timestamp = ""
        self.error = None
        self.lock = threading.Lock()

    def update(self, workload_type, confidence, probabilities, timestamp):
        with self.lock:
            self.workload_type = workload_type
            self.confidence = confidence
            self.probabilities = probabilities
            self.timestamp = timestamp
            self.error = None

    def set_error(self, error_msg):
        with self.lock:
            self.error = error_msg

    def to_dict(self):
        with self.lock:
            return {
                "workload_type": self.workload_type,
                "confidence": self.confidence,
                "probabilities": self.probabilities,
                "timestamp": self.timestamp,
                "error": self.error,
                "service_status": "running",
            }


class LiveClassifier:
    """Continuously classifies workloads from System's Eye telemetry"""

    def __init__(
        self, models_path="/models", eye_api="http://localhost:8080/api/metrics"
    ):
        self.models_path = Path(models_path)
        self.eye_api = eye_api
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.result = ClassificationResult()
        self.running = False
        self.sample_history = []  # Keep rolling window of samples
        self.history_size = 10  # Number of samples to keep

    def load_models(self):
        """Load trained ML models"""
        print("[CLASSIFIER] Loading models...")
        try:
            with open(self.models_path / "model.pkl", "rb") as f:
                self.model = pickle.load(f)
            with open(self.models_path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            with open(self.models_path / "encoder.pkl", "rb") as f:
                self.encoder = pickle.load(f)

            # Try to load feature selector (required for enhanced ensemble model)
            try:
                with open(self.models_path / "feature_selector.pkl", "rb") as f:
                    self.feature_selector = pickle.load(f)
                print(
                    "[CLASSIFIER] Enhanced ensemble model with feature selector loaded"
                )
            except FileNotFoundError:
                print("[WARN] Feature selector not found - using legacy model format")
                self.feature_selector = None

            print(f"[OK] Models loaded successfully")
            print(f"[INFO] Model classes: {self.encoder.classes_}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            self.result.set_error(f"Model loading failed: {e}")
            return False

    def fetch_telemetry(self):
        """Fetch current telemetry from System's Eye API"""
        try:
            with urllib.request.urlopen(self.eye_api, timeout=5) as response:
                data = json.loads(response.read().decode())
            return data
        except Exception as e:
            print(f"[ERROR] Failed to fetch telemetry: {e}")
            return None

    def transform_eye_to_brain_format(self, eye_data):
        """
        Transform System's Eye API format to Brain's expected format
        Eye sends single snapshot, Brain expects list of samples
        Maintains a rolling window of real samples for proper time-series features
        """
        # Create current sample from Eye data
        sample = {
            "timestamp": eye_data.get("timestamp", datetime.now().isoformat()),
            "cpu": {
                "overall": eye_data.get("cpu", {}).get("overall", 0),
                "per_core": eye_data.get("cpu", {}).get("per_core", []),
                "load_avg_1": eye_data.get("cpu", {}).get("load_avg_1", 0),
                "load_avg_5": eye_data.get("cpu", {}).get("load_avg_5", 0),
                "load_avg_15": eye_data.get("cpu", {}).get("load_avg_15", 0),
            },
            "memory": {
                "usage_percent": eye_data.get("memory", {}).get("usage_percent", 0),
                "used": eye_data.get("memory", {}).get("used", 0),
                "total": eye_data.get("memory", {}).get("total", 0),
            },
            "gpu": {"devices": []},
            "disk": eye_data.get("disk", {}),
            "network": eye_data.get("network", {}),
        }

        # Transform GPU data if available
        if "gpu" in eye_data and eye_data["gpu"]:
            gpu_devices = eye_data["gpu"].get("devices", [])
            for gpu in gpu_devices:
                sample["gpu"]["devices"].append(
                    {
                        "utilization": gpu.get("utilization", 0),
                        "memory": {
                            "usage_percent": gpu.get("memory", {}).get(
                                "usage_percent", 0
                            ),
                            "used": gpu.get("memory", {}).get("used", 0),
                            "total": gpu.get("memory", {}).get("total", 0),
                        },
                        "temperature": gpu.get("temperature", 0),
                        "power_usage": gpu.get("power_usage", 0),
                    }
                )

        # Add current sample to history
        self.sample_history.append(sample)

        # Keep only last N samples (rolling window)
        if len(self.sample_history) > self.history_size:
            self.sample_history = self.sample_history[-self.history_size :]

        # If we don't have enough history yet, replicate to fill window
        samples_to_use = self.sample_history.copy()
        while len(samples_to_use) < self.history_size:
            samples_to_use.insert(0, samples_to_use[0])

        brain_format = {"samples": samples_to_use}

        return brain_format

    def classify(self, telemetry_data):
        """Classify workload from telemetry"""
        try:
            # Extract features
            features = extract_features(telemetry_data).reshape(1, -1)

            # Apply feature selection if available
            if self.feature_selector is not None:
                features = self.feature_selector.transform(features)

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Predict
            pred = self.model.predict(features_scaled)[0]

            if hasattr(self.model, "predict_proba"):
                probas = self.model.predict_proba(features_scaled)[0]
                class_indices = getattr(self.model, "classes_", [])
                probabilities = {
                    self.encoder.inverse_transform([cls_idx])[0]: float(prob)
                    for cls_idx, prob in zip(class_indices, probas)
                }
            else:
                probabilities = self._estimate_vote_distribution(features_scaled)

            # Decode prediction
            pred_class = self.encoder.inverse_transform([pred])[0]
            confidence = float(max(probabilities.values())) if probabilities else 0.0

            return pred_class, confidence, probabilities

        except Exception as e:
            print(f"[ERROR] Classification failed: {e}")
            raise

    def _estimate_vote_distribution(self, features_scaled):
        """Approximate class probabilities from ensemble votes"""
        if not hasattr(self.model, "estimators_"):
            return {cls: 0.0 for cls in self.encoder.classes_}

        vote_counts = {cls: 0 for cls in self.encoder.classes_}
        total_votes = 0

        for estimator in self.model.estimators_:
            est_pred = estimator.predict(features_scaled)[0]
            label = self.encoder.inverse_transform([est_pred])[0]
            vote_counts[label] = vote_counts.get(label, 0) + 1
            total_votes += 1

        if total_votes == 0:
            return {cls: 0.0 for cls in self.encoder.classes_}

        return {cls: count / total_votes for cls, count in vote_counts.items()}

    def apply_heuristic_corrections(
        self, workload_type, confidence, probabilities, eye_data
    ):
        """
        Apply rule-based heuristics to correct ML predictions based on GPU patterns.
        This helps overcome model biases and improve accuracy without retraining.
        """
        if (
            "gpu" not in eye_data
            or not eye_data["gpu"]
            or not eye_data["gpu"]["devices"]
        ):
            return workload_type, confidence, probabilities

        gpu = eye_data["gpu"]["devices"][0]
        gpu_util = gpu.get("utilization", 0)
        gpu_mem_pct = gpu.get("memory", {}).get("usage_percent", 0)

        # Rule 1: Very low GPU activity (<5%) = likely idle or misclassification
        if gpu_util < 5 and gpu_mem_pct < 3:
            # Penalize training predictions for idle GPU
            if "training" in workload_type:
                # Redistribute to inference with lower confidence
                new_probs = probabilities.copy()
                inference_variant = workload_type.replace("_training", "_inference")
                if inference_variant in new_probs:
                    # Reduce training probability, increase inference
                    new_probs[workload_type] *= 0.3
                    new_probs[inference_variant] *= 1.5

                    # Renormalize
                    total = sum(new_probs.values())
                    new_probs = {k: v / total for k, v in new_probs.items()}

                    # Get new prediction
                    workload_type = max(new_probs, key=new_probs.get)
                    confidence = new_probs[workload_type]
                    probabilities = new_probs

        # Rule 2: High GPU utilization (>50%) with very low memory (<3%) = likely inference
        elif gpu_util > 50 and gpu_mem_pct < 3:
            if "training" in workload_type:
                new_probs = probabilities.copy()
                inference_variant = workload_type.replace("_training", "_inference")
                if inference_variant in new_probs:
                    # Boost inference probability significantly
                    new_probs[inference_variant] *= 3.0
                    new_probs[workload_type] *= 0.3

                    # Renormalize
                    total = sum(new_probs.values())
                    new_probs = {k: v / total for k, v in new_probs.items()}

                    workload_type = max(new_probs, key=new_probs.get)
                    confidence = new_probs[workload_type]
                    probabilities = new_probs

        # Rule 3: High GPU utilization (>60%) with moderate-high memory (>5%) = likely training
        elif gpu_util > 60 and gpu_mem_pct > 5:
            if "inference" in workload_type:
                new_probs = probabilities.copy()
                training_variant = workload_type.replace("_inference", "_training")
                if training_variant in new_probs:
                    # Boost training probability
                    new_probs[training_variant] *= 2.5
                    new_probs[workload_type] *= 0.4

                    # Renormalize
                    total = sum(new_probs.values())
                    new_probs = {k: v / total for k, v in new_probs.items()}

                    workload_type = max(new_probs, key=new_probs.get)
                    confidence = new_probs[workload_type]
                    probabilities = new_probs

        # Rule 4: Medium GPU util (30-60%) with low memory (1-4%) = likely CNN inference
        elif 30 < gpu_util <= 60 and 1 < gpu_mem_pct < 4:
            if workload_type in ["transformer_training", "transformer_inference"]:
                new_probs = probabilities.copy()
                # Boost CNN inference
                if "cnn_inference" in new_probs:
                    new_probs["cnn_inference"] *= 2.0
                    new_probs[workload_type] *= 0.5

                    # Renormalize
                    total = sum(new_probs.values())
                    new_probs = {k: v / total for k, v in new_probs.items()}

                    workload_type = max(new_probs, key=new_probs.get)
                    confidence = new_probs[workload_type]
                    probabilities = new_probs

        return workload_type, confidence, probabilities

    def run(self):
        """Main classification loop"""
        print("[CLASSIFIER] Starting live classification service")
        print(f"[CLASSIFIER] Polling Eye API: {self.eye_api}")

        if not self.load_models():
            print("[ERROR] Cannot start - model loading failed")
            return

        self.running = True
        classification_count = 0

        while self.running:
            try:
                # Fetch telemetry from Eye
                eye_data = self.fetch_telemetry()

                if eye_data is None:
                    print("[WARN] No telemetry data available")
                    time.sleep(5)
                    continue

                # Transform to Brain format
                brain_data = self.transform_eye_to_brain_format(eye_data)

                # Debug: Print GPU metrics
                if (
                    classification_count % 10 == 0
                    and "gpu" in eye_data
                    and eye_data["gpu"]
                ):
                    gpu_dev = (
                        eye_data["gpu"]["devices"][0]
                        if eye_data["gpu"]["devices"]
                        else {}
                    )
                    print(
                        f"[DEBUG] GPU: {gpu_dev.get('utilization', 0)}% util, {gpu_dev.get('memory', {}).get('usage_percent', 0):.1f}% mem, {gpu_dev.get('temperature', 0)}°C"
                    )

                # Classify
                workload_type, confidence, probabilities = self.classify(brain_data)

                # Apply rule-based corrections to improve predictions
                workload_type, confidence, probabilities = (
                    self.apply_heuristic_corrections(
                        workload_type, confidence, probabilities, eye_data
                    )
                )

                # Check GPU utilization and memory threshold - if truly idle, don't classify
                gpu_util = 0
                gpu_mem_pct = 0
                if "gpu" in eye_data and eye_data["gpu"] and eye_data["gpu"]["devices"]:
                    gpu_util = eye_data["gpu"]["devices"][0].get("utilization", 0)
                    gpu_mem_pct = (
                        eye_data["gpu"]["devices"][0]
                        .get("memory", {})
                        .get("usage_percent", 0)
                    )

                # Only reject if both GPU util AND memory are very low (truly idle)
                if gpu_util < 3 and gpu_mem_pct < 5:
                    workload_type = "insufficient_confidence"
                    confidence = 0.0
                    print(
                        f"[CLASSIFY] GPU idle ({gpu_util}% util, {gpu_mem_pct:.1f}% mem) - No active workload detected"
                    )
                # Apply confidence threshold (lowered to 30% for better demo performance)
                elif confidence < 0.30:
                    workload_type = "insufficient_confidence"
                    print(
                        f"[CLASSIFY] Confidence too low ({confidence * 100:.2f}%) - Unable to classify workload"
                    )

                # Update result
                timestamp = eye_data.get("timestamp", datetime.now().isoformat())
                self.result.update(workload_type, confidence, probabilities, timestamp)

                classification_count += 1

                if classification_count % 10 == 0:
                    print(
                        f"[CLASSIFY] {classification_count}: {workload_type} ({confidence:.2%})"
                    )
                else:
                    print(f"[CLASSIFY] {workload_type} ({confidence:.2%})")

                # Sleep before next classification
                time.sleep(3)  # Classify every 3 seconds

            except Exception as e:
                print(f"[ERROR] Classification loop error: {e}")
                self.result.set_error(str(e))
                time.sleep(5)

    def stop(self):
        """Stop the classifier"""
        print("[CLASSIFIER] Stopping...")
        self.running = False


class ClassificationHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for classification API"""

    classifier = None  # Will be set by main

    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/api/classification":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            result = self.classifier.result.to_dict()
            self.wfile.write(json.dumps(result, indent=2).encode())

        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"status": "healthy", "service": "classifier"}).encode()
            )

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress HTTP logging"""
        pass


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Live Workload Classification Service")
    parser.add_argument(
        "--models-path", default="/models", help="Path to trained models"
    )
    parser.add_argument(
        "--eye-api",
        default="http://localhost:8080/api/metrics",
        help="System's Eye API endpoint",
    )
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    args = parser.parse_args()

    print("=" * 60)
    print("LIVE WORKLOAD CLASSIFICATION SERVICE")
    print("=" * 60)
    print(f"Models path: {args.models_path}")
    print(f"Eye API: {args.eye_api}")
    print(f"HTTP port: {args.port}")
    print()

    # Create classifier
    classifier = LiveClassifier(models_path=args.models_path, eye_api=args.eye_api)

    # Set classifier for HTTP handler
    ClassificationHTTPHandler.classifier = classifier

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), ClassificationHTTPHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[HTTP] Classification API server started on port {args.port}")
    print(f"[HTTP] Endpoint: http://localhost:{args.port}/api/classification")
    print()

    # Start classifier loop
    try:
        classifier.run()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Received interrupt signal")
    finally:
        classifier.stop()
        server.shutdown()
        print("[SHUTDOWN] Complete")


if __name__ == "__main__":
    main()
