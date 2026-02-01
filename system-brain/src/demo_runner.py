#!/usr/bin/env python3
"""
Live demo runner for System's Brain ML workload classification.
Demonstrates real-time workload classification capabilities.
"""

import pickle
import numpy as np
import time
from datetime import datetime

from workload_executor import RealWorkloadExecutor
from feature_extractor import extract_features


class MLClassifierService:
    """Service for real-time ML workload classification"""

    def __init__(self, models_path='/models'):
        self.models_path = models_path
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_selector = None
        self.load_model()

    def load_model(self):
        """Load trained model and preprocessors"""
        try:
            with open(f'{self.models_path}/model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open(f'{self.models_path}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f'{self.models_path}/encoder.pkl', 'rb') as f:
                self.encoder = pickle.load(f)

            # Load feature selector (required for new enhanced model)
            try:
                with open(f'{self.models_path}/feature_selector.pkl', 'rb') as f:
                    self.feature_selector = pickle.load(f)
                print('[OK] Enhanced ensemble model loaded successfully')

                # Display ensemble information
                if hasattr(self.model, 'estimators_'):
                    print(f'   Model: Enhanced Ensemble with {len(self.model.estimators_)} components')
                    for name, estimator in self.model.named_estimators_.items():
                        if hasattr(estimator, 'n_estimators'):
                            print(f'     - {name}: {estimator.n_estimators} estimators')
                        else:
                            print(f'     - {name}: {type(estimator).__name__}')
                    print(f'   Features: {self.feature_selector.k} selected from {len(self.feature_selector.scores_)} total')
                else:
                    # Fallback for older models
                    if hasattr(self.model, 'n_estimators'):
                        print(f'   Model: {self.model.n_estimators} trees')

            except FileNotFoundError:
                print('[WARN] Feature selector not found - using legacy model format')
                self.feature_selector = None

            return True
        except Exception as e:
            print(f'[ERROR] Failed to load model: {e}')
            return False

    def classify_workload(self, telemetry, model_info=None):
        """Classify workload from telemetry data.

        New behavior:
        - Build a human-readable classification based on model parameter-count bins
          and execution phase (training/inference).
        - Only emit the human-readable classification if the underlying model
          prediction confidence is >= 0.75. Otherwise return an unverified result.
        """
        # Extract numeric features for ML prediction (if available)
        try:
            features = extract_features(telemetry).reshape(1, -1)
        except Exception:
            features = None

        # Apply feature selection if available
        if features is not None:
            if self.feature_selector is not None:
                features_selected = self.feature_selector.transform(features)
            else:
                features_selected = features

            features_scaled = self.scaler.transform(features_selected)

            # Attempt to get model probabilities
            probabilities = None
            confidence = 0.0
            try:
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(features_scaled)[0]
                    confidence = float(np.max(probabilities))
                    prediction = int(self.model.predict(features_scaled)[0])
                else:
                    # Fallback: use predict (no probabilities)
                    prediction = int(self.model.predict(features_scaled)[0])
                    confidence = 0.0
            except Exception:
                probabilities = None
                confidence = 0.0
        else:
            probabilities = None
            confidence = 0.0

        # Helper: human readable param formatting and binning
        def _human_readable_params(n):
            if n is None:
                return 'unknown'
            if n >= 10**11:
                return f"{n/10**9:.0f}B"
            if n >= 10**9:
                val = n/10**9
                if abs(val - round(val)) < 1e-6:
                    return f"{int(round(val))}B"
                return f"{val:.1f}B"
            if n >= 10**6:
                return f"{n/10**6:.1f}M".rstrip('0').rstrip('.')
            return str(n)

        def _param_bin(n):
            # Return (bin_name, range_str)
            if n is None:
                return ('Unknown size', 'unknown')
            if n < 1_000_000_000:
                return ('Very Small', '< 1B')
            if 1_000_000_000 <= n < 10_000_000_000:
                return ('Small', '1–10B')
            if 10_000_000_000 <= n < 30_000_000_000:
                return ('Medium', '10–30B')
            if 30_000_000_000 <= n < 100_000_000_000:
                return ('Large', '30–100B')
            return ('Very Large', '> 100B')

        # Determine parameter count and phase
        param_count = None
        phase = None
        if model_info and isinstance(model_info, dict):
            param_count = model_info.get('parameter_count')
            phase = model_info.get('phase')
        # telemetry may embed model_info
        if param_count is None and isinstance(telemetry, dict):
            mi = telemetry.get('model_info')
            if isinstance(mi, dict):
                param_count = mi.get('parameter_count')
                phase = phase or mi.get('phase')

        # Fallback: try to infer phase from predicted class label if available
        inferred_phase = None
        try:
            if 'prediction' in locals() and hasattr(self.encoder, 'inverse_transform'):
                pred_label = self.encoder.inverse_transform([prediction])[0]
                if pred_label.endswith('_training'):
                    inferred_phase = 'training'
                elif pred_label.endswith('_inference'):
                    inferred_phase = 'inference'
        except Exception:
            inferred_phase = None

        if phase is None:
            phase = inferred_phase or 'unknown'

        # Build human-readable classification string
        bin_name, range_str = _param_bin(param_count)
        human_params = _human_readable_params(param_count)
        phase_text = phase.lower() if phase else 'unknown'
        classification_string = f"{bin_name} model ({range_str} parameters) being {phase_text}"

        # Only emit classification if confidence is high enough
        verified = False
        if confidence >= 0.75:
            verified = True
            return {
                'workload_type': classification_string,
                'confidence': float(confidence),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.encoder.classes_, probabilities)
                } if probabilities is not None else {},
                'parameter_count': int(param_count) if param_count is not None else None,
                'parameter_count_human': _human_readable_params(param_count),
                'verified': True
            }

        # Low confidence: return unverified response (do not claim classification)
        # Provide best-effort info for debugging
        best_guess = None
        if 'prediction' in locals() and hasattr(self.encoder, 'inverse_transform'):
            try:
                best_guess = self.encoder.inverse_transform([prediction])[0]
            except Exception:
                best_guess = None

        return {
            'workload_type': 'unverified',
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.encoder.classes_, probabilities)
            } if probabilities is not None else {},
            'parameter_count': int(param_count) if param_count is not None else None,
            'parameter_count_human': _human_readable_params(param_count),
            'verified': False,
            'best_guess': best_guess
        }


class LiveDemoRunner:
    """Runner for live classification demonstrations"""

    def __init__(self):
        self.executor = RealWorkloadExecutor()
        self.classifier = MLClassifierService()

    def run_demo_scenarios(self, scenarios=None):
        """Run predefined demo scenarios"""
        if scenarios is None:
            scenarios = [
                ('transformer_training', 'Training a large language model'),
                ('cnn_inference', 'Running image classification'),
                ('transformer_inference', 'LLM text generation'),
                ('cnn_training', 'Training computer vision model')
            ]

        results = []

        for i, (workload, description) in enumerate(scenarios, 1):
            print(f'\n[ACTION] DEMO SCENARIO {i}/{len(scenarios)}: {description}')
            print('=' * 60)

            # Execute workload and capture telemetry
            telemetry = self.executor.execute_real_workload(workload, duration_seconds=30)

            # Build model_info from executor config so classification can use parameter count
            cfg = self.executor.workload_configs.get(workload, {})
            model_info = None
            try:
                if cfg.get('model_type') == 'transformer':
                    tmp_model = self.executor.create_transformer_model(cfg)
                else:
                    tmp_model = self.executor.create_cnn_model(cfg)
                param_count = sum(p.numel() for p in tmp_model.parameters())
                def _hr(n):
                    if n >= 10**9:
                        val = n/10**9
                        if abs(val - round(val)) < 1e-6:
                            return f"{int(round(val))}B"
                        return f"{val:.1f}B"
                    if n >= 10**6:
                        return f"{n/10**6:.1f}M".rstrip('0').rstrip('.')
                    return str(n)
                model_info = {
                    'parameter_count': int(param_count),
                    'parameter_count_human': _hr(param_count),
                    'phase': cfg.get('mode', 'unknown')
                }
            except Exception:
                model_info = None

            # Attach model_info into telemetry if possible
            if isinstance(telemetry, dict):
                try:
                    telemetry['model_info'] = model_info
                except Exception:
                    pass

            # Classify the workload
            result = self.classifier.classify_workload(telemetry, model_info=model_info)
            results.append((workload, result))

            # Display results
            print('[TARGET] CLASSIFICATION RESULTS:')
            print(f'   Predicted: {result["workload_type"]}')
            print(f'   Confidence: {result["confidence"]:.1%}')
            print(f'   Expected: {workload}')

            status = '[OK] CORRECT' if result['workload_type'] == workload else '[ERROR] INCORRECT'
            print(f'   Status: {status}')

            # Show probability distribution
            print(f'   Probabilities:')
            for class_name, prob in result['probabilities'].items():
                bar = '█' * int(prob * 20)
                print(f'     {class_name:25} {bar:20} {prob:.2%}')

            time.sleep(5)

        # Summary
        self._print_summary(results)
        return results

    def run_continuous_monitoring(self, duration=300):
        """Run continuous workload monitoring"""
        print(f'[MONITOR] Starting continuous monitoring for {duration}s...')

        start_time = time.time()
        classifications = []

        while time.time() - start_time < duration:
            # Capture 10 seconds of telemetry
            print(f'\n[CAPTURE] Collecting telemetry... ({int(time.time() - start_time)}/{duration}s)')
            telemetry = self._capture_current_telemetry(10)

            # Classify current workload
            result = self.classifier.classify_workload(telemetry)
            classifications.append({
                'timestamp': datetime.now().isoformat(),
                'classification': result
            })

            # Display result
            print(f'[LIVE] Current workload: {result["workload_type"]} (confidence: {result["confidence"]:.1%})')

            # Wait before next capture
            time.sleep(5)

        print('\n[MONITOR] Continuous monitoring complete')
        self._analyze_monitoring_results(classifications)

        return classifications

    def _capture_current_telemetry(self, duration):
        """Capture current system telemetry without running specific workload"""
        # This would capture whatever is currently running on the system
        # For demo purposes, we'll run a random workload
        import random
        workload_types = list(self.executor.workload_configs.keys())
        workload = random.choice(workload_types)
        telemetry = self.executor.execute_real_workload(workload, duration)

        # Try to attach model_info from the workload config
        cfg = self.executor.workload_configs.get(workload, {})
        try:
            if cfg.get('model_type') == 'transformer':
                tmp_model = self.executor.create_transformer_model(cfg)
            else:
                tmp_model = self.executor.create_cnn_model(cfg)
            param_count = sum(p.numel() for p in tmp_model.parameters())
            def _hr(n):
                if n >= 10**9:
                    val = n/10**9
                    if abs(val - round(val)) < 1e-6:
                        return f"{int(round(val))}B"
                    return f"{val:.1f}B"
                if n >= 10**6:
                    return f"{n/10**6:.1f}M".rstrip('0').rstrip('.')
                return str(n)
            model_info = {
                'parameter_count': int(param_count),
                'parameter_count_human': _hr(param_count),
                'phase': cfg.get('mode', 'unknown')
            }
        except Exception:
            model_info = None

        if isinstance(telemetry, dict):
            try:
                telemetry['model_info'] = model_info
            except Exception:
                pass

        return telemetry

    def _print_summary(self, results):
        """Print demo summary"""
        print('\n' + '=' * 60)
        print('[STATS] DEMO SUMMARY')
        correct = sum(1 for expected, result in results if expected == result['workload_type'])
        accuracy = correct / len(results)
        print(f'[TARGET] Overall Accuracy: {accuracy:.1%} ({correct}/{len(results)})')

        # Confidence analysis
        confidences = [result['confidence'] for _, result in results]
        print(f'[CONF] Average Confidence: {np.mean(confidences):.1%}')
        print(f'[CONF] Min Confidence: {np.min(confidences):.1%}')
        print(f'[CONF] Max Confidence: {np.max(confidences):.1%}')

    def _analyze_monitoring_results(self, classifications):
        """Analyze continuous monitoring results"""
        from collections import Counter

        # Count workload types
        workload_counts = Counter(c['classification']['workload_type'] for c in classifications)

        print('\n[ANALYSIS] Workload Distribution:')
        total = len(classifications)
        for workload, count in workload_counts.most_common():
            percentage = (count / total) * 100
            print(f'   {workload:25}: {count:3} ({percentage:.1f}%)')

        # Confidence statistics
        confidences = [c['classification']['confidence'] for c in classifications]
        print(f'\n[CONF] Confidence Statistics:')
        print(f'   Mean: {np.mean(confidences):.1%}')
        print(f'   Std:  {np.std(confidences):.1%}')
        print(f'   Min:  {np.min(confidences):.1%}')
        print(f'   Max:  {np.max(confidences):.1%}')


def main():
    """Main entry point for demo runner"""
    import argparse

    parser = argparse.ArgumentParser(description='System\'s Brain Live Demo')
    parser.add_argument('--mode', choices=['demo', 'monitor', 'wait'], default='demo',
                       help='Demo mode: demo (run scenarios), monitor (continuous), wait (keep alive)')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration for monitoring mode (seconds)')
    parser.add_argument('--models-path', default='/models',
                       help='Path to models directory')

    args = parser.parse_args()

    if args.mode == 'wait':
        print('[WAIT] Waiting for model to be available...')
        import os
        while not os.path.exists(f'{args.models_path}/model.pkl'):
            print('[WAIT] Model not found, waiting...')
            time.sleep(10)
        print('[OK] Model found!')

    # Create demo runner
    runner = LiveDemoRunner()

    if args.mode == 'demo':
        runner.run_demo_scenarios()
    elif args.mode == 'monitor':
        runner.run_continuous_monitoring(args.duration)

    # Keep container alive if needed
    if args.mode == 'wait':
        print('[TARGET] Demo completed. Container staying alive for interaction...')
        import signal
        signal.pause()


if __name__ == '__main__':
    main()