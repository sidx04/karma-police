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

    def classify_workload(self, telemetry):
        """Classify workload from telemetry data"""
        features = extract_features(telemetry).reshape(1, -1)

        # Apply feature selection if available
        if self.feature_selector is not None:
            features_selected = self.feature_selector.transform(features)
        else:
            features_selected = features

        features_scaled = self.scaler.transform(features_selected)

        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)

        workload_type = self.encoder.inverse_transform([prediction])[0]

        return {
            'workload_type': workload_type,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in zip(self.encoder.classes_, probabilities)
            }
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

            # Classify the workload
            result = self.classifier.classify_workload(telemetry)
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
        return self.executor.execute_real_workload(workload, duration)

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