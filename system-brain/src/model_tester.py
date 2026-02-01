#!/usr/bin/env python3
"""
Comprehensive testing suite for ML workload classification models.
Includes validation, testing, and performance analysis.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from collections import Counter

from feature_extractor import extract_features


class ModelTester:
    """Comprehensive model testing and validation suite"""

    def __init__(self, models_path='/models', data_path='/data'):
        self.models_path = models_path
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_selector = None

    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            with open(f'{self.models_path}/model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open(f'{self.models_path}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f'{self.models_path}/encoder.pkl', 'rb') as f:
                self.encoder = pickle.load(f)

            # Load feature selector (required for enhanced ensemble model)
            try:
                with open(f'{self.models_path}/feature_selector.pkl', 'rb') as f:
                    self.feature_selector = pickle.load(f)
                print('[OK] Enhanced ensemble model with feature selector loaded')
            except FileNotFoundError:
                print('[WARN] Feature selector not found - using legacy model format')
                self.feature_selector = None

            return True
        except Exception as e:
            print(f'[ERROR] Failed to load models: {e}')
            return False

    def test_inference(self):
        """Basic inference test with sample data"""
        if not self.load_models():
            return None

        # Test sample
        test_data = {
            'samples': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'gpu': {'devices': [{'utilization': 85, 'memory': {'usage_percent': 60},
                                        'temperature': 75, 'power_usage': 280}]},
                    'memory': {'usage_percent': 60},
                    'cpu': {'overall': 40}
                } for _ in range(10)
            ]
        }

        features = extract_features(test_data).reshape(1, -1)

        # Apply feature selection if available
        if self.feature_selector is not None:
            features_selected = self.feature_selector.transform(features)
        else:
            features_selected = features

        features_scaled = self.scaler.transform(features_selected)
        pred, confidence, _ = self._predict_with_confidence(features_scaled)
        pred_class = self.encoder.inverse_transform([pred])[0]

        # Map legacy workload label to parameter-count classification for readability
        def _label_to_classification(label):
            phase = 'training' if label.endswith('_training') else 'inference' if label.endswith('_inference') else 'unknown'
            mapping = {
                'transformer_training': 20_000_000_000,
                'transformer_inference': 5_000_000_000,
                'cnn_training': 50_000_000_000,
                'cnn_inference': 500_000_000
            }
            pc = mapping.get(label)
            if pc is None:
                return f'Unknown model being {phase}'
            # binning
            if pc < 1_000_000_000:
                bin_name, range_str = 'Very Small', '< 1B'
            elif pc < 10_000_000_000:
                bin_name, range_str = 'Small', '1–10B'
            elif pc < 30_000_000_000:
                bin_name, range_str = 'Medium', '10–30B'
            elif pc < 100_000_000_000:
                bin_name, range_str = 'Large', '30–100B'
            else:
                bin_name, range_str = 'Very Large', '> 100B'
            return f"{bin_name} model ({range_str} parameters) being {phase}"

        readable = _label_to_classification(pred_class)
        print(f'[TEST] Basic inference result: {readable} (confidence: {confidence:.3f})')
        return pred_class

    def comprehensive_model_test(self):
        """Run comprehensive model testing with multiple workload patterns"""
        if not self.load_models():
            return None, None

        # Define test cases with expected workload patterns
        test_cases = [
            # (test_name, gpu_range, mem_range, mem_growth, expected_class)
            ('Transformer Training Test', (55, 75), (40, 70), True, 'transformer_training'),
            ('CNN Training Test', (85, 95), (35, 50), False, 'cnn_training'),
            ('Transformer Inference Test', (40, 60), (30, 45), False, 'transformer_inference'),
            ('CNN Inference Test', (70, 82), (35, 50), False, 'cnn_inference'),
            # Edge cases
            ('High CNN Training', (90, 99), (35, 45), False, 'cnn_training'),
            ('Low Transformer Inference', (42, 50), (25, 35), False, 'transformer_inference'),
            # Memory growth pattern test
            ('Memory Growing Training', (60, 75), (40, 80), True, 'transformer_training'),
            # Mixed patterns
            ('Variable GPU Training', (50, 90), (30, 60), True, 'transformer_training'),
            ('Stable CNN Inference', (75, 78), (40, 42), False, 'cnn_inference'),
            # Stress tests
            ('Low Resource Inference', (20, 30), (15, 20), False, 'transformer_inference'),
            ('High Resource Training', (95, 99), (80, 95), True, 'cnn_training'),
        ]

        results = []
        total_tests = len(test_cases)
        correct = 0

        print(f'[TEST] Running {total_tests} comprehensive model tests...')
        print('=' * 80)

        for test_name, gpu_range, mem_range, mem_growth, expected in test_cases:
            # Generate test sample
            samples = self._generate_test_sample(gpu_range, mem_range, mem_growth)
            test_data = {'samples': samples}

            # Extract features and predict
            features = extract_features(test_data).reshape(1, -1)

            # Apply feature selection if available
            if self.feature_selector is not None:
                features_selected = self.feature_selector.transform(features)
            else:
                features_selected = features

            features_scaled = self.scaler.transform(features_selected)
            pred, confidence, probabilities = self._predict_with_confidence(features_scaled)
            pred_class = self.encoder.inverse_transform([pred])[0]

            is_correct = pred_class == expected
            if is_correct:
                correct += 1
                status = '[PASS]'
            else:
                status = '[FAIL]'

            print(f'{status} {test_name}')
            print(f'     GPU: {gpu_range[0]}-{gpu_range[1]}%, Mem: {mem_range[0]}-{mem_range[1]}%, Growth: {mem_growth}')
            # Map legacy labels to human-readable parameter-count classification
            def _label_to_classification(label):
                phase = 'training' if label.endswith('_training') else 'inference' if label.endswith('_inference') else 'unknown'
                mapping = {
                    'transformer_training': 20_000_000_000,
                    'transformer_inference': 5_000_000_000,
                    'cnn_training': 50_000_000_000,
                    'cnn_inference': 500_000_000
                }
                pc = mapping.get(label)
                if pc is None:
                    return f'Unknown model being {phase}'
                if pc < 1_000_000_000:
                    bin_name, range_str = 'Very Small', '< 1B'
                elif pc < 10_000_000_000:
                    bin_name, range_str = 'Small', '1–10B'
                elif pc < 30_000_000_000:
                    bin_name, range_str = 'Medium', '10–30B'
                elif pc < 100_000_000_000:
                    bin_name, range_str = 'Large', '30–100B'
                else:
                    bin_name, range_str = 'Very Large', '> 100B'
                return f"{bin_name} model ({range_str} parameters) being {phase}"

            expected_readable = _label_to_classification(expected)
            predicted_readable = _label_to_classification(pred_class)

            print(f'     Expected: {expected} -> {expected_readable}')
            print(f'     Predicted: {pred_class} -> {predicted_readable} (confidence: {confidence:.3f})')

            # Show probability distribution for failed tests
            if not is_correct and probabilities:
                print(f'     Probability distribution:')
                for class_name in self.encoder.classes_:
                    prob = probabilities.get(class_name, 0.0)
                    print(f'       {class_name}: {prob:.3f}')
            print()

            results.append({
                'test_name': test_name,
                'expected': expected,
                'predicted': pred_class,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'correct': is_correct,
                'gpu_range': gpu_range,
                'mem_range': mem_range,
                'mem_growth': mem_growth
            })

        accuracy = correct / total_tests
        self._print_test_summary(total_tests, correct, accuracy)
        self._save_test_results(results, accuracy)

        return accuracy, results

    def validate_training_data(self):
        """Validate training data quality and distribution"""
        data_file = f'{self.data_path}/training_data.pkl'
        if not os.path.exists(data_file):
            print('[ERROR] No training data found!')
            return False

        with open(data_file, 'rb') as f:
            data, labels = pickle.load(f)

        print('[DATA] Training Data Validation:')
        print(f'  Total samples: {len(data)}')
        print(f'  Unique classes: {len(set(labels))}')

        # Count class distribution
        class_counts = Counter(labels)
        print(f'  Class distribution:')
        total_samples = len(labels)
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100
            balance_status = '[OK]' if 20 <= percentage <= 30 else '[WARN]'
            print(f'    {balance_status} {class_name:25}: {count:4} samples ({percentage:.1f}%)')

        # Validate feature extraction
        print(f'\n  Feature validation:')
        sample_features = extract_features(data[0])
        print(f'    Feature vector length: {len(sample_features)}')
        print(f'    Feature range: [{np.min(sample_features):.3f}, {np.max(sample_features):.3f}]')

        # Check for NaN or infinite values
        has_nan = np.any(np.isnan(sample_features))
        has_inf = np.any(np.isinf(sample_features))
        print(f'    Contains NaN: {has_nan}')
        print(f'    Contains Inf: {has_inf}')

        # Data quality metrics
        print(f'\n  Data quality metrics:')

        # Check sample consistency
        sample_lengths = [len(d.get('samples', [])) for d in data]
        print(f'    Sample length range: {min(sample_lengths)} - {max(sample_lengths)}')
        print(f'    Average sample length: {np.mean(sample_lengths):.1f}')

        # Check telemetry completeness
        complete_samples = sum(1 for d in data if self._check_telemetry_completeness(d))
        completeness = (complete_samples / len(data)) * 100
        print(f'    Telemetry completeness: {completeness:.1f}%')

        return True

    def performance_dashboard(self):
        """Generate comprehensive performance dashboard"""
        history_file = f'{self.models_path}/training_history.json'
        if not os.path.exists(history_file):
            print('[WARN] No training history found!')
            return

        with open(history_file, 'r') as f:
            history = json.load(f)

        print('\n' + '='*80)
        print('[STATS] PERFORMANCE DASHBOARD')
        print('='*80)

        # Overall progress
        total_sessions = len(history)
        latest = history[-1] if history else {}

        print('[STATS] TRAINING PROGRESS:')
        print(f'   Total Training Sessions: {total_sessions}')
        print(f'   Current Model: {latest.get("model_info", {}).get("n_estimators", 0)} trees')
        print(f'   Model Size: {latest.get("model_info", {}).get("model_size_mb", 0):.2f}MB')
        print(f'   Latest Accuracy: {latest.get("accuracy", 0)*100:.1f}%')

        # Trend analysis
        if len(history) >= 3:
            recent_sessions = history[-3:]
            accuracies = [s.get('accuracy', 0) for s in recent_sessions]
            model_sizes = [s.get('model_info', {}).get('n_estimators', 0) for s in recent_sessions]

            acc_trend = '[UP]' if accuracies[-1] > accuracies[0] else '[STABLE]' if accuracies[-1] == accuracies[0] else '[DOWN]'
            size_growth = model_sizes[-1] - model_sizes[0]

            print(f'   Accuracy Trend (last 3): {acc_trend}')
            print(f'   Model Growth: +{size_growth} trees')

        # Per-class performance details
        if 'per_class_accuracy' in latest:
            print(f'\n[TARGET] PER-CLASS PERFORMANCE:')
            per_class = latest['per_class_accuracy']
            per_conf = latest.get('per_class_confidence', {})

            for class_name, accuracy in per_class.items():
                confidence = per_conf.get(class_name, 0) * 100
                status = '[GOOD]' if accuracy > 0.9 else '[WARN]' if accuracy > 0.7 else '[BAD]'
                print(f'   {status} {class_name:25}: {accuracy*100:5.1f}% acc | {confidence:5.1f}% conf')

        # Feature importance (if available)
        if 'feature_importance' in latest:
            print(f'\n[INFO] TOP FEATURES (Most Important):')
            features = latest['feature_importance']
            for i, (feature, importance) in enumerate(list(features.items())[:5], 1):
                bar = '█' * int(importance * 30)
                print(f'   {i}. {feature:12}: {bar:30} ({importance:.3f})')

        # Performance timeline
        print(f'\n[DATA] SESSION HISTORY (Last 5):')
        recent_history = history[-5:] if len(history) > 5 else history
        print(f'   {"#":>3} {"Time":>20} {"Trees":>6} {"Accuracy":>8} {"Progressive":>12}')
        print(f'   {"-"*60}')

        for session in recent_history:
            session_num = session.get('session_number', 0)
            timestamp = session.get('timestamp', '')[:19]
            trees = session.get('model_info', {}).get('n_estimators', 0)
            accuracy = session.get('accuracy', 0) * 100
            train_time = session.get('training_time_seconds', 0)
            progressive = 'Yes' if session.get('progressive', False) else 'No'

            print(f'   {session_num:>3} {timestamp:>20} {trees:>6} {accuracy:>7.1f}% {progressive:>12}')

        # Recommendations
        self._generate_recommendations(latest, history)

        print('='*80)

    def _generate_test_sample(self, gpu_range, mem_range, mem_growth, num_samples=50):
        """Generate synthetic test sample with specified characteristics"""
        samples = []
        base_mem = np.random.uniform(mem_range[0], mem_range[1])

        for t in range(num_samples):
            # Add growth pattern if specified
            if mem_growth:
                mem = base_mem + (t * 0.5) + np.random.uniform(-3, 3)
            else:
                mem = np.random.uniform(mem_range[0], mem_range[1])

            gpu_util = np.random.uniform(gpu_range[0], gpu_range[1])
            cpu_overall = np.random.uniform(20, 50)

            # Generate per-core CPU data (16 cores)
            per_core = []
            for core in range(16):
                core_util = max(0, cpu_overall + np.random.uniform(-10, 10))
                per_core.append(core_util)

            # Generate process data
            num_processes = 15 if mem_growth else 8
            top_cpu = []
            top_memory = []
            for i in range(num_processes):
                top_cpu.append({
                    'pid': 1000 + i,
                    'name': f'process_{i}',
                    'cpu_percent': np.random.uniform(0, 25),
                    'memory_percent': np.random.uniform(0, 15),
                    'cmdline': f'python training.py' if i < 3 and mem_growth else f'worker_{i}'
                })
                top_memory.append({
                    'pid': 2000 + i,
                    'name': f'mem_proc_{i}',
                    'cpu_percent': np.random.uniform(0, 10),
                    'memory_percent': np.random.uniform(0, 20),
                    'cmdline': f'pytorch_{i}'
                })

            # GPU processes
            gpu_processes = []
            if gpu_util > 30:
                num_gpu_procs = 3 if mem_growth else 1
                for i in range(num_gpu_procs):
                    gpu_processes.append({
                        'pid': 3000 + i,
                        'name': 'python',
                        'gpu_memory_mb': int(gpu_util * 40 + np.random.uniform(-200, 200))
                    })

            samples.append({
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'overall': cpu_overall,
                    'per_core': per_core
                },
                'memory': {
                    'usage_percent': min(95, max(10, mem)),
                    'used_gb': min(95, max(10, mem)) * 64 / 100,
                    'total_gb': 64.0
                },
                'gpu': {
                    'devices': [{
                        'index': 0,
                        'utilization': gpu_util,
                        'memory': {
                            'used': gpu_util * 80,
                            'total': 80.0,
                            'usage_percent': min(95, max(10, mem + 5))
                        },
                        'temperature': np.random.uniform(65, 85),
                        'power_usage': gpu_util * 3.5
                    }]
                },
                'processes': {
                    'top_cpu': top_cpu,
                    'top_memory': top_memory,
                    'gpu_processes': gpu_processes
                },
                'system': {
                    'load_avg': [cpu_overall/20, cpu_overall/25, cpu_overall/30],
                    'uptime': 3600 + t
                }
            })

        return samples

    def _check_telemetry_completeness(self, data):
        """Check if telemetry data has all required fields"""
        samples = data.get('samples', [])
        if not samples:
            return False

        required_fields = ['timestamp', 'cpu', 'memory', 'gpu']
        for sample in samples:
            if not all(field in sample for field in required_fields):
                return False

        return True

    def _print_test_summary(self, total_tests, correct, accuracy):
        """Print test summary statistics"""
        print('=' * 80)
        print('[SUMMARY] TEST RESULTS:')
        print(f'  Total Tests: {total_tests}')
        print(f'  Passed: {correct}')
        print(f'  Failed: {total_tests - correct}')
        print(f'  Accuracy: {accuracy:.1%}')

        # Grade the model
        if accuracy >= 0.95:
            grade = 'A+ - Excellent'
        elif accuracy >= 0.90:
            grade = 'A - Very Good'
        elif accuracy >= 0.85:
            grade = 'B - Good'
        elif accuracy >= 0.75:
            grade = 'C - Acceptable'
        else:
            grade = 'F - Needs Improvement'

        print(f'  Grade: {grade}')
        print('=' * 80)

    def _save_test_results(self, results, accuracy):
        """Save test results to file"""
        test_report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'passed': sum(1 for r in results if r['correct']),
            'accuracy': accuracy,
            'results': results
        }

        test_file = f'{self.data_path}/test_results.json'
        with open(test_file, 'w') as f:
            json.dump(test_report, f, indent=2)

        print(f'[SAVE] Test results saved to {test_file}')

    def _generate_recommendations(self, latest, history):
        """Generate improvement recommendations"""
        print(f'\n[RECOMMENDATIONS]:')

        recommendations = []

        # Accuracy recommendations
        current_acc = latest.get('accuracy', 0)
        if current_acc < 0.95:
            recommendations.append('Consider more training sessions to improve accuracy')

        # Model size recommendations
        n_trees = latest.get('model_info', {}).get('n_estimators', 0)
        if n_trees < 200 and current_acc < 0.95:
            recommendations.append('Model has room to grow - more trees could help')

        # Class balance recommendations
        if 'per_class_accuracy' in latest:
            worst_class = min(latest['per_class_accuracy'].items(), key=lambda x: x[1], default=('', 1))
            if worst_class[1] < 0.9:
                recommendations.append(f'Focus on improving {worst_class[0]} classification')

        # Training trend recommendations
        if len(history) >= 3:
            recent_accs = [h.get('accuracy', 0) for h in history[-3:]]
            if recent_accs[-1] < recent_accs[-2]:
                recommendations.append('Accuracy decreased in last session - check for overfitting')

        # Display recommendations
        if recommendations:
            for rec in recommendations:
                print(f'   • {rec}')
        else:
            print('   • Model performance is excellent!')

    def _predict_with_confidence(self, features_scaled):
        """Return prediction index, confidence, and probability mapping"""
        pred = self.model.predict(features_scaled)[0]

        probabilities = {}
        confidence = 0.0

        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(features_scaled)[0]
            classes = getattr(self.model, 'classes_', [])
            for cls_idx, prob in zip(classes, probas):
                label = self.encoder.inverse_transform([cls_idx])[0]
                probabilities[label] = float(prob)
            if probabilities:
                confidence = max(probabilities.values())
        elif hasattr(self.model, 'estimators_'):
            vote_counts = {}
            total_votes = 0
            for estimator in self.model.estimators_:
                est_pred = estimator.predict(features_scaled)[0]
                label = self.encoder.inverse_transform([est_pred])[0]
                vote_counts[label] = vote_counts.get(label, 0) + 1
                total_votes += 1
            if total_votes > 0:
                for label in self.encoder.classes_:
                    probabilities[label] = vote_counts.get(label, 0) / total_votes
                confidence = max(probabilities.values())

        return pred, confidence, probabilities


def main():
    """Main entry point for model tester"""
    import argparse

    parser = argparse.ArgumentParser(description='Model Testing Suite')
    parser.add_argument('--test', choices=['basic', 'comprehensive', 'validate', 'dashboard', 'all'],
                       default='all', help='Test type to run')
    parser.add_argument('--models-path', default='/models', help='Path to models')
    parser.add_argument('--data-path', default='/data', help='Path to data')

    args = parser.parse_args()

    tester = ModelTester(args.models_path, args.data_path)

    if args.test in ['basic', 'all']:
        print('\n[TEST 1] Running basic inference test...')
        tester.test_inference()

    if args.test in ['comprehensive', 'all']:
        print('\n[TEST 2] Running comprehensive model tests...')
        accuracy, results = tester.comprehensive_model_test()

    if args.test in ['validate', 'all']:
        print('\n[TEST 3] Validating training data...')
        tester.validate_training_data()

    if args.test in ['dashboard', 'all']:
        print('\n[TEST 4] Generating performance dashboard...')
        tester.performance_dashboard()


if __name__ == '__main__':
    main()
