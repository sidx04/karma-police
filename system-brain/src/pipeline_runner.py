#!/usr/bin/env python3
"""
Pipeline runner for complete ML training pipeline.
Coordinates data generation, training, and testing.
"""

import os
import pickle
import json
from datetime import datetime

from workload_executor import RealWorkloadExecutor
from model_trainer import train_model, generate_training_data
from model_tester import ModelTester
from demo_runner import LiveDemoRunner


class PipelineRunner:
    """Coordinates the complete ML pipeline"""

    def __init__(self, models_path='/models', data_path='/data', sample_duration=30, runs_per_config=3):
        self.models_path = models_path
        self.data_path = data_path
        self.sample_duration = sample_duration
        self.runs_per_config = runs_per_config
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        self.executor = RealWorkloadExecutor()
        self.tester = ModelTester(models_path, data_path)

    def run_step1_data_generation(self, force_new=False):
        """Step 1: Generate new training data and combine with existing"""
        import time
        print('='*60)
        print('STEP 1: GENERATING REAL TRAINING DATA WITH GPU')
        print('='*60)

        data_file = f'{self.data_path}/training_data.pkl'
        existing_data, existing_labels = [], []

        # Load existing data if it exists
        print(f'[DATA] Checking for existing data at {data_file}...')
        if os.path.exists(data_file):
            print('[DATA] Loading existing training data...')
            load_start = time.time()
            with open(data_file, 'rb') as f:
                existing_data, existing_labels = pickle.load(f)
            load_time = time.time() - load_start
            print(f'[OK] Loaded {len(existing_data)} existing samples in {load_time:.1f}s')
        else:
            print('[DATA] No existing data found, will generate from scratch')

        # ALWAYS generate new data
        print(f'[DATA] Generating NEW real workload data with GPU... [{time.strftime("%H:%M:%S")}]')
        print('[DATA] This involves running actual PyTorch workloads and collecting telemetry')
        print(f'[DATA] Configuration: duration={self.sample_duration}s, runs_per_config={self.runs_per_config}')
        gen_start = time.time()

        try:
            new_data, new_labels = generate_training_data(
                sample_duration=self.sample_duration,
                runs_per_config=self.runs_per_config
            )
            gen_time = time.time() - gen_start
            print(f'[OK] Generated {len(new_data)} NEW training samples in {gen_time:.1f}s')
        except Exception as e:
            gen_time = time.time() - gen_start
            print(f'[ERROR] Data generation failed after {gen_time:.1f}s: {e}')
            raise

        # Combine existing and new data
        print('[DATA] Combining existing and new datasets...')
        data = existing_data + new_data
        labels = existing_labels + new_labels

        # Save combined dataset
        print('[DATA] Saving combined dataset...')
        save_start = time.time()
        with open(data_file, 'wb') as f:
            pickle.dump((data, labels), f)
        save_time = time.time() - save_start

        print(f'[OK] Total dataset: {len(data)} samples ({len(existing_data)} existing + {len(new_data)} new)')
        print(f'[OK] Classes: {set(labels)}')
        print(f'[OK] Combined data saved to {data_file} in {save_time:.1f}s')

        return data, labels

    def run_step2_training(self, progressive=None):
        """Step 2: Train models with optional progressive training"""
        print('='*60)
        print('STEP 2: TRAINING ML MODELS')
        print('='*60)

        # Determine if progressive training should be used
        if progressive is None:
            progressive = os.path.exists(f'{self.models_path}/model.pkl')
            print(f'Progressive training: {progressive}')

        # Load training data
        data_file = f'{self.data_path}/training_data.pkl'

        if os.path.exists(data_file):
            print('Loading existing training data...')
            with open(data_file, 'rb') as f:
                existing_data, existing_labels = pickle.load(f)
            print(f'Loaded {len(existing_data)} existing samples')
        else:
            existing_data, existing_labels = [], []

        # Generate additional training data for progressive training
        if progressive and len(existing_data) < 100:
            print('Generating additional training data...')
            new_data, new_labels = generate_training_data(
                sample_duration=self.sample_duration,
                runs_per_config=self.runs_per_config
            )
            print(f'Generated {len(new_data)} new samples')

            # Combine existing and new data
            data = existing_data + new_data
            labels = existing_labels + new_labels

            # Save combined dataset
            with open(data_file, 'wb') as f:
                pickle.dump((data, labels), f)
            print(f'Saved expanded dataset with {len(data)} samples')
        else:
            data = existing_data
            labels = existing_labels

        print(f'Total dataset size: {len(data)} samples')

        # Train with progressive option
        model, validation_accuracy = train_model(data, labels, progressive=progressive)
        print(f'Validation accuracy: {validation_accuracy:.4f}')

        # Show training history
        self._display_training_history()
        print('[OK] Models saved to /models/')

        return validation_accuracy

    def run_step3_inference_test(self):
        """Step 3: Test inference capabilities"""
        print('='*60)
        print('STEP 3: TESTING INFERENCE')
        print('='*60)

        # Ensure model exists
        if not os.path.exists(f'{self.models_path}/model.pkl'):
            print('[ERROR] No trained model found! Training first...')
            self.run_step2_training()

        print('\n[STEP 3a] Testing basic inference...')
        prediction = self.tester.test_inference()
        print(f'[OK] Basic test prediction: {prediction}')

        print('\n[STEP 3b] Running comprehensive model tests...')
        accuracy, _ = self.tester.comprehensive_model_test()
        print(f'[OK] Comprehensive test accuracy: {accuracy:.1%}')

        print('\n[STEP 3c] Validating training data...')
        self.tester.validate_training_data()
        print('[OK] Data validation complete')

        print('\n[OK] All inference tests successful')
        return accuracy

    def run_step4_comprehensive_testing(self):
        """Step 4: Comprehensive model testing and analysis"""
        print('='*60)
        print('STEP 4: COMPREHENSIVE MODEL TESTING')
        print('='*60)

        # Check if models exist
        if not os.path.exists(f'{self.models_path}/model.pkl'):
            print('[ERROR] No trained model found!')
            print('Please run training first: ./deploy.sh train')
            return None

        print('\n[TEST 1] Running comprehensive model tests...')
        accuracy, results = self.tester.comprehensive_model_test()
        print(f'[OK] Model test accuracy: {accuracy:.1%}')

        print('\n[TEST 2] Validating training data...')
        self.tester.validate_training_data()
        print('[OK] Data validation passed')

        print('\n[TEST 3] Basic inference test...')
        prediction = self.tester.test_inference()
        print(f'[OK] Basic inference: {prediction}')

        # Performance Dashboard
        self.tester.performance_dashboard()

        # Performance analysis
        self._analyze_test_results(results)

        print('\n[OK] Comprehensive testing complete')
        return accuracy

    def run_full_pipeline(self):
        """Run the complete ML pipeline"""
        import time
        start_time = time.time()

        print('='*60)
        print('COMPLETE ML PIPELINE')
        print('='*60)
        print(f'[PIPELINE] Starting full pipeline at {time.strftime("%H:%M:%S")}')
        print(f'[PIPELINE] Models path: {self.models_path}')
        print(f'[PIPELINE] Data path: {self.data_path}')

        # Force output flush for real-time logging
        import sys
        sys.stdout.flush()

        try:
            # Step 1: Generate/Load Data
            print(f'\n[STEP 1] Loading or generating training data... [{time.strftime("%H:%M:%S")}]')
            step1_start = time.time()
            data, labels = self.run_step1_data_generation()
            step1_time = time.time() - step1_start
            print(f'[STEP 1] Completed in {step1_time:.1f}s - Got {len(data)} samples')

            # Step 2: Train Models
            print(f'\n[STEP 2] Training models... [{time.strftime("%H:%M:%S")}]')
            step2_start = time.time()
            validation_accuracy = self.run_step2_training()
            step2_time = time.time() - step2_start
            print(f'[STEP 2] Completed in {step2_time:.1f}s - Validation Accuracy: {validation_accuracy:.4f}')

            # Step 3: Test Inference
            print(f'\n[STEP 3] Testing inference... [{time.strftime("%H:%M:%S")}]')
            step3_start = time.time()
            self.run_step3_inference_test()
            step3_time = time.time() - step3_start
            print(f'[STEP 3] Completed in {step3_time:.1f}s')

            # Step 4: Comprehensive Testing
            print(f'\n[STEP 4] Running comprehensive model tests... [{time.strftime("%H:%M:%S")}]')
            step4_start = time.time()
            test_accuracy = self.run_step4_comprehensive_testing()
            step4_time = time.time() - step4_start
            print(f'[STEP 4] Completed in {step4_time:.1f}s - Test Accuracy: {test_accuracy:.1%}')

            # Summary
            total_time = time.time() - start_time
            print(f'\n[PIPELINE] Total execution time: {total_time:.1f}s')
            self._print_pipeline_summary(data, labels, validation_accuracy, test_accuracy)

            print(f'\n[OK] Pipeline completed successfully at {time.strftime("%H:%M:%S")}')
            return validation_accuracy, test_accuracy

        except Exception as e:
            elapsed = time.time() - start_time
            print(f'\n[ERROR] Pipeline failed after {elapsed:.1f}s: {e}')
            import traceback
            traceback.print_exc()
            raise

    def _display_training_history(self):
        """Display recent training history"""
        history_file = f'{self.models_path}/training_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)

            print(f'Training sessions: {len(history)}')
            for h in history[-3:]:  # Show last 3 sessions
                timestamp = h['timestamp'][:19]
                accuracy = h['accuracy']
                progressive = h.get('progressive', False)  # Default to False for old entries
                n_trees = h.get('model_info', {}).get('n_estimators', 0)
                print(f"  {timestamp}: Accuracy={accuracy:.4f}, Trees={n_trees}, Progressive={progressive}")

    def _analyze_test_results(self, results):
        """Analyze and report on test results"""
        print('\n[ANALYSIS] Model Performance Summary:')
        print('=' * 50)

        if not results:
            print('No test results to analyze')
            return

        failed_tests = [r for r in results if not r['correct']]

        if failed_tests:
            print('Failed test cases:')
            for test in failed_tests:
                test_name = test['test_name']
                expected = test['expected']
                predicted = test['predicted']
                confidence = test['confidence']
                print(f'  - {test_name}: Expected {expected}, Got {predicted} (conf: {confidence:.2f})')
        else:
            print('[OK] All test cases passed!')

        # Confidence analysis
        confidences = [r['confidence'] for r in results]
        import numpy as np
        print(f'\nConfidence Statistics:')
        print(f'  Mean: {np.mean(confidences):.3f}')
        print(f'  Std:  {np.std(confidences):.3f}')
        print(f'  Min:  {np.min(confidences):.3f}')
        print(f'  Max:  {np.max(confidences):.3f}')

    def _print_pipeline_summary(self, data, labels, validation_accuracy, test_accuracy):
        """Print pipeline execution summary"""
        print('\n' + '='*60)
        print('PIPELINE COMPLETED SUCCESSFULLY')
        print('='*60)
        print('Results:')
        print(f'  Training Samples: {len(data)}')
        print(f'  Workload Classes: {len(set(labels))}')

        if validation_accuracy is not None:
            print(f'  Validation Accuracy: {validation_accuracy:.4f}')

        if test_accuracy:
            print(f'  Comprehensive Test Accuracy: {test_accuracy:.1%}')

            # Determine status
            if test_accuracy >= 0.9:
                status = '[OK] All tests passed - Model is production ready'
            elif test_accuracy >= 0.8:
                status = '[WARN] Model needs improvement'
            else:
                status = '[ERROR] Model performance insufficient'

            print(f'  Status: {status}')

        print('='*60)


def main():
    """Main entry point for pipeline runner"""
    import argparse

    parser = argparse.ArgumentParser(description='ML Pipeline Runner')
    parser.add_argument('step', nargs='?', default='all',
                       choices=['1', '2', '3', '4', 'all', 'data', 'train', 'test', 'full'],
                       help='Pipeline step to run')
    parser.add_argument('--progressive', action='store_true',
                       help='Use progressive training')
    parser.add_argument('--force-new-data', action='store_true',
                       help='Force generation of new training data')
    parser.add_argument('--models-path', default='/models',
                       help='Path to models directory')
    parser.add_argument('--data-path', default='/data',
                       help='Path to data directory')
    parser.add_argument('--duration', type=int, default=30,
                       help='Sample duration in seconds for each workload run (step 1)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per workload configuration (step 1)')

    args = parser.parse_args()

    runner = PipelineRunner(
        models_path=args.models_path,
        data_path=args.data_path,
        sample_duration=args.duration,
        runs_per_config=args.runs
    )

    # Map step names to functions
    step_map = {
        '1': runner.run_step1_data_generation,
        'data': runner.run_step1_data_generation,
        '2': lambda: runner.run_step2_training(args.progressive),
        'train': lambda: runner.run_step2_training(args.progressive),
        '3': runner.run_step3_inference_test,
        'test': runner.run_step3_inference_test,
        '4': runner.run_step4_comprehensive_testing,
        'all': runner.run_full_pipeline,
        'full': runner.run_full_pipeline,
    }

    # Execute the requested step
    if args.step in step_map:
        if args.step in ['1', 'data']:
            step_map[args.step](args.force_new_data)
        else:
            step_map[args.step]()
    else:
        print(f'Unknown step: {args.step}')


if __name__ == '__main__':
    main()
