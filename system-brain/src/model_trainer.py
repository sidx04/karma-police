#!/usr/bin/env python3
"""
Model training pipeline for workload classification.
Supports both fresh training and progressive learning.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

from feature_extractor import extract_features
from workload_executor import RealWorkloadExecutor
from sklearn.utils._tags import _DEFAULT_TAGS

# Fix for Scikit-Learn 1.6+ VotingClassifier AttributeError
def patch_voting_classifier():
    if not hasattr(VotingClassifier, "__sklearn_tags__"):
        def __sklearn_tags__(self):
            return _DEFAULT_TAGS
        VotingClassifier.__sklearn_tags__ = __sklearn_tags__

patch_voting_classifier()


def create_enhanced_model(n_samples, n_features, progressive=False):
    """
    Create enhanced ensemble model with XGBoost, Random Forest, and SVM.

    Args:
        n_samples: Number of training samples
        n_features: Number of features after selection
        progressive: Whether to use progressive training

    Returns:
        Ensemble model optimized for telemetry classification
    """
    print(f"[MODEL] Creating enhanced ensemble model for {n_samples} samples, {n_features} features")

    # XGBoost - Primary model optimized for balanced bias/variance
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )

    # Random Forest - Increased capacity with stronger regularization
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        warm_start=progressive
    )

    # SVM - For different decision boundary
    svm_model = SVC(
        kernel='rbf',
        C=0.3,
        gamma='scale',
        probability=False,
        random_state=42
    )

    # Create ensemble with hard voting and reduced SVM influence
    ensemble = VotingClassifier([
        ('xgboost', xgb_model),
        ('random_forest', rf_model),
        ('svm', svm_model)
    ], voting='hard', weights=[2, 2, 1])
    
    ensemble._estimator_type = "classifier"
    
    print(f"[MODEL] Ensemble components:")
    print(f"  - XGBoost: {xgb_model.n_estimators} estimators, lr={xgb_model.learning_rate}")
    print(f"  - Random Forest: {rf_model.n_estimators} trees, max_features={rf_model.max_features}")
    print(f"  - SVM: RBF kernel, C={svm_model.C}, probability={svm_model.probability}")

    return ensemble


def select_best_features(X, y, n_features=60):
    """
    Select the most informative features using statistical tests.

    Args:
        X: Feature matrix
        y: Target labels
        n_features: Number of features to select

    Returns:
        Feature selector and selected feature indices
    """
    # Ensure we don't select more features than available
    n_features = min(n_features, X.shape[1])

    print(f"[FEATURES] Selecting top {n_features} features from {X.shape[1]} total")

    # Use ANOVA F-statistic for feature selection
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_selected = selector.fit_transform(X, y)

    # Get feature scores for analysis
    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)

    print(f"[FEATURES] Selected features: {len(selected_indices)}")
    print(f"[FEATURES] Feature score range: {np.min(scores[selected_indices]):.2f} - {np.max(scores[selected_indices]):.2f}")

    return selector, X_selected


def generate_training_data(sample_duration=30, runs_per_config=3):
    """
    Generate training data using real workload execution.

    Args:
        sample_duration: Duration in seconds for each workload sample
        runs_per_config: Number of runs per workload configuration

    Returns:
        tuple of (data, labels)
    """
    executor = RealWorkloadExecutor()
    data = []
    labels = []

    # Vary configurations for better generalization
    config_variations = {
        'transformer_training': [
            {'batch_size': 16, 'sequence_length': 512},
            {'batch_size': 32, 'sequence_length': 1024},
            {'batch_size': 8, 'sequence_length': 2048},
        ],
        'cnn_training': [
            {'batch_size': 64, 'image_size': 224},
            {'batch_size': 128, 'image_size': 128},
            {'batch_size': 32, 'image_size': 512},
        ],
        'transformer_inference': [
            {'batch_size': 1, 'sequence_length': 256},
            {'batch_size': 8, 'sequence_length': 512},
            {'batch_size': 16, 'sequence_length': 128},
        ],
        'cnn_inference': [
            {'batch_size': 32, 'image_size': 224},
            {'batch_size': 64, 'image_size': 128},
            {'batch_size': 128, 'image_size': 224},
        ]
    }

    import time
    total_start = time.time()
    print("Generating training data with real GPU workloads...")

    total_workloads = sum(len(variations) * runs_per_config for variations in config_variations.values())
    current_workload = 0

    print(f"[WORKLOAD] Total workloads to execute: {total_workloads}")
    print(f"[WORKLOAD] Sample duration: {sample_duration}s each")
    print(f"[WORKLOAD] Estimated total time: {total_workloads * sample_duration / 60:.1f} minutes")

    for workload_type, variations in config_variations.items():
        print(f"\n[WORKLOAD TYPE] Starting {workload_type} ({len(variations)} variations)")

        for var_idx, variation in enumerate(variations):
            print(f"[VARIATION] {workload_type} variant {var_idx + 1}/{len(variations)}: {variation}")
            # Update executor config with variation
            executor.workload_configs[workload_type].update(variation)

            for run in range(runs_per_config):
                current_workload += 1
                run_start = time.time()

                print(f"[RUN {current_workload}/{total_workloads}] {workload_type} (variant {var_idx + 1}, run {run + 1}/{runs_per_config}) [{time.strftime('%H:%M:%S')}]")

                try:
                    # Execute workload and collect telemetry
                    telemetry = executor.execute_real_workload(workload_type, sample_duration)
                    data.append(telemetry)
                    labels.append(workload_type)

                    run_time = time.time() - run_start
                    progress = (current_workload / total_workloads) * 100
                    elapsed_total = time.time() - total_start
                    estimated_remaining = (elapsed_total / current_workload) * (total_workloads - current_workload)

                    print(f"[OK] Run completed in {run_time:.1f}s | Progress: {progress:.1f}% | ETA: {estimated_remaining/60:.1f}min")

                except Exception as e:
                    run_time = time.time() - run_start
                    print(f"[ERROR] Run failed after {run_time:.1f}s: {e}")
                    raise

    total_time = time.time() - total_start
    print(f"\n[WORKLOAD] Generated {len(data)} training samples in {total_time/60:.1f} minutes")
    return data, labels


def train_model(data, labels, progressive=False, model_path='/models/model.pkl'):
    """
    Train the classification model.

    Args:
        data: List of telemetry data samples
        labels: List of workload type labels
        progressive: If True, improve existing model; if False, train from scratch
        model_path: Path to save/load model

    Returns:
        tuple of (model, accuracy)
    """
    # Extract features
    print("Extracting features...")
    X = np.array([extract_features(d) for d in data])
    y = np.array(labels)

    # Encode labels
    encoder_path = Path(model_path).parent / 'encoder.pkl'
    selector_path = Path(model_path).parent / 'feature_selector.pkl'
    scaler_path = Path(model_path).parent / 'scaler.pkl'

    n_features_to_select = 45
    if progressive and selector_path.exists():
        with open(selector_path, 'rb') as f:
            previous_selector = pickle.load(f)
        if hasattr(previous_selector, 'k') and previous_selector.k != 'all':
            n_features_to_select = previous_selector.k

    if progressive and encoder_path.exists():
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        # Fit any new labels
        encoder.classes_ = np.unique(np.concatenate([encoder.classes_, np.unique(y)]))
    else:
        encoder = LabelEncoder()
        encoder.fit(y)

    y_encoded = encoder.transform(y)

    print(f"Using Stratified 3-fold cross-validation with {n_features_to_select} selected features")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_train_scores = []
    fold_val_scores = []
    last_fold_report = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), start=1):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        feature_selector_fold, X_train_selected = select_best_features(
            X_train_raw, y_train, n_features=n_features_to_select
        )
        X_val_selected = feature_selector_fold.transform(X_val_raw)

        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train_selected)
        X_val_scaled = scaler_fold.transform(X_val_selected)

        if progressive and os.path.exists(model_path):
            print("[INFO] Progressive ensemble training not yet supported, creating fresh ensemble")
            progressive = False

        model_fold = create_enhanced_model(
            n_samples=len(X_train_scaled),
            n_features=X_train_scaled.shape[1],
            progressive=progressive
        )
        model_fold.fit(X_train_scaled, y_train)

        y_train_pred = model_fold.predict(X_train_scaled)
        y_val_pred = model_fold.predict(X_val_scaled)

        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        fold_train_scores.append(train_acc)
        fold_val_scores.append(val_acc)

        print(f"[CV] Fold {fold_idx}: train={train_acc:.2%}, val={val_acc:.2%}")

        last_fold_report = classification_report(
            y_val, y_val_pred, target_names=encoder.classes_, digits=3
        )

    train_mean = float(np.mean(fold_train_scores))
    train_std = float(np.std(fold_train_scores))
    val_mean = float(np.mean(fold_val_scores))
    val_std = float(np.std(fold_val_scores))

    print(f"\nCross-validation summary:")
    print(f"  Train Accuracy: {train_mean:.2%} ± {train_std:.2%}")
    print(f"  Validation Accuracy: {val_mean:.2%} ± {val_std:.2%}")

    if last_fold_report is not None:
        print("\nClassification Report (last fold):")
        print(last_fold_report)

    # Refit transformers on full dataset for the persisted model
    feature_selector = SelectKBest(score_func=f_classif, k=min(n_features_to_select, X.shape[1]))
    feature_selector.fit(X, y_encoded)
    X_full_selected = feature_selector.transform(X)

    scaler = StandardScaler()
    scaler.fit(X_full_selected)
    X_full_scaled = scaler.transform(X_full_selected)

    final_model = create_enhanced_model(
        n_samples=len(X_full_scaled),
        n_features=X_full_scaled.shape[1],
        progressive=False
    )
    final_model.fit(X_full_scaled, y_encoded)

    full_training_pred = final_model.predict(X_full_scaled)
    full_training_accuracy = accuracy_score(y_encoded, full_training_pred)
    print(f"[INFO] Final model trained on full dataset - Training Accuracy: {full_training_accuracy:.2%}")

    # Save models and preprocessors
    models_dir = Path(model_path).parent
    models_dir.mkdir(exist_ok=True, parents=True)

    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoder, f)
    with open(selector_path, 'wb') as f:
        pickle.dump(feature_selector, f)

    print(f"[SAVE] Enhanced ensemble model saved to {models_dir}")
    print(f"[SAVE] Components: model, scaler, encoder, feature_selector")

    # Update training history
    history_file = models_dir / 'training_history.json'
    history = []
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)

    # Get ensemble information
    ensemble_info = {}
    if hasattr(final_model, 'estimators_'):
        for name, estimator in final_model.named_estimators_.items():
            if hasattr(estimator, 'n_estimators'):
                ensemble_info[name] = {'type': 'tree_ensemble', 'n_estimators': estimator.n_estimators}
            elif hasattr(estimator, 'kernel'):
                ensemble_info[name] = {'type': 'svm', 'kernel': estimator.kernel}
            else:
                ensemble_info[name] = {'type': 'other'}

    history.append({
        'timestamp': datetime.now().isoformat(),
        'accuracy': float(val_mean),
        'validation_accuracy_mean': float(val_mean),
        'validation_accuracy_std': float(val_std),
        'train_accuracy_mean': float(train_mean),
        'train_accuracy_std': float(train_std),
        'full_training_accuracy': float(full_training_accuracy),
        'n_samples': len(data),
        'progressive': progressive,
        'model_info': {
            'type': 'ensemble',
            'n_features_original': X.shape[1],
            'n_features_selected': X_full_selected.shape[1],
            'ensemble_components': ensemble_info
        }
    })

    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nModels saved to {models_dir}")

    return final_model, val_mean


def main():
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description='Train workload classification model')
    parser.add_argument('--progressive', action='store_true', help='Progressive training mode')
    parser.add_argument('--duration', type=int, default=30, help='Sample duration in seconds')
    parser.add_argument('--runs', type=int, default=3, help='Runs per configuration')
    parser.add_argument('--model-path', default='/models/model.pkl', help='Model save path')
    args = parser.parse_args()

    print("=" * 60)
    print("WORKLOAD CLASSIFICATION TRAINING")
    print("=" * 60)

    if args.progressive:
        print("Mode: Progressive Training")
    else:
        print("Mode: Fresh Training")

    # Generate data
    data, labels = generate_training_data(args.duration, args.runs)

    # Save raw data
    data_dir = Path('/data')
    data_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_file = data_dir / f'training_data_{timestamp}.pkl'

    with open(data_file, 'wb') as f:
        pickle.dump((data, labels), f)
    print(f"Data saved to {data_file}")

    # Train model
    model, accuracy = train_model(data, labels, progressive=args.progressive, model_path=args.model_path)

    print("\n" + "=" * 60)
    print(f"Training Complete - Accuracy: {accuracy:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
