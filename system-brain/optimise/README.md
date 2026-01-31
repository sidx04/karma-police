# System Brain Optimisation Toolkit

This package houses experiments that compress and refine System's Brain models.
The focus is on *distilling* the existing ensemble into a smaller student while
re-using the production feature pipeline, telemetry datasets, and evaluation
suite under `work/system-brain/src/`.

## Key Ideas

- Load the teacher ensemble (`model.pkl`, `scaler.pkl`, `feature_selector.pkl`,
  `encoder.pkl`) exactly as the runtime does in `demo_runner.MLClassifierService`.
- Generate soft labels from saved telemetry windows (`/data`) or freshly
  captured workloads via `RealWorkloadExecutor`.
- Train a compact student model (single XGBoost booster by default) on those
  soft labels.
- Evaluate the student using `ModelTester` to ensure behaviour matches the
  teacher before swapping models.

## Directory Layout

- `configs/`: Hyper-parameters and tolerances for distillation experiments.
- `data/`: Cached distilled datasets (features + teacher probabilities).
- `scripts/`: Command-line tools that implement each optimisation stage.
- `utils/`: Shared helpers that wrap existing modules from `src/`.
- `reports/`: Evaluation outputs for auditability.
- `artifacts/`: Candidate student models ready for packaging.

## Typical Workflow

Run the orchestrator to mirror the main training pipeline:

```bash
# Run all steps (extract → train → evaluate)
./optimise/run_pipeline.py all

# Or run individual stages
./optimise/run_pipeline.py extract
./optimise/run_pipeline.py train
./optimise/run_pipeline.py evaluate
```

Each stage accepts the same overrides as the CLI scripts, e.g.

```bash
./optimise/run_pipeline.py extract \
  --models-path /models \
  --data-path /data/training_data.pkl \
  --dataset optimise/data/distilled_dataset.npz
```

Behind the scenes the pipeline reuses the teacher loader, feature selector,
scaler, and evaluation harness from the runtime code, ensuring the student
model sees exactly the same data representation as production.
