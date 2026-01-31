#!/bin/bash
# Create ConfigMap from Python source files

NAMESPACE="eidf219ns"

echo "Creating ConfigMap from Python source files..."

SCRIPT_DIR="$(dirname "$0")"
SRC_DIR="$SCRIPT_DIR/../src"

kubectl create configmap teamovercooked-ml-src \
  --from-file=workload_executor.py=$SRC_DIR/workload_executor.py \
  --from-file=feature_extractor.py=$SRC_DIR/feature_extractor.py \
  --from-file=model_trainer.py=$SRC_DIR/model_trainer.py \
  --from-file=model_tester.py=$SRC_DIR/model_tester.py \
  --from-file=pipeline_runner.py=$SRC_DIR/pipeline_runner.py \
  --from-file=demo_runner.py=$SRC_DIR/demo_runner.py \
  -n $NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

echo "ConfigMap 'teamovercooked-ml-src' created/updated"