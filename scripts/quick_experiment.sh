#!/bin/bash
# Quick experiment script for testing model performance

echo "Starting H&M Recommendation Model Experiments"
echo "==========================================="

# Set experiment parameters
SAMPLE_FRACTION=0.01  # Use 1% of data for quick testing
EPOCHS=5              # Reduced epochs for quick testing

# Create experiment directory
EXPERIMENT_DIR="experiments/quick_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $EXPERIMENT_DIR

echo "Experiment directory: $EXPERIMENT_DIR"
echo "Sample fraction: $SAMPLE_FRACTION"
echo "Epochs: $EPOCHS"
echo ""

# Function to run experiment
run_experiment() {
    MODEL=$1
    NAME=$2
    EXTRA_ARGS=$3
    
    echo "----------------------------------------"
    echo "Running experiment: $NAME"
    echo "Model: $MODEL"
    echo "----------------------------------------"
    
    python scripts/train.py \
        model=$MODEL \
        data.sample_fraction=$SAMPLE_FRACTION \
        training.epochs=$EPOCHS \
        paths.output_dir=$EXPERIMENT_DIR/$NAME \
        run_name=$NAME \
        training.val_check_interval=1.0 \
        $EXTRA_ARGS
    
    if [ $? -eq 0 ]; then
        echo "✅ $NAME completed successfully"
    else
        echo "❌ $NAME failed"
    fi
    echo ""
}

# Run experiments
echo "Starting experiments..."
echo ""

# 1. Popularity Baseline (should be very fast)
run_experiment "popularity_baseline" "popularity_baseline" ""

# 2. Matrix Factorization
run_experiment "matrix_factorization" "matrix_factorization" "training.batch_size=4096"

# 3. Neural CF
run_experiment "neural_cf" "neural_cf" "model.mlp_dims=[64,32,16]"

# 4. Wide & Deep (with features)
run_experiment "wide_deep" "wide_deep" "data.use_features=true model.deep_layers=[256,128,64]"

# 5. LightGCN (with BPR)
run_experiment "lightgcn" "lightgcn" "data.dataset_type=bpr training.batch_size=2048"

echo "==========================================="
echo "All experiments completed!"
echo "Results saved to: $EXPERIMENT_DIR"
echo ""

# Create simple summary
echo "Creating summary..."
python -c "
import os
import json
import pandas as pd
from pathlib import Path

exp_dir = Path('$EXPERIMENT_DIR')
results = []

for model_dir in exp_dir.iterdir():
    if model_dir.is_dir():
        results_files = list(model_dir.glob('*_results.yaml'))
        if results_files:
            import yaml
            with open(results_files[0], 'r') as f:
                data = yaml.safe_load(f)
                data['model'] = model_dir.name
                results.append(data)

if results:
    df = pd.DataFrame(results)
    print('\nModel Performance Summary:')
    print('=' * 60)
    if 'test_map' in df.columns:
        summary = df[['model', 'test_map', 'test_recall', 'test_precision', 'test_ndcg']].round(4)
        summary = summary.sort_values('test_map', ascending=False)
        print(summary.to_string(index=False))
    else:
        print('No test results found')
else:
    print('No results found')
"

echo ""
echo "To run full experiments with more data, use:"
echo "python scripts/run_experiments.py --sample-fraction 0.1"