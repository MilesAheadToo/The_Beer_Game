#!/bin/bash

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create necessary directories
mkdir -p data/processed logs_cpu checkpoints_cpu

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Please run setup_training_env.sh first."
    exit 1
fi

# Run the training script
python scripts/train.py \
    --config scripts/training/config/cpu_config.yaml \
    --data-dir data/processed \
    --output-dir checkpoints_cpu \
    --device cpu

# Deactivate virtual environment
deactivate
