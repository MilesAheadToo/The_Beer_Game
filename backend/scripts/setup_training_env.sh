#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    
    # Upgrade pip and setuptools
    pip install --upgrade pip setuptools wheel
    
    # Install PyTorch for CPU
    echo "Installing PyTorch for CPU..."
    pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
    
    # Install PyTorch Geometric and its dependencies
    echo "Installing PyTorch Geometric dependencies..."
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    pip install torch-geometric
    
    # Install other requirements
    echo "Installing other requirements..."
    pip install -r scripts/requirements-train.txt
    
    echo "Environment setup complete!"
    echo "Activate the virtual environment with: source venv/bin/activate"
else
    echo "Virtual environment already exists. Activate it with: source venv/bin/activate"
fi
