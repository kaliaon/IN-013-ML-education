#!/bin/bash

# Quick install script for dashboard dependencies
# Run this once to install all required packages

echo "========================================="
echo "Installing Dashboard Dependencies"
echo "========================================="
echo ""

# Activate conda environment
echo "Activating conda environment 'env'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment 'env'"
    exit 1
fi

echo "✅ Conda environment activated"
echo ""

# Install all required packages
echo "Installing required packages..."
echo ""

pip install streamlit plotly pillow

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Installation Complete!"
    echo "========================================="
    echo ""
    echo "You can now run the dashboard with:"
    echo "  ./run_dashboard.sh"
    echo "  OR"
    echo "  python run_dashboard.py"
    echo ""
else
    echo ""
    echo "❌ Installation failed"
    echo "Please check the error messages above"
    exit 1
fi
