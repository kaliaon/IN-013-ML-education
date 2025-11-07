#!/bin/bash

# OULAD Learning Analytics Dashboard Launcher
# Phase 5: Visualizations and Dashboard

echo "========================================="
echo "OULAD Learning Analytics Dashboard"
echo "========================================="
echo ""

# Check if we're in the correct directory
if [ ! -d "src/visualization" ]; then
    echo "❌ Error: Please run this script from the Project directory"
    echo "   Usage: ./run_dashboard.sh"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment 'env'..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment 'env'"
    echo "   Please ensure conda is properly installed"
    exit 1
fi

echo "✅ Conda environment activated"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found in environment"
    echo "Installing Streamlit..."
    pip install streamlit

    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install Streamlit"
        exit 1
    fi
    echo "✅ Streamlit installed"
fi

# Check for required Python packages
echo "Checking required packages..."
python -c "import plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing required packages, installing..."
    pip install plotly pillow
    if [ $? -eq 0 ]; then
        echo "✅ Additional packages installed"
    else
        echo "⚠️  Warning: Some packages may have failed to install"
    fi
fi

# Check for required data files
echo "Checking data files..."

if [ ! -f "data/processed/oulad/oulad_with_clusters.csv" ]; then
    echo "⚠️  Warning: oulad_with_clusters.csv not found"
    echo "   Please run Phase 1-2 notebooks first"
fi

# Check for model files
echo "Checking model files..."

MODEL_COUNT=$(ls models/*.pkl 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -lt 6 ]; then
    echo "⚠️  Warning: Some model files missing (found $MODEL_COUNT/6)"
    echo "   Please run Phase 3 notebook to train models"
fi

echo ""
echo "✅ Pre-flight checks complete"
echo ""
echo "Starting Streamlit dashboard..."
echo "Dashboard will open in your default browser"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "========================================="
echo ""

# Launch Streamlit
streamlit run src/visualization/dashboard.py
