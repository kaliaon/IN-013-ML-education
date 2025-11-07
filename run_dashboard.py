#!/usr/bin/env python3
"""
OULAD Learning Analytics Dashboard Launcher
Alternative Python-based launcher for easier execution
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 60)
    print("OULAD Learning Analytics Dashboard")
    print("=" * 60)
    print()

    # Check if we're in the correct directory
    if not Path("src/visualization").exists():
        print("❌ Error: Please run this script from the Project directory")
        print("   Usage: python run_dashboard.py")
        sys.exit(1)

    # Check for data files
    print("Checking data files...")
    data_file = Path("data/processed/oulad/oulad_with_clusters.csv")
    if not data_file.exists():
        print("⚠️  Warning: oulad_with_clusters.csv not found")
        print("   Please run Phase 1-2 notebooks first")

    # Check for model files
    print("Checking model files...")
    model_dir = Path("models")
    if model_dir.exists():
        model_count = len(list(model_dir.glob("*.pkl")))
        if model_count < 6:
            print(f"⚠️  Warning: Some model files missing (found {model_count}/6)")
            print("   Please run Phase 3 notebook to train models")
    else:
        print("⚠️  Warning: models/ directory not found")

    print()
    print("✅ Pre-flight checks complete")
    print()

    # Check if streamlit is available
    try:
        result = subprocess.run(
            ["streamlit", "--version"],
            capture_output=True,
            text=True
        )
        print(f"✅ Streamlit found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Streamlit not found!")
        print()
        print("Installing Streamlit...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "streamlit"],
                check=True
            )
            print("✅ Streamlit installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit")
            print("   Please install manually: pip install streamlit")
            sys.exit(1)

    # Check for required packages
    print()
    print("Checking required packages...")
    required_packages = ["plotly", "pillow"]
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_").lower())
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + missing_packages,
                check=True
            )
            print("✅ All packages installed")
        except subprocess.CalledProcessError:
            print("⚠️  Warning: Some packages may have failed to install")
    else:
        print("✅ All required packages found")

    print()
    print("Starting Streamlit dashboard...")
    print("Dashboard will open in your default browser")
    print()
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    print()

    # Launch Streamlit
    try:
        subprocess.run(
            ["streamlit", "run", "src/visualization/dashboard.py"],
            check=True
        )
    except KeyboardInterrupt:
        print()
        print("Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
