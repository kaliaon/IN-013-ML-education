# Dashboard Installation & Launch Guide

## Quick Start (3 Options)

### ✅ Option 1: Python Launcher (Recommended)
Works with any Python environment (including conda):

```bash
cd Project
python run_dashboard.py
```

This will:
- Check for required files
- Auto-install Streamlit if missing
- Launch the dashboard

---

### ✅ Option 2: Bash Script with Conda
For conda environment users:

```bash
cd Project
./run_dashboard.sh
```

This will:
- Activate conda environment 'env'
- Check/install Streamlit
- Launch the dashboard

---

### ✅ Option 3: Direct Command
If Streamlit is already installed:

```bash
cd Project
streamlit run src/visualization/dashboard.py
```

---

## First-Time Setup

### 1. Install Streamlit (if not installed)

**Using conda:**
```bash
conda activate env
pip install streamlit
```

**Using pip:**
```bash
pip install streamlit
```

**Verify installation:**
```bash
streamlit --version
```

Should show: `Streamlit, version 1.x.x`

---

### 2. Install Additional Dependencies (if missing)

The dashboard requires these packages (likely already installed from Phase 1-4):

```bash
conda activate env
pip install plotly pandas numpy scikit-learn xgboost lightgbm pillow
```

Or using the project requirements:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### ❌ "streamlit: command not found"

**Solution 1 - Activate conda environment:**
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env
streamlit run src/visualization/dashboard.py
```

**Solution 2 - Install Streamlit:**
```bash
pip install streamlit
```

**Solution 3 - Use Python launcher:**
```bash
python run_dashboard.py  # Auto-installs if missing
```

---

### ❌ "Unable to load dataset" in dashboard

**Cause:** Phase 1-2 not completed

**Solution:**
```bash
cd notebooks
jupyter notebook 01_data_cleaning_eda.ipynb    # Run Phase 1
jupyter notebook 02_clustering.ipynb           # Run Phase 2
```

This generates: `data/processed/oulad/oulad_with_clusters.csv`

---

### ❌ "Unable to load models" in dashboard

**Cause:** Phase 3 not completed

**Solution:**
```bash
cd notebooks
jupyter notebook 03_prediction_models.ipynb   # Run Phase 3
```

This generates model files in `models/` directory.

---

### ❌ Import errors (ModuleNotFoundError)

**Cause:** Missing dependencies

**Solution:**
```bash
conda activate env
pip install streamlit plotly pandas numpy scikit-learn xgboost lightgbm pillow
```

---

### ❌ Dashboard shows "Missing Required Data"

Check the sidebar "System Status" section to see which files are missing:

**If "Processed Data" is ❌:**
- Run Phase 1 notebook (`01_data_cleaning_eda.ipynb`)

**If "Clustered Data" is ❌:**
- Run Phase 2 notebook (`02_clustering.ipynb`)

**If "ML Models" is ❌:**
- Run Phase 3 notebook (`03_prediction_models.ipynb`)

---

### ❌ Port already in use

**Error:** `Address already in use`

**Solution 1 - Use different port:**
```bash
streamlit run src/visualization/dashboard.py --server.port 8502
```

**Solution 2 - Stop existing Streamlit:**
```bash
pkill -f streamlit
streamlit run src/visualization/dashboard.py
```

---

## Verification Checklist

Before launching, ensure:

- [ ] Conda environment 'env' exists and is activated
- [ ] Streamlit is installed (`streamlit --version` works)
- [ ] File exists: `data/processed/oulad/oulad_with_clusters.csv`
- [ ] 6 files exist in `models/` directory (*.pkl)
- [ ] Python packages installed: plotly, pandas, numpy, scikit-learn

**Quick check script:**
```bash
cd Project
ls data/processed/oulad/oulad_with_clusters.csv
ls models/*.pkl | wc -l  # Should show 6
python -c "import streamlit, plotly, pandas; print('✅ All imports OK')"
```

---

## Manual Installation Steps

If automated launchers don't work:

### Step 1: Activate environment
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env
```

### Step 2: Install Streamlit
```bash
pip install streamlit
```

### Step 3: Install other dependencies (if needed)
```bash
pip install plotly pandas numpy scikit-learn xgboost lightgbm pillow
```

### Step 4: Launch dashboard
```bash
cd /home/galym/Code/Work/Projects/IN-013/Project
streamlit run src/visualization/dashboard.py
```

---

## Browser Access

Once launched, the dashboard is available at:

- **Local:** http://localhost:8501
- **Network:** http://YOUR_IP:8501 (if accessible from other machines)

**Supported Browsers:**
- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Edge
- ⚠️ Safari (limited support)

---

## Stopping the Dashboard

**Method 1:** Press `Ctrl+C` in the terminal

**Method 2:** Close the terminal window

**Method 3:** Kill process:
```bash
pkill -f streamlit
```

---

## Performance Tips

**First Launch:**
- Takes ~5-10 seconds to load data and models
- Caching improves subsequent loads

**Subsequent Loads:**
- Takes <1 second (cached)
- Much faster navigation

**For Better Performance:**
- Use Chrome browser
- Close unused tabs
- Let first page fully load before navigating

---

## Development Mode

To run with auto-reload on file changes:

```bash
streamlit run src/visualization/dashboard.py --server.runOnSave true
```

To run with debugging:
```bash
streamlit run src/visualization/dashboard.py --logger.level=debug
```

---

## Environment Variables (Optional)

Set custom paths if needed:

```bash
export STREAMLIT_SERVER_PORT=8502
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
streamlit run src/visualization/dashboard.py
```

---

## Getting Help

**Check logs:**
```bash
tail -f ~/.streamlit/logs/streamlit.log
```

**Test imports:**
```bash
python -c "from src.visualization import config, utils; print('OK')"
```

**System status:**
- Check sidebar in dashboard for detailed status
- Green checkmarks = all good
- Red X = missing files

---

## Success Indicators

Dashboard is working correctly if you see:

1. ✅ Dashboard opens in browser
2. ✅ Sidebar shows "All systems operational"
3. ✅ Overview page displays dataset statistics
4. ✅ All 5 pages are accessible
5. ✅ No error messages in terminal

---

**Need More Help?**
- Check: `src/visualization/README.md` (technical docs)
- Check: `DASHBOARD_GUIDE.md` (user guide)
- Check: `CLAUDE.md` (project structure)
