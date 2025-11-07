# OULAD Learning Analytics Dashboard

Interactive Streamlit dashboard for exploring student performance predictions, clustering analysis, and model insights from the Open University Learning Analytics Dataset (OULAD).

## 📋 Overview

This dashboard provides comprehensive visualization and interaction with machine learning models trained on OULAD data. It enables educators and researchers to:

- Explore dataset statistics and distributions
- Predict individual student performance
- Analyze behavioral clustering patterns
- Compare model performance metrics
- Understand feature importance and SHAP values

## 🚀 Quick Start

### Prerequisites

Ensure you have completed **Phases 1-4** of the project:
- ✅ Phase 1: Data Cleaning and EDA
- ✅ Phase 2: Clustering Models
- ✅ Phase 3: Prediction Models
- ✅ Phase 4: Feature Importance Experiments

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r ../../requirements.txt
   ```

2. **Verify data files exist:**
   ```bash
   # Required data files
   ls ../../data/processed/oulad/oulad_with_clusters.csv

   # Required model files
   ls ../../models/*.pkl
   ```

### Running the Dashboard

From the `Project` directory:

```bash
cd Project
streamlit run src/visualization/dashboard.py
```

Or from the visualization directory:

```bash
cd src/visualization
streamlit run dashboard.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`.

## 📊 Dashboard Pages

### 1. 🏠 Overview
**Purpose:** Dataset exploration and summary statistics

**Features:**
- Total student count and feature summary
- Student outcome distribution (Pass, Fail, Distinction, Withdrawn)
- Demographics analysis (gender, age, education)
- VLE activity patterns
- Assessment performance distributions
- Feature correlation heatmap
- Data quality report
- Export capabilities

**Use Cases:**
- Initial data exploration
- Understanding dataset composition
- Identifying data quality issues
- Exporting summary statistics

---

### 2. 🎯 Predictions
**Purpose:** Interactive student performance prediction

**Features:**

**Manual Input Tab:**
- Select prediction model (LightGBM, XGBoost, Random Forest, Decision Tree)
- Input student demographics, VLE activity, and assessment scores
- Get predicted outcome with confidence scores
- View probability distribution across all outcomes
- Receive intervention recommendations

**Batch Prediction Tab:**
- Upload CSV file or use existing data
- Predict multiple students simultaneously
- Filter results by outcome and confidence
- View prediction distribution
- Export results

**What-If Analysis Tab:**
- Select baseline student
- Adjust engagement, assessment, and submission factors
- Compare baseline vs. modified predictions
- Visualize probability changes
- Understand feature impact

**Use Cases:**
- Identifying at-risk students
- Testing intervention strategies
- Exploring "what-if" scenarios
- Batch processing for entire cohorts

---

### 3. 👥 Clustering
**Purpose:** Student behavioral group analysis

**Features:**
- K-Means and DBSCAN clustering results
- Cluster size distribution
- PCA and t-SNE visualizations
- Cluster profile comparison
- Feature heatmaps by cluster
- Outcome distribution within clusters
- Cluster-specific recommendations

**Use Cases:**
- Identifying student behavioral groups
- Tailoring interventions by cluster
- Understanding engagement patterns
- Resource allocation planning

---

### 4. 📈 Model Performance
**Purpose:** ML model comparison and evaluation

**Features:**
- Performance metrics table (Accuracy, F1, Precision, Recall, ROC-AUC)
- Interactive metric comparison charts
- Radar chart for multi-metric comparison
- Confusion matrices
- ROC curves (One-vs-Rest)
- Model complexity analysis
- Selection recommendations

**Use Cases:**
- Choosing the best model for deployment
- Understanding model trade-offs
- Justifying model selection to stakeholders
- Identifying areas for improvement

---

### 5. ⭐ Feature Importance
**Purpose:** Understanding predictive features

**Features:**
- Top feature rankings
- Feature importance by category (Demographics, VLE, Assessments)
- Feature correlation analysis
- Interactive feature explorer
- Distribution by outcome
- Complete feature importance table with percentiles
- Export capabilities

**Use Cases:**
- Identifying key success indicators
- Prioritizing data collection
- Feature selection for model optimization
- Communicating insights to educators

## 🏗️ Architecture

### Directory Structure

```
src/visualization/
├── dashboard.py           # Main Streamlit app
├── config.py             # Configuration settings
├── utils.py              # Utility functions
├── README.md             # This file
├── pages/
│   ├── __init__.py
│   ├── overview.py       # Overview page
│   ├── predictions.py    # Predictions page
│   ├── clustering.py     # Clustering page
│   ├── performance.py    # Model performance page
│   └── importance.py     # Feature importance page
```

### Design Patterns

**1. Modular Architecture**
- Each page is a separate module
- Shared utilities in `utils.py`
- Centralized configuration in `config.py`

**2. Caching Strategy**
- `@st.cache_data` for data loading (TTL: 1 hour)
- `@st.cache_resource` for model loading (persistent)
- Improves performance and reduces loading times

**3. Error Handling**
- Graceful degradation on missing files
- User-friendly error messages
- System status indicators in sidebar

**4. Responsive Design**
- Column layouts for optimal space usage
- Expanders for detailed information
- Mobile-friendly (within Streamlit constraints)

## 🔧 Configuration

### Customization

Edit `config.py` to customize:

**Paths:**
```python
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
```

**Colors:**
```python
COLORS = {
    "primary": "#3498db",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
}
```

**Feature Categories:**
```python
FEATURE_CATEGORIES = {
    "Demographics": [...],
    "VLE Activity": [...],
    "Assessment Performance": [...],
}
```

## 📈 Performance Optimization

### Caching
The dashboard uses aggressive caching to minimize loading times:
- Dataset loaded once per hour
- Models loaded once per session
- Visualization data cached

### Memory Management
For large datasets (>100K students):
- Consider sampling for visualizations
- Use pagination for data tables
- Implement lazy loading for figures

### Speed Tips
- Use LightGBM for fastest predictions
- Limit t-SNE perplexity for faster computation
- Pre-compute and cache expensive operations

## 🐛 Troubleshooting

### Common Issues

**1. "Unable to load dataset"**
- **Cause:** Data files not found
- **Solution:** Run Phases 1-2 notebooks to generate `oulad_with_clusters.csv`
- **Location:** `data/processed/oulad/`

**2. "Unable to load models"**
- **Cause:** Model files not found
- **Solution:** Run Phase 3 notebook to train and save models
- **Location:** `models/`

**3. "Module not found"**
- **Cause:** Missing dependencies
- **Solution:** `pip install -r requirements.txt`

**4. Slow loading times**
- **Cause:** First load or cache expired
- **Solution:** Wait for initial load; subsequent loads will be faster

**5. Plotly charts not displaying**
- **Cause:** Browser compatibility
- **Solution:** Use Chrome/Firefox; clear browser cache

### Debug Mode

Enable Streamlit debug mode:
```bash
streamlit run dashboard.py --logger.level=debug
```

## 📊 Data Requirements

### Required Files

**Data:**
- `data/processed/oulad/oulad_with_clusters.csv` (32K+ rows, ~50 columns)

**Models:**
- `models/decision_tree_model.pkl`
- `models/random_forest_model.pkl`
- `models/xgboost_model.pkl`
- `models/lightgbm_model.pkl`
- `models/label_encoder.pkl`
- `models/feature_names.pkl`

**Optional (for visualizations):**
- `models/model_comparison_results.csv`
- `models/feature_importance.csv`
- `figures/*.png` (17 figures from Phases 1-4)

### Data Format

The dashboard expects datasets with these columns:
- Demographics: `gender`, `age_band`, `region`, `highest_education`, `disability`, etc.
- VLE Activity: `total_clicks`, `clicks_*` (by resource type)
- Assessments: `avg_assessment_score`, `assessment_submission_rate`, etc.
- Clustering: `kmeans_cluster`, `dbscan_cluster`
- Target: `final_result` (Pass, Fail, Distinction, Withdrawn)

## 🔐 Security Considerations

**Data Privacy:**
- Student IDs are not displayed in public views
- Ensure data files contain anonymized data only
- Do not deploy publicly without proper access controls

**Deployment:**
- Use Streamlit authentication for production
- Implement role-based access control (RBAC)
- Use HTTPS for secure communication
- Set appropriate CORS policies

## 🚢 Deployment

### Local Deployment
```bash
streamlit run dashboard.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect Streamlit Cloud to repository
3. Configure secrets and environment variables
4. Deploy

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "src/visualization/dashboard.py"]
```

### Production Considerations
- Use environment variables for paths
- Implement logging and monitoring
- Set up error tracking (e.g., Sentry)
- Configure resource limits
- Enable authentication

## 📚 API Reference

### Key Functions

**utils.py:**
```python
load_dataset(dataset_type="clustered") -> pd.DataFrame
load_model(model_name: str) -> Any
load_all_models() -> Dict[str, Any]
prepare_features_for_prediction(input_data, feature_names, df_reference) -> np.ndarray
```

**Page rendering:**
```python
overview.render()      # Render overview page
predictions.render()   # Render predictions page
clustering.render()    # Render clustering page
performance.render()   # Render performance page
importance.render()    # Render importance page
```

## 🤝 Contributing

To add new features:

1. **New Page:**
   - Create `pages/newpage.py`
   - Implement `render()` function
   - Add to `config.PAGES`
   - Import in `dashboard.py`

2. **New Visualization:**
   - Use Plotly for interactivity
   - Follow existing color scheme
   - Add tooltips and labels
   - Make responsive

3. **New Utility:**
   - Add to `utils.py`
   - Use caching when appropriate
   - Include error handling
   - Add docstrings

## 📖 References

- **OULAD Dataset:** https://analyse.kmi.open.ac.uk/open_dataset
- **Streamlit Docs:** https://docs.streamlit.io
- **Plotly Docs:** https://plotly.com/python/
- **Dissertation Spec:** `../../spec.txt`

## 📧 Support

For issues or questions:
- Check CLAUDE.md for project structure
- Review spec.txt for requirements
- Consult Phase 1-4 notebooks for data pipeline

## 📄 License

This project is part of a dissertation on learning analytics using the OULAD dataset (CC BY 4.0).

---

**Version:** 1.0.0
**Last Updated:** 2025-01-06
**Phase:** 5 - Visualizations and Dashboard
**Status:** ✅ Complete
