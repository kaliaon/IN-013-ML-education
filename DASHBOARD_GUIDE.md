# OULAD Learning Analytics Dashboard - Quick Start Guide

## 🚀 Launch Dashboard

### Option 1: Using Launch Script (Recommended)
```bash
cd Project
./run_dashboard.sh
```

### Option 2: Direct Streamlit Command
```bash
cd Project
streamlit run src/visualization/dashboard.py
```

### Option 3: From Visualization Directory
```bash
cd Project/src/visualization
streamlit run dashboard.py
```

The dashboard will automatically open in your browser at **http://localhost:8501**

## 📋 Prerequisites

Before launching, ensure you have:

1. ✅ **Completed Phases 1-4** - Run all notebooks:
   - `notebooks/01_data_cleaning_eda.ipynb`
   - `notebooks/02_clustering.ipynb`
   - `notebooks/03_prediction_models.ipynb`
   - `notebooks/04_feature_importance_experiments.ipynb`

2. ✅ **Required Files Exist:**
   ```
   data/processed/oulad/oulad_with_clusters.csv
   models/decision_tree_model.pkl
   models/random_forest_model.pkl
   models/xgboost_model.pkl
   models/lightgbm_model.pkl
   models/label_encoder.pkl
   models/feature_names.pkl
   ```

3. ✅ **Dependencies Installed:**
   ```bash
   pip install streamlit plotly pandas numpy scikit-learn xgboost lightgbm pillow
   ```

## 🎯 Dashboard Pages Overview

### 1. 🏠 Overview Page
**What it does:** Provides comprehensive dataset statistics and exploratory visualizations

**Key Features:**
- Student count, features, average metrics
- Outcome distribution (Pass/Fail/Distinction/Withdrawn)
- Demographics breakdown by gender, age, education
- VLE activity patterns
- Assessment performance distributions
- Feature correlation heatmap
- Data quality report

**Use it to:**
- Understand dataset composition
- Identify patterns in student demographics
- Explore engagement and performance distributions
- Export summary statistics

---

### 2. 🎯 Predictions Page
**What it does:** Predict student performance using trained ML models

**3 Modes:**

#### Manual Input Mode
1. Select a model (LightGBM recommended - 88.83% accuracy)
2. Fill in student information:
   - Demographics (gender, age, region, education)
   - VLE activity (total clicks, resource usage)
   - Assessment performance (scores, submission rate)
3. Click "Predict Performance"
4. View predicted outcome, confidence, and recommendations

**Use it to:**
- Identify at-risk students early
- Get intervention recommendations
- Understand prediction confidence

#### Batch Prediction Mode
1. Choose data source: upload CSV or use existing dataset
2. Select sample size (if using existing data)
3. Choose prediction model
4. Click "Run Batch Prediction"
5. View results, filter by outcome/confidence
6. Download predictions

**Use it to:**
- Process entire student cohorts
- Identify high-risk groups
- Export predictions for further analysis

#### What-If Analysis Mode
1. Select baseline student by index
2. Adjust factors:
   - VLE Clicks Multiplier (0-3x)
   - Assessment Score Boost (-30 to +30)
   - Submission Rate Boost (-0.5 to +0.5)
3. Compare baseline vs. modified predictions
4. View probability changes

**Use it to:**
- Test intervention scenarios
- Understand feature impact
- Explore "what would happen if..." questions

---

### 3. 👥 Clustering Page
**What it does:** Analyze student behavioral groups from K-Means and DBSCAN clustering

**Key Features:**
- Cluster size distribution
- PCA and t-SNE visualizations (2D projections)
- Cluster profile comparison
- Feature heatmaps by cluster
- Outcome distribution within each cluster
- Cluster-specific recommendations

**How to use:**
1. Select clustering method (K-Means or DBSCAN)
2. View cluster distribution and sizes
3. Choose dimensionality reduction (PCA or t-SNE)
4. Select a cluster to analyze in detail
5. Compare clusters using feature heatmaps
6. Review intervention recommendations per cluster

**Use it to:**
- Identify distinct student behavioral groups
- Tailor interventions by cluster characteristics
- Understand engagement patterns
- Allocate support resources effectively

---

### 4. 📈 Model Performance Page
**What it does:** Compare ML model performance across multiple metrics

**Key Features:**
- Performance metrics table (Accuracy, F1, Precision, Recall, ROC-AUC)
- Interactive metric comparison charts
- Multi-metric radar chart
- Confusion matrices (all 4 models)
- ROC curves (One-vs-Rest multiclass)
- Model complexity analysis
- Selection recommendations

**How to use:**
1. Review performance overview (best model highlighted)
2. Select metrics to compare interactively
3. Choose models for radar chart comparison
4. Examine confusion matrices and ROC curves
5. Review model complexity trade-offs
6. Read recommendations for production deployment

**Use it to:**
- Choose the best model for your use case
- Understand model trade-offs (accuracy vs. interpretability)
- Justify model selection to stakeholders
- Identify improvement opportunities

---

### 5. ⭐ Feature Importance Page
**What it does:** Understand which features drive predictions

**Key Features:**
- Top feature rankings (adjustable: top 5-50)
- Feature importance by category (Demographics, VLE, Assessments)
- Feature correlation heatmap
- Interactive feature explorer
- Distribution analysis by outcome
- Complete feature importance table with percentiles

**How to use:**
1. Adjust slider to view top N important features
2. Explore importance by category
3. Expand categories to see individual feature scores
4. Review correlation heatmap to identify redundancies
5. Use interactive explorer to dive into specific features
6. Download importance rankings

**Use it to:**
- Identify key predictors of student success
- Prioritize data collection efforts
- Perform feature selection for model optimization
- Communicate insights to educators and administrators

---

## 💡 Common Use Cases

### Use Case 1: Identify At-Risk Students
1. Go to **Predictions > Batch Prediction**
2. Select "Use existing dataset" or upload student CSV
3. Run prediction with LightGBM model
4. Filter by "Fail" or "Withdrawn" outcomes
5. Sort by prediction confidence
6. Download high-risk student list
7. Cross-reference with **Clustering** page to understand behavioral patterns

### Use Case 2: Evaluate Intervention Impact
1. Go to **Predictions > What-If Analysis**
2. Select a student currently predicted to fail
3. Increase VLE Clicks Multiplier to 2.0 (simulating increased engagement)
4. Boost Assessment Score by +10 (simulating tutoring impact)
5. Compare probability changes
6. If outcome improves to "Pass", this indicates potential intervention value

### Use Case 3: Understand Student Segments
1. Go to **Clustering**
2. Select K-Means clustering
3. Review cluster size distribution
4. For each cluster:
   - Check dominant outcome
   - Review avg VLE clicks and assessment scores
   - Read cluster-specific recommendations
5. Use insights to design targeted support programs

### Use Case 4: Communicate Model Performance
1. Go to **Model Performance**
2. Screenshot performance metrics table
3. Save confusion matrices and ROC curves
4. Use radar chart to show multi-metric comparison
5. Reference model complexity analysis
6. Include recommendations in stakeholder presentation

### Use Case 5: Optimize Data Collection
1. Go to **Feature Importance**
2. Review top 20 important features
3. Note which categories dominate (e.g., Assessments)
4. Check "Feature Importance by Category"
5. Prioritize collecting/monitoring top features
6. Consider removing low-importance features (<10th percentile)

---

## 🔍 Tips & Best Practices

### Performance Tips
- **First load is slow** - Subsequent loads are cached and much faster
- **Use LightGBM** for fastest predictions (XGBoost also fast)
- **Limit t-SNE samples** if dataset is very large (>50K students)
- **Download results** instead of viewing all in browser for large batches

### Interpretation Tips
- **High confidence (>80%)** predictions are more reliable
- **Withdrawn outcomes** may indicate early dropout - intervene ASAP
- **"Distinction" predictions** with low confidence suggest borderline high performers
- **Check multiple models** if predictions disagree significantly

### Data Quality
- **Missing values** are automatically imputed with median/zero
- **Extreme values** may indicate data errors - check Overview page distributions
- **Correlation heatmap** helps identify redundant features

### Export & Share
- Most pages have **Download buttons** for CSV exports
- **Screenshot visualizations** for reports (high-quality PNGs from notebooks also available)
- **Share filtered results** from batch predictions with instructors

---

## 🐛 Troubleshooting

### Dashboard won't start
```bash
# Check Streamlit is installed
streamlit --version

# Reinstall if needed
pip install streamlit

# Try with explicit Python
python -m streamlit run src/visualization/dashboard.py
```

### "Unable to load dataset" error
- Ensure `data/processed/oulad/oulad_with_clusters.csv` exists
- Run Phase 1-2 notebooks to generate data
- Check file permissions

### "Unable to load models" error
- Ensure all `.pkl` files exist in `models/` directory
- Run Phase 3 notebook to train models
- Check for corrupted pickle files

### Slow performance
- First load takes 30-60 seconds (normal)
- Clear browser cache if visualizations don't render
- Reduce sample sizes for batch predictions
- Use PCA instead of t-SNE for faster clustering viz

### Charts not displaying
- Use Chrome or Firefox (best Plotly support)
- Check JavaScript is enabled
- Try refreshing page (Ctrl+F5)
- Check browser console for errors

---

## 📊 System Status Indicators

Check the sidebar for system status:
- ✅ **Green checkmarks** = All systems operational
- ❌ **Red X** = Missing required files
- ⚠️ **Yellow warning** = Some features unavailable

---

## 🔐 Data Privacy Note

The dashboard displays aggregated statistics and predictions. Individual student IDs are included in exports but not prominently displayed. For production use:
- Ensure data is anonymized
- Implement access controls
- Use HTTPS for deployment
- Follow institutional data policies

---

## 📚 Additional Resources

- **Full Documentation:** `src/visualization/README.md`
- **Project Overview:** `CLAUDE.md`
- **Technical Spec:** `spec.txt`
- **Source Code:** `src/visualization/`
- **Notebooks:** `notebooks/` (Phases 1-4)

---

## 🎓 For Dissertation Defense

**Key Points to Highlight:**

1. **Comprehensive Analytics** - 5 integrated pages covering all aspects
2. **Production-Quality** - Modular architecture, error handling, caching
3. **Interactive Exploration** - What-if analysis, batch prediction, filtering
4. **Visual Excellence** - Plotly charts, heatmaps, PCA/t-SNE projections
5. **Actionable Insights** - Recommendations for each cluster and prediction
6. **Export Capabilities** - Download all results for further analysis

**Demo Flow for Defense:**
1. Start with Overview to show dataset understanding
2. Use Predictions to demonstrate practical application
3. Show Clustering to prove behavioral pattern discovery
4. Display Model Performance to justify algorithm selection
5. Explain Feature Importance for interpretability

---

**Version:** 1.0.0
**Last Updated:** 2025-01-06
**Phase:** 5 - Complete ✅
