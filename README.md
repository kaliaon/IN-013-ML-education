# OULAD Learning Analytics - Dissertation Project

## Тема: «Үлкен деректер негізінде оқу процесін оңтайлы басқарудың әдістері мен алгоритмдерін әзірлеу»

---

## 📋 Project Overview

This project implements a comprehensive learning analytics system using the **Open University Learning Analytics Dataset (OULAD)** to:
- Predict student performance early in the semester
- Cluster students by behavioral and demographic patterns
- Provide personalized recommendations for intervention

---

## 🗂️ Project Structure

```
Project/
├── data/
│   ├── raw/oulad/              # Original OULAD CSV files (7 tables)
│   ├── processed/              # Processed datasets
│   └── external/               # Additional datasets
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb    ✅ COMPLETED
│   ├── 02_clustering.ipynb           🔜 Next
│   ├── 03_prediction_models.ipynb    📅 Planned
│   └── 04_recommendations.ipynb      📅 Planned
├── external/                   # Reference Kaggle notebooks
├── src/                        # Python modules (future)
├── figures/                    # Generated visualizations
└── README.md                   # This file
```

---

## ✅ Phase 1: Data Cleaning and EDA (COMPLETED)

### Implemented in: `notebooks/01_data_cleaning_eda.ipynb`

**Key Achievements:**
- ✅ Loaded all 7 OULAD tables (~32K students, 10M+ interactions)
- ✅ Fixed semantic errors (assessment weights for modules CCC & GGG)
- ✅ Engineered 50+ features (marks, VLE activity, demographics)
- ✅ Generated 11 publication-ready visualizations
- ✅ Discovered counterintuitive findings (IMD effect, age patterns)
- ✅ Saved processed dataset: `data/processed/oulad_processed.csv`

---

## 🔍 Key Findings from Phase 1

### 1. **🔥 Counterintuitive IMD Finding**
**Students from MORE deprived areas perform BETTER in online learning!**
- Opposite to traditional classroom education
- IMD correlation with mark: positive
- Online learning removes geographic/transportation barriers

### 2. **👴 Age Effect**
Older students perform significantly better:
- Age 55+: Highest marks
- Age 0-35: Lowest marks

### 3. **💻 VLE Engagement**
Strong correlation between clicks and success:
- Pass/Distinction: ~1,500+ avg clicks
- Fail/Withdrawn: ~400-800 avg clicks

### 4. **📚 Most Important VLE Resources**
Top 5: oucontent, homepage, resource, quiz, forumng

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn missingno jupyter scikit-learn xgboost lightgbm
```

### Run Phase 1 Analysis
```bash
cd Project/notebooks
jupyter notebook 01_data_cleaning_eda.ipynb
```

The notebook will:
1. Load OULAD data from `../data/raw/oulad/`
2. Clean and validate (fix assessment weights, handle missing values)
3. Engineer 50+ features
4. Generate 11 visualizations → `../figures/`
5. Save processed data → `../data/processed/oulad_processed.csv`

---

## 📊 Dataset Statistics

```
Total unique students: ~28,000
Number of features: 50+
Modules: 7 (AAA-GGG)
Course presentations: 22

Target Distribution:
  Pass:        43%
  Withdrawn:   29%
  Fail:        18%
  Distinction: 11%
```

---

## 🔜 Next Steps

**Phase 2: Clustering** (Ready to implement)
- K-Means clustering (elbow method, silhouette score)
- DBSCAN clustering
- Profile each cluster
- Reference: `oulad-open-university-learning-analytics-dataset.ipynb`

**Phase 3: Predictive Modeling**
- Decision Tree, Random Forest, XGBoost, LightGBM
- LSTM (optional)
- Model comparison with metrics
- Reference: `oulad-random-forest.ipynb`

**Phase 4: Recommendations**
- Cluster-based recommendations
- Risk-based interventions
- Reference: `oulad-personalized-learning-path-recommender-sys.ipynb`

---

## 📚 Data Source

**OULAD (Open University Learning Analytics Dataset)**
- Source: https://analyse.kmi.open.ac.uk/open_dataset
- Citation: Kuzilek J., et al. (2017) Sci. Data 4:170171
- License: CC BY 4.0

---

## 🛠️ Technologies

- Python 3.8+, pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, lightgbm
- jupyter, missingno

---

## 👥 Author

**Galym** - Dissertation Research
Topic: "Методы и алгоритмы оптимального управления учебным процессом на основе больших данных"

---

**Last Updated**: 2025-10-25
**Version**: Phase 1 Complete
**Status**: ✅ Ready for Phase 2


