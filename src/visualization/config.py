"""
Dashboard Configuration
Centralized configuration for the Streamlit dashboard.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
PROCESSED_DIR = DATA_DIR / "processed" / "oulad"

# Data files
DATA_FILES = {
    "processed": PROCESSED_DIR / "oulad_processed.csv",
    "clustered": PROCESSED_DIR / "oulad_with_clusters.csv",
}

# Model files
MODEL_FILES = {
    "decision_tree": MODELS_DIR / "decision_tree_model.pkl",
    "random_forest": MODELS_DIR / "random_forest_model.pkl",
    "xgboost": MODELS_DIR / "xgboost_model.pkl",
    "lightgbm": MODELS_DIR / "lightgbm_model.pkl",
    "label_encoder": MODELS_DIR / "label_encoder.pkl",
    "feature_names": MODELS_DIR / "feature_names.pkl",
}

# Dashboard configuration
PAGE_CONFIG = {
    "page_title": "OULAD Learning Analytics Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Color schemes
COLORS = {
    "primary": "#3498db",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#9b59b6",
    "distinction": "#2ecc71",
    "pass": "#3498db",
    "fail": "#e74c3c",
    "withdrawn": "#f39c12",
}

# Class labels and colors
CLASS_COLORS = {
    "Distinction": COLORS["distinction"],
    "Pass": COLORS["pass"],
    "Fail": COLORS["danger"],
    "Withdrawn": COLORS["warning"],
}

# Feature categories for organized display
FEATURE_CATEGORIES = {
    "Demographics": [
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
        "num_of_prev_attempts",
        "studied_credits",
    ],
    "VLE Activity": [
        "total_clicks",
        "num_unique_activities",
        "clicks_per_activity",
        "clicks_homepage",
        "clicks_oucontent",
        "clicks_resource",
        "clicks_forumng",
        "clicks_quiz",
        "clicks_subpage",
        "clicks_url",
    ],
    "Assessment Performance": [
        "avg_assessment_score",
        "num_assessments_submitted",
        "total_assessments",
        "assessment_submission_rate",
        "avg_score_TMA",
        "avg_score_CMA",
        "avg_score_Exam",
    ],
    "Registration": [
        "date_unregistration",
        "did_unregister",
    ],
    "Clustering": [
        "kmeans_cluster",
        "dbscan_cluster",
    ],
}

# Categorical features and their possible values
CATEGORICAL_FEATURES = {
    "gender": ["M", "F"],
    "region": [
        "East Anglian Region",
        "East Midlands Region",
        "Ireland",
        "London Region",
        "North Region",
        "North Western Region",
        "Scotland",
        "South East Region",
        "South Region",
        "South West Region",
        "Wales",
        "West Midlands Region",
        "Yorkshire Region",
    ],
    "highest_education": [
        "A Level or Equivalent",
        "HE Qualification",
        "Lower Than A Level",
        "No Formal quals",
        "Post Graduate Qualification",
    ],
    "imd_band": [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ],
    "age_band": ["0-35", "35-55", "55<="],
    "disability": ["Y", "N"],
}

# Tooltips and descriptions
FEATURE_DESCRIPTIONS = {
    "total_clicks": "Total number of clicks on VLE resources",
    "num_unique_activities": "Number of unique activity types accessed",
    "clicks_per_activity": "Average clicks per unique activity",
    "avg_assessment_score": "Average score across all assessments",
    "assessment_submission_rate": "Percentage of assessments submitted",
    "avg_score_TMA": "Average score on Tutor Marked Assessments",
    "avg_score_CMA": "Average score on Computer Marked Assessments",
    "avg_score_Exam": "Score on final exam",
    "kmeans_cluster": "K-Means cluster assignment",
    "dbscan_cluster": "DBSCAN cluster assignment (-1 = noise)",
}

# Dashboard pages
PAGES = {
    "🏠 Overview": "overview",
    "🎯 Predictions": "predictions",
    "👥 Clustering": "clustering",
    "📈 Model Performance": "performance",
    "⭐ Feature Importance": "importance",
}

# Chart settings
CHART_HEIGHT = 500
CHART_WIDTH = 800

# Caching settings
CACHE_TTL = 3600  # 1 hour in seconds
