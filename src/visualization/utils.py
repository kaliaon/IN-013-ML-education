"""
Utility Functions for Dashboard
Data loading, caching, and helper functions.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from visualization.config import MODEL_FILES, DATA_FILES, CACHE_TTL


@st.cache_data(ttl=CACHE_TTL)
def load_dataset(dataset_type: str = "clustered") -> pd.DataFrame:
    """
    Load processed dataset with caching.

    Args:
        dataset_type: Type of dataset ('processed' or 'clustered')

    Returns:
        Loaded DataFrame
    """
    try:
        file_path = DATA_FILES[dataset_type]
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        df = pd.read_csv(file_path, low_memory=False)

        # Convert object columns to appropriate types to avoid PyArrow issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass

        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_model(model_name: str) -> Any:
    """
    Load trained model with caching.

    Args:
        model_name: Name of model to load

    Returns:
        Loaded model object
    """
    try:
        file_path = MODEL_FILES[model_name]
        if not file_path.exists():
            raise FileNotFoundError(f"Model not found: {file_path}")

        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None


@st.cache_resource
def load_all_models() -> Dict[str, Any]:
    """
    Load all trained models.

    Returns:
        Dictionary of model name to model object
    """
    models = {}
    model_names = ["decision_tree", "random_forest", "xgboost", "lightgbm"]

    for name in model_names:
        model = load_model(name)
        if model is not None:
            models[name] = model

    return models


@st.cache_resource
def load_label_encoder() -> LabelEncoder:
    """
    Load label encoder for target variable.

    Returns:
        Fitted LabelEncoder
    """
    return load_model("label_encoder")


@st.cache_resource
def load_feature_names() -> list:
    """
    Load feature names used for model training.

    Returns:
        List of feature names
    """
    return load_model("feature_names")


@st.cache_data(ttl=CACHE_TTL)
def load_model_comparison_results() -> pd.DataFrame:
    """
    Load model comparison results from CSV.

    Returns:
        DataFrame with model comparison metrics
    """
    try:
        file_path = MODEL_FILES["decision_tree"].parent / "model_comparison_results.csv"
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading model comparison results: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL)
def load_feature_importance() -> pd.Series:
    """
    Load feature importance rankings.

    Returns:
        Series with feature importance scores
    """
    try:
        file_path = MODEL_FILES["decision_tree"].parent / "feature_importance.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0)
            return df.iloc[:, 0]  # First column contains importance scores
        else:
            return pd.Series()
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")
        return pd.Series()


def prepare_features_for_prediction(
    input_data: Dict[str, Any],
    feature_names: list,
    df_reference: pd.DataFrame
) -> np.ndarray:
    """
    Prepare input features for model prediction.

    Args:
        input_data: Dictionary of feature values
        feature_names: Expected feature names from training
        df_reference: Reference DataFrame for encoding

    Returns:
        Numpy array of features ready for prediction
    """
    # Create a DataFrame with input data
    input_df = pd.DataFrame([input_data])

    # Get categorical columns that need encoding
    categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()

    # One-hot encode categorical features (same as training)
    if categorical_cols:
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    else:
        input_encoded = input_df.copy()

    # Ensure all expected features are present
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Select only the features used in training (in correct order)
    input_encoded = input_encoded[feature_names]

    # Fill any NaN values with median from reference data
    for col in input_encoded.columns:
        if input_encoded[col].isnull().values.any():
            if col in df_reference.columns:
                # Calculate median and ensure it's a scalar
                try:
                    median_val = float(df_reference[col].median())
                except:
                    median_val = 0.0
                input_encoded[col] = input_encoded[col].fillna(median_val)
            else:
                input_encoded[col] = input_encoded[col].fillna(0)

    return input_encoded.values


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage string.

    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with thousands separator.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    return f"{value:,.{decimals}f}"


def get_risk_level(prediction: str) -> Tuple[str, str]:
    """
    Get risk level and color based on prediction.

    Args:
        prediction: Predicted class label

    Returns:
        Tuple of (risk_level, color)
    """
    risk_map = {
        "Distinction": ("Low Risk", "success"),
        "Pass": ("Low Risk", "success"),
        "Fail": ("High Risk", "error"),
        "Withdrawn": ("High Risk", "warning"),
    }
    return risk_map.get(prediction, ("Unknown", "info"))


def compute_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for dataset.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary of statistics
    """
    stats = {
        "n_students": len(df),
        "n_features": len(df.columns),
        "target_distribution": df["final_result"].value_counts().to_dict() if "final_result" in df.columns else {},
        "avg_clicks": df["total_clicks"].mean() if "total_clicks" in df.columns else 0,
        "avg_assessment_score": df["avg_assessment_score"].mean() if "avg_assessment_score" in df.columns else 0,
        "submission_rate": df["assessment_submission_rate"].mean() if "assessment_submission_rate" in df.columns else 0,
        "n_clusters_kmeans": df["kmeans_cluster"].nunique() if "kmeans_cluster" in df.columns else 0,
        "n_clusters_dbscan": len(df[df["dbscan_cluster"] != -1]["dbscan_cluster"].unique()) if "dbscan_cluster" in df.columns else 0,
    }
    return stats


def get_cluster_profile(df: pd.DataFrame, cluster_col: str, cluster_id: int) -> pd.Series:
    """
    Get profile statistics for a specific cluster.

    Args:
        df: DataFrame with cluster assignments
        cluster_col: Name of cluster column
        cluster_id: Cluster ID to profile

    Returns:
        Series with cluster statistics
    """
    cluster_data = df[df[cluster_col] == cluster_id]

    numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
    profile = cluster_data[numeric_cols].mean()
    profile["cluster_size"] = len(cluster_data)
    profile["cluster_percentage"] = len(cluster_data) / len(df) * 100

    return profile


def display_metric_card(label: str, value: Any, delta: Optional[Any] = None, help_text: Optional[str] = None):
    """
    Display a metric card with consistent styling.

    Args:
        label: Metric label
        value: Metric value
        delta: Change/comparison value
        help_text: Tooltip text
    """
    st.metric(label=label, value=value, delta=delta, help=help_text)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division fails

    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default


def get_figure_path(figure_name: str) -> Path:
    """
    Get path to saved figure.

    Args:
        figure_name: Name of figure file

    Returns:
        Path to figure
    """
    from visualization.config import FIGURES_DIR
    return FIGURES_DIR / figure_name


def check_data_availability() -> Dict[str, bool]:
    """
    Check availability of required data files.

    Returns:
        Dictionary indicating which files are available
    """
    availability = {
        "processed_data": DATA_FILES["processed"].exists(),
        "clustered_data": DATA_FILES["clustered"].exists(),
        "models": all(MODEL_FILES[m].exists() for m in ["decision_tree", "random_forest", "xgboost", "lightgbm"]),
        "label_encoder": MODEL_FILES["label_encoder"].exists(),
        "feature_names": MODEL_FILES["feature_names"].exists(),
    }
    return availability
