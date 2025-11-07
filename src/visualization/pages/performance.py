"""
Model Performance Page
Comprehensive model comparison and evaluation metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

from visualization.utils import (
    load_model_comparison_results,
    load_all_models,
    load_label_encoder,
    load_dataset,
    format_percentage,
    format_number,
    get_figure_path
)
from visualization.config import CLASS_COLORS, FIGURES_DIR


def render():
    """Render the model performance page."""
    st.title("📈 Model Performance Comparison")
    st.markdown("---")

    # Load comparison results
    with st.spinner("Loading model performance data..."):
        results_df = load_model_comparison_results()
        models = load_all_models()

    if results_df.empty:
        st.error("❌ Unable to load model comparison results.")
        return

    # Performance overview
    st.header("🏆 Performance Overview")

    # Display best model prominently
    best_model = results_df.iloc[0]  # Assuming sorted by accuracy

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.success("**🥇 Best Model**")
        st.metric(
            label=best_model["Model"],
            value=format_percentage(best_model["Accuracy"]),
            delta="Highest Accuracy",
            help="Model with the best overall accuracy"
        )

    with col2:
        st.metric(
            label="F1 Score",
            value=format_percentage(best_model["F1 Score"])
        )

    with col3:
        st.metric(
            label="ROC-AUC",
            value=format_percentage(best_model["ROC-AUC"])
        )

    st.markdown("---")

    # Model comparison table
    st.header("📊 Performance Metrics Comparison")

    # Format the dataframe for display
    display_df = results_df.copy()
    for col in ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Highlight best performers
    col1, col2, col3 = st.columns(3)

    with col1:
        best_f1 = results_df.loc[results_df["F1 Score"].idxmax()]
        st.info(f"**Best F1:** {best_f1['Model']} ({format_percentage(best_f1['F1 Score'])})")

    with col2:
        best_precision = results_df.loc[results_df["Precision"].idxmax()]
        st.info(f"**Best Precision:** {best_precision['Model']} ({format_percentage(best_precision['Precision'])})")

    with col3:
        best_recall = results_df.loc[results_df["Recall"].idxmax()]
        st.info(f"**Best Recall:** {best_recall['Model']} ({format_percentage(best_recall['Recall'])})")

    st.markdown("---")

    # Interactive metric comparison
    st.header("📈 Interactive Metric Comparison")

    # Metric selector
    selected_metrics = st.multiselect(
        "Select metrics to compare:",
        options=["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"],
        default=["Accuracy", "F1 Score", "ROC-AUC"]
    )

    if selected_metrics:
        # Create subplots
        n_metrics = len(selected_metrics)
        cols_per_row = 2
        n_rows = (n_metrics + cols_per_row - 1) // cols_per_row

        fig = make_subplots(
            rows=n_rows,
            cols=cols_per_row,
            subplot_titles=selected_metrics,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        colors = px.colors.qualitative.Set2

        for idx, metric in enumerate(selected_metrics):
            row = idx // cols_per_row + 1
            col = idx % cols_per_row + 1

            data = results_df.sort_values(metric, ascending=False)

            fig.add_trace(
                go.Bar(
                    x=data["Model"],
                    y=data[metric] * 100,
                    name=metric,
                    marker_color=colors[idx % len(colors)],
                    text=[f"{v*100:.2f}%" for v in data[metric]],
                    textposition='outside',
                    showlegend=False
                ),
                row=row,
                col=col
            )

            fig.update_yaxes(title_text="Score (%)", range=[0, 105], row=row, col=col)
            fig.update_xaxes(tickangle=-45, row=row, col=col)

        fig.update_layout(height=300 * n_rows, title_text="Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Radar chart comparison
    st.header("🎯 Multi-Metric Radar Chart")

    # Select models to compare
    selected_models = st.multiselect(
        "Select models to compare:",
        options=results_df["Model"].tolist(),
        default=results_df["Model"].tolist()[:3]
    )

    if selected_models:
        fig = go.Figure()

        metrics = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC"]

        for model_name in selected_models:
            model_data = results_df[results_df["Model"] == model_name].iloc[0]
            values = [model_data[m] * 100 for m in metrics]
            values.append(values[0])  # Close the polygon

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name,
                opacity=0.6
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            title="Multi-Metric Performance Comparison",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Confusion matrices
    st.header("🔍 Confusion Matrices")

    st.markdown("Confusion matrices show how well each model classifies different student outcomes.")

    # Display confusion matrix image if available
    confusion_fig_path = get_figure_path("14_confusion_matrices.png")

    if confusion_fig_path.exists():
        try:
            image = Image.open(confusion_fig_path)
            st.image(image, caption="Confusion Matrices for All Models", use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load confusion matrix image: {e}")
    else:
        st.info("Confusion matrix visualization not found. Run Phase 3 notebook to generate.")

    st.markdown("---")

    # ROC Curves
    st.header("📉 ROC Curves")

    st.markdown("ROC curves illustrate the trade-off between true positive rate and false positive rate.")

    roc_fig_path = get_figure_path("15_roc_curves.png")

    if roc_fig_path.exists():
        try:
            image = Image.open(roc_fig_path)
            st.image(image, caption="ROC Curves (One-vs-Rest)", use_column_width=True)
        except Exception as e:
            st.warning(f"Could not load ROC curve image: {e}")
    else:
        st.info("ROC curve visualization not found. Run Phase 3 notebook to generate.")

    st.markdown("---")

    # Model complexity comparison
    st.header("⚙️ Model Complexity Analysis")

    st.markdown("""
    Different models have different computational requirements and interpretability characteristics.
    """)

    complexity_data = {
        "Model": ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
        "Training Speed": ["Fast", "Moderate", "Slow", "Fast"],
        "Prediction Speed": ["Very Fast", "Fast", "Moderate", "Very Fast"],
        "Interpretability": ["High", "Medium", "Low", "Low"],
        "Memory Usage": ["Low", "High", "High", "Medium"],
        "Overfitting Risk": ["High", "Low", "Low", "Low"]
    }

    complexity_df = pd.DataFrame(complexity_data)

    # Add performance data
    if not results_df.empty:
        for _, row in results_df.iterrows():
            model_name = row["Model"]
            if model_name in complexity_df["Model"].values:
                idx = complexity_df[complexity_df["Model"] == model_name].index[0]
                complexity_df.loc[idx, "Accuracy"] = f"{row['Accuracy']*100:.2f}%"

    st.dataframe(complexity_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Model recommendations
    st.header("💡 Model Selection Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Best for Production")
        st.success("**LightGBM** or **XGBoost**")
        st.markdown("""
        **Reasons:**
        - Highest accuracy and F1 scores
        - Good balance of speed and performance
        - Handle large datasets efficiently
        - Robust to overfitting
        - Strong ROC-AUC scores (>0.97)
        """)

    with col2:
        st.subheader("🔍 Best for Interpretability")
        st.info("**Decision Tree** or **Random Forest**")
        st.markdown("""
        **Reasons:**
        - Easy to visualize decision paths
        - Feature importance is intuitive
        - Can explain individual predictions
        - Good for stakeholder communication
        - Acceptable accuracy trade-off
        """)

    st.markdown("---")

    # Performance insights
    st.header("📊 Key Performance Insights")

    with st.expander("🎯 Overall Findings", expanded=True):
        st.markdown(f"""
        1. **Best Overall Model:** {best_model['Model']} achieved {format_percentage(best_model['Accuracy'])} accuracy

        2. **High Performance Across Board:** All models achieved >87% accuracy, indicating:
           - Well-engineered features
           - Quality data preprocessing
           - Appropriate model selection

        3. **ROC-AUC Excellence:** All models achieved >0.96 ROC-AUC, showing:
           - Strong discriminative power
           - Good probability calibration
           - Reliable across all classes

        4. **Ensemble Methods Win:** Random Forest, XGBoost, and LightGBM outperform single Decision Tree:
           - Better generalization
           - Reduced overfitting
           - More robust predictions
        """)

    with st.expander("⚠️ Challenges & Limitations"):
        st.markdown("""
        1. **Class Imbalance:**
           - "Distinction" class is underrepresented (~9%)
           - May affect precision/recall for this class
           - Consider SMOTE or class weights for improvement

        2. **Feature Complexity:**
           - 67 features after encoding
           - Some features may be redundant
           - Feature selection could improve efficiency

        3. **Interpretability Trade-off:**
           - Best models (XGBoost/LightGBM) are less interpretable
           - Important for educational stakeholder buy-in
           - SHAP values can help bridge this gap
        """)

    with st.expander("🚀 Recommendations for Improvement"):
        st.markdown("""
        1. **Hyperparameter Tuning:**
           - Use GridSearchCV or Bayesian optimization
           - Could gain 1-2% accuracy improvement

        2. **Ensemble Stacking:**
           - Combine predictions from multiple models
           - Potential for marginal gains

        3. **Deep Learning:**
           - Try LSTM for temporal patterns
           - Transformer models for sequence modeling
           - May capture complex interactions

        4. **Early Prediction:**
           - Train models on partial semester data
           - Enable intervention before course completion
           - Evaluate accuracy vs. timing trade-off

        5. **Real-time Updates:**
           - Implement online learning
           - Update predictions as new data arrives
           - Adapt to changing student behavior
        """)

    st.markdown("---")

    # Export section
    st.header("📥 Export Performance Data")

    col1, col2 = st.columns(2)

    with col1:
        if not results_df.empty:
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Performance Metrics (CSV)",
                data=csv,
                file_name="model_performance_comparison.csv",
                mime="text/csv"
            )

    with col2:
        if not complexity_df.empty:
            csv = complexity_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Complexity Analysis (CSV)",
                data=csv,
                file_name="model_complexity_analysis.csv",
                mime="text/csv"
            )
