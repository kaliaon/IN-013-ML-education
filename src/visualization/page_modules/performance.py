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
from visualization.i18n import t


def render():
    """Render the model performance page."""
    st.title(f"📈 {t('performance.title')}")
    st.markdown("---")

    # Load comparison results
    with st.spinner(t('performance.loading')):
        results_df = load_model_comparison_results()
        models = load_all_models()

    if results_df.empty:
        st.error(t('performance.unable_to_load_results'))
        return

    # Performance overview
    st.header(t('performance.overview_title'))

    # Display best model prominently
    best_model = results_df.iloc[0]  # Assuming sorted by accuracy

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.success(t('performance.best_model_label'))
        st.metric(
            label=best_model["Model"],
            value=format_percentage(best_model["Accuracy"]),
            delta=t('performance.highest_accuracy'),
            help=t('performance.best_model_help')
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
    st.header(f"📊 {t('performance.metrics_comparison')}")

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
        st.info(t('performance.best_f1').format(model=best_f1['Model'], score=format_percentage(best_f1['F1 Score'])))

    with col2:
        best_precision = results_df.loc[results_df["Precision"].idxmax()]
        st.info(t('performance.best_precision').format(model=best_precision['Model'], score=format_percentage(best_precision['Precision'])))

    with col3:
        best_recall = results_df.loc[results_df["Recall"].idxmax()]
        st.info(t('performance.best_recall').format(model=best_recall['Model'], score=format_percentage(best_recall['Recall'])))

    st.markdown("---")

    # Interactive metric comparison
    st.header(t('performance.interactive_title'))

    # Metric selector
    selected_metrics = st.multiselect(
        t('performance.select_metrics'),
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

            fig.update_yaxes(title_text=t('performance.score_percent'), range=[0, 105], row=row, col=col)
            fig.update_xaxes(tickangle=-45, row=row, col=col)

        fig.update_layout(height=300 * n_rows, title_text=t('performance.comparison_title'))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Radar chart comparison
    st.header(t('performance.radar_chart_title'))

    # Select models to compare
    selected_models = st.multiselect(
        t('performance.select_models'),
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
            title=t('performance.radar_title'),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Confusion matrices
    st.header(t('performance.confusion_title'))

    st.markdown(t('performance.confusion_desc'))

    # Display confusion matrix image if available
    confusion_fig_path = get_figure_path("14_confusion_matrices.png")

    if confusion_fig_path.exists():
        try:
            image = Image.open(confusion_fig_path)
            st.image(image, caption=t('performance.confusion_caption'), use_column_width=True)
        except Exception as e:
            st.warning(t('performance.confusion_load_error').format(e=e))
    else:
        st.info(t('performance.confusion_not_found'))

    st.markdown("---")

    # ROC Curves
    st.header(t('performance.roc_title'))

    st.markdown(t('performance.roc_desc'))

    roc_fig_path = get_figure_path("15_roc_curves.png")

    if roc_fig_path.exists():
        try:
            image = Image.open(roc_fig_path)
            st.image(image, caption=t('performance.roc_caption'), use_column_width=True)
        except Exception as e:
            st.warning(t('performance.roc_load_error').format(e=e))
    else:
        st.info(t('performance.roc_not_found'))

    st.markdown("---")

    # Model complexity comparison
    st.header(t('performance.complexity_title'))

    st.markdown(t('performance.complexity_desc'))

    complexity_data = {
        t('performance.complexity_headers.model'): ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
        t('performance.complexity_headers.training_speed'): [
            t('performance.complexity_values.fast'),
            t('performance.complexity_values.moderate'),
            t('performance.complexity_values.slow'),
            t('performance.complexity_values.fast')
        ],
        t('performance.complexity_headers.prediction_speed'): [
            t('performance.complexity_values.very_fast'),
            t('performance.complexity_values.fast'),
            t('performance.complexity_values.moderate'),
            t('performance.complexity_values.very_fast')
        ],
        t('performance.complexity_headers.interpretability'): [
            t('performance.complexity_values.high'),
            t('performance.complexity_values.medium'),
            t('performance.complexity_values.low'),
            t('performance.complexity_values.low')
        ],
        t('performance.complexity_headers.memory_usage'): [
            t('performance.complexity_values.low'),
            t('performance.complexity_values.high'),
            t('performance.complexity_values.high'),
            t('performance.complexity_values.medium')
        ],
        t('performance.complexity_headers.overfitting_risk'): [
            t('performance.complexity_values.high'),
            t('performance.complexity_values.low'),
            t('performance.complexity_values.low'),
            t('performance.complexity_values.low')
        ]
    }

    complexity_df = pd.DataFrame(complexity_data)

    # Add performance data
    if not results_df.empty:
        model_col = t('performance.complexity_headers.model')
        accuracy_col = t('performance.complexity_headers.accuracy')
        for _, row in results_df.iterrows():
            model_name = row["Model"]
            if model_name in complexity_df[model_col].values:
                idx = complexity_df[complexity_df[model_col] == model_name].index[0]
                complexity_df.loc[idx, accuracy_col] = f"{row['Accuracy']*100:.2f}%"

    st.dataframe(complexity_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Model recommendations
    st.header(t('performance.recommendations_title'))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t('performance.best_production_title'))
        st.success(t('performance.best_production_models'))
        st.markdown(t('performance.best_production_reasons'))

    with col2:
        st.subheader(t('performance.best_interpretability_title'))
        st.info(t('performance.best_interpretability_models'))
        st.markdown(t('performance.best_interpretability_reasons'))

    st.markdown("---")

    # Performance insights
    st.header(t('performance.insights_title'))

    with st.expander(t('performance.insights_overall_title'), expanded=True):
        st.markdown(t('performance.insights_overall_content').format(
            model=best_model['Model'],
            accuracy=format_percentage(best_model['Accuracy'])
        ))

    with st.expander(t('performance.insights_challenges_title')):
        st.markdown(t('performance.insights_challenges_content'))

    with st.expander(t('performance.insights_recommendations_title')):
        st.markdown(t('performance.insights_recommendations_content'))

    st.markdown("---")

    # Export section
    st.header(t('performance.export_title'))

    col1, col2 = st.columns(2)

    with col1:
        if not results_df.empty:
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=t('performance.download_metrics'),
                data=csv,
                file_name="model_performance_comparison.csv",
                mime="text/csv"
            )

    with col2:
        if not complexity_df.empty:
            csv = complexity_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=t('performance.download_complexity'),
                data=csv,
                file_name="model_complexity_analysis.csv",
                mime="text/csv"
            )
