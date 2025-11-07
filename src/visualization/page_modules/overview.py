"""
Overview Page
Dataset statistics, distributions, and key insights.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.utils import (
    load_dataset,
    compute_dataset_statistics,
    format_number,
    format_percentage,
    get_figure_path
)
from visualization.config import CLASS_COLORS, FEATURE_CATEGORIES
from visualization.i18n import t


def render():
    """Render the overview page."""
    st.title(f"🏠 {t('overview.title')}")
    st.markdown("---")

    # Load data
    with st.spinner(t('overview.loading_dataset')):
        df = load_dataset("clustered")

    if df.empty:
        st.error(f"❌ {t('overview.unable_to_load_dataset')}")
        return

    # Compute statistics
    stats = compute_dataset_statistics(df)

    # Display key metrics
    st.header(f"📊 {t('overview.dataset_summary')}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=t('overview.total_students'),
            value=format_number(stats["n_students"], 0),
            help=t('overview.total_students_help')
        )

    with col2:
        st.metric(
            label=t('overview.features'),
            value=stats["n_features"],
            help=t('overview.features_help')
        )

    with col3:
        st.metric(
            label=t('overview.avg_vle_clicks'),
            value=format_number(stats["avg_clicks"], 0),
            help=t('overview.avg_vle_clicks_help')
        )

    with col4:
        st.metric(
            label=t('overview.avg_assessment_score'),
            value=format_number(stats["avg_assessment_score"], 1),
            help=t('overview.avg_assessment_score_help')
        )

    st.markdown("---")

    # Target distribution
    st.header(f"🎯 {t('overview.outcomes_distribution')}")

    col1, col2 = st.columns([2, 1])

    with col1:
        if "final_result" in df.columns:
            # Create interactive pie chart
            result_counts = df["final_result"].value_counts()
            fig = px.pie(
                values=result_counts.values,
                names=result_counts.index,
                title=t('overview.final_results_distribution'),
                color=result_counts.index,
                color_discrete_map=CLASS_COLORS,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(t('overview.outcome_statistics'))
        for outcome, count in stats["target_distribution"].items():
            percentage = (count / stats["n_students"]) * 100
            st.metric(
                label=outcome,
                value=format_number(count, 0),
                delta=f"{percentage:.1f}%"
            )

    st.markdown("---")

    # Demographics analysis
    st.header(f"👥 {t('overview.demographics')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "gender" in df.columns:
            fig = px.histogram(
                df,
                x="gender",
                color="final_result",
                title=t('overview.gender_distribution'),
                color_discrete_map=CLASS_COLORS,
                barmode="group"
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "age_band" in df.columns:
            fig = px.histogram(
                df,
                x="age_band",
                color="final_result",
                title=t('overview.age_distribution'),
                color_discrete_map=CLASS_COLORS,
                barmode="group",
                category_orders={"age_band": ["0-35", "35-55", "55<="]}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if "num_of_prev_attempts" in df.columns:
            fig = px.histogram(
                df,
                x="num_of_prev_attempts",
                color="final_result",
                title=t('overview.previous_attempts'),
                color_discrete_map=CLASS_COLORS,
                barmode="group"
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # VLE Activity Analysis
    st.header(f"💻 {t('overview.vle_activity')}")

    col1, col2 = st.columns(2)

    with col1:
        # VLE clicks distribution by outcome
        if "total_clicks" in df.columns:
            fig = px.box(
                df,
                x="final_result",
                y="total_clicks",
                color="final_result",
                title=t('overview.vle_clicks_distribution'),
                color_discrete_map=CLASS_COLORS
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxes(title=t('overview.total_clicks'))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Assessment submission rate by outcome
        if "assessment_submission_rate" in df.columns:
            fig = px.box(
                df,
                x="final_result",
                y="assessment_submission_rate",
                color="final_result",
                title=t('overview.assessment_submission_rate'),
                color_discrete_map=CLASS_COLORS
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxes(title=t('overview.submission_rate'))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation analysis
    st.header(f"🔗 {t('overview.correlations')}")

    # Select numeric features for correlation
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Remove ID columns
    numeric_cols = [col for col in numeric_cols if "id" not in col.lower() and col not in ["kmeans_cluster", "dbscan_cluster"]]

    # Let user select features to correlate
    selected_features = st.multiselect(
        t('overview.select_features'),
        options=numeric_cols,
        default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
    )

    if selected_features:
        corr_matrix = df[selected_features].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title=t('overview.correlation_heatmap'),
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Data quality section
    st.header(f"✅ {t('overview.data_quality')}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t('overview.missing_values'))
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            "Feature": missing.index,
            "Missing Count": missing.values,
            "Percentage": missing_pct.values
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)

        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True, height=300)
        else:
            st.success(f"✨ {t('overview.no_missing_values')}")

    with col2:
        st.subheader(t('overview.dataset_info'))
        info_data = {
            "Metric": [
                t('overview.total_records'),
                t('overview.total_features'),
                t('overview.numeric_features'),
                t('overview.categorical_features'),
                t('overview.memory_usage'),
                t('overview.duplicate_rows')
            ],
            "Value": [
                len(df),
                len(df.columns),
                len(df.select_dtypes(include=["number"]).columns),
                len(df.select_dtypes(include=["object"]).columns),
                f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}",
                df.duplicated().sum()
            ]
        }
        st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Download section
    st.header(f"📥 {t('overview.data_export')}")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=t('overview.download_full_dataset'),
            data=csv,
            file_name="oulad_data.csv",
            mime="text/csv"
        )

    with col2:
        # Create summary statistics
        summary = df.describe().T
        summary_csv = summary.to_csv().encode('utf-8')
        st.download_button(
            label=t('overview.download_summary_stats'),
            data=summary_csv,
            file_name="oulad_summary_stats.csv",
            mime="text/csv"
        )

    with col3:
        # Create correlation matrix export
        if selected_features:
            corr_csv = corr_matrix.to_csv().encode('utf-8')
            st.download_button(
                label=t('overview.download_correlation_matrix'),
                data=corr_csv,
                file_name="oulad_correlations.csv",
                mime="text/csv"
            )
