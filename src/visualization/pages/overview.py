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


def render():
    """Render the overview page."""
    st.title("🏠 OULAD Learning Analytics - Overview")
    st.markdown("---")

    # Load data
    with st.spinner("Loading dataset..."):
        df = load_dataset("clustered")

    if df.empty:
        st.error("❌ Unable to load dataset. Please check data files.")
        return

    # Compute statistics
    stats = compute_dataset_statistics(df)

    # Display key metrics
    st.header("📊 Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Students",
            value=format_number(stats["n_students"], 0),
            help="Total number of student records"
        )

    with col2:
        st.metric(
            label="Features",
            value=stats["n_features"],
            help="Total number of features in dataset"
        )

    with col3:
        st.metric(
            label="Avg VLE Clicks",
            value=format_number(stats["avg_clicks"], 0),
            help="Average number of Virtual Learning Environment interactions"
        )

    with col4:
        st.metric(
            label="Avg Assessment Score",
            value=format_number(stats["avg_assessment_score"], 1),
            help="Average score across all assessments"
        )

    st.markdown("---")

    # Target distribution
    st.header("🎯 Student Outcomes Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        if "final_result" in df.columns:
            # Create interactive pie chart
            result_counts = df["final_result"].value_counts()
            fig = px.pie(
                values=result_counts.values,
                names=result_counts.index,
                title="Final Results Distribution",
                color=result_counts.index,
                color_discrete_map=CLASS_COLORS,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Outcome Statistics")
        for outcome, count in stats["target_distribution"].items():
            percentage = (count / stats["n_students"]) * 100
            st.metric(
                label=outcome,
                value=format_number(count, 0),
                delta=f"{percentage:.1f}%"
            )

    st.markdown("---")

    # Demographics analysis
    st.header("👥 Demographics Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "gender" in df.columns:
            fig = px.histogram(
                df,
                x="gender",
                color="final_result",
                title="Gender Distribution by Outcome",
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
                title="Age Distribution by Outcome",
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
                title="Previous Attempts by Outcome",
                color_discrete_map=CLASS_COLORS,
                barmode="group"
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # VLE Activity Analysis
    st.header("💻 Virtual Learning Environment Activity")

    col1, col2 = st.columns(2)

    with col1:
        # VLE clicks distribution by outcome
        if "total_clicks" in df.columns:
            fig = px.box(
                df,
                x="final_result",
                y="total_clicks",
                color="final_result",
                title="VLE Clicks Distribution by Outcome",
                color_discrete_map=CLASS_COLORS
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxes(title="Total Clicks")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Assessment submission rate by outcome
        if "assessment_submission_rate" in df.columns:
            fig = px.box(
                df,
                x="final_result",
                y="assessment_submission_rate",
                color="final_result",
                title="Assessment Submission Rate by Outcome",
                color_discrete_map=CLASS_COLORS
            )
            fig.update_layout(height=400, showlegend=False)
            fig.update_yaxes(title="Submission Rate")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Correlation analysis
    st.header("🔗 Feature Correlations")

    # Select numeric features for correlation
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Remove ID columns
    numeric_cols = [col for col in numeric_cols if "id" not in col.lower() and col not in ["kmeans_cluster", "dbscan_cluster"]]

    # Let user select features to correlate
    selected_features = st.multiselect(
        "Select features to analyze correlations:",
        options=numeric_cols,
        default=numeric_cols[:10] if len(numeric_cols) >= 10 else numeric_cols
    )

    if selected_features:
        corr_matrix = df[selected_features].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Data quality section
    st.header("✅ Data Quality Report")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Values")
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
            st.success("✨ No missing values found!")

    with col2:
        st.subheader("Dataset Info")
        info_data = {
            "Metric": [
                "Total Records",
                "Total Features",
                "Numeric Features",
                "Categorical Features",
                "Memory Usage (MB)",
                "Duplicate Rows"
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
    st.header("📥 Data Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Dataset (CSV)",
            data=csv,
            file_name="oulad_data.csv",
            mime="text/csv"
        )

    with col2:
        # Create summary statistics
        summary = df.describe().T
        summary_csv = summary.to_csv().encode('utf-8')
        st.download_button(
            label="Download Summary Statistics (CSV)",
            data=summary_csv,
            file_name="oulad_summary_stats.csv",
            mime="text/csv"
        )

    with col3:
        # Create correlation matrix export
        if selected_features:
            corr_csv = corr_matrix.to_csv().encode('utf-8')
            st.download_button(
                label="Download Correlation Matrix (CSV)",
                data=corr_csv,
                file_name="oulad_correlations.csv",
                mime="text/csv"
            )
