"""
Feature Importance Page
Feature importance analysis, SHAP values, and feature selection insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

from visualization.utils import (
    load_feature_importance,
    load_all_models,
    load_dataset,
    format_number,
    get_figure_path
)
from visualization.config import FEATURE_CATEGORIES, FIGURES_DIR
from visualization.i18n import t


def render():
    """Render the feature importance page."""
    st.title(f"⭐ {t('importance.title')}")
    st.markdown("---")

    # Load data
    with st.spinner(t('importance.loading')):
        importance_scores = load_feature_importance()
        models = load_all_models()
        df = load_dataset("clustered")

    if importance_scores.empty:
        st.error("❌ Unable to load feature importance data.")
        return

    # Overview
    st.header("📊 Feature Importance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Features", len(importance_scores))

    with col2:
        top_10_contribution = importance_scores.head(10).sum() / importance_scores.sum() * 100
        st.metric("Top 10 Contribution", f"{top_10_contribution:.1f}%")

    with col3:
        top_20_contribution = importance_scores.head(20).sum() / importance_scores.sum() * 100
        st.metric("Top 20 Contribution", f"{top_20_contribution:.1f}%")

    with col4:
        # Count features with very low importance
        low_importance = (importance_scores < importance_scores.quantile(0.1)).sum()
        st.metric("Low Importance Features", low_importance)

    st.markdown("---")

    # Top features visualization
    st.header(f"🏆 {t('importance.top_features')}")

    # Slider to select number of features
    n_features = st.slider(
        "Number of top features to display:",
        min_value=5,
        max_value=min(50, len(importance_scores)),
        value=20,
        step=5
    )

    top_features = importance_scores.head(n_features)

    # Horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top_features.index[::-1],  # Reverse for top-to-bottom display
        x=top_features.values[::-1],
        orientation='h',
        marker=dict(
            color=top_features.values[::-1],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f"{v:.1f}" for v in top_features.values[::-1]],
        textposition='outside'
    ))

    fig.update_layout(
        title=f"Top {n_features} Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(400, n_features * 20),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature categories analysis
    st.header("📂 Feature Importance by Category")

    # Categorize features
    category_importance = {}

    for category, features in FEATURE_CATEGORIES.items():
        category_scores = []
        for feature in features:
            if feature in importance_scores.index:
                category_scores.append(importance_scores[feature])

        if category_scores:
            category_importance[category] = {
                "total": sum(category_scores),
                "average": np.mean(category_scores),
                "count": len(category_scores)
            }

    # Create category comparison
    if category_importance:
        cat_df = pd.DataFrame(category_importance).T
        cat_df = cat_df.sort_values("total", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                x=cat_df.index,
                y=cat_df["total"],
                title="Total Importance by Category",
                labels={"x": "Category", "y": "Total Importance"},
                color=cat_df["total"],
                color_continuous_scale="Blues"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                x=cat_df.index,
                y=cat_df["average"],
                title="Average Importance by Category",
                labels={"x": "Category", "y": "Average Importance"},
                color=cat_df["average"],
                color_continuous_scale="Greens"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Category details
        st.subheader("Category Details")

        for category in cat_df.index:
            with st.expander(f"{category} ({cat_df.loc[category, 'count']} features)"):
                features = FEATURE_CATEGORIES[category]
                feature_scores = [(f, importance_scores[f]) for f in features if f in importance_scores.index]
                feature_scores.sort(key=lambda x: x[1], reverse=True)

                feature_df = pd.DataFrame(feature_scores, columns=["Feature", "Importance"])

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.dataframe(feature_df, use_container_width=True, hide_index=True)

                with col2:
                    st.metric("Total Importance", format_number(cat_df.loc[category, "total"], 1))
                    st.metric("Average Importance", format_number(cat_df.loc[category, "average"], 1))
                    st.metric("Feature Count", int(cat_df.loc[category, "count"]))

    st.markdown("---")

    # Feature importance images from notebooks
    st.header("📊 Detailed Visualizations")

    tab1, tab2, tab3 = st.tabs(["Feature Rankings", "Model Comparison", "SHAP Analysis"])

    with tab1:
        st.subheader("Feature Importance Rankings")

        fig_path = get_figure_path("16_feature_importance.png")

        if fig_path.exists():
            try:
                image = Image.open(fig_path)
                st.image(image, caption="Top 20 Most Important Features (Averaged Across Models)", use_column_width=True)
            except Exception as e:
                st.warning(f"Could not load feature importance image: {e}")
        else:
            st.info("Feature importance visualization not found. Run Phase 3 notebook to generate.")

    with tab2:
        st.subheader("Feature Importance by Model")

        fig_path = get_figure_path("17_feature_importance_by_model.png")

        if fig_path.exists():
            try:
                image = Image.open(fig_path)
                st.image(image, caption="Feature Importance Comparison Across Models", use_column_width=True)
            except Exception as e:
                st.warning(f"Could not load model comparison image: {e}")
        else:
            st.info("Model comparison visualization not found. Run Phase 3 notebook to generate.")

    with tab3:
        st.subheader("SHAP Analysis")

        st.markdown("""
        SHAP (SHapley Additive exPlanations) values provide detailed insights into how each feature
        contributes to individual predictions.
        """)

        st.info("SHAP visualizations are available in Phase 4 notebook. Run the notebook to generate detailed SHAP plots.")

    st.markdown("---")

    # Feature correlation analysis
    st.header(f"🔗 {t('importance.correlations')}")

    st.markdown("Understanding correlations helps identify redundant features and feature engineering opportunities.")

    if not df.empty:
        # Select numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Get top features
        top_feature_names = importance_scores.head(20).index.tolist()
        available_features = [f for f in top_feature_names if f in numeric_cols]

        if available_features:
            # Compute correlation matrix
            corr_matrix = df[available_features].corr()

            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Matrix - Top 20 Important Features",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)

            # Highlight highly correlated pairs
            st.subheader("Highly Correlated Feature Pairs")

            # Find pairs with |correlation| > 0.7
            high_corr_pairs = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            "Feature 1": corr_matrix.columns[i],
                            "Feature 2": corr_matrix.columns[j],
                            "Correlation": corr_val
                        })

            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs)
                high_corr_df = high_corr_df.sort_values("Correlation", key=abs, ascending=False)
                st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No highly correlated feature pairs found (|r| > 0.7)")

    st.markdown("---")

    # Feature selection recommendations
    st.header("💡 Feature Selection Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Key Findings")

        st.markdown(f"""
        1. **Top Feature:** {importance_scores.index[0]}
           - Importance: {format_number(importance_scores.iloc[0], 1)}

        2. **Top 10 Features** account for {top_10_contribution:.1f}% of total importance
           - These features are critical for accurate predictions

        3. **Top 20 Features** account for {top_20_contribution:.1f}% of total importance
           - Could use reduced feature set with minimal accuracy loss

        4. **Assessment Features Dominate:**
           - Multiple assessment-related features in top 10
           - Student performance history is strongest predictor

        5. **VLE Engagement Matters:**
           - Homepage clicks, content access, and resource usage are important
           - Activity level correlates with outcomes
        """)

    with col2:
        st.subheader("🎯 Recommendations")

        st.markdown("""
        **For Model Optimization:**
        - Consider training with top 30 features for efficiency
        - Remove features with importance < 1st percentile
        - Use feature selection for faster inference

        **For Data Collection:**
        - Prioritize collecting assessment scores early
        - Track VLE engagement metrics closely
        - Monitor homepage and content access patterns

        **For Intervention:**
        - Focus on students with low assessment scores
        - Encourage VLE engagement (especially homepage/content)
        - Monitor submission rates as early warning signal

        **For Feature Engineering:**
        - Create interaction features between top predictors
        - Add temporal features (trend analysis)
        - Engineer ratio features (e.g., clicks per day)
        """)

    st.markdown("---")

    # Interactive feature explorer
    st.header("🔍 Interactive Feature Explorer")

    selected_feature = st.selectbox(
        "Select a feature to explore:",
        options=importance_scores.index.tolist()
    )

    if selected_feature and not df.empty and selected_feature in df.columns:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Importance Rank", importance_scores.index.tolist().index(selected_feature) + 1)

        with col2:
            st.metric("Importance Score", format_number(importance_scores[selected_feature], 2))

        with col3:
            pct_contribution = (importance_scores[selected_feature] / importance_scores.sum()) * 100
            st.metric("% of Total", f"{pct_contribution:.2f}%")

        # Distribution by outcome
        if "final_result" in df.columns and pd.api.types.is_numeric_dtype(df[selected_feature]):
            fig = px.box(
                df,
                x="final_result",
                y=selected_feature,
                color="final_result",
                title=f"{selected_feature} Distribution by Outcome",
                color_discrete_map={
                    "Distinction": "#2ecc71",
                    "Pass": "#3498db",
                    "Fail": "#e74c3c",
                    "Withdrawn": "#f39c12"
                }
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Statistics table
        st.subheader("Feature Statistics")

        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            stats_df = df.groupby("final_result")[selected_feature].describe().T

            st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")

    # Feature importance comparison table
    st.header("📋 Complete Feature Importance Table")

    # Add percentile and cumulative information
    importance_df = pd.DataFrame({
        "Feature": importance_scores.index,
        "Importance": importance_scores.values,
        "Rank": range(1, len(importance_scores) + 1),
        "Percentile": [(1 - i/len(importance_scores)) * 100 for i in range(len(importance_scores))],
        "Cumulative %": [importance_scores.iloc[:i+1].sum() / importance_scores.sum() * 100
                        for i in range(len(importance_scores))]
    })

    # Add category information
    def get_category(feature):
        for cat, features in FEATURE_CATEGORIES.items():
            if feature in features:
                return cat
        return "Other"

    importance_df["Category"] = importance_df["Feature"].apply(get_category)

    # Format for display
    display_importance = importance_df.copy()
    display_importance["Importance"] = display_importance["Importance"].apply(lambda x: format_number(x, 2))
    display_importance["Percentile"] = display_importance["Percentile"].apply(lambda x: f"{x:.1f}%")
    display_importance["Cumulative %"] = display_importance["Cumulative %"].apply(lambda x: f"{x:.1f}%")

    st.dataframe(
        display_importance,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    st.markdown("---")

    # Export options
    st.header("📥 Export Feature Importance Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = importance_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Table (CSV)",
            data=csv,
            file_name="feature_importance_full.csv",
            mime="text/csv"
        )

    with col2:
        top_50 = importance_df.head(50)
        csv = top_50.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Top 50 (CSV)",
            data=csv,
            file_name="feature_importance_top50.csv",
            mime="text/csv"
        )

    with col3:
        if category_importance:
            cat_csv = cat_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Category Summary (CSV)",
                data=cat_csv,
                file_name="feature_importance_by_category.csv",
                mime="text/csv"
            )
