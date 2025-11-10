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
        st.error(t('importance.unable_to_load_data'))
        return

    # Overview
    st.header(t('importance.overview_title'))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(t('importance.total_features'), len(importance_scores))

    with col2:
        top_10_contribution = importance_scores.head(10).sum() / importance_scores.sum() * 100
        st.metric(t('importance.top_10_contribution'), f"{top_10_contribution:.1f}%")

    with col3:
        top_20_contribution = importance_scores.head(20).sum() / importance_scores.sum() * 100
        st.metric(t('importance.top_20_contribution'), f"{top_20_contribution:.1f}%")

    with col4:
        # Count features with very low importance
        low_importance = (importance_scores < importance_scores.quantile(0.1)).sum()
        st.metric(t('importance.low_importance_features'), low_importance)

    st.markdown("---")

    # Top features visualization
    st.header(f"🏆 {t('importance.top_features')}")

    # Slider to select number of features
    n_features = st.slider(
        t('importance.num_features_slider'),
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
            colorbar=dict(title=t('importance.importance_colorbar'))
        ),
        text=[f"{v:.1f}" for v in top_features.values[::-1]],
        textposition='outside'
    ))

    fig.update_layout(
        title=t('importance.top_n_features_title').format(n=n_features),
        xaxis_title=t('importance.importance_score'),
        yaxis_title=t('importance.feature_label'),
        height=max(400, n_features * 20),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature categories analysis
    st.header(t('importance.by_category_title'))

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
                title=t('importance.total_by_category'),
                labels={"x": t('importance.category_label'), "y": t('importance.total_importance_label')},
                color=cat_df["total"],
                color_continuous_scale="Blues"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                x=cat_df.index,
                y=cat_df["average"],
                title=t('importance.avg_by_category'),
                labels={"x": t('importance.category_label'), "y": t('importance.avg_importance_label')},
                color=cat_df["average"],
                color_continuous_scale="Greens"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Category details
        st.subheader(t('importance.category_details'))

        for category in cat_df.index:
            with st.expander(t('importance.category_expander').format(category=category, count=int(cat_df.loc[category, 'count']))):
                features = FEATURE_CATEGORIES[category]
                feature_scores = [(f, importance_scores[f]) for f in features if f in importance_scores.index]
                feature_scores.sort(key=lambda x: x[1], reverse=True)

                feature_df = pd.DataFrame(feature_scores, columns=[t('importance.feature_label'), t('importance.importance_colorbar')])

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.dataframe(feature_df, use_container_width=True, hide_index=True)

                with col2:
                    st.metric(t('importance.total_importance_metric'), format_number(cat_df.loc[category, "total"], 1))
                    st.metric(t('importance.avg_importance_metric'), format_number(cat_df.loc[category, "average"], 1))
                    st.metric(t('importance.feature_count'), int(cat_df.loc[category, "count"]))

    st.markdown("---")

    # Feature importance images from notebooks
    st.header(t('importance.detailed_viz_title'))

    tab1, tab2, tab3 = st.tabs([t('importance.tabs.rankings'), t('importance.tabs.model_comparison'), t('importance.tabs.shap')])

    with tab1:
        st.subheader(t('importance.rankings_title'))

        fig_path = get_figure_path("16_feature_importance.png")

        if fig_path.exists():
            try:
                image = Image.open(fig_path)
                st.image(image, caption=t('importance.rankings_caption'), use_column_width=True)
            except Exception as e:
                st.warning(t('importance.viz_load_error').format(e=e))
        else:
            st.info(t('importance.viz_not_found'))

    with tab2:
        st.subheader(t('importance.by_model_title'))

        fig_path = get_figure_path("17_feature_importance_by_model.png")

        if fig_path.exists():
            try:
                image = Image.open(fig_path)
                st.image(image, caption=t('importance.by_model_caption'), use_column_width=True)
            except Exception as e:
                st.warning(t('importance.model_viz_load_error').format(e=e))
        else:
            st.info(t('importance.model_viz_not_found'))

    with tab3:
        st.subheader(t('importance.shap_title'))

        st.markdown(t('importance.shap_desc'))

        st.info(t('importance.shap_info'))

    st.markdown("---")

    # Feature correlation analysis
    st.header(f"🔗 {t('importance.correlations')}")

    st.markdown(t('importance.correlations_desc'))

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
                title=t('importance.correlation_matrix_title'),
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)

            # Highlight highly correlated pairs
            st.subheader(t('importance.high_corr_pairs'))

            # Find pairs with |correlation| > 0.7
            high_corr_pairs = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr_pairs.append({
                            t('importance.feature_1'): corr_matrix.columns[i],
                            t('importance.feature_2'): corr_matrix.columns[j],
                            t('importance.correlation'): corr_val
                        })

            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs)
                high_corr_df = high_corr_df.sort_values(t('importance.correlation'), key=abs, ascending=False)
                st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
            else:
                st.success(t('importance.no_high_corr'))

    st.markdown("---")

    # Feature selection recommendations
    st.header(t('importance.insights_title'))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t('importance.findings_title'))

        st.markdown(t('importance.findings_content').format(
            top_feature=importance_scores.index[0],
            top_importance=format_number(importance_scores.iloc[0], 1),
            top_10=f"{top_10_contribution:.1f}",
            top_20=f"{top_20_contribution:.1f}"
        ))

    with col2:
        st.subheader(t('importance.recommendations_title'))

        st.markdown(t('importance.recommendations_content'))

    st.markdown("---")

    # Interactive feature explorer
    st.header(t('importance.explorer_title'))

    selected_feature = st.selectbox(
        t('importance.select_feature'),
        options=importance_scores.index.tolist()
    )

    if selected_feature and not df.empty and selected_feature in df.columns:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(t('importance.importance_rank'), importance_scores.index.tolist().index(selected_feature) + 1)

        with col2:
            st.metric(t('importance.importance_score_metric'), format_number(importance_scores[selected_feature], 2))

        with col3:
            pct_contribution = (importance_scores[selected_feature] / importance_scores.sum()) * 100
            st.metric(t('importance.percent_of_total'), f"{pct_contribution:.2f}%")

        # Distribution by outcome
        if "final_result" in df.columns and pd.api.types.is_numeric_dtype(df[selected_feature]):
            fig = px.box(
                df,
                x="final_result",
                y=selected_feature,
                color="final_result",
                title=t('importance.distribution_by_outcome').format(feature=selected_feature),
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
        st.subheader(t('importance.feature_statistics'))

        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            stats_df = df.groupby("final_result")[selected_feature].describe().T

            st.dataframe(stats_df, use_container_width=True)

    st.markdown("---")

    # Feature importance comparison table
    st.header(t('importance.complete_table_title'))

    # Add percentile and cumulative information
    importance_df = pd.DataFrame({
        t('importance.feature_label'): importance_scores.index,
        t('importance.importance_colorbar'): importance_scores.values,
        t('common.rank'): range(1, len(importance_scores) + 1),
        t('importance.percentile'): [(1 - i/len(importance_scores)) * 100 for i in range(len(importance_scores))],
        t('importance.cumulative_percent'): [importance_scores.iloc[:i+1].sum() / importance_scores.sum() * 100
                        for i in range(len(importance_scores))]
    })

    # Add category information
    def get_category(feature):
        for cat, features in FEATURE_CATEGORIES.items():
            if feature in features:
                return cat
        return t('importance.category_other')

    importance_df[t('importance.category_label')] = importance_df[t('importance.feature_label')].apply(get_category)

    # Format for display
    display_importance = importance_df.copy()
    display_importance[t('importance.importance_colorbar')] = display_importance[t('importance.importance_colorbar')].apply(lambda x: format_number(x, 2))
    display_importance[t('importance.percentile')] = display_importance[t('importance.percentile')].apply(lambda x: f"{x:.1f}%")
    display_importance[t('importance.cumulative_percent')] = display_importance[t('importance.cumulative_percent')].apply(lambda x: f"{x:.1f}%")

    st.dataframe(
        display_importance,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    st.markdown("---")

    # Export options
    st.header(t('importance.export_title'))

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = importance_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=t('importance.download_full'),
            data=csv,
            file_name="feature_importance_full.csv",
            mime="text/csv"
        )

    with col2:
        top_50 = importance_df.head(50)
        csv = top_50.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=t('importance.download_top50'),
            data=csv,
            file_name="feature_importance_top50.csv",
            mime="text/csv"
        )

    with col3:
        if category_importance:
            cat_csv = cat_df.to_csv().encode('utf-8')
            st.download_button(
                label=t('importance.download_category'),
                data=cat_csv,
                file_name="feature_importance_by_category.csv",
                mime="text/csv"
            )
