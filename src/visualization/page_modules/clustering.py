"""
Clustering Page
Student behavioral clustering analysis and insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from visualization.utils import (
    load_dataset,
    get_cluster_profile,
    format_number,
    format_percentage,
    get_figure_path
)
from visualization.config import CLASS_COLORS
from visualization.i18n import t


def render():
    """Render the clustering page."""
    st.title(f"👥 {t('clustering.title')}")
    st.markdown("---")

    # Load data
    with st.spinner(t('clustering.loading_dataset')):
        df = load_dataset("clustered")

    if df.empty:
        st.error(t('clustering.unable_to_load_dataset'))
        return

    # Check for clustering columns
    has_kmeans = "kmeans_cluster" in df.columns
    has_dbscan = "dbscan_cluster" in df.columns

    if not has_kmeans and not has_dbscan:
        st.error(t('clustering.no_data_found'))
        return

    # Clustering method selection
    clustering_methods = []
    if has_kmeans:
        clustering_methods.append(t('clustering.methods.kmeans'))
    if has_dbscan:
        clustering_methods.append(t('clustering.methods.dbscan'))

    selected_method = st.selectbox(
        t('clustering.select_method'),
        options=clustering_methods,
        help=t('clustering.select_method_help')
    )

    cluster_col = "kmeans_cluster" if selected_method == t('clustering.methods.kmeans') else "dbscan_cluster"

    st.markdown("---")

    # Clustering overview
    st.header(t('clustering.overview_title'))

    # Basic statistics
    n_clusters = df[cluster_col].nunique()
    if selected_method == t('clustering.methods.dbscan'):
        # Exclude noise points (-1) from cluster count
        n_clusters = len(df[df[cluster_col] != -1][cluster_col].unique())
        n_noise = len(df[df[cluster_col] == -1])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(t('clustering.total_students'), format_number(len(df), 0))

    with col2:
        st.metric(t('clustering.num_clusters'), n_clusters)

    with col3:
        avg_cluster_size = len(df) / n_clusters if n_clusters > 0 else 0
        st.metric(t('clustering.avg_cluster_size'), format_number(avg_cluster_size, 0))

    with col4:
        if selected_method == t('clustering.methods.dbscan'):
            st.metric(t('clustering.noise_points'), format_number(n_noise, 0), delta=format_percentage(n_noise / len(df)))
        else:
            largest_cluster = df[cluster_col].value_counts().max()
            st.metric(t('clustering.largest_cluster'), format_number(largest_cluster, 0))

    st.markdown("---")

    # Cluster distribution
    st.header(t('clustering.distribution_title'))

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart of cluster sizes
        cluster_counts = df[cluster_col].value_counts().sort_index()

        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={"x": t('clustering.cluster_id'), "y": t('clustering.number_of_students')},
            title=t('clustering.cluster_sizes_title').format(method=selected_method),
            color=cluster_counts.values,
            color_continuous_scale="viridis"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            values=cluster_counts.values,
            names=[t('clustering.cluster_label').format(id=i) for i in cluster_counts.index],
            title=t('clustering.cluster_proportions'),
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Cluster visualization
    st.header(f"🎨 {t('clustering.visualization')}")

    viz_method = st.radio(
        t('clustering.viz_method'),
        options=[t('clustering.viz.pca'), t('clustering.viz.tsne')],
        horizontal=True,
        help=t('clustering.viz_method_help')
    )

    with st.spinner(t('clustering.computing_viz').format(method=viz_method)):
        # Select numeric features for visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove cluster columns and IDs
        viz_features = [col for col in numeric_cols
                       if col not in [cluster_col, "kmeans_cluster", "dbscan_cluster", "id_student"]]

        # Prepare data
        X_viz = df[viz_features].fillna(df[viz_features].median())

        try:
            if viz_method == t('clustering.viz.pca'):
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X_viz)
                explained_var = reducer.explained_variance_ratio_

                st.info(t('clustering.pca_explained_variance').format(
                    pc1=f"{explained_var[0]:.1%}",
                    pc2=f"{explained_var[1]:.1%}",
                    total=f"{sum(explained_var):.1%}"
                ))
            else:  # t-SNE
                # Use PCA first to reduce to 50 dimensions for speed
                if len(viz_features) > 50:
                    pca = PCA(n_components=50, random_state=42)
                    X_pca = pca.fit_transform(X_viz)
                else:
                    X_pca = X_viz

                reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
                X_reduced = reducer.fit_transform(X_pca)

            # Create visualization
            viz_df = pd.DataFrame({
                t('clustering.component_1'): X_reduced[:, 0],
                t('clustering.component_2'): X_reduced[:, 1],
                t('clustering.cluster_column'): df[cluster_col].astype(str),
                t('clustering.final_result_column'): df["final_result"] if "final_result" in df.columns else t('common.unknown')
            })

            # Plot by cluster
            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter(
                    viz_df,
                    x=t('clustering.component_1'),
                    y=t('clustering.component_2'),
                    color=t('clustering.cluster_column'),
                    title=t('clustering.viz.colored_by_cluster').format(method=viz_method),
                    opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    viz_df,
                    x=t('clustering.component_1'),
                    y=t('clustering.component_2'),
                    color=t('clustering.final_result_column'),
                    title=t('clustering.viz.colored_by_outcome').format(method=viz_method),
                    opacity=0.6,
                    color_discrete_map=CLASS_COLORS
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(t('clustering.viz_error').format(e=e))

    st.markdown("---")

    # Cluster profiles
    st.header(t('clustering.profiles_title'))

    # Select cluster to profile
    cluster_ids = sorted(df[cluster_col].unique())
    selected_cluster = st.selectbox(
        t('clustering.select_cluster'),
        options=cluster_ids,
        format_func=lambda x: t('clustering.cluster_format').format(id=x) if x != -1 else t('clustering.noise_points_label')
    )

    cluster_data = df[df[cluster_col] == selected_cluster]

    # Cluster summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(t('clustering.profile.cluster_size'), len(cluster_data))

    with col2:
        cluster_pct = len(cluster_data) / len(df) * 100
        st.metric(t('clustering.profile.percentage'), f"{cluster_pct:.1f}%")

    with col3:
        if "total_clicks" in cluster_data.columns:
            st.metric(t('clustering.profile.avg_vle_clicks'), format_number(cluster_data["total_clicks"].mean(), 0))

    with col4:
        if "avg_assessment_score" in cluster_data.columns:
            st.metric(t('clustering.profile.avg_score'), format_number(cluster_data["avg_assessment_score"].mean(), 1))

    # Outcome distribution within cluster
    st.subheader(t('clustering.outcomes_in_cluster').format(id=selected_cluster))

    if "final_result" in cluster_data.columns:
        col1, col2 = st.columns([2, 1])

        with col1:
            outcome_counts = cluster_data["final_result"].value_counts()

            fig = px.bar(
                x=outcome_counts.index,
                y=outcome_counts.values,
                color=outcome_counts.index,
                color_discrete_map=CLASS_COLORS,
                title=t('clustering.results_distribution').format(id=selected_cluster),
                labels={"x": t('clustering.outcome_label'), "y": t('clustering.count_label')}
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            outcome_pct = (outcome_counts / len(cluster_data) * 100).round(1)
            st.markdown(t('clustering.percentages_title'))
            for outcome, pct in outcome_pct.items():
                st.metric(outcome, f"{pct}%")

    # Feature comparison across clusters
    st.subheader(t('clustering.feature_comparison_title'))

    # Select features to compare
    comparison_features = st.multiselect(
        t('clustering.select_features'),
        options=viz_features,
        default=viz_features[:6] if len(viz_features) >= 6 else viz_features
    )

    if comparison_features:
        # Compute mean values per cluster
        cluster_profiles = df.groupby(cluster_col)[comparison_features].mean()

        # Normalize for heatmap
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())

        fig = px.imshow(
            cluster_profiles_norm.T,
            labels=dict(x=t('clustering.heatmap.cluster_axis'), y=t('clustering.heatmap.feature_axis'), color=t('clustering.heatmap.normalized_value')),
            x=[t('clustering.cluster_label').format(id=c) for c in cluster_profiles.index],
            y=cluster_profiles.columns,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title=t('clustering.heatmap.title')
        )
        fig.update_layout(height=max(400, len(comparison_features) * 30))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Cluster comparison
    st.header(t('clustering.comparison_title'))

    # Select two clusters to compare
    col1, col2 = st.columns(2)

    with col1:
        cluster_1 = st.selectbox(
            t('clustering.first_cluster'),
            options=cluster_ids,
            format_func=lambda x: t('clustering.cluster_format').format(id=x),
            key="cluster_1"
        )

    with col2:
        cluster_2 = st.selectbox(
            t('clustering.second_cluster'),
            options=cluster_ids,
            format_func=lambda x: t('clustering.cluster_format').format(id=x),
            key="cluster_2"
        )

    if cluster_1 != cluster_2:
        data_1 = df[df[cluster_col] == cluster_1]
        data_2 = df[df[cluster_col] == cluster_2]

        # Select features for comparison
        comparison_feature = st.selectbox(
            t('clustering.select_feature_compare'),
            options=viz_features,
            index=0
        )

        # Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(t('clustering.cluster_heading').format(id=cluster_1))
            st.metric(t('clustering.stats.mean'), format_number(data_1[comparison_feature].mean(), 2))
            st.metric(t('clustering.stats.median'), format_number(data_1[comparison_feature].median(), 2))
            st.metric(t('clustering.stats.std_dev'), format_number(data_1[comparison_feature].std(), 2))

        with col2:
            st.markdown(t('clustering.cluster_heading').format(id=cluster_2))
            st.metric(t('clustering.stats.mean'), format_number(data_2[comparison_feature].mean(), 2))
            st.metric(t('clustering.stats.median'), format_number(data_2[comparison_feature].median(), 2))
            st.metric(t('clustering.stats.std_dev'), format_number(data_2[comparison_feature].std(), 2))

        # Distribution comparison
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=data_1[comparison_feature],
            name=t('clustering.cluster_format').format(id=cluster_1),
            opacity=0.7,
            marker_color="blue"
        ))

        fig.add_trace(go.Histogram(
            x=data_2[comparison_feature],
            name=t('clustering.cluster_format').format(id=cluster_2),
            opacity=0.7,
            marker_color="red"
        ))

        fig.update_layout(
            title=t('clustering.distribution_comparison').format(feature=comparison_feature),
            barmode="overlay",
            height=400,
            xaxis_title=comparison_feature,
            yaxis_title=t('clustering.count_label')
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Cluster insights and recommendations
    st.header(t('clustering.insights_title'))

    for cluster_id in cluster_ids[:5]:  # Show top 5 clusters
        with st.expander(t('clustering.insights.cluster_expander').format(id=cluster_id) if cluster_id != -1 else t('clustering.insights.noise_expander')):
            cluster_subset = df[df[cluster_col] == cluster_id]

            # Calculate key metrics
            avg_clicks = cluster_subset["total_clicks"].mean() if "total_clicks" in cluster_subset.columns else 0
            avg_score = cluster_subset["avg_assessment_score"].mean() if "avg_assessment_score" in cluster_subset.columns else 0
            submission_rate = cluster_subset["assessment_submission_rate"].mean() if "assessment_submission_rate" in cluster_subset.columns else 0

            # Dominant outcome
            if "final_result" in cluster_subset.columns:
                dominant_outcome = cluster_subset["final_result"].value_counts().index[0]
                dominant_pct = cluster_subset["final_result"].value_counts().iloc[0] / len(cluster_subset) * 100
            else:
                dominant_outcome = t('common.unknown')
                dominant_pct = 0

            # Display insights
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(t('clustering.insights.size'), len(cluster_subset))
                st.metric(t('clustering.insights.avg_clicks'), format_number(avg_clicks, 0))

            with col2:
                st.metric(t('clustering.insights.avg_score'), format_number(avg_score, 1))
                st.metric(t('clustering.insights.submission_rate'), format_percentage(submission_rate))

            with col3:
                st.metric(t('clustering.insights.dominant_outcome'), dominant_outcome)
                st.metric(t('clustering.insights.outcome_percent'), f"{dominant_pct:.1f}%")

            # Provide recommendations
            st.markdown(t('clustering.insights.recommendations_title'))

            if avg_clicks < 500 and submission_rate < 0.5:
                st.warning(t('clustering.insights.low_engagement'))
                st.markdown(t('clustering.insights.low_engagement_actions'))
            elif avg_score > 75 and submission_rate > 0.8:
                st.success(t('clustering.insights.high_performance'))
                st.markdown(t('clustering.insights.high_performance_actions'))
            elif submission_rate < 0.7:
                st.info(t('clustering.insights.moderate_engagement'))
                st.markdown(t('clustering.insights.moderate_engagement_actions'))
            else:
                st.info(t('clustering.insights.average_performance'))
                st.markdown(t('clustering.insights.average_performance_actions'))

    st.markdown("---")

    # Export cluster data
    st.header(t('clustering.export_title'))

    col1, col2 = st.columns(2)

    with col1:
        # Export selected cluster
        if selected_cluster is not None:
            cluster_csv = cluster_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=t('clustering.download_cluster_data').format(id=selected_cluster),
                data=cluster_csv,
                file_name=f"cluster_{selected_cluster}_data.csv",
                mime="text/csv"
            )

    with col2:
        # Export cluster profiles
        if comparison_features:
            profiles_csv = cluster_profiles.to_csv().encode('utf-8')
            st.download_button(
                label=t('clustering.download_profiles'),
                data=profiles_csv,
                file_name="cluster_profiles.csv",
                mime="text/csv"
            )
