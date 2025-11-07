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
    with st.spinner("Loading dataset..."):
        df = load_dataset("clustered")

    if df.empty:
        st.error("❌ Unable to load clustered dataset.")
        return

    # Check for clustering columns
    has_kmeans = "kmeans_cluster" in df.columns
    has_dbscan = "dbscan_cluster" in df.columns

    if not has_kmeans and not has_dbscan:
        st.error("❌ No clustering data found in dataset.")
        return

    # Clustering method selection
    clustering_methods = []
    if has_kmeans:
        clustering_methods.append("K-Means")
    if has_dbscan:
        clustering_methods.append("DBSCAN")

    selected_method = st.selectbox(
        "Select clustering method:",
        options=clustering_methods,
        help="Choose which clustering algorithm results to analyze"
    )

    cluster_col = "kmeans_cluster" if selected_method == "K-Means" else "dbscan_cluster"

    st.markdown("---")

    # Clustering overview
    st.header("📊 Clustering Overview")

    # Basic statistics
    n_clusters = df[cluster_col].nunique()
    if selected_method == "DBSCAN":
        # Exclude noise points (-1) from cluster count
        n_clusters = len(df[df[cluster_col] != -1][cluster_col].unique())
        n_noise = len(df[df[cluster_col] == -1])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Students", format_number(len(df), 0))

    with col2:
        st.metric("Number of Clusters", n_clusters)

    with col3:
        avg_cluster_size = len(df) / n_clusters if n_clusters > 0 else 0
        st.metric("Avg Cluster Size", format_number(avg_cluster_size, 0))

    with col4:
        if selected_method == "DBSCAN":
            st.metric("Noise Points", format_number(n_noise, 0), delta=format_percentage(n_noise / len(df)))
        else:
            largest_cluster = df[cluster_col].value_counts().max()
            st.metric("Largest Cluster", format_number(largest_cluster, 0))

    st.markdown("---")

    # Cluster distribution
    st.header("📈 Cluster Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Bar chart of cluster sizes
        cluster_counts = df[cluster_col].value_counts().sort_index()

        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={"x": "Cluster ID", "y": "Number of Students"},
            title=f"{selected_method} Cluster Sizes",
            color=cluster_counts.values,
            color_continuous_scale="viridis"
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart
        fig = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i}" for i in cluster_counts.index],
            title="Cluster Proportions",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Cluster visualization
    st.header(f"🎨 {t('clustering.visualization')}")

    viz_method = st.radio(
        "Dimensionality reduction method:",
        options=["PCA", "t-SNE"],
        horizontal=True,
        help="PCA is faster, t-SNE often provides better separation"
    )

    with st.spinner(f"Computing {viz_method} visualization..."):
        # Select numeric features for visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove cluster columns and IDs
        viz_features = [col for col in numeric_cols
                       if col not in [cluster_col, "kmeans_cluster", "dbscan_cluster", "id_student"]]

        # Prepare data
        X_viz = df[viz_features].fillna(df[viz_features].median())

        try:
            if viz_method == "PCA":
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X_viz)
                explained_var = reducer.explained_variance_ratio_

                st.info(f"ℹ️ PCA explained variance: {explained_var[0]:.1%} (PC1) + {explained_var[1]:.1%} (PC2) = {sum(explained_var):.1%} total")
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
                "Component 1": X_reduced[:, 0],
                "Component 2": X_reduced[:, 1],
                "Cluster": df[cluster_col].astype(str),
                "Final Result": df["final_result"] if "final_result" in df.columns else "Unknown"
            })

            # Plot by cluster
            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter(
                    viz_df,
                    x="Component 1",
                    y="Component 2",
                    color="Cluster",
                    title=f"{viz_method} Projection - Colored by Cluster",
                    opacity=0.6,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    viz_df,
                    x="Component 1",
                    y="Component 2",
                    color="Final Result",
                    title=f"{viz_method} Projection - Colored by Outcome",
                    opacity=0.6,
                    color_discrete_map=CLASS_COLORS
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Visualization failed: {e}")

    st.markdown("---")

    # Cluster profiles
    st.header("🔍 Cluster Profiles")

    # Select cluster to profile
    cluster_ids = sorted(df[cluster_col].unique())
    selected_cluster = st.selectbox(
        t('clustering.select_cluster'),
        options=cluster_ids,
        format_func=lambda x: f"Cluster {x}" if x != -1 else "Noise Points"
    )

    cluster_data = df[df[cluster_col] == selected_cluster]

    # Cluster summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Cluster Size", len(cluster_data))

    with col2:
        cluster_pct = len(cluster_data) / len(df) * 100
        st.metric("Percentage", f"{cluster_pct:.1f}%")

    with col3:
        if "total_clicks" in cluster_data.columns:
            st.metric("Avg VLE Clicks", format_number(cluster_data["total_clicks"].mean(), 0))

    with col4:
        if "avg_assessment_score" in cluster_data.columns:
            st.metric("Avg Score", format_number(cluster_data["avg_assessment_score"].mean(), 1))

    # Outcome distribution within cluster
    st.subheader(f"Outcomes in Cluster {selected_cluster}")

    if "final_result" in cluster_data.columns:
        col1, col2 = st.columns([2, 1])

        with col1:
            outcome_counts = cluster_data["final_result"].value_counts()

            fig = px.bar(
                x=outcome_counts.index,
                y=outcome_counts.values,
                color=outcome_counts.index,
                color_discrete_map=CLASS_COLORS,
                title=f"Final Results Distribution - Cluster {selected_cluster}",
                labels={"x": "Outcome", "y": "Count"}
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            outcome_pct = (outcome_counts / len(cluster_data) * 100).round(1)
            st.markdown("**Percentages:**")
            for outcome, pct in outcome_pct.items():
                st.metric(outcome, f"{pct}%")

    # Feature comparison across clusters
    st.subheader("Feature Comparison Across Clusters")

    # Select features to compare
    comparison_features = st.multiselect(
        "Select features to compare:",
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
            labels=dict(x="Cluster", y="Feature", color="Normalized Value"),
            x=[f"Cluster {c}" for c in cluster_profiles.index],
            y=cluster_profiles.columns,
            color_continuous_scale="RdYlGn",
            aspect="auto",
            title="Cluster Profile Heatmap (Normalized)"
        )
        fig.update_layout(height=max(400, len(comparison_features) * 30))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Cluster comparison
    st.header("⚖️ Cluster Comparison")

    # Select two clusters to compare
    col1, col2 = st.columns(2)

    with col1:
        cluster_1 = st.selectbox(
            "First cluster:",
            options=cluster_ids,
            format_func=lambda x: f"Cluster {x}",
            key="cluster_1"
        )

    with col2:
        cluster_2 = st.selectbox(
            "Second cluster:",
            options=cluster_ids,
            format_func=lambda x: f"Cluster {x}",
            key="cluster_2"
        )

    if cluster_1 != cluster_2:
        data_1 = df[df[cluster_col] == cluster_1]
        data_2 = df[df[cluster_col] == cluster_2]

        # Select features for comparison
        comparison_feature = st.selectbox(
            "Select feature to compare:",
            options=viz_features,
            index=0
        )

        # Side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Cluster {cluster_1}**")
            st.metric("Mean", format_number(data_1[comparison_feature].mean(), 2))
            st.metric("Median", format_number(data_1[comparison_feature].median(), 2))
            st.metric("Std Dev", format_number(data_1[comparison_feature].std(), 2))

        with col2:
            st.markdown(f"**Cluster {cluster_2}**")
            st.metric("Mean", format_number(data_2[comparison_feature].mean(), 2))
            st.metric("Median", format_number(data_2[comparison_feature].median(), 2))
            st.metric("Std Dev", format_number(data_2[comparison_feature].std(), 2))

        # Distribution comparison
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=data_1[comparison_feature],
            name=f"Cluster {cluster_1}",
            opacity=0.7,
            marker_color="blue"
        ))

        fig.add_trace(go.Histogram(
            x=data_2[comparison_feature],
            name=f"Cluster {cluster_2}",
            opacity=0.7,
            marker_color="red"
        ))

        fig.update_layout(
            title=f"Distribution Comparison: {comparison_feature}",
            barmode="overlay",
            height=400,
            xaxis_title=comparison_feature,
            yaxis_title="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Cluster insights and recommendations
    st.header("💡 Cluster Insights & Recommendations")

    for cluster_id in cluster_ids[:5]:  # Show top 5 clusters
        with st.expander(f"Cluster {cluster_id}" if cluster_id != -1 else "Noise Points"):
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
                dominant_outcome = "Unknown"
                dominant_pct = 0

            # Display insights
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Size", len(cluster_subset))
                st.metric("Avg VLE Clicks", format_number(avg_clicks, 0))

            with col2:
                st.metric("Avg Assessment Score", format_number(avg_score, 1))
                st.metric("Submission Rate", format_percentage(submission_rate))

            with col3:
                st.metric("Dominant Outcome", dominant_outcome)
                st.metric("Outcome %", f"{dominant_pct:.1f}%")

            # Provide recommendations
            st.markdown("**Recommendations:**")

            if avg_clicks < 500 and submission_rate < 0.5:
                st.warning("⚠️ **Low engagement cluster** - High risk group requiring immediate intervention")
                st.markdown("- Send engagement reminders\n- Offer additional support\n- Monitor closely for withdrawal")
            elif avg_score > 75 and submission_rate > 0.8:
                st.success("✅ **High-performing cluster** - Maintain current support")
                st.markdown("- Acknowledge achievements\n- Provide advanced materials\n- Encourage peer mentoring")
            elif submission_rate < 0.7:
                st.info("ℹ️ **Moderate engagement** - Focus on improving assessment completion")
                st.markdown("- Send deadline reminders\n- Check for barriers to submission\n- Provide feedback on partial work")
            else:
                st.info("ℹ️ **Average performance cluster** - Standard support and monitoring")
                st.markdown("- Continue regular check-ins\n- Provide consistent feedback\n- Monitor for any decline")

    st.markdown("---")

    # Export cluster data
    st.header("📥 Export Cluster Data")

    col1, col2 = st.columns(2)

    with col1:
        # Export selected cluster
        if selected_cluster is not None:
            cluster_csv = cluster_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download Cluster {selected_cluster} Data",
                data=cluster_csv,
                file_name=f"cluster_{selected_cluster}_data.csv",
                mime="text/csv"
            )

    with col2:
        # Export cluster profiles
        if comparison_features:
            profiles_csv = cluster_profiles.to_csv().encode('utf-8')
            st.download_button(
                label="Download Cluster Profiles",
                data=profiles_csv,
                file_name="cluster_profiles.csv",
                mime="text/csv"
            )
