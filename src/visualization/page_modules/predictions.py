"""
Predictions Page
Interactive student risk prediction and what-if analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from visualization.utils import (
    load_dataset,
    load_all_models,
    load_label_encoder,
    load_feature_names,
    prepare_features_for_prediction,
    get_risk_level,
    format_percentage
)
from visualization.config import CATEGORICAL_FEATURES, CLASS_COLORS, FEATURE_CATEGORIES
from visualization.i18n import t


def render():
    """Render the predictions page."""
    st.title(f"🎯 {t('predictions.title')}")
    st.markdown("---")

    # Load required data
    with st.spinner(t('predictions.loading')):
        models = load_all_models()
        label_encoder = load_label_encoder()
        feature_names = load_feature_names()
        df = load_dataset("clustered")

    if not models or label_encoder is None or feature_names is None or df.empty:
        st.error(f"❌ {t('predictions.unable_to_load_models')}")
        return

    # Tabs for different prediction modes
    tab1, tab2, tab3 = st.tabs([
        t('predictions.tabs.manual'),
        t('predictions.tabs.batch'),
        t('predictions.tabs.whatif')
    ])

    # Tab 1: Manual Input Prediction
    with tab1:
        render_manual_prediction(models, label_encoder, feature_names, df)

    # Tab 2: Batch Prediction
    with tab2:
        render_batch_prediction(models, label_encoder, feature_names, df)

    # Tab 3: What-If Analysis
    with tab3:
        render_whatif_analysis(models, label_encoder, feature_names, df)


def render_manual_prediction(models, label_encoder, feature_names, df):
    """Render manual input prediction interface."""
    st.header(f"📝 {t('predictions.manual_title')}")
    st.markdown(t('predictions.manual_desc'))

    # Model selection
    model_options = {
        t('predictions.models.lightgbm_best'): "lightgbm",
        t('predictions.models.xgboost'): "xgboost",
        t('predictions.models.random_forest'): "random_forest",
        t('predictions.models.decision_tree'): "decision_tree"
    }

    selected_model_name = st.selectbox(
        t('predictions.select_model'),
        options=list(model_options.keys()),
        help=t('predictions.model_help')
    )
    selected_model = models[model_options[selected_model_name]]

    st.markdown("---")

    # Create input form organized by categories
    st.subheader(t('predictions.student_info'))

    input_data = {}

    # Demographics
    with st.expander(f"👤 {t('predictions.demographics')}", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            input_data["gender"] = st.selectbox(t('predictions.form.gender'), options=CATEGORICAL_FEATURES["gender"])
            input_data["age_band"] = st.selectbox(t('predictions.form.age_band'), options=CATEGORICAL_FEATURES["age_band"])

        with col2:
            input_data["region"] = st.selectbox(t('predictions.form.region'), options=CATEGORICAL_FEATURES["region"])
            input_data["disability"] = st.selectbox(t('predictions.form.disability'), options=CATEGORICAL_FEATURES["disability"])

        with col3:
            input_data["highest_education"] = st.selectbox(
                t('predictions.form.highest_education'),
                options=CATEGORICAL_FEATURES["highest_education"]
            )
            input_data["imd_band"] = st.selectbox(
                t('predictions.form.imd_band'),
                options=CATEGORICAL_FEATURES["imd_band"]
            )

        col4, col5 = st.columns(2)
        with col4:
            input_data["num_of_prev_attempts"] = st.number_input(
                t('predictions.form.previous_attempts'),
                min_value=0,
                max_value=6,
                value=0,
                help=t('predictions.form.previous_attempts_help')
            )

        with col5:
            input_data["studied_credits"] = st.number_input(
                t('predictions.form.studied_credits'),
                min_value=0,
                max_value=360,
                value=60,
                step=30,
                help=t('predictions.form.studied_credits_help')
            )

    # VLE Activity
    with st.expander(f"💻 {t('predictions.vle_activity')}", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            input_data["total_clicks"] = st.number_input(
                t('predictions.form.total_vle_clicks'),
                min_value=0,
                max_value=50000,
                value=1000,
                step=100,
                help=t('predictions.form.total_vle_clicks_help')
            )

        with col2:
            input_data["num_unique_activities"] = st.number_input(
                t('predictions.form.unique_activities'),
                min_value=0,
                max_value=100,
                value=20,
                help=t('predictions.form.unique_activities_help')
            )

        with col3:
            input_data["clicks_per_activity"] = st.number_input(
                t('predictions.form.clicks_per_activity'),
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                help=t('predictions.form.clicks_per_activity_help')
            )

        # Specific VLE components
        st.markdown(t('predictions.form.specific_resources_title'))
        col1, col2, col3, col4 = st.columns(4)

        vle_components = [
            (t('predictions.form.resource_homepage'), "clicks_homepage"),
            (t('predictions.form.resource_content'), "clicks_oucontent"),
            (t('predictions.form.resource_resources'), "clicks_resource"),
            (t('predictions.form.resource_forum'), "clicks_forumng"),
            (t('predictions.form.resource_quiz'), "clicks_quiz"),
            (t('predictions.form.resource_subpage'), "clicks_subpage"),
            (t('predictions.form.resource_url'), "clicks_url"),
            (t('predictions.form.resource_wiki'), "clicks_ouwiki")
        ]

        for idx, (label, key) in enumerate(vle_components):
            col = [col1, col2, col3, col4][idx % 4]
            with col:
                input_data[key] = st.number_input(
                    label,
                    min_value=0,
                    max_value=10000,
                    value=100,
                    key=key
                )

        # Set other VLE clicks to 0 (for features not in form)
        other_vle_features = [
            "clicks_dataplus", "clicks_dualpane", "clicks_externalquiz",
            "clicks_folder", "clicks_glossary", "clicks_htmlactivity",
            "clicks_oucollaborate", "clicks_ouelluminate", "clicks_page",
            "clicks_questionnaire", "clicks_repeatactivity", "clicks_sharedsubpage"
        ]
        for feat in other_vle_features:
            input_data[feat] = 0

    # Assessment Performance
    with st.expander(f"📊 {t('predictions.assessment_performance')}", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            input_data["avg_assessment_score"] = st.slider(
                t('predictions.form.avg_assessment_score'),
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                help=t('predictions.form.avg_assessment_score_help')
            )

        with col2:
            input_data["total_assessments"] = st.number_input(
                t('predictions.form.total_assessments'),
                min_value=0,
                max_value=20,
                value=6
            )

        with col3:
            input_data["num_assessments_submitted"] = st.number_input(
                t('predictions.form.assessments_submitted'),
                min_value=0,
                max_value=20,
                value=5
            )

        # Calculate submission rate
        if input_data["total_assessments"] > 0:
            input_data["assessment_submission_rate"] = input_data["num_assessments_submitted"] / input_data["total_assessments"]
        else:
            input_data["assessment_submission_rate"] = 0.0

        col1, col2, col3 = st.columns(3)

        with col1:
            input_data["avg_score_TMA"] = st.slider(
                t('predictions.form.avg_tma_score'),
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                help=t('predictions.form.avg_tma_score_help')
            )

        with col2:
            input_data["avg_score_CMA"] = st.slider(
                t('predictions.form.avg_cma_score'),
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                help=t('predictions.form.avg_cma_score_help')
            )

        with col3:
            input_data["avg_score_Exam"] = st.slider(
                t('predictions.form.exam_score'),
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                help=t('predictions.form.exam_score_help')
            )

    # Registration info
    with st.expander(f"📝 {t('predictions.registration')}", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            input_data["did_unregister"] = st.selectbox(
                t('predictions.form.did_unregister'),
                options=[0, 1],
                format_func=lambda x: t('common.no') if x == 0 else t('common.yes')
            )

        with col2:
            input_data["date_unregistration"] = st.number_input(
                t('predictions.form.date_unregistration'),
                min_value=0,
                max_value=1000,
                value=999,
                help=t('predictions.form.date_unregistration_help')
            )

    # Clustering (use median values)
    input_data["kmeans_cluster"] = int(df["kmeans_cluster"].median())
    input_data["dbscan_cluster"] = int(df["dbscan_cluster"].median())

    # Binary gender and disability encoding
    input_data["gender_M"] = 1 if input_data["gender"] == "M" else 0
    input_data["disability_Y"] = 1 if input_data["disability"] == "Y" else 0

    st.markdown("---")

    # Predict button
    if st.button(f"🔮 {t('predictions.predict_button')}", type="primary", use_container_width=True):
        with st.spinner(t('predictions.running_prediction')):
            try:
                # Prepare features
                X_input = prepare_features_for_prediction(input_data, feature_names, df)

                # Make prediction
                prediction_encoded = selected_model.predict(X_input)[0]
                prediction_proba = selected_model.predict_proba(X_input)[0]

                # Decode prediction
                prediction = label_encoder.inverse_transform([prediction_encoded])[0]
                risk_level, risk_color = get_risk_level(prediction)

                # Display results
                st.success(f"✅ {t('predictions.prediction_complete')}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label=t('predictions.result.predicted_outcome'),
                        value=prediction,
                        help=t('predictions.result.predicted_outcome_help')
                    )

                with col2:
                    st.metric(
                        label=t('predictions.result.risk_level'),
                        value=risk_level,
                        help=t('predictions.result.risk_level_help')
                    )

                with col3:
                    confidence = prediction_proba[prediction_encoded] * 100
                    st.metric(
                        label=t('predictions.result.confidence'),
                        value=f"{confidence:.1f}%",
                        help=t('predictions.result.confidence_help')
                    )

                # Probability distribution
                st.subheader(t('predictions.result.probabilities_title'))

                prob_df = pd.DataFrame({
                    t('predictions.result.outcome_label'): label_encoder.classes_,
                    t('predictions.result.probability_label'): prediction_proba * 100
                })

                fig = px.bar(
                    prob_df,
                    x=t('predictions.result.outcome_label'),
                    y=t('predictions.result.probability_label'),
                    color=t('predictions.result.outcome_label'),
                    color_discrete_map=CLASS_COLORS,
                    title=t('predictions.result.probability_distribution')
                )
                fig.update_layout(showlegend=False, height=400)
                fig.update_yaxes(title=t('predictions.result.probability_percent'))
                st.plotly_chart(fig, use_container_width=True)

                # Recommendations based on prediction
                st.subheader(t('predictions.recommendations_section.title'))

                if prediction in ["Fail", "Withdrawn"]:
                    st.error(t('predictions.recommendations_section.high_risk'))
                    st.markdown(f"""
                    {t('predictions.recommendations_section.suggested_actions')}
                    - {t('predictions.recommendations_section.action_1')}
                    - {t('predictions.recommendations_section.action_2')}
                    - {t('predictions.recommendations_section.action_3')}
                    - {t('predictions.recommendations_section.action_4')}
                    - {t('predictions.recommendations_section.action_5')}
                    """)
                elif prediction == "Pass":
                    st.info(t('predictions.recommendations_section.moderate'))
                    st.markdown(f"""
                    {t('predictions.recommendations_section.suggested_actions')}
                    - {t('predictions.recommendations_section.action_6')}
                    - {t('predictions.recommendations_section.action_7')}
                    - {t('predictions.recommendations_section.action_8')}
                    - {t('predictions.recommendations_section.action_9')}
                    """)
                else:  # Distinction
                    st.success(t('predictions.recommendations_section.high_performance'))
                    st.markdown(f"""
                    {t('predictions.recommendations_section.suggested_actions')}
                    - {t('predictions.recommendations_section.action_10')}
                    - {t('predictions.recommendations_section.action_11')}
                    - {t('predictions.recommendations_section.action_12')}
                    - {t('predictions.recommendations_section.action_13')}
                    """)

            except Exception as e:
                st.error(t('predictions.prediction_error').format(e=str(e)))


def render_batch_prediction(models, label_encoder, feature_names, df):
    """Render batch prediction interface."""
    st.header(f"📊 {t('predictions.batch_title')}")
    st.markdown(t('predictions.batch.description'))

    # Data source selection
    data_source = st.radio(
        t('predictions.batch.select_source'),
        options=[t('predictions.batch.use_existing'), t('predictions.batch.upload_csv')],
        horizontal=True
    )

    if data_source == t('predictions.batch.upload_csv'):
        uploaded_file = st.file_uploader(
            t('predictions.batch.upload_label'),
            type=["csv"],
            help=t('predictions.batch.upload_help')
        )

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(t('predictions.batch.loaded_records').format(count=len(batch_df)))
        else:
            st.info(t('predictions.batch.upload_prompt'))
            return
    else:
        # Use existing data - sample for demonstration
        sample_size = st.slider(t('predictions.batch.sample_size'), min_value=10, max_value=1000, value=100)
        batch_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Model selection
    model_choice = st.selectbox(
        t('predictions.batch.select_model'),
        options=["lightgbm", "xgboost", "random_forest", "decision_tree"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    if st.button(t('predictions.batch.run_button'), type="primary"):
        with st.spinner(t('predictions.batch.spinner').format(count=len(batch_df))):
            try:
                # Prepare features by encoding each row
                X_batch_list = []

                for idx in range(len(batch_df)):
                    student_dict = batch_df.iloc[idx].to_dict()
                    X_student = prepare_features_for_prediction(student_dict, feature_names, df)
                    X_batch_list.append(X_student[0])  # Extract the single row

                X_batch = np.array(X_batch_list)

                # Make predictions
                model = models[model_choice]
                predictions_encoded = model.predict(X_batch)
                predictions_proba = model.predict_proba(X_batch)

                # Decode predictions
                predictions = label_encoder.inverse_transform(predictions_encoded)
                confidences = predictions_proba.max(axis=1)

                # Add predictions to dataframe
                results_df = batch_df.copy()
                results_df["predicted_outcome"] = predictions
                results_df["prediction_confidence"] = confidences

                # Display results summary
                st.success(t('predictions.batch.success_complete'))

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(t('predictions.batch.total_predicted'), len(predictions))

                with col2:
                    high_risk = sum(p in ["Fail", "Withdrawn"] for p in predictions)
                    st.metric(t('predictions.batch.high_risk'), high_risk, delta=f"{high_risk/len(predictions)*100:.1f}%")

                with col3:
                    avg_confidence = confidences.mean() * 100
                    st.metric(t('predictions.batch.avg_confidence'), f"{avg_confidence:.1f}%")

                with col4:
                    distinction = sum(p == "Distinction" for p in predictions)
                    st.metric(t('predictions.batch.distinction'), distinction)

                # Prediction distribution
                st.subheader(t('predictions.batch.distribution_title'))
                pred_counts = pd.Series(predictions).value_counts()

                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title=t('predictions.batch.distribution_chart'),
                    color=pred_counts.index,
                    color_discrete_map=CLASS_COLORS,
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show results table
                st.subheader(t('predictions.batch.results_title'))

                # Filter and display options
                col1, col2 = st.columns(2)

                with col1:
                    filter_outcome = st.multiselect(
                        t('predictions.batch.filter_outcome'),
                        options=list(label_encoder.classes_),
                        default=list(label_encoder.classes_)
                    )

                with col2:
                    min_confidence = st.slider(
                        t('predictions.batch.min_confidence'),
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05
                    )

                # Apply filters
                filtered_results = results_df[
                    (results_df["predicted_outcome"].isin(filter_outcome)) &
                    (results_df["prediction_confidence"] >= min_confidence)
                ]

                st.dataframe(
                    filtered_results[[
                        "predicted_outcome", "prediction_confidence",
                        "total_clicks", "avg_assessment_score", "assessment_submission_rate"
                    ]].head(100),
                    use_container_width=True
                )

                # Download results
                csv = filtered_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=t('predictions.batch.download_button'),
                    data=csv,
                    file_name=f"batch_predictions_{model_choice}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(t('predictions.batch.error_failed').format(e=e))


def render_whatif_analysis(models, label_encoder, feature_names, df):
    """Render what-if analysis interface."""
    st.header(f"🔮 {t('predictions.whatif_title')}")
    st.markdown(t('predictions.whatif.description'))

    st.info(t('predictions.whatif.tip'))

    # Select a baseline student
    st.subheader(t('predictions.whatif.step1_title'))

    col1, col2 = st.columns([1, 2])

    with col1:
        student_idx = st.number_input(
            t('predictions.whatif.student_index'),
            min_value=0,
            max_value=len(df)-1,
            value=0,
            help=t('predictions.whatif.student_index_help')
        )

    baseline_student = df.iloc[student_idx]

    with col2:
        if "final_result" in baseline_student:
            st.metric(t('predictions.whatif.actual_outcome'), baseline_student["final_result"])

    st.markdown("---")

    # Select model
    model = models["lightgbm"]  # Use best model

    # What-if scenarios
    st.subheader(t('predictions.whatif.step2_title'))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(t('predictions.whatif.engagement_title'))
        clicks_multiplier = st.slider(
            t('predictions.whatif.clicks_multiplier'),
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help=t('predictions.whatif.clicks_multiplier_help')
        )

    with col2:
        st.markdown(t('predictions.whatif.assessment_title'))
        assessment_boost = st.slider(
            t('predictions.whatif.assessment_boost'),
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=5.0,
            help=t('predictions.whatif.assessment_boost_help')
        )

    with col3:
        st.markdown(t('predictions.whatif.submission_title'))
        submission_boost = st.slider(
            t('predictions.whatif.submission_boost'),
            min_value=-0.5,
            max_value=0.5,
            value=0.0,
            step=0.1,
            help=t('predictions.whatif.submission_boost_help')
        )

    # Apply what-if changes
    modified_student = baseline_student.copy()

    # Adjust clicks
    click_cols = [col for col in df.columns if "clicks" in col and col in feature_names]
    for col in click_cols:
        if col in modified_student.index:
            modified_student[col] = modified_student[col] * clicks_multiplier

    # Adjust assessment scores
    score_cols = ["avg_assessment_score", "avg_score_TMA", "avg_score_CMA", "avg_score_Exam"]
    for col in score_cols:
        if col in modified_student.index:
            modified_student[col] = np.clip(modified_student[col] + assessment_boost, 0, 100)

    # Adjust submission rate
    if "assessment_submission_rate" in modified_student.index:
        modified_student["assessment_submission_rate"] = np.clip(
            modified_student["assessment_submission_rate"] + submission_boost,
            0, 1
        )

    st.markdown("---")

    # Run predictions
    st.subheader(t('predictions.whatif.step3_title'))

    col1, col2 = st.columns(2)

    try:
        # Convert Series to dict for prepare_features_for_prediction
        baseline_dict = baseline_student.to_dict()
        modified_dict = modified_student.to_dict()

        # Baseline prediction
        X_baseline = prepare_features_for_prediction(baseline_dict, feature_names, df)

        pred_baseline_encoded = model.predict(X_baseline)[0]
        pred_baseline_proba = model.predict_proba(X_baseline)[0]
        pred_baseline = label_encoder.inverse_transform([pred_baseline_encoded])[0]

        # Modified prediction
        X_modified = prepare_features_for_prediction(modified_dict, feature_names, df)

        pred_modified_encoded = model.predict(X_modified)[0]
        pred_modified_proba = model.predict_proba(X_modified)[0]
        pred_modified = label_encoder.inverse_transform([pred_modified_encoded])[0]

        with col1:
            st.markdown(t('predictions.whatif.baseline_title'))
            st.metric(t('predictions.whatif.outcome'), pred_baseline)
            st.metric(t('predictions.whatif.confidence'), format_percentage(pred_baseline_proba[pred_baseline_encoded]))

            # Probability chart
            fig1 = go.Figure(data=[
                go.Bar(
                    x=label_encoder.classes_,
                    y=pred_baseline_proba,
                    marker_color=[CLASS_COLORS.get(c, "#999") for c in label_encoder.classes_]
                )
            ])
            fig1.update_layout(title=t('predictions.whatif.baseline_probabilities'), height=300, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown(t('predictions.whatif.modified_title'))
            delta_color = "normal"
            if pred_modified != pred_baseline:
                delta_color = "inverse" if pred_modified in ["Fail", "Withdrawn"] else "normal"

            st.metric(t('predictions.whatif.outcome'), pred_modified, delta=None if pred_modified == pred_baseline else t('predictions.whatif.changed'))
            st.metric(t('predictions.whatif.confidence'), format_percentage(pred_modified_proba[pred_modified_encoded]))

            # Probability chart
            fig2 = go.Figure(data=[
                go.Bar(
                    x=label_encoder.classes_,
                    y=pred_modified_proba,
                    marker_color=[CLASS_COLORS.get(c, "#999") for c in label_encoder.classes_]
                )
            ])
            fig2.update_layout(title=t('predictions.whatif.modified_probabilities'), height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Impact analysis
        st.markdown("---")
        st.subheader(t('predictions.whatif.impact_title'))

        if pred_baseline != pred_modified:
            st.success(t('predictions.whatif.outcome_changed').format(from_val=pred_baseline, to_val=pred_modified))
        else:
            st.info(t('predictions.whatif.no_change').format(val=pred_baseline))

        # Show probability changes
        prob_changes = pd.DataFrame({
            t('predictions.whatif.outcome'): label_encoder.classes_,
            t('predictions.whatif.baseline'): pred_baseline_proba * 100,
            t('predictions.whatif.modified'): pred_modified_proba * 100,
            t('predictions.whatif.change'): (pred_modified_proba - pred_baseline_proba) * 100
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name=t('predictions.whatif.baseline'), x=prob_changes[t('predictions.whatif.outcome')], y=prob_changes[t('predictions.whatif.baseline')],
                            marker_color="lightblue"))
        fig.add_trace(go.Bar(name=t('predictions.whatif.modified'), x=prob_changes[t('predictions.whatif.outcome')], y=prob_changes[t('predictions.whatif.modified')],
                            marker_color="darkblue"))
        fig.update_layout(title=t('predictions.whatif.prob_comparison'), barmode="group", height=400)
        fig.update_yaxes(title=t('predictions.result.probability_percent'))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(prob_changes, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(t('predictions.whatif.error_failed').format(e=e))
