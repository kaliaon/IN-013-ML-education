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
    tab1, tab2, tab3 = st.tabs(["📝 Manual Input", "📊 Batch Prediction", "🔮 What-If Analysis"])

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
        "LightGBM (Best)": "lightgbm",
        "XGBoost": "xgboost",
        "Random Forest": "random_forest",
        "Decision Tree": "decision_tree"
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
            input_data["gender"] = st.selectbox("Gender", options=CATEGORICAL_FEATURES["gender"])
            input_data["age_band"] = st.selectbox("Age Band", options=CATEGORICAL_FEATURES["age_band"])

        with col2:
            input_data["region"] = st.selectbox("Region", options=CATEGORICAL_FEATURES["region"])
            input_data["disability"] = st.selectbox("Disability", options=CATEGORICAL_FEATURES["disability"])

        with col3:
            input_data["highest_education"] = st.selectbox(
                "Highest Education",
                options=CATEGORICAL_FEATURES["highest_education"]
            )
            input_data["imd_band"] = st.selectbox(
                "IMD Band (Deprivation Index)",
                options=CATEGORICAL_FEATURES["imd_band"]
            )

        col4, col5 = st.columns(2)
        with col4:
            input_data["num_of_prev_attempts"] = st.number_input(
                "Previous Attempts",
                min_value=0,
                max_value=6,
                value=0,
                help="Number of previous attempts at this module"
            )

        with col5:
            input_data["studied_credits"] = st.number_input(
                "Studied Credits",
                min_value=0,
                max_value=360,
                value=60,
                step=30,
                help="Total credits being studied"
            )

    # VLE Activity
    with st.expander(f"💻 {t('predictions.vle_activity')}", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            input_data["total_clicks"] = st.number_input(
                "Total VLE Clicks",
                min_value=0,
                max_value=50000,
                value=1000,
                step=100,
                help="Total interactions with VLE resources"
            )

        with col2:
            input_data["num_unique_activities"] = st.number_input(
                "Unique Activities",
                min_value=0,
                max_value=100,
                value=20,
                help="Number of different activity types accessed"
            )

        with col3:
            input_data["clicks_per_activity"] = st.number_input(
                "Clicks per Activity",
                min_value=0.0,
                max_value=1000.0,
                value=50.0,
                help="Average clicks per unique activity"
            )

        # Specific VLE components
        st.markdown("**Specific Resource Interactions:**")
        col1, col2, col3, col4 = st.columns(4)

        vle_components = [
            ("Homepage", "clicks_homepage"),
            ("Content", "clicks_oucontent"),
            ("Resources", "clicks_resource"),
            ("Forum", "clicks_forumng"),
            ("Quiz", "clicks_quiz"),
            ("Subpage", "clicks_subpage"),
            ("URL", "clicks_url"),
            ("Wiki", "clicks_ouwiki")
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
                "Avg Assessment Score",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                help="Average score across all assessments"
            )

        with col2:
            input_data["total_assessments"] = st.number_input(
                "Total Assessments",
                min_value=0,
                max_value=20,
                value=6
            )

        with col3:
            input_data["num_assessments_submitted"] = st.number_input(
                "Assessments Submitted",
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
                "Avg TMA Score",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                help="Average on Tutor Marked Assessments"
            )

        with col2:
            input_data["avg_score_CMA"] = st.slider(
                "Avg CMA Score",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                help="Average on Computer Marked Assessments"
            )

        with col3:
            input_data["avg_score_Exam"] = st.slider(
                "Exam Score",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                help="Final exam score"
            )

    # Registration info
    with st.expander(f"📝 {t('predictions.registration')}", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            input_data["did_unregister"] = st.selectbox(
                "Did Unregister?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        with col2:
            input_data["date_unregistration"] = st.number_input(
                "Date Unregistration",
                min_value=0,
                max_value=1000,
                value=999,
                help="Days until unregistration (999 = never unregistered)"
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
        with st.spinner("Running prediction..."):
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
                        label="Predicted Outcome",
                        value=prediction,
                        help="Most likely final result"
                    )

                with col2:
                    st.metric(
                        label="Risk Level",
                        value=risk_level,
                        help="Student risk category"
                    )

                with col3:
                    confidence = prediction_proba[prediction_encoded] * 100
                    st.metric(
                        label="Confidence",
                        value=f"{confidence:.1f}%",
                        help="Model confidence in prediction"
                    )

                # Probability distribution
                st.subheader("Prediction Probabilities")

                prob_df = pd.DataFrame({
                    "Outcome": label_encoder.classes_,
                    "Probability": prediction_proba * 100
                })

                fig = px.bar(
                    prob_df,
                    x="Outcome",
                    y="Probability",
                    color="Outcome",
                    color_discrete_map=CLASS_COLORS,
                    title="Probability Distribution Across Outcomes"
                )
                fig.update_layout(showlegend=False, height=400)
                fig.update_yaxes(title="Probability (%)")
                st.plotly_chart(fig, use_container_width=True)

                # Recommendations based on prediction
                st.subheader("💡 Recommendations")

                if prediction in ["Fail", "Withdrawn"]:
                    st.error(f"⚠️ **High Risk Student** - Immediate intervention recommended")
                    st.markdown("""
                    **Suggested Actions:**
                    - Schedule one-on-one meeting with tutor
                    - Review assessment submission patterns
                    - Check VLE engagement levels
                    - Offer additional support resources
                    - Monitor progress closely
                    """)
                elif prediction == "Pass":
                    st.info("ℹ️ **Moderate Performance** - Monitoring recommended")
                    st.markdown("""
                    **Suggested Actions:**
                    - Encourage continued engagement
                    - Provide feedback on assessments
                    - Suggest peer study groups
                    - Monitor for any decline in activity
                    """)
                else:  # Distinction
                    st.success("✅ **High Performance** - On track for excellence")
                    st.markdown("""
                    **Suggested Actions:**
                    - Acknowledge strong performance
                    - Offer advanced learning materials
                    - Encourage peer mentoring
                    - Maintain current support level
                    """)

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")


def render_batch_prediction(models, label_encoder, feature_names, df):
    """Render batch prediction interface."""
    st.header(f"📊 {t('predictions.batch_title')}")
    st.markdown("Upload a CSV file or use existing data to predict multiple students at once.")

    # Data source selection
    data_source = st.radio(
        "Select data source:",
        options=["Use existing dataset", "Upload CSV file"],
        horizontal=True
    )

    if data_source == "Upload CSV file":
        uploaded_file = st.file_uploader(
            "Upload student data (CSV)",
            type=["csv"],
            help="CSV should contain the same features as training data"
        )

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_df)} records")
        else:
            st.info("Please upload a CSV file to continue.")
            return
    else:
        # Use existing data - sample for demonstration
        sample_size = st.slider("Number of students to predict:", min_value=10, max_value=1000, value=100)
        batch_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Model selection
    model_choice = st.selectbox(
        "Select model:",
        options=["lightgbm", "xgboost", "random_forest", "decision_tree"],
        format_func=lambda x: x.replace("_", " ").title()
    )

    if st.button("🚀 Run Batch Prediction", type="primary"):
        with st.spinner(f"Predicting {len(batch_df)} students..."):
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
                st.success("✅ Batch prediction complete!")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Predicted", len(predictions))

                with col2:
                    high_risk = sum(p in ["Fail", "Withdrawn"] for p in predictions)
                    st.metric("High Risk", high_risk, delta=f"{high_risk/len(predictions)*100:.1f}%")

                with col3:
                    avg_confidence = confidences.mean() * 100
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")

                with col4:
                    distinction = sum(p == "Distinction" for p in predictions)
                    st.metric("Distinction", distinction)

                # Prediction distribution
                st.subheader("Prediction Distribution")
                pred_counts = pd.Series(predictions).value_counts()

                fig = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Predicted Outcomes Distribution",
                    color=pred_counts.index,
                    color_discrete_map=CLASS_COLORS,
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show results table
                st.subheader("Prediction Results")

                # Filter and display options
                col1, col2 = st.columns(2)

                with col1:
                    filter_outcome = st.multiselect(
                        "Filter by predicted outcome:",
                        options=list(label_encoder.classes_),
                        default=list(label_encoder.classes_)
                    )

                with col2:
                    min_confidence = st.slider(
                        "Minimum confidence:",
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
                    label="📥 Download Predictions (CSV)",
                    data=csv,
                    file_name=f"batch_predictions_{model_choice}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Batch prediction failed: {e}")


def render_whatif_analysis(models, label_encoder, feature_names, df):
    """Render what-if analysis interface."""
    st.header(f"🔮 {t('predictions.whatif_title')}")
    st.markdown("Explore how changes in student behavior affect predictions.")

    st.info("💡 **Tip:** Adjust the sliders to see how different factors impact the predicted outcome.")

    # Select a baseline student
    st.subheader("1️⃣ Select Baseline Student")

    col1, col2 = st.columns([1, 2])

    with col1:
        student_idx = st.number_input(
            "Student Index",
            min_value=0,
            max_value=len(df)-1,
            value=0,
            help="Select a student from the dataset as baseline"
        )

    baseline_student = df.iloc[student_idx]

    with col2:
        if "final_result" in baseline_student:
            st.metric("Actual Outcome", baseline_student["final_result"])

    st.markdown("---")

    # Select model
    model = models["lightgbm"]  # Use best model

    # What-if scenarios
    st.subheader("2️⃣ Adjust Key Factors")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📚 Engagement**")
        clicks_multiplier = st.slider(
            "VLE Clicks Multiplier",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Multiply baseline VLE clicks"
        )

    with col2:
        st.markdown("**✍️ Assessment**")
        assessment_boost = st.slider(
            "Assessment Score Boost",
            min_value=-30.0,
            max_value=30.0,
            value=0.0,
            step=5.0,
            help="Add/subtract from baseline scores"
        )

    with col3:
        st.markdown("**📝 Submission**")
        submission_boost = st.slider(
            "Submission Rate Boost",
            min_value=-0.5,
            max_value=0.5,
            value=0.0,
            step=0.1,
            help="Add/subtract from submission rate"
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
    st.subheader("3️⃣ Compare Predictions")

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
            st.markdown("**📊 Baseline Prediction**")
            st.metric("Outcome", pred_baseline)
            st.metric("Confidence", format_percentage(pred_baseline_proba[pred_baseline_encoded]))

            # Probability chart
            fig1 = go.Figure(data=[
                go.Bar(
                    x=label_encoder.classes_,
                    y=pred_baseline_proba,
                    marker_color=[CLASS_COLORS.get(c, "#999") for c in label_encoder.classes_]
                )
            ])
            fig1.update_layout(title="Baseline Probabilities", height=300, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("**🔮 Modified Prediction**")
            delta_color = "normal"
            if pred_modified != pred_baseline:
                delta_color = "inverse" if pred_modified in ["Fail", "Withdrawn"] else "normal"

            st.metric("Outcome", pred_modified, delta=None if pred_modified == pred_baseline else "Changed")
            st.metric("Confidence", format_percentage(pred_modified_proba[pred_modified_encoded]))

            # Probability chart
            fig2 = go.Figure(data=[
                go.Bar(
                    x=label_encoder.classes_,
                    y=pred_modified_proba,
                    marker_color=[CLASS_COLORS.get(c, "#999") for c in label_encoder.classes_]
                )
            ])
            fig2.update_layout(title="Modified Probabilities", height=300, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Impact analysis
        st.markdown("---")
        st.subheader("📊 Impact Analysis")

        if pred_baseline != pred_modified:
            st.success(f"✨ **Outcome Changed:** {pred_baseline} → {pred_modified}")
        else:
            st.info(f"ℹ️ **No Change:** Prediction remains {pred_baseline}")

        # Show probability changes
        prob_changes = pd.DataFrame({
            "Outcome": label_encoder.classes_,
            "Baseline": pred_baseline_proba * 100,
            "Modified": pred_modified_proba * 100,
            "Change": (pred_modified_proba - pred_baseline_proba) * 100
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Baseline", x=prob_changes["Outcome"], y=prob_changes["Baseline"],
                            marker_color="lightblue"))
        fig.add_trace(go.Bar(name="Modified", x=prob_changes["Outcome"], y=prob_changes["Modified"],
                            marker_color="darkblue"))
        fig.update_layout(title="Probability Comparison", barmode="group", height=400)
        fig.update_yaxes(title="Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(prob_changes, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ What-if analysis failed: {e}")
