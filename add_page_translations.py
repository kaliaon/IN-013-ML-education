#!/usr/bin/env python3
"""
Script to add comprehensive translation keys for all dashboard pages.
This will extend the existing translation files with all necessary keys.
"""

import json
from pathlib import Path

# Define all translation keys needed for the remaining pages
ADDITIONAL_KEYS = {
    "predictions": {
        # Existing keys preserved, adding new ones
        "title": "Student Performance Prediction",
        "loading": "Loading models and data...",
        "unable_to_load_models": "Unable to load required models or data. Please check model files.",

        # Tabs
        "tab_manual": "Manual Input",
        "tab_batch": "Batch Prediction",
        "tab_whatif": "What-If Analysis",

        # Manual prediction
        "manual_title": "Predict Individual Student Performance",
        "manual_desc": "Enter student characteristics to predict their likely outcome.",
        "select_model": "Select prediction model:",
        "model_help": "LightGBM achieved the best performance (88.83% accuracy)",
        "student_info": "Student Information",

        # Demographics
        "demographics": "Demographics",
        "gender": "Gender",
        "age_band": "Age Band",
        "region": "Region",
        "disability": "Disability",
        "highest_education": "Highest Education",
        "imd_band": "IMD Band",

        # VLE Activity
        "vle_activity": "VLE Activity",
        "total_clicks": "Total Clicks",
        "specific_resources": "Specific Resource Interactions:",

        # Assessment
        "assessment_performance": "Assessment Performance",
        "avg_score": "Avg Score",
        "submission_rate": "Submission Rate",

        # Registration
        "registration": "Registration",
        "did_unregister": "Did Unregister",
        "num_prev_attempts": "Num Prev Attempts",

        # Prediction button and results
        "predict_button": "Predict Performance",
        "prediction_complete": "Prediction Complete!",
        "predicted_outcome": "Predicted Outcome",
        "confidence": "Confidence",
        "prediction_probabilities": "Prediction Probabilities",
        "recommendations": "Recommendations",
        "high_risk_title": "High Risk Student",
        "high_risk_desc": "Immediate intervention recommended",
        "moderate_perf_title": "Moderate Performance",
        "moderate_perf_desc": "Monitoring recommended",
        "high_perf_title": "High Performance",
        "high_perf_desc": "On track for excellence",
        "prediction_failed": "Prediction failed",

        # Batch prediction
        "batch_title": "Batch Student Prediction",
        "batch_desc": "Upload a CSV file or use existing data to predict multiple students at once.",
        "upload_csv": "Upload CSV File",
        "use_sample": "Or use sample data:",
        "loaded_records": "Loaded {count} records",
        "upload_file_prompt": "Please upload a CSV file to continue.",
        "select_model_batch": "Select model for batch prediction:",
        "run_batch_button": "Run Batch Prediction",
        "batch_complete": "Batch prediction complete!",
        "prediction_distribution": "Prediction Distribution",
        "prediction_results": "Prediction Results",
        "download_results": "Download Results (CSV)",
        "batch_failed": "Batch prediction failed",

        # What-if analysis
        "whatif_title": "What-If Analysis",
        "whatif_desc": "Explore how changes in student behavior affect predictions.",
        "whatif_tip": "Tip: Adjust the sliders to see how different factors impact the predicted outcome.",
        "select_baseline": "Select Baseline Student",
        "adjust_factors": "Adjust Key Factors",
        "engagement": "Engagement",
        "assessment": "Assessment",
        "baseline_prediction": "Baseline Prediction",
        "modified_prediction": "Modified Prediction",
        "impact_analysis": "Impact Analysis",
        "outcome_changed": "Outcome changed from {from_outcome} to {to_outcome}",
        "outcome_unchanged": "Outcome remains {outcome}",
        "confidence_change": "Confidence change: {change}",
        "whatif_failed": "What-if analysis failed"
    },

    "clustering": {
        "title": "Student Clustering Analysis",
        "loading": "Loading clustering data...",
        "unable_to_load": "Unable to load clustering data. Please check data files.",
        "overview": "Clustering Overview",
        "algorithm": "Algorithm",
        "num_clusters": "Number of Clusters",
        "silhouette_score": "Silhouette Score",
        "distribution": "Cluster Distribution",
        "visualization": "Cluster Visualization",
        "method": "Visualization Method",
        "pca": "PCA (Principal Component Analysis)",
        "tsne": "t-SNE (t-Distributed Stochastic Neighbor Embedding)",
        "cluster_profiles": "Cluster Profiles",
        "select_cluster": "Select cluster to analyze:",
        "cluster_size": "Cluster Size",
        "percentage": "Percentage",
        "dominant_outcome": "Dominant Outcome",
        "avg_metrics": "Average Metrics",
        "characteristics": "Key Characteristics",
        "comparison": "Cluster Comparison",
        "insights": "Cluster Insights & Recommendations",
        "export_data": "Export Cluster Data",
        "download_clusters": "Download Cluster Data (CSV)"
    },

    "performance": {
        "title": "Model Performance Comparison",
        "loading": "Loading model performance data...",
        "unable_to_load": "Unable to load model performance data. Please check files.",
        "overview": "Performance Overview",
        "best_model": "Best Model",
        "metrics_comparison": "Performance Metrics Comparison",
        "select_metric": "Select metric to compare:",
        "metric_charts": "Interactive Metric Comparison",
        "radar_chart": "Multi-Metric Radar Chart",
        "confusion_matrices": "Confusion Matrices",
        "select_model": "Select model:",
        "roc_curves": "ROC Curves",
        "complexity": "Model Complexity Analysis",
        "training_time": "Training Time",
        "prediction_time": "Prediction Time",
        "model_size": "Model Size",
        "recommendations": "Model Selection Recommendations",
        "insights": "Key Performance Insights",
        "export_metrics": "Export Metrics (CSV)"
    },

    "importance": {
        "title": "Feature Importance Analysis",
        "loading": "Loading feature importance data...",
        "unable_to_load": "Unable to load feature importance data. Please check files.",
        "overview": "Feature Importance Overview",
        "top_features": "Top Important Features",
        "num_features": "Number of features to display:",
        "by_category": "Feature Importance by Category",
        "correlations": "Feature Correlations",
        "with_target": "Correlation with Target",
        "explorer": "Interactive Feature Explorer",
        "select_feature": "Select feature to explore:",
        "distribution": "Feature Distribution",
        "by_outcome": "Distribution by Outcome",
        "complete_table": "Complete Feature Importance Table",
        "export_data": "Export Feature Importance Data",
        "download_importance": "Download Importance (CSV)"
    }
}

# Kazakh translations
ADDITIONAL_KEYS_KK = {
    "predictions": {
        "title": "Студенттердің Үлгерімін Болжау",
        "loading": "Модельдер мен деректер жүктелуде...",
        "unable_to_load_models": "Қажетті модельдер немесе деректерді жүктеу мүмкін емес. Модель файлдарын тексеріңіз.",

        "tab_manual": "Қолмен Енгізу",
        "tab_batch": "Топтық Болжау",
        "tab_whatif": "\"Не Болса\" Талдауы",

        "manual_title": "Жеке Студенттің Үлгерімін Болжау",
        "manual_desc": "Студенттің мүмкін нәтижесін болжау үшін сипаттамаларын енгізіңіз.",
        "select_model": "Болжау үшін модельді таңдаңыз:",
        "model_help": "LightGBM ең жақсы нәтиже көрсетті (88.83% дәлдік)",
        "student_info": "Студент Туралы Ақпарат",

        "demographics": "Демография",
        "gender": "Жынысы",
        "age_band": "Жас Тобы",
        "region": "Аймақ",
        "disability": "Мүгедектік",
        "highest_education": "Ең Жоғары Білім",
        "imd_band": "IMD Тобы",

        "vle_activity": "VLE Белсенділігі",
        "total_clicks": "Барлық Басулар",
        "specific_resources": "Нақты Ресурстармен Өзара Әрекет:",

        "assessment_performance": "Бағалау Үлгерімі",
        "avg_score": "Орташа Ұпай",
        "submission_rate": "Тапсыру Жылдамдығы",

        "registration": "Тіркелу",
        "did_unregister": "Тіркелуден Шықты",
        "num_prev_attempts": "Алдыңғы Әрекеттер Саны",

        "predict_button": "Үлгерімді Болжау",
        "prediction_complete": "Болжау Аяқталды!",
        "predicted_outcome": "Болжанған Нәтиже",
        "confidence": "Сенімділік",
        "prediction_probabilities": "Болжау Ықтималдықтары",
        "recommendations": "Ұсыныстар",
        "high_risk_title": "Жоғары Тәуекелді Студент",
        "high_risk_desc": "Дереу араласу қажет",
        "moderate_perf_title": "Орташа Үлгерім",
        "moderate_perf_desc": "Бақылау ұсынылады",
        "high_perf_title": "Жоғары Үлгерім",
        "high_perf_desc": "Үздік нәтижеге бағыт алуда",
        "prediction_failed": "Болжау сәтсіз аяқталды",

        "batch_title": "Топтық Студенттерді Болжау",
        "batch_desc": "Бірнеше студентті бір мезгілде болжау үшін CSV файлын жүктеңіз немесе қолданыстағы деректерді пайдаланыңыз.",
        "upload_csv": "CSV Файлын Жүктеу",
        "use_sample": "Немесе үлгі деректерді пайдаланыңыз:",
        "loaded_records": "{count} жазба жүктелді",
        "upload_file_prompt": "Жалғастыру үшін CSV файлын жүктеңіз.",
        "select_model_batch": "Топтық болжау үшін модельді таңдаңыз:",
        "run_batch_button": "Топтық Болжауды Іске Қосу",
        "batch_complete": "Топтық болжау аяқталды!",
        "prediction_distribution": "Болжау Үлестірімі",
        "prediction_results": "Болжау Нәтижелері",
        "download_results": "Нәтижелерді Жүктеп Алу (CSV)",
        "batch_failed": "Топтық болжау сәтсіз аяқталды",

        "whatif_title": "\"Не Болса\" Талдауы",
        "whatif_desc": "Студенттің мінез-құлқындағы өзгерістер болжамға қалай әсер ететінін зерттеңіз.",
        "whatif_tip": "Кеңес: Әртүрлі факторлардың болжанған нәтижеге қалай әсер ететінін көру үшін жүгіргілерді реттеңіз.",
        "select_baseline": "Базалық Студентті Таңдау",
        "adjust_factors": "Негізгі Факторларды Реттеу",
        "engagement": "Қатысу",
        "assessment": "Бағалау",
        "baseline_prediction": "Базалық Болжам",
        "modified_prediction": "Өзгертілген Болжам",
        "impact_analysis": "Әсер Талдауы",
        "outcome_changed": "Нәтиже {from_outcome} деп өзгерді {to_outcome}",
        "outcome_unchanged": "Нәтиже {outcome} күйінде қалады",
        "confidence_change": "Сенімділік өзгерісі: {change}",
        "whatif_failed": "\"Не болса\" талдауы сәтсіз аяқталды"
    },

    "clustering": {
        "title": "Студенттерді Кластерлеу Талдауы",
        "loading": "Кластерлеу деректері жүктелуде...",
        "unable_to_load": "Кластерлеу деректерін жүктеу мүмкін емес. Деректер файлдарын тексеріңіз.",
        "overview": "Кластерлеу Шолуы",
        "algorithm": "Алгоритм",
        "num_clusters": "Кластерлер Саны",
        "silhouette_score": "Силуэт Көрсеткіші",
        "distribution": "Кластерлер Үлестірімі",
        "visualization": "Кластерлерді Визуализациялау",
        "method": "Визуализация Әдісі",
        "pca": "PCA (Басты Компоненттер Талдауы)",
        "tsne": "t-SNE (t-Үлестірілген Стохастикалық Көршілерді Ендіру)",
        "cluster_profiles": "Кластер Профильдері",
        "select_cluster": "Талдау үшін кластерді таңдаңыз:",
        "cluster_size": "Кластер Өлшемі",
        "percentage": "Пайыз",
        "dominant_outcome": "Басым Нәтиже",
        "avg_metrics": "Орташа Көрсеткіштер",
        "characteristics": "Негізгі Сипаттамалар",
        "comparison": "Кластерлерді Салыстыру",
        "insights": "Кластер Талдауы және Ұсыныстар",
        "export_data": "Кластер Деректерін Экспорттау",
        "download_clusters": "Кластер Деректерін Жүктеп Алу (CSV)"
    },

    "performance": {
        "title": "Модельдердің Өнімділігін Салыстыру",
        "loading": "Модель өнімділігі деректері жүктелуде...",
        "unable_to_load": "Модель өнімділігі деректерін жүктеу мүмкін емес. Файлдарды тексеріңіз.",
        "overview": "Өнімділік Шолуы",
        "best_model": "Ең Жақсы Модель",
        "metrics_comparison": "Өнімділік Көрсеткіштерін Салыстыру",
        "select_metric": "Салыстыру үшін көрсеткішті таңдаңыз:",
        "metric_charts": "Интерактивті Көрсеткіштерді Салыстыру",
        "radar_chart": "Көп Көрсеткішті Радар Кестесі",
        "confusion_matrices": "Шатасу Матрицалары",
        "select_model": "Модельді таңдаңыз:",
        "roc_curves": "ROC Қисықтары",
        "complexity": "Модель Күрделілігін Талдау",
        "training_time": "Үйрету Уақыты",
        "prediction_time": "Болжау Уақыты",
        "model_size": "Модель Өлшемі",
        "recommendations": "Модельді Таңдау Ұсыныстары",
        "insights": "Негізгі Өнімділік Түсініктері",
        "export_metrics": "Көрсеткіштерді Экспорттау (CSV)"
    },

    "importance": {
        "title": "Белгілердің Маңыздылығын Талдау",
        "loading": "Белгілердің маңыздылығы деректері жүктелуде...",
        "unable_to_load": "Белгілердің маңыздылығы деректерін жүктеу мүмкін емес. Файлдарды тексеріңіз.",
        "overview": "Белгілердің Маңыздылығы Шолуы",
        "top_features": "Ең Маңызды Белгілер",
        "num_features": "Көрсетілетін белгілер саны:",
        "by_category": "Санат бойынша Белгілердің Маңыздылығы",
        "correlations": "Белгілердің Корреляциясы",
        "with_target": "Мақсатпен Корреляция",
        "explorer": "Интерактивті Белгілерді Зерттеуші",
        "select_feature": "Зерттеу үшін белгіні таңдаңыз:",
        "distribution": "Белгі Үлестірімі",
        "by_outcome": "Нәтижелер бойынша Үлестірім",
        "complete_table": "Белгілер Маңыздылығының Толық Кестесі",
        "export_data": "Белгілер Маңыздылығын Экспорттау",
        "download_importance": "Маңыздылықты Жүктеп Алу (CSV)"
    }
}


def update_translation_file(file_path: Path, additional_keys: dict):
    """Update translation file with additional keys while preserving existing ones."""
    # Load existing translations
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Update with new keys (deep merge)
    for section, keys in additional_keys.items():
        if section in data:
            data[section].update(keys)
        else:
            data[section] = keys

    # Save updated translations
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Updated {file_path}")


def main():
    """Main function to update both translation files."""
    locales_dir = Path(__file__).parent / "src/visualization/locales"

    # Update English translations
    en_file = locales_dir / "en.json"
    update_translation_file(en_file, ADDITIONAL_KEYS)

    # Update Kazakh translations
    kk_file = locales_dir / "kk.json"
    update_translation_file(kk_file, ADDITIONAL_KEYS_KK)

    print("\n🎉 All translation files updated successfully!")
    print("📊 Added comprehensive keys for:")
    print("   - Predictions page")
    print("   - Clustering page")
    print("   - Performance page")
    print("   - Importance page")


if __name__ == "__main__":
    main()
