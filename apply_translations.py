#!/usr/bin/env python3
"""
Script to apply translations to page modules by replacing hardcoded strings.
"""

import re
from pathlib import Path

# Define replacements for each page
REPLACEMENTS = {
    "predictions.py": [
        (r'st\.title\("🎯 Student Performance Prediction"\)', r'st.title(f"🎯 {t(\'predictions.title\')}")'),
        (r'"Loading models and data\.\.\."', r't(\'predictions.loading\')'),
        (r'"❌ Unable to load required models or data\. Please check model files\."', r'f"❌ {t(\'predictions.unable_to_load_models\')}"'),
        (r'st\.header\("📝 Predict Individual Student Performance"\)', r'st.header(f"📝 {t(\'predictions.manual_title\')}")'),
        (r'"Enter student characteristics to predict their likely outcome\."', r't(\'predictions.manual_desc\')'),
        (r'"Select prediction model:"', r't(\'predictions.select_model\')'),
        (r'"LightGBM achieved the best performance \(88\.83% accuracy\)"', r't(\'predictions.model_help\')'),
        (r'st\.subheader\("Student Information"\)', r'st.subheader(t(\'predictions.student_info\'))'),
        (r'st\.expander\("👤 Demographics"', r'st.expander(f"👤 {t(\'predictions.demographics\')}"'),
        (r'st\.expander\("💻 VLE Activity"', r'st.expander(f"💻 {t(\'predictions.vle_activity\')}"'),
        (r'st\.expander\("📊 Assessment Performance"', r'st.expander(f"📊 {t(\'predictions.assessment_performance\')}"'),
        (r'st\.expander\("📝 Registration"', r'st.expander(f"📝 {t(\'predictions.registration\')}"'),
        (r'st\.button\("🔮 Predict Performance"', r'st.button(f"🔮 {t(\'predictions.predict_button\')}"'),
        (r'st\.success\("✅ Prediction Complete!"\)', r'st.success(f"✅ {t(\'predictions.prediction_complete\')}")'),
        (r'st\.header\("📊 Batch Student Prediction"\)', r'st.header(f"📊 {t(\'predictions.batch_title\')}")'),
        (r'st\.header\("🔮 What-If Analysis"\)', r'st.header(f"🔮 {t(\'predictions.whatif_title\')}")'),
    ],

    "clustering.py": [
        (r'st\.title\("👥 Student Clustering Analysis"\)', r'st.title(f"👥 {t(\'clustering.title\')}")'),
        (r'"Loading clustering data\.\.\."', r't(\'clustering.loading\')'),
        (r'st\.header\("🔍 Clustering Overview"\)', r'st.header(f"🔍 {t(\'clustering.overview\')}")'),
        (r'st\.header\("📊 Cluster Distribution"\)', r'st.header(f"📊 {t(\'clustering.distribution\')}")'),
        (r'st\.header\("🎨 Cluster Visualization"\)', r'st.header(f"🎨 {t(\'clustering.visualization\')}")'),
        (r'st\.header\("👥 Cluster Profiles"\)', r'st.header(f"👥 {t(\'clustering.cluster_profiles\')}")'),
        (r'"Select cluster to analyze:"', r't(\'clustering.select_cluster\')'),
    ],

    "performance.py": [
        (r'st\.title\("📈 Model Performance Comparison"\)', r'st.title(f"📈 {t(\'performance.title\')}")'),
        (r'"Loading model performance data\.\.\."', r't(\'performance.loading\')'),
        (r'st\.header\("🎯 Performance Overview"\)', r'st.header(f"🎯 {t(\'performance.overview\')}")'),
        (r'st\.header\("📊 Performance Metrics Comparison"\)', r'st.header(f"📊 {t(\'performance.metrics_comparison\')}")'),
        (r'st\.header\("🎭 Confusion Matrices"\)', r'st.header(f"🎭 {t(\'performance.confusion_matrices\')}")'),
        (r'st\.header\("📈 ROC Curves"\)', r'st.header(f"📈 {t(\'performance.roc_curves\')}")'),
        (r'"Select model:"', r't(\'performance.select_model\')'),
    ],

    "importance.py": [
        (r'st\.title\("⭐ Feature Importance Analysis"\)', r'st.title(f"⭐ {t(\'importance.title\')}")'),
        (r'"Loading feature importance data\.\.\."', r't(\'importance.loading\')'),
        (r'st\.header\("🔍 Feature Importance Overview"\)', r'st.header(f"🔍 {t(\'importance.overview\')}")'),
        (r'st\.header\("🏆 Top Important Features"\)', r'st.header(f"🏆 {t(\'importance.top_features\')}")'),
        (r'st\.header\("📊 Feature Importance by Category"\)', r'st.header(f"📊 {t(\'importance.by_category\')}")'),
        (r'st\.header\("🔗 Feature Correlations"\)', r'st.header(f"🔗 {t(\'importance.correlations\')}")'),
    ],
}


def apply_replacements_to_file(file_path: Path, replacements: list):
    """Apply regex replacements to a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    replacements_made = 0

    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            replacements_made += 1
            content = new_content

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {file_path.name}: Applied {replacements_made} replacements")
    else:
        print(f"ℹ️  {file_path.name}: No changes needed")

    return replacements_made


def main():
    """Main function to apply translations to all page modules."""
    pages_dir = Path(__file__).parent / "src/visualization/page_modules"

    total_replacements = 0
    for filename, replacements in REPLACEMENTS.items():
        file_path = pages_dir / filename
        if file_path.exists():
            count = apply_replacements_to_file(file_path, replacements)
            total_replacements += count
        else:
            print(f"⚠️  {filename}: File not found")

    print(f"\n🎉 Translation complete! Applied {total_replacements} replacements across all pages.")
    print("📝 Note: Some complex UI elements may still need manual translation.")
    print("   Test the dashboard to identify any remaining untranslated text.")


if __name__ == "__main__":
    main()
