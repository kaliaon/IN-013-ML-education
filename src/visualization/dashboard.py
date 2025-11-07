"""
OULAD Learning Analytics Dashboard
Main Streamlit application for interactive data exploration and visualization.

Phase 5: Visualizations and Dashboard
Dissertation Project: Methods and Algorithms for Optimal Educational Process Management
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root and src directory to path
project_root = Path(__file__).parent.parent.parent
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Import using absolute paths
from visualization.config import PAGE_CONFIG, PAGES
from visualization.utils import check_data_availability
from visualization.page_modules import overview, predictions, clustering, performance, importance
from visualization.i18n import t, language_selector_sidebar, get_current_language


def main():
    """Main dashboard application."""

    # Configure page
    st.set_page_config(**PAGE_CONFIG)

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #3498db;
            text-align: center;
            padding: 1rem 0;
        }
        .subheader {
            font-size: 1.2rem;
            color: #7f8c8d;
            text-align: center;
            padding-bottom: 2rem;
        }
        .sidebar-info {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .metric-container {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown(f"""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='color: #3498db; margin: 0;'>📊 OULAD</h1>
                <h3 style='color: #7f8c8d; margin: 0;'>{t('app.subtitle')}</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        st.subheader(f"📑 {t('navigation.title')}")

        # Translate page names
        page_translations = {
            "🏠 Overview": f"🏠 {t('pages.overview')}",
            "🎯 Predictions": f"🎯 {t('pages.predictions')}",
            "👥 Clustering": f"👥 {t('pages.clustering')}",
            "📈 Model Performance": f"📈 {t('pages.performance')}",
            "⭐ Feature Importance": f"⭐ {t('pages.importance')}"
        }

        translated_pages = [page_translations[page] for page in PAGES.keys()]
        selected_page_translated = st.radio(
            t('navigation.select_page'),
            options=translated_pages,
            label_visibility="collapsed"
        )

        # Map back to original key
        reverse_map = {v: k for k, v in page_translations.items()}
        selected_page = reverse_map[selected_page_translated]

        st.markdown("---")

        # Language selector
        language_selector_sidebar()

        st.markdown("---")

        # System status
        st.subheader(f"⚙️ {t('system_status.title')}")

        availability = check_data_availability()

        status_items = {
            t('system_status.processed_data'): availability["processed_data"],
            t('system_status.clustered_data'): availability["clustered_data"],
            t('system_status.ml_models'): availability["models"],
            t('system_status.label_encoder'): availability["label_encoder"],
            t('system_status.feature_names'): availability["feature_names"]
        }

        all_available = all(status_items.values())

        if all_available:
            st.success(f"✅ {t('system_status.all_operational')}")
        else:
            st.warning(f"⚠️ {t('system_status.some_missing')}")

        for item, status in status_items.items():
            icon = "✅" if status else "❌"
            st.text(f"{icon} {item}")

        st.markdown("---")

        # Project information
        with st.expander(f"ℹ️ {t('help.about_title')}"):
            st.markdown(f"""
                **{t('app.title')}**

                {t('help.about_description')}

                **{t('help.features_title')}**
                - {t('help.feature_1')}
                - {t('help.feature_2')}
                - {t('help.feature_3')}
                - {t('help.feature_4')}
                - {t('help.feature_5')}

                **{t('help.phase')}** 5 - {t('help.phase_name')}
                **{t('help.dataset')}** OULAD (32K+ {t('help.students')})
                **{t('help.models')}** Decision Tree, Random Forest, XGBoost, LightGBM
            """)

        with st.expander(f"📚 {t('help.how_to_use_title')}"):
            st.markdown(f"""
                **{t('pages.overview')}:**
                - {t('help.overview_1')}
                - {t('help.overview_2')}
                - {t('help.overview_3')}

                **{t('pages.predictions')}:**
                - {t('help.predictions_1')}
                - {t('help.predictions_2')}
                - {t('help.predictions_3')}

                **{t('pages.clustering')}:**
                - {t('help.clustering_1')}
                - {t('help.clustering_2')}
                - {t('help.clustering_3')}

                **{t('pages.performance')}:**
                - {t('help.performance_1')}
                - {t('help.performance_2')}
                - {t('help.performance_3')}

                **{t('pages.importance')}:**
                - {t('help.importance_1')}
                - {t('help.importance_2')}
                - {t('help.importance_3')}
            """)

        with st.expander(f"🎓 {t('help.dissertation_title')}"):
            st.markdown(f"""
                **{t('help.topic')}**
                Методы и алгоритмы оптимального управления
                учебным процессом на основе больших данных

                **{t('help.translation')}**
                Methods and algorithms for optimal educational
                process management based on big data

                **{t('help.university')}** Open University
                **{t('help.dataset')}** OULAD (Open University Learning Analytics Dataset)
                **{t('help.technologies')}** Python, Scikit-learn, XGBoost, LightGBM, Streamlit
            """)

        st.markdown("---")

        # Footer
        st.markdown(f"""
            <div style='text-align: center; color: #7f8c8d; font-size: 0.8rem;'>
                <p>{t('footer.built_with')} Streamlit</p>
                <p>{t('footer.copyright')}</p>
            </div>
        """, unsafe_allow_html=True)

    # Main content area
    if not all_available:
        st.error(f"""
            ⚠️ **{t('messages.missing_data_title')}**

            {t('messages.missing_data_desc')}

            {t('messages.ensure_steps')}
            1. {t('messages.step_1')}
            2. {t('messages.step_2')}
            3. {t('messages.step_3')}
            4. {t('messages.step_4')}

            {t('messages.check_sidebar')}
        """)
        st.stop()

    # Route to selected page
    page_map = {
        "🏠 Overview": overview,
        "🎯 Predictions": predictions,
        "👥 Clustering": clustering,
        "📈 Model Performance": performance,
        "⭐ Feature Importance": importance
    }

    # Render selected page
    try:
        page_module = page_map[selected_page]
        page_module.render()
    except Exception as e:
        st.error(f"❌ {t('messages.error_rendering')}: {e}")
        st.exception(e)

    # Footer in main area
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
            <p><strong>{t('app.title')}</strong></p>
            <p>{t('footer.tagline')}</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
