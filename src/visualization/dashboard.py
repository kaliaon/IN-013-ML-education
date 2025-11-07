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
from visualization.pages import overview, predictions, clustering, performance, importance


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
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='color: #3498db; margin: 0;'>📊 OULAD</h1>
                <h3 style='color: #7f8c8d; margin: 0;'>Learning Analytics</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        st.subheader("📑 Navigation")
        selected_page = st.radio(
            "Select a page:",
            options=list(PAGES.keys()),
            label_visibility="collapsed"
        )

        st.markdown("---")

        # System status
        st.subheader("⚙️ System Status")

        availability = check_data_availability()

        status_items = {
            "Processed Data": availability["processed_data"],
            "Clustered Data": availability["clustered_data"],
            "ML Models": availability["models"],
            "Label Encoder": availability["label_encoder"],
            "Feature Names": availability["feature_names"]
        }

        all_available = all(status_items.values())

        if all_available:
            st.success("✅ All systems operational")
        else:
            st.warning("⚠️ Some data missing")

        for item, status in status_items.items():
            icon = "✅" if status else "❌"
            st.text(f"{icon} {item}")

        st.markdown("---")

        # Project information
        with st.expander("ℹ️ About This Dashboard"):
            st.markdown("""
                **OULAD Learning Analytics Dashboard**

                This interactive dashboard provides comprehensive analysis of student
                performance using the Open University Learning Analytics Dataset.

                **Features:**
                - Dataset exploration and statistics
                - Student performance predictions
                - Behavioral clustering analysis
                - Model performance comparison
                - Feature importance insights

                **Phase:** 5 - Visualizations & Dashboard
                **Dataset:** OULAD (32K+ students)
                **Models:** Decision Tree, Random Forest, XGBoost, LightGBM
            """)

        with st.expander("📚 How to Use"):
            st.markdown("""
                **Overview Page:**
                - View dataset statistics and distributions
                - Explore demographic patterns
                - Analyze VLE activity and assessments

                **Predictions Page:**
                - Predict individual student outcomes
                - Run batch predictions
                - Perform what-if analysis

                **Clustering Page:**
                - Explore student behavioral groups
                - Analyze cluster characteristics
                - View PCA/t-SNE visualizations

                **Performance Page:**
                - Compare model metrics
                - View confusion matrices and ROC curves
                - Understand model trade-offs

                **Feature Importance Page:**
                - Identify key predictive features
                - Analyze feature categories
                - Explore SHAP explanations
            """)

        with st.expander("🎓 Dissertation Info"):
            st.markdown("""
                **Topic:**
                Методы и алгоритмы оптимального управления
                учебным процессом на основе больших данных

                **Translation:**
                Methods and algorithms for optimal educational
                process management based on big data

                **University:** Open University
                **Dataset:** OULAD (Open University Learning Analytics Dataset)
                **Technologies:** Python, Scikit-learn, XGBoost, LightGBM, Streamlit
            """)

        st.markdown("---")

        # Footer
        st.markdown("""
            <div style='text-align: center; color: #7f8c8d; font-size: 0.8rem;'>
                <p>Built with Streamlit</p>
                <p>© 2025 Learning Analytics Project</p>
            </div>
        """, unsafe_allow_html=True)

    # Main content area
    if not all_available:
        st.error("""
            ⚠️ **Missing Required Data**

            Some required data files or models are not available. Please ensure:
            1. Phase 1-4 notebooks have been executed
            2. Data files are in `data/processed/oulad/`
            3. Model files are in `models/`
            4. All dependencies are installed

            Check the sidebar for detailed status.
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
        st.error(f"❌ Error rendering page: {e}")
        st.exception(e)

    # Footer in main area
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
            <p><strong>OULAD Learning Analytics Dashboard</strong></p>
            <p>Empowering educators with data-driven insights for student success</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
