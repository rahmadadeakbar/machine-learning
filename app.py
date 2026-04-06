"""
=================================================================
  KULIAH TAMU: Membangun Model Prediksi
  Klasifikasi Data dengan Decision Tree & Naive Bayes
=================================================================
  Streamlit Interactive Application
  Author: Guest Lecturer
=================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from modules.presentation import render_presentation
from modules.eda import render_eda
from modules.preprocessing import render_preprocessing
from modules.modeling import render_modeling
from modules.comparison import render_comparison
from modules.prediction import render_prediction
from modules.dataset_loader import load_dataset

# ─── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Kuliah Tamu — Decision Tree & Naive Bayes",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #e0d4f5; margin: 0.3rem 0 0 0; font-size: 1rem; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #e0e0e0 !important;
    }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; color: #666; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #333; }

    /* Slide styling */
    .slide-container {
        background: white;
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        min-height: 500px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .slide-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .slide-subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
    }

    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e8f0 100%);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196F3;
    }
    .warning-box {
        background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #FF9800;
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }

    /* Code blocks */
    .stCodeBlock { border-radius: 10px !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Kuliah Tamu")
    st.markdown("**Machine Learning**")
    st.markdown("---")

    menu = st.radio(
        "📋 **Navigasi Materi**",
        [
            "🎤 Presentasi Materi",
            "📊 Eksplorasi Data (EDA)",
            "🔧 Preprocessing Data",
            "🤖 Pemodelan & Evaluasi",
            "⚖️ Perbandingan Model",
            "🔮 Prediksi Interaktif",
        ],
        index=0,
    )

    st.markdown("---")

    # Dataset selection
    st.markdown("### 📁 Pilihan Dataset")
    dataset_choice = st.selectbox(
        "Dataset",
        ["Heart Disease", "Breast Cancer", "Iris"],
        help="Pilih dataset untuk demonstrasi",
    )

    st.markdown("---")
    st.markdown(
        """
        ### 📌 Tentang Aplikasi
        Aplikasi ini dibuat sebagai materi
        **Kuliah Tamu** tentang membangun
        model prediksi menggunakan:
        - 🌳 **Decision Tree**
        - 📊 **Naive Bayes**

        ---
        *Built with Streamlit & Scikit-learn*
        """
    )

# ─── Load Dataset ───────────────────────────────────────────────
df, feature_names, target_name, target_labels, dataset_info = load_dataset(dataset_choice)

# ─── Header ─────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🎓 Membangun Model Prediksi</h1>
        <p>Klasifikasi Data dengan Decision Tree & Naive Bayes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Route Pages ────────────────────────────────────────────────
if menu == "🎤 Presentasi Materi":
    render_presentation()
elif menu == "📊 Eksplorasi Data (EDA)":
    render_eda(df, feature_names, target_name, target_labels, dataset_info)
elif menu == "🔧 Preprocessing Data":
    render_preprocessing(df, feature_names, target_name)
elif menu == "🤖 Pemodelan & Evaluasi":
    render_modeling(df, feature_names, target_name, target_labels, dataset_choice)
elif menu == "⚖️ Perbandingan Model":
    render_comparison(df, feature_names, target_name, target_labels, dataset_choice)
elif menu == "🔮 Prediksi Interaktif":
    render_prediction(df, feature_names, target_name, target_labels, dataset_choice)
