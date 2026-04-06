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

# Page Configuration
st.set_page_config(
    page_title="Kuliah Tamu — Decision Tree & Naive Bayes",
    page_icon="./",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — flat, professional, no gradients, no emojis
st.markdown("""
<style>
    .main-header {
        background: #1a1a2e;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        border-bottom: 3px solid #3a86ff;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; font-weight: 600; letter-spacing: -0.02em; }
    .main-header p  { color: #a0aec0; margin: 0.3rem 0 0 0; font-size: 1rem; }

    [data-testid="stSidebar"] {
        background: #f7f8fa;
        border-right: 1px solid #e2e8f0;
    }

    .info-box {
        background: #eef4fb;
        border-radius: 6px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3a86ff;
    }
    .warning-box {
        background: #fef9eb;
        border-radius: 6px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 3px solid #d69e2e;
    }
    .success-box {
        background: #edf7ee;
        border-radius: 6px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 3px solid #2e8b3a;
    }

    .stCodeBlock { border-radius: 6px !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; padding: 8px 20px; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## Kuliah Tamu")
    st.markdown("**Machine Learning**")
    st.markdown("---")

    menu = st.radio(
        "**Navigasi Materi**",
        [
            "Presentasi Materi",
            "Eksplorasi Data (EDA)",
            "Preprocessing Data",
            "Pemodelan & Evaluasi",
            "Perbandingan Model",
            "Prediksi Interaktif",
        ],
        index=0,
    )

    st.markdown("---")

    st.markdown("### Pilihan Dataset")
    dataset_choice = st.selectbox(
        "Dataset",
        ["Heart Disease", "Breast Cancer", "Iris"],
        help="Pilih dataset untuk demonstrasi",
    )

    st.markdown("---")
    st.markdown(
        """
        ### Tentang Aplikasi
        Aplikasi ini dibuat sebagai materi
        **Kuliah Tamu** tentang membangun
        model prediksi menggunakan:
        - **Decision Tree**
        - **Naive Bayes**

        ---
        *Built with Streamlit & Scikit-learn*
        """
    )

# Load Dataset
df, feature_names, target_name, target_labels, dataset_info = load_dataset(dataset_choice)

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>Membangun Model Prediksi</h1>
        <p>Klasifikasi Data dengan Decision Tree & Naive Bayes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Route Pages
if menu == "Presentasi Materi":
    render_presentation()
elif menu == "Eksplorasi Data (EDA)":
    render_eda(df, feature_names, target_name, target_labels, dataset_info)
elif menu == "Preprocessing Data":
    render_preprocessing(df, feature_names, target_name)
elif menu == "Pemodelan & Evaluasi":
    render_modeling(df, feature_names, target_name, target_labels, dataset_choice)
elif menu == "Perbandingan Model":
    render_comparison(df, feature_names, target_name, target_labels, dataset_choice)
elif menu == "Prediksi Interaktif":
    render_prediction(df, feature_names, target_name, target_labels, dataset_choice)
