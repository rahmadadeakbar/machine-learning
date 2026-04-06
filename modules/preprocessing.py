"""
Data Preprocessing module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def render_preprocessing(df, feature_names, target_name):
    """Render preprocessing section."""

    st.markdown("## 🔧 Preprocessing Data")
    st.markdown(
        """
        <div class="info-box">
            <strong>Preprocessing</strong> adalah tahap penting untuk menyiapkan data sebelum
            dimasukkan ke model. Data mentah sering memiliki masalah yang perlu ditangani.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "1️⃣ Penanganan Missing Values",
        "2️⃣ Feature Scaling",
        "3️⃣ Encoding",
        "4️⃣ Train-Test Split",
    ])

    with tab1:
        _render_missing_values(df, feature_names)

    with tab2:
        _render_scaling(df, feature_names)

    with tab3:
        _render_encoding(df, feature_names)

    with tab4:
        _render_split(df, feature_names, target_name)


def _render_missing_values(df, feature_names):
    """Handle missing values."""
    st.markdown("### 1️⃣ Penanganan Missing Values")

    missing = df[feature_names].isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        st.success("✅ Dataset ini **tidak memiliki missing values**!")
        st.markdown(
            """
            Namun, dalam praktik nyata, missing values sangat umum. Berikut strategi penanganannya:

            | Strategi | Kapan Digunakan | Kode |
            |----------|----------------|------|
            | **Drop rows** | Missing sedikit (<5%) | `df.dropna()` |
            | **Drop columns** | Kolom banyak missing (>50%) | `df.drop(columns=[...])` |
            | **Mean/Median imputation** | Data numerik | `df.fillna(df.mean())` |
            | **Mode imputation** | Data kategorikal | `df.fillna(df.mode()[0])` |
            | **KNN Imputer** | Hubungan antar fitur | `KNNImputer(n_neighbors=5)` |
            """
        )
    else:
        st.warning(f"⚠️ Terdapat **{total_missing}** missing values!")
        st.dataframe(
            missing[missing > 0].reset_index().rename(
                columns={'index': 'Fitur', 0: 'Missing Count'}
            ),
            use_container_width=True,
            hide_index=True,
        )

    # Code example
    st.markdown("#### 💻 Contoh Kode")
    st.code(
        """
import pandas as pd
from sklearn.impute import SimpleImputer

# Cek missing values
print(df.isnull().sum())

# Strategi 1: Drop rows dengan missing values
df_clean = df.dropna()

# Strategi 2: Imputation dengan mean
imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Strategi 3: Imputation dengan mode (untuk kategorikal)
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
        """,
        language="python",
    )


def _render_scaling(df, feature_names):
    """Feature scaling demonstration."""
    st.markdown("### 2️⃣ Feature Scaling")

    st.markdown(
        """
        <div class="warning-box">
            <strong>⚠️ Mengapa Scaling Penting?</strong><br>
            Fitur dengan skala berbeda bisa mempengaruhi performa algoritma.
            Contoh: <em>age</em> (20-80) vs <em>income</em> (1.000.000-50.000.000).<br><br>
            <strong>Catatan:</strong> Decision Tree <strong>tidak memerlukan</strong> scaling,
            tetapi Naive Bayes (Gaussian) bisa <strong>terpengaruh</strong> oleh skala fitur.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            #### StandardScaler (Z-Score)
            $$X_{scaled} = \\frac{X - \\mu}{\\sigma}$$
            - Mean = 0, Std = 1
            - Baik jika data berdistribusi normal
            """
        )

    with col2:
        st.markdown(
            """
            #### MinMaxScaler
            $$X_{scaled} = \\frac{X - X_{min}}{X_{max} - X_{min}}$$
            - Range [0, 1]
            - Baik jika distribusi tidak normal
            """
        )

    # Interactive demo
    st.markdown("#### 🔬 Demo Interaktif Scaling")
    demo_feat = st.selectbox("Pilih fitur:", feature_names[:5])

    original = df[demo_feat].values.reshape(-1, 1)
    standard_scaled = StandardScaler().fit_transform(original)
    minmax_scaled = MinMaxScaler().fit_transform(original)

    demo_df = pd.DataFrame({
        'Original': original.flatten(),
        'StandardScaler': standard_scaled.flatten(),
        'MinMaxScaler': minmax_scaled.flatten(),
    })

    fig = px.histogram(
        demo_df.melt(var_name='Metode', value_name='Nilai'),
        x='Nilai', color='Metode', barmode='overlay',
        facet_col='Metode', opacity=0.7,
        title=f"Perbandingan Scaling untuk fitur: {demo_feat}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=350, margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.code(
        """
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
        """,
        language="python",
    )


def _render_encoding(df, feature_names):
    """Encoding demonstration."""
    st.markdown("### 3️⃣ Encoding Fitur Kategorikal")

    st.markdown(
        """
        Algoritma ML memerlukan input berupa **angka**. Fitur kategorikal perlu di-encode.

        | Metode | Penjelasan | Contoh |
        |--------|-----------|--------|
        | **Label Encoding** | Assign angka unik per kategori | Red→0, Blue→1, Green→2 |
        | **One-Hot Encoding** | Buat kolom biner per kategori | Red→[1,0,0], Blue→[0,1,0] |
        | **Ordinal Encoding** | Untuk data berurutan | Low→0, Medium→1, High→2 |
        """
    )

    st.markdown(
        """
        <div class="info-box">
            <strong>💡 Tips:</strong><br>
            - <strong>Label Encoding</strong>: Untuk fitur ordinal atau algoritma tree-based<br>
            - <strong>One-Hot Encoding</strong>: Untuk fitur nominal (tanpa urutan)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        """
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Label Encoding
le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)
        """,
        language="python",
    )


def _render_split(df, feature_names, target_name):
    """Train-test split demonstration."""
    st.markdown("### 4️⃣ Train-Test Split")

    st.markdown(
        """
        <div class="info-box">
            Data dibagi menjadi <strong>training set</strong> (untuk melatih model) dan
            <strong>test set</strong> (untuk menguji performa model pada data yang belum pernah dilihat).
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5)
        random_state = st.number_input("Random State", 0, 100, 42)
        stratify = st.checkbox("Stratified Split", value=True)

    with col2:
        X = df[feature_names]
        y = df[target_name]

        strat = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=random_state, stratify=strat,
        )

        sizes = pd.DataFrame({
            'Set': ['Training', 'Testing', 'Total'],
            'Jumlah': [len(X_train), len(X_test), len(df)],
            'Persentase': [
                f"{len(X_train)/len(df)*100:.0f}%",
                f"{len(X_test)/len(df)*100:.0f}%",
                "100%",
            ],
        })
        st.dataframe(sizes, use_container_width=True, hide_index=True)

        # Show class distribution
        if stratify:
            train_dist = y_train.value_counts(normalize=True).round(3)
            test_dist = y_test.value_counts(normalize=True).round(3)

            dist_df = pd.DataFrame({
                'Kelas': train_dist.index,
                'Train (%)': (train_dist.values * 100).round(1),
                'Test (%)': (test_dist.values * 100).round(1),
            })
            st.markdown("**Distribusi kelas (stratified):**")
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # Visualization
    fig = px.pie(
        sizes.head(2), values='Jumlah', names='Set',
        title=f"Split Ratio: {100-test_size}% Training / {test_size}% Testing",
        color_discrete_sequence=['#667eea', '#f093fb'],
        hole=0.4,
    )
    fig.update_layout(height=300, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.code(
        f"""
from sklearn.model_selection import train_test_split

X = df[feature_columns]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size={test_size/100},
    random_state={random_state},
    stratify={'y' if stratify else 'None'}
)

print(f"Training: {{X_train.shape[0]}} samples")
print(f"Testing:  {{X_test.shape[0]}} samples")
        """,
        language="python",
    )
