"""
Interactive Prediction module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def render_prediction(df, feature_names, target_name, target_labels, dataset_choice):
    """Render interactive prediction section."""

    st.markdown("## 🔮 Prediksi Interaktif")
    st.markdown(
        """
        <div class="info-box">
            Masukkan nilai fitur secara manual dan lihat hasil prediksi dari kedua model
            secara <strong>real-time</strong>!
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Train models in background ───────────────────────────
    X = df[feature_names].values
    y = df[target_name].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)

    # ─── Input Form ────────────────────────────────────────────
    st.markdown("### 📝 Masukkan Data untuk Prediksi")

    input_method = st.radio(
        "Metode Input:", ["Manual Input", "Random Sample dari Dataset"],
        horizontal=True,
    )

    input_values = {}

    if input_method == "Manual Input":
        n_cols = 3
        cols = st.columns(n_cols)

        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                mean_val = float(df[feat].mean())

                # Check if feature appears to be integer
                if df[feat].dtype in ['int64', 'int32'] and df[feat].nunique() < 10:
                    unique_vals = sorted(df[feat].unique())
                    input_values[feat] = st.selectbox(
                        f"**{feat}**",
                        unique_vals,
                        index=unique_vals.index(int(df[feat].mode()[0])),
                        key=f"pred_{feat}",
                    )
                else:
                    input_values[feat] = st.slider(
                        f"**{feat}**",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100,
                        key=f"pred_{feat}",
                    )
    else:
        if st.button("🎲 Ambil Sample Acak", use_container_width=True):
            st.session_state['random_sample_idx'] = np.random.randint(0, len(df))

        sample_idx = st.session_state.get('random_sample_idx', 0)
        sample = df.iloc[sample_idx]

        st.markdown(f"**Sample #{sample_idx}** (Actual: **{target_labels.get(int(sample[target_name]), str(sample[target_name]))}**)")

        n_cols = 3
        cols = st.columns(n_cols)

        for i, feat in enumerate(feature_names):
            with cols[i % n_cols]:
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())

                if df[feat].dtype in ['int64', 'int32'] and df[feat].nunique() < 10:
                    unique_vals = sorted(df[feat].unique())
                    val = int(sample[feat])
                    input_values[feat] = st.selectbox(
                        f"**{feat}**",
                        unique_vals,
                        index=unique_vals.index(val) if val in unique_vals else 0,
                        key=f"pred_{feat}",
                    )
                else:
                    input_values[feat] = st.slider(
                        f"**{feat}**",
                        min_value=min_val,
                        max_value=max_val,
                        value=float(sample[feat]),
                        step=(max_val - min_val) / 100,
                        key=f"pred_{feat}",
                    )

    st.markdown("---")

    # ─── Make Prediction ───────────────────────────────────────
    if st.button("🚀 **Prediksi Sekarang!**", use_container_width=True, type="primary"):
        input_array = np.array([[input_values[f] for f in feature_names]])
        input_scaled = scaler.transform(input_array)

        # Decision Tree prediction
        dt_pred = dt.predict(input_array)[0]
        dt_proba = dt.predict_proba(input_array)[0]

        # Naive Bayes prediction
        nb_pred = nb.predict(input_scaled)[0]
        nb_proba = nb.predict_proba(input_scaled)[0]

        labels = [target_labels.get(i, str(i)) for i in sorted(target_labels.keys())]

        st.markdown("### 🎯 Hasil Prediksi")

        col1, col2 = st.columns(2)

        with col1:
            dt_label = target_labels.get(dt_pred, str(dt_pred))
            dt_conf = dt_proba.max() * 100

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                     border-radius: 16px; padding: 2rem; text-align: center; color: white;">
                    <h3 style="color: white; margin-bottom: 0.5rem;">🌳 Decision Tree</h3>
                    <h2 style="color: white; font-size: 2rem;">{dt_label}</h2>
                    <p style="color: #e0d4f5;">Confidence: <strong>{dt_conf:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            nb_label = target_labels.get(nb_pred, str(nb_pred))
            nb_conf = nb_proba.max() * 100

            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                     border-radius: 16px; padding: 2rem; text-align: center; color: white;">
                    <h3 style="color: white; margin-bottom: 0.5rem;">📊 Naive Bayes</h3>
                    <h2 style="color: white; font-size: 2rem;">{nb_label}</h2>
                    <p style="color: #fde0e0;">Confidence: <strong>{nb_conf:.1f}%</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Agreement check
        if dt_pred == nb_pred:
            st.markdown(
                """
                <div class="success-box">
                    ✅ <strong>Kedua model sepakat!</strong> Prediksi konsisten dari Decision Tree dan Naive Bayes.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="warning-box">
                    ⚠️ <strong>Model tidak sepakat!</strong> Decision Tree dan Naive Bayes memberikan prediksi berbeda.
                    Perhatikan probabilitas masing-masing untuk keputusan lebih lanjut.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ─── Probability Details ───────────────────────────────
        st.markdown("### 📊 Detail Probabilitas")

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(go.Bar(
                x=labels, y=dt_proba,
                marker_color=['#667eea' if i == dt_pred else '#c4c4c4'
                               for i in sorted(target_labels.keys())],
                text=[f"{p:.3f}" for p in dt_proba],
                textposition='outside',
            ))
            fig.update_layout(
                title="Decision Tree — Probabilitas",
                yaxis_range=[0, 1.2], height=350,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Bar(
                x=labels, y=nb_proba,
                marker_color=['#f093fb' if i == nb_pred else '#c4c4c4'
                               for i in sorted(target_labels.keys())],
                text=[f"{p:.3f}" for p in nb_proba],
                textposition='outside',
            ))
            fig.update_layout(
                title="Naive Bayes — Probabilitas",
                yaxis_range=[0, 1.2], height=350,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ─── Input Summary ─────────────────────────────────────
        st.markdown("### 📋 Ringkasan Input")
        summary_df = pd.DataFrame({
            'Fitur': feature_names,
            'Nilai Input': [input_values[f] for f in feature_names],
            'Mean Dataset': [df[f].mean().round(3) for f in feature_names],
            'Std Dataset': [df[f].std().round(3) for f in feature_names],
            'Status': ['⬆️ Di atas rata-rata' if input_values[f] > df[f].mean() else '⬇️ Di bawah rata-rata'
                        for f in feature_names],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
