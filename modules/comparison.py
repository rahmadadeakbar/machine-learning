"""
Model Comparison module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)
import time


DT_COLOR = "#3a86ff"
NB_COLOR = "#8338ec"


def render_comparison(df, feature_names, target_name, target_labels, dataset_choice):
    """Render model comparison."""

    st.markdown("## Perbandingan Model")
    st.markdown(
        """
        <div class="info-box">
            Mari bandingkan performa <strong>Decision Tree</strong> dan <strong>Naive Bayes</strong>
            secara head-to-head pada dataset yang sama.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5, key="comp_test_size")
    with col2:
        cv_folds = st.slider("CV Folds", 3, 10, 5, key="comp_cv")

    # Prepare Data
    X = df[feature_names].values
    y = df[target_name].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Both Models
    t0 = time.time()
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_time = time.time() - t0

    t0 = time.time()
    dt_pred = dt.predict(X_test)
    dt_pred_time = time.time() - t0

    dt_cv = cross_val_score(dt, X_train, y_train, cv=cv_folds, scoring='accuracy')

    t0 = time.time()
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    nb_train_time = time.time() - t0

    t0 = time.time()
    nb_pred = nb.predict(X_test_scaled)
    nb_pred_time = time.time() - t0

    nb_cv = cross_val_score(nb, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy')

    st.markdown("---")

    # Head-to-Head Metrics
    st.markdown("### Head-to-Head Comparison")

    metrics_data = {
        'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1-Score',
                    'CV Mean Accuracy', 'CV Std', 'Training Time (ms)', 'Prediction Time (ms)'],
        'Decision Tree': [
            f"{accuracy_score(y_test, dt_pred):.4f}",
            f"{precision_score(y_test, dt_pred, average='weighted', zero_division=0):.4f}",
            f"{recall_score(y_test, dt_pred, average='weighted', zero_division=0):.4f}",
            f"{f1_score(y_test, dt_pred, average='weighted', zero_division=0):.4f}",
            f"{dt_cv.mean():.4f}",
            f"{dt_cv.std():.4f}",
            f"{dt_train_time*1000:.2f}",
            f"{dt_pred_time*1000:.2f}",
        ],
        'Naive Bayes': [
            f"{accuracy_score(y_test, nb_pred):.4f}",
            f"{precision_score(y_test, nb_pred, average='weighted', zero_division=0):.4f}",
            f"{recall_score(y_test, nb_pred, average='weighted', zero_division=0):.4f}",
            f"{f1_score(y_test, nb_pred, average='weighted', zero_division=0):.4f}",
            f"{nb_cv.mean():.4f}",
            f"{nb_cv.std():.4f}",
            f"{nb_train_time*1000:.2f}",
            f"{nb_pred_time*1000:.2f}",
        ],
    }

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Winner
    dt_f1 = f1_score(y_test, dt_pred, average='weighted', zero_division=0)
    nb_f1 = f1_score(y_test, nb_pred, average='weighted', zero_division=0)

    if dt_f1 > nb_f1:
        winner = "Decision Tree"
        winner_f1 = dt_f1
    elif nb_f1 > dt_f1:
        winner = "Naive Bayes"
        winner_f1 = nb_f1
    else:
        winner = "Seri"
        winner_f1 = dt_f1

    st.markdown(
        f"""
        <div class="success-box">
            <strong>Pemenang (berdasarkan F1-Score):</strong> {winner}
            dengan F1-Score = <strong>{winner_f1:.4f}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Visual Comparisons
    tab1, tab2, tab3, tab4 = st.tabs([
        "Bar Chart Metrik",
        "Learning Curves",
        "Confusion Matrices",
        "ROC Curves",
    ])

    with tab1:
        _render_metric_bars(y_test, dt_pred, nb_pred)

    with tab2:
        _render_learning_curves(X_train, X_train_scaled, y_train, cv_folds)

    with tab3:
        _render_confusion_matrices(y_test, dt_pred, nb_pred, target_labels)

    with tab4:
        _render_roc_curves(X_test, X_test_scaled, y_test, dt, nb, target_labels)

    # Summary
    st.markdown("---")
    st.markdown("### Kesimpulan Perbandingan")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            #### Decision Tree
            - Accuracy: **{accuracy_score(y_test, dt_pred):.4f}**
            - Kelebihan: Interpretable, no scaling needed
            - Train time: **{dt_train_time*1000:.2f}ms**
            - CV Score: **{dt_cv.mean():.4f}** +/- {dt_cv.std():.4f}
            """
        )

    with col2:
        st.markdown(
            f"""
            #### Naive Bayes
            - Accuracy: **{accuracy_score(y_test, nb_pred):.4f}**
            - Kelebihan: Sangat cepat, baik untuk data kecil
            - Train time: **{nb_train_time*1000:.2f}ms**
            - CV Score: **{nb_cv.mean():.4f}** +/- {nb_cv.std():.4f}
            """
        )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Insight:</strong> Tidak ada satu algoritma yang <strong>selalu terbaik</strong>
            untuk semua kasus. Pemilihan model tergantung pada:
            <ul>
                <li>Karakteristik data</li>
                <li>Kebutuhan interpretabilitas</li>
                <li>Kecepatan yang dibutuhkan</li>
                <li>Trade-off precision vs recall</li>
            </ul>
            Inilah yang disebut <strong>"No Free Lunch Theorem"</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_bars(y_test, dt_pred, nb_pred):
    """Bar chart comparison."""
    st.markdown("#### Perbandingan Metrik")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    dt_scores = [
        accuracy_score(y_test, dt_pred),
        precision_score(y_test, dt_pred, average='weighted', zero_division=0),
        recall_score(y_test, dt_pred, average='weighted', zero_division=0),
        f1_score(y_test, dt_pred, average='weighted', zero_division=0),
    ]
    nb_scores = [
        accuracy_score(y_test, nb_pred),
        precision_score(y_test, nb_pred, average='weighted', zero_division=0),
        recall_score(y_test, nb_pred, average='weighted', zero_division=0),
        f1_score(y_test, nb_pred, average='weighted', zero_division=0),
    ]

    fig = go.Figure(data=[
        go.Bar(name='Decision Tree', x=metrics, y=dt_scores,
               marker_color=DT_COLOR, text=[f"{s:.4f}" for s in dt_scores], textposition='outside'),
        go.Bar(name='Naive Bayes', x=metrics, y=nb_scores,
               marker_color=NB_COLOR, text=[f"{s:.4f}" for s in nb_scores], textposition='outside'),
    ])

    fig.update_layout(
        barmode='group', title="Perbandingan Metrik Evaluasi",
        yaxis_range=[0, 1.15], height=450,
        margin=dict(t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_learning_curves(X_train, X_train_scaled, y_train, cv_folds):
    """Learning curves."""
    st.markdown("#### Learning Curves")

    st.markdown(
        """
        <div class="info-box">
            <strong>Learning Curve</strong> menunjukkan performa model terhadap jumlah data training.
            Berguna untuk mendeteksi <strong>overfitting</strong> atau <strong>underfitting</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Menghitung learning curves..."):
        train_sizes, dt_train_scores, dt_test_scores = learning_curve(
            DecisionTreeClassifier(max_depth=5, random_state=42),
            X_train, y_train, cv=cv_folds, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy',
        )

        _, nb_train_scores, nb_test_scores = learning_curve(
            GaussianNB(), X_train_scaled, y_train, cv=cv_folds, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy',
        )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_sizes, y=dt_train_scores.mean(axis=1),
            mode='lines+markers', name='Train Score',
            line=dict(color=DT_COLOR),
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes, y=dt_test_scores.mean(axis=1),
            mode='lines+markers', name='Validation Score',
            line=dict(color=NB_COLOR),
        ))
        fig.update_layout(
            title="Learning Curve — Decision Tree",
            xaxis_title="Training Samples",
            yaxis_title="Accuracy",
            yaxis_range=[0.4, 1.05],
            height=400, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=train_sizes, y=nb_train_scores.mean(axis=1),
            mode='lines+markers', name='Train Score',
            line=dict(color=DT_COLOR),
        ))
        fig.add_trace(go.Scatter(
            x=train_sizes, y=nb_test_scores.mean(axis=1),
            mode='lines+markers', name='Validation Score',
            line=dict(color=NB_COLOR),
        ))
        fig.update_layout(
            title="Learning Curve — Naive Bayes",
            xaxis_title="Training Samples",
            yaxis_title="Accuracy",
            yaxis_range=[0.4, 1.05],
            height=400, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Interpretasi:**
        - Gap besar antara train dan validation = **Overfitting** (model terlalu kompleks)
        - Keduanya rendah = **Underfitting** (model terlalu sederhana)
        - Keduanya tinggi dan berdekatan = **Good fit**
        """
    )


def _render_confusion_matrices(y_test, dt_pred, nb_pred, target_labels):
    """Side-by-side confusion matrices."""
    st.markdown("#### Confusion Matrix Side-by-Side")

    labels = [target_labels.get(i, str(i)) for i in sorted(target_labels.keys())]

    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_test, dt_pred)
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=labels, y=labels,
            labels={'x': 'Predicted', 'y': 'Actual'},
            title="Decision Tree",
        )
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cm = confusion_matrix(y_test, nb_pred)
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Oranges",
            x=labels, y=labels,
            labels={'x': 'Predicted', 'y': 'Actual'},
            title="Naive Bayes",
        )
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


def _render_roc_curves(X_test, X_test_scaled, y_test, dt, nb, target_labels):
    """ROC curves."""
    st.markdown("#### ROC Curve")

    n_classes = len(set(y_test))
    if n_classes > 2:
        st.info("ROC Curve ditampilkan untuk klasifikasi biner (2 kelas). Dataset ini memiliki "
                f"{n_classes} kelas — menampilkan One-vs-Rest untuk setiap kelas.")

        labels = [target_labels.get(i, str(i)) for i in sorted(target_labels.keys())]
        fig = go.Figure()

        for i, label in enumerate(labels):
            y_binary = (y_test == i).astype(int)

            if hasattr(dt, 'predict_proba'):
                dt_prob = dt.predict_proba(X_test)[:, i]
                fpr, tpr, _ = roc_curve(y_binary, dt_prob)
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"DT - {label} (AUC={roc_auc:.3f})",
                                          line=dict(dash='solid')))

            nb_prob = nb.predict_proba(X_test_scaled)[:, i]
            fpr, tpr, _ = roc_curve(y_binary, nb_prob)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"NB - {label} (AUC={roc_auc:.3f})",
                                      line=dict(dash='dash')))

        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                  line=dict(color='grey', dash='dot')))

        fig.update_layout(
            title="ROC Curves (One-vs-Rest)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    # Binary classification ROC
    fig = go.Figure()

    if hasattr(dt, 'predict_proba'):
        dt_prob = dt.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, dt_prob)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                  name=f"Decision Tree (AUC = {roc_auc:.4f})",
                                  line=dict(color=DT_COLOR, width=2)))

    nb_prob = nb.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, nb_prob)
    roc_auc = auc(fpr, tpr)
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                              name=f"Naive Bayes (AUC = {roc_auc:.4f})",
                              line=dict(color=NB_COLOR, width=2)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                              name="Random (AUC = 0.5)",
                              line=dict(color='grey', dash='dash')))

    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500, margin=dict(t=40, b=20),
        legend=dict(x=0.5, y=0.05),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="info-box">
            <strong>Interpretasi ROC dan AUC:</strong><br>
            - <strong>AUC = 1.0</strong> — Model sempurna<br>
            - <strong>AUC = 0.5</strong> — Model acak (tidak lebih baik dari tebakan)<br>
            - <strong>AUC > 0.8</strong> — Model baik<br>
            - Semakin kurva mendekati sudut kiri atas, semakin baik model
        </div>
        """,
        unsafe_allow_html=True,
    )
