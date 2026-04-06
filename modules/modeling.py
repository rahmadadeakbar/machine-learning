"""
Modeling & Evaluation module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
)
import io


def render_modeling(df, feature_names, target_name, target_labels, dataset_choice):
    """Render modeling section."""

    st.markdown("## 🤖 Pemodelan & Evaluasi")
    st.markdown(
        """
        <div class="info-box">
            Saatnya melatih model <strong>Decision Tree</strong> dan <strong>Naive Bayes</strong>
            pada dataset, lalu mengevaluasi performanya.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Configuration ─────────────────────────────────────────
    st.markdown("### ⚙️ Konfigurasi")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5, key="model_test_size")
        random_state = st.number_input("Random State", 0, 100, 42, key="model_rs")

    with col2:
        use_scaling = st.checkbox("Gunakan StandardScaler", value=True,
                                   help="Scaling bermanfaat untuk Naive Bayes")
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

    # ─── Prepare Data ──────────────────────────────────────────
    X = df[feature_names].values
    y = df[target_name].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=random_state, stratify=y,
    )

    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    st.markdown("---")

    # ─── Tabs ──────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["🌳 Decision Tree", "📊 Naive Bayes"])

    with tab1:
        _render_decision_tree(
            X_train, X_test, y_train, y_test,
            feature_names, target_labels, cv_folds, random_state,
        )

    with tab2:
        _render_naive_bayes(
            X_train_scaled, X_test_scaled, y_train, y_test,
            feature_names, target_labels, cv_folds, use_scaling,
        )


def _render_decision_tree(X_train, X_test, y_train, y_test,
                            feature_names, target_labels, cv_folds, random_state):
    """Decision Tree training and evaluation."""
    st.markdown("### 🌳 Decision Tree Classifier")

    # Hyperparameters
    st.markdown("#### 🔧 Hyperparameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        criterion = st.selectbox("Criterion", ["gini", "entropy"], index=0)
        max_depth = st.slider("Max Depth", 1, 20, 5,
                               help="Kedalaman maksimum tree. Lebih dalam = lebih kompleks.")

    with col2:
        min_samples_split = st.slider("Min Samples Split", 2, 50, 2,
                                       help="Minimum sampel untuk split node.")
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 20, 1,
                                      help="Minimum sampel di leaf node.")

    with col3:
        max_features = st.selectbox("Max Features", [None, "sqrt", "log2"], index=0)

    # Train model
    dt_model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
    )
    dt_model.fit(X_train, y_train)

    # Predictions
    y_pred = dt_model.predict(X_test)
    y_pred_train = dt_model.predict(X_train)

    # Cross-validation
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=cv_folds, scoring='accuracy')

    st.markdown("---")

    # ─── Results ───────────────────────────────────────────────
    st.markdown("#### 📊 Hasil Evaluasi")

    col1, col2, col3, col4 = st.columns(4)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    col1.metric("🎯 Accuracy", f"{acc:.4f}")
    col2.metric("🔍 Precision", f"{prec:.4f}")
    col3.metric("📋 Recall", f"{rec:.4f}")
    col4.metric("⚖️ F1-Score", f"{f1:.4f}")

    # Cross-validation
    st.markdown(
        f"""
        <div class="success-box">
            <strong>Cross-Validation ({cv_folds}-fold):</strong>
            Mean Accuracy = <strong>{cv_scores.mean():.4f}</strong> ± {cv_scores.std():.4f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overfitting check
    train_acc = accuracy_score(y_train, y_pred_train)
    if train_acc - acc > 0.1:
        st.warning(
            f"⚠️ **Kemungkinan Overfitting!** "
            f"Train Accuracy: {train_acc:.4f} vs Test Accuracy: {acc:.4f} "
            f"(Gap: {train_acc - acc:.4f}). Coba kurangi max_depth atau tingkatkan min_samples_leaf."
        )

    # Confusion Matrix
    st.markdown("#### 📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    labels = [target_labels.get(i, str(i)) for i in sorted(target_labels.keys())]

    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=labels, y=labels,
        labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
        title="Confusion Matrix — Decision Tree",
    )
    fig.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    st.markdown("#### 📋 Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=labels,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

    # Feature Importance
    st.markdown("#### 🏆 Feature Importance")
    importance = pd.DataFrame({
        'Fitur': feature_names,
        'Importance': dt_model.feature_importances_,
    }).sort_values('Importance', ascending=True)

    fig = px.bar(
        importance, x='Importance', y='Fitur', orientation='h',
        title="Feature Importance — Decision Tree",
        color='Importance', color_continuous_scale="Viridis",
    )
    fig.update_layout(height=max(400, len(feature_names) * 25), margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Tree rules (text)
    st.markdown("#### 🌲 Struktur Decision Tree (Text)")
    with st.expander("Lihat aturan tree"):
        tree_text = export_text(dt_model, feature_names=feature_names, max_depth=5)
        st.code(tree_text, language="text")

    # Generated code
    st.markdown("#### 💻 Kode Python")
    with st.expander("Lihat kode lengkap"):
        st.code(
            f"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={st.session_state.get('model_test_size', 20)/100}, random_state=42, stratify=y
)

# Inisialisasi model
dt = DecisionTreeClassifier(
    criterion='{criterion}',
    max_depth={max_depth},
    min_samples_split={min_samples_split},
    min_samples_leaf={min_samples_leaf},
    random_state=42
)

# Training
dt.fit(X_train, y_train)

# Prediksi
y_pred = dt.predict(X_test)

# Evaluasi
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(dt, X_train, y_train, cv={cv_folds})
print(f"CV Accuracy: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")
            """,
            language="python",
        )


def _render_naive_bayes(X_train, X_test, y_train, y_test,
                          feature_names, target_labels, cv_folds, use_scaling):
    """Naive Bayes training and evaluation."""
    st.markdown("### 📊 Gaussian Naive Bayes")

    st.markdown(
        """
        <div class="info-box">
            <strong>Gaussian Naive Bayes</strong> mengasumsikan fitur berdistribusi normal (Gaussian).
            Cocok untuk data kontinu seperti dataset yang kita gunakan.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Hyperparameters
    st.markdown("#### 🔧 Hyperparameters")
    var_smoothing_exp = st.slider(
        "Var Smoothing (10^x)", -12, 0, -9,
        help="Smoothing parameter. Nilai lebih besar = lebih smooth.",
    )
    var_smoothing = 10 ** var_smoothing_exp

    st.markdown(f"**var_smoothing** = {var_smoothing:.2e}")

    # Train model
    nb_model = GaussianNB(var_smoothing=var_smoothing)
    nb_model.fit(X_train, y_train)

    # Predictions
    y_pred = nb_model.predict(X_test)
    y_pred_train = nb_model.predict(X_train)
    y_prob = nb_model.predict_proba(X_test)

    # Cross-validation
    cv_scores = cross_val_score(nb_model, X_train, y_train, cv=cv_folds, scoring='accuracy')

    st.markdown("---")

    # ─── Results ───────────────────────────────────────────────
    st.markdown("#### 📊 Hasil Evaluasi")

    col1, col2, col3, col4 = st.columns(4)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    col1.metric("🎯 Accuracy", f"{acc:.4f}")
    col2.metric("🔍 Precision", f"{prec:.4f}")
    col3.metric("📋 Recall", f"{rec:.4f}")
    col4.metric("⚖️ F1-Score", f"{f1:.4f}")

    # Cross-validation
    st.markdown(
        f"""
        <div class="success-box">
            <strong>Cross-Validation ({cv_folds}-fold):</strong>
            Mean Accuracy = <strong>{cv_scores.mean():.4f}</strong> ± {cv_scores.std():.4f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Confusion Matrix
    st.markdown("#### 📊 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    labels = [target_labels.get(i, str(i)) for i in sorted(target_labels.keys())]

    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Oranges",
        x=labels, y=labels,
        labels={'x': 'Predicted', 'y': 'Actual', 'color': 'Count'},
        title="Confusion Matrix — Naive Bayes",
    )
    fig.update_layout(height=400, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Classification Report
    st.markdown("#### 📋 Classification Report")
    report = classification_report(
        y_test, y_pred,
        target_names=labels,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

    # Probability Distribution
    st.markdown("#### 📈 Distribusi Probabilitas Prediksi")
    prob_df = pd.DataFrame(y_prob, columns=labels)
    prob_df['Actual'] = [labels[int(i)] for i in y_test]
    prob_df['Predicted'] = [labels[int(i)] for i in y_pred]
    prob_df['Correct'] = prob_df['Actual'] == prob_df['Predicted']

    fig = px.histogram(
        prob_df, x=labels[0], color='Correct',
        title=f"Distribusi Probabilitas untuk kelas '{labels[0]}'",
        color_discrete_map={True: '#4CAF50', False: '#F44336'},
        barmode='overlay', opacity=0.7,
    )
    fig.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Class prior & parameters
    st.markdown("#### 📊 Parameter Model")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Class Prior (Probabilitas Prior):**")
        prior_df = pd.DataFrame({
            'Kelas': labels,
            'Prior P(C)': nb_model.class_prior_.round(4),
        })
        st.dataframe(prior_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Mean per Kelas (μ) — 5 fitur pertama:**")
        mean_df = pd.DataFrame(
            nb_model.theta_[:, :5].round(3),
            columns=feature_names[:5],
            index=labels,
        )
        st.dataframe(mean_df, use_container_width=True)

    # Generated code
    st.markdown("#### 💻 Kode Python")
    with st.expander("Lihat kode lengkap"):
        st.code(
            f"""
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling {'(opsional tapi direkomendasikan)' if use_scaling else '(tidak digunakan)'}
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inisialisasi model
nb = GaussianNB(var_smoothing={var_smoothing:.2e})

# Training
nb.fit(X_train_scaled, y_train)

# Prediksi
y_pred = nb.predict(X_test_scaled)
y_prob = nb.predict_proba(X_test_scaled)

# Evaluasi
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(nb, X_train_scaled, y_train, cv={cv_folds})
print(f"CV Accuracy: {{cv_scores.mean():.4f}} ± {{cv_scores.std():.4f}}")
            """,
            language="python",
        )
