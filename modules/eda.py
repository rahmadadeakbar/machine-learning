"""
Exploratory Data Analysis (EDA) module
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def render_eda(df, feature_names, target_name, target_labels, dataset_info):
    """Render EDA section."""

    st.markdown("## Eksplorasi Data (EDA)")
    st.markdown(
        """
        <div class="info-box">
            <strong>EDA</strong> adalah proses awal untuk memahami data sebelum membangun model.
            Kita akan mengeksplorasi distribusi, korelasi, dan pola dalam data.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Informasi Dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Data", f"{len(df):,}")
    col2.metric("Jumlah Fitur", len(feature_names))
    col3.metric("Jumlah Kelas", dataset_info['n_classes'])
    col4.metric("Sumber", dataset_info['source'].split('(')[0].strip()[:20])

    st.markdown(f"> **{dataset_info['name']}**: {dataset_info['description']}")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Preview",
        "Statistik Deskriptif",
        "Distribusi Fitur",
        "Korelasi",
        "Distribusi Target",
    ])

    with tab1:
        _render_data_preview(df, feature_names, dataset_info)

    with tab2:
        _render_statistics(df, feature_names)

    with tab3:
        _render_distributions(df, feature_names, target_name, target_labels)

    with tab4:
        _render_correlation(df, feature_names)

    with tab5:
        _render_target_distribution(df, target_name, target_labels)


def _render_data_preview(df, feature_names, dataset_info):
    """Show data preview."""
    st.markdown("#### Preview Data (10 baris pertama)")
    st.dataframe(df.head(10), use_container_width=True, height=400)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Tipe Data")
        dtype_df = pd.DataFrame({
            'Fitur': df.columns,
            'Tipe Data': df.dtypes.astype(str).values,
            'Non-Null': df.notnull().sum().values,
            'Null': df.isnull().sum().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Deskripsi Fitur")
        if 'features_desc' in dataset_info:
            desc_df = pd.DataFrame({
                'Fitur': list(dataset_info['features_desc'].keys()),
                'Keterangan': list(dataset_info['features_desc'].values()),
            })
            st.dataframe(desc_df, use_container_width=True, hide_index=True)

    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"Terdapat {missing.sum()} missing values!")
        st.dataframe(missing[missing > 0].reset_index(), use_container_width=True)
    else:
        st.success("Tidak ada missing values dalam dataset.")


def _render_statistics(df, feature_names):
    """Show descriptive statistics."""
    st.markdown("#### Statistik Deskriptif")
    st.dataframe(df[feature_names].describe().round(2), use_container_width=True)

    st.markdown("#### Penjelasan")
    st.markdown(
        """
        | Statistik | Penjelasan |
        |-----------|-----------|
        | **count** | Jumlah data non-null |
        | **mean** | Rata-rata |
        | **std** | Standar deviasi (sebaran data) |
        | **min** | Nilai minimum |
        | **25%** | Kuartil pertama (Q1) |
        | **50%** | Median (Q2) |
        | **75%** | Kuartil ketiga (Q3) |
        | **max** | Nilai maksimum |
        """
    )


def _render_distributions(df, feature_names, target_name, target_labels):
    """Show feature distributions."""
    st.markdown("#### Distribusi Fitur")

    selected_features = st.multiselect(
        "Pilih fitur untuk divisualisasikan:",
        feature_names,
        default=feature_names[:4],
    )

    if not selected_features:
        st.warning("Pilih minimal satu fitur!")
        return

    chart_type = st.radio(
        "Tipe chart:", ["Histogram", "Box Plot", "Violin Plot"],
        horizontal=True,
    )

    df_plot = df.copy()
    df_plot['Kelas'] = df_plot[target_name].map(target_labels)

    palette = ["#3a86ff", "#8338ec", "#ff006e", "#fb5607", "#ffbe0b"]

    if chart_type == "Histogram":
        for i in range(0, len(selected_features), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(selected_features):
                    feat = selected_features[idx]
                    with col:
                        fig = px.histogram(
                            df_plot, x=feat, color='Kelas',
                            barmode='overlay', opacity=0.7,
                            title=f"Distribusi: {feat}",
                            color_discrete_sequence=palette,
                        )
                        fig.update_layout(height=350, margin=dict(t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        for i in range(0, len(selected_features), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(selected_features):
                    feat = selected_features[idx]
                    with col:
                        fig = px.box(
                            df_plot, x='Kelas', y=feat, color='Kelas',
                            title=f"Box Plot: {feat}",
                            color_discrete_sequence=palette,
                        )
                        fig.update_layout(height=350, margin=dict(t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Violin Plot":
        for i in range(0, len(selected_features), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(selected_features):
                    feat = selected_features[idx]
                    with col:
                        fig = px.violin(
                            df_plot, x='Kelas', y=feat, color='Kelas',
                            box=True, points="outliers",
                            title=f"Violin Plot: {feat}",
                            color_discrete_sequence=palette,
                        )
                        fig.update_layout(height=350, margin=dict(t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)


def _render_correlation(df, feature_names):
    """Show correlation analysis."""
    st.markdown("#### Analisis Korelasi")

    corr = df[feature_names].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Heatmap Korelasi Antar Fitur",
    )
    fig.update_layout(height=600, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="info-box">
            <strong>Interpretasi:</strong><br>
            - Nilai mendekati <strong>+1</strong> → Korelasi positif kuat<br>
            - Nilai mendekati <strong>-1</strong> → Korelasi negatif kuat<br>
            - Nilai mendekati <strong>0</strong> → Tidak ada korelasi linear
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Top 10 Korelasi Tertinggi (absolut)")
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_pairs = upper_tri.stack().reset_index()
    corr_pairs.columns = ['Fitur 1', 'Fitur 2', 'Korelasi']
    corr_pairs['|Korelasi|'] = corr_pairs['Korelasi'].abs()
    corr_pairs = corr_pairs.sort_values('|Korelasi|', ascending=False).head(10)
    st.dataframe(corr_pairs.round(3), use_container_width=True, hide_index=True)


def _render_target_distribution(df, target_name, target_labels):
    """Show target distribution."""
    st.markdown("#### Distribusi Target/Label")

    target_counts = df[target_name].value_counts()
    target_df = pd.DataFrame({
        'Kelas': [target_labels.get(k, str(k)) for k in target_counts.index],
        'Jumlah': target_counts.values,
        'Persentase (%)': (target_counts.values / len(df) * 100).round(1),
    })

    palette = ["#3a86ff", "#8338ec", "#ff006e"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            target_df, values='Jumlah', names='Kelas',
            title="Distribusi Kelas Target",
            color_discrete_sequence=palette,
            hole=0.4,
        )
        fig.update_layout(height=400, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            target_df, x='Kelas', y='Jumlah', color='Kelas',
            title="Jumlah Data Per Kelas",
            color_discrete_sequence=palette,
            text='Jumlah',
        )
        fig.update_layout(height=400, margin=dict(t=40, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(target_df, use_container_width=True, hide_index=True)

    ratio = target_counts.min() / target_counts.max()
    if ratio < 0.5:
        st.warning(
            f"Dataset **tidak seimbang** (rasio = {ratio:.2f}). "
            "Pertimbangkan teknik balancing seperti SMOTE, undersampling, atau class_weight."
        )
    else:
        st.success(f"Dataset cukup **seimbang** (rasio = {ratio:.2f}).")
