"""
Presentation / Slide module — Materi Kuliah Tamu
"""

import streamlit as st


def render_presentation():
    """Render interactive presentation slides."""

    slides = [
        _slide_cover,
        _slide_outline,
        _slide_what_is_ml,
        _slide_supervised_learning,
        _slide_classification_intro,
        _slide_decision_tree_concept,
        _slide_decision_tree_how,
        _slide_decision_tree_splitting,
        _slide_decision_tree_pros_cons,
        _slide_naive_bayes_concept,
        _slide_naive_bayes_theorem,
        _slide_naive_bayes_types,
        _slide_naive_bayes_pros_cons,
        _slide_evaluation_metrics,
        _slide_confusion_matrix,
        _slide_workflow,
        _slide_comparison,
        _slide_real_world,
        _slide_demo_intro,
        _slide_thank_you,
    ]

    if "slide_idx" not in st.session_state:
        st.session_state.slide_idx = 0

    total = len(slides)
    idx = st.session_state.slide_idx

    col_prev, col_info, col_next = st.columns([1, 3, 1])

    with col_prev:
        if st.button("Sebelumnya", disabled=(idx == 0), use_container_width=True):
            st.session_state.slide_idx -= 1
            st.rerun()

    with col_info:
        st.markdown(
            f"<div style='text-align:center; padding:8px; color:#3a86ff; font-weight:600;'>"
            f"Slide {idx + 1} / {total}</div>",
            unsafe_allow_html=True,
        )

    with col_next:
        if st.button("Selanjutnya", disabled=(idx == total - 1), use_container_width=True):
            st.session_state.slide_idx += 1
            st.rerun()

    st.progress((idx + 1) / total)
    st.markdown("---")

    slides[idx]()

    with st.expander("Daftar Semua Slide"):
        slide_names = [
            "1. Cover",
            "2. Outline Materi",
            "3. Apa itu Machine Learning?",
            "4. Supervised Learning",
            "5. Pengenalan Klasifikasi",
            "6. Konsep Decision Tree",
            "7. Cara Kerja Decision Tree",
            "8. Splitting Criteria",
            "9. Pro & Kontra Decision Tree",
            "10. Konsep Naive Bayes",
            "11. Teorema Bayes",
            "12. Tipe-tipe Naive Bayes",
            "13. Pro & Kontra Naive Bayes",
            "14. Metrik Evaluasi",
            "15. Confusion Matrix",
            "16. Workflow ML",
            "17. Perbandingan DT vs NB",
            "18. Aplikasi Dunia Nyata",
            "19. Intro Demo",
            "20. Terima Kasih",
        ]
        selected = st.selectbox("Pilih slide:", slide_names, index=idx)
        new_idx = slide_names.index(selected)
        if new_idx != idx:
            st.session_state.slide_idx = new_idx
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# SLIDE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _slide_cover():
    st.markdown(
        """
        <div style="text-align:center; padding: 40px 20px;">
            <h1 style="font-size:2.8rem; color:#1a1a2e; margin-bottom:0.5rem;">
                Kuliah Tamu
            </h1>
            <h2 style="font-size:1.8rem; color:#3a86ff; margin-bottom:1rem;">
                Membangun Model Prediksi
            </h2>
            <h3 style="font-size:1.4rem; color:#555; font-weight:400;">
                Klasifikasi Data dengan<br>
                <strong>Decision Tree</strong> & <strong>Naive Bayes</strong>
            </h3>
            <hr style="width:200px; margin:30px auto; border-color:#3a86ff;">
            <p style="font-size:1.1rem; color:#888;">
                Kamis, 2026<br>
                Sesi Kuliah Tamu — Program Studi Informatika
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_outline():
    st.markdown("## Outline Materi")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Bagian I — Teori
            1. **Pengenalan Machine Learning**
               - Apa itu ML? Supervised vs Unsupervised
            2. **Konsep Klasifikasi**
               - Definisi & contoh kasus
            3. **Decision Tree**
               - Konsep, cara kerja, splitting criteria
            4. **Naive Bayes**
               - Teorema Bayes, asumsi, tipe-tipe
            """
        )

    with col2:
        st.markdown(
            """
            ### Bagian II — Praktik
            5. **Evaluasi Model**
               - Accuracy, Precision, Recall, F1-Score
            6. **Hands-on Demo**
               - Eksplorasi data (EDA)
               - Preprocessing
               - Training & evaluasi
               - Perbandingan model
            7. **Prediksi Interaktif**
               - Live prediction dengan model
            """
        )


def _slide_what_is_ml():
    st.markdown("## Apa itu Machine Learning?")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Definisi:</strong><br>
            Machine Learning adalah cabang dari Artificial Intelligence (AI) yang memungkinkan
            komputer untuk <strong>belajar dari data</strong> dan membuat keputusan atau prediksi
            <strong>tanpa diprogram secara eksplisit</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Supervised Learning
            - Belajar dari data **berlabel**
            - Contoh: Klasifikasi, Regresi
            - *"Guru memberi contoh soal & jawaban"*
            """
        )

    with col2:
        st.markdown(
            """
            ### Unsupervised Learning
            - Belajar dari data **tanpa label**
            - Contoh: Clustering, Dimensionality Reduction
            - *"Siswa mengelompokkan sendiri"*
            """
        )

    with col3:
        st.markdown(
            """
            ### Reinforcement Learning
            - Belajar dari **reward/punishment**
            - Contoh: Game AI, Robotics
            - *"Belajar dari trial & error"*
            """
        )


def _slide_supervised_learning():
    st.markdown("## Supervised Learning")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            Supervised Learning menggunakan dataset yang memiliki <strong>input (fitur/X)</strong>
            dan <strong>output (label/Y)</strong> untuk melatih model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Klasifikasi
            Output berupa **kategori/kelas diskrit**

            | Input | Output |
            |-------|--------|
            | Email | Spam / Tidak Spam |
            | Gambar | Kucing / Anjing |
            | Data Pasien | Sakit / Sehat |
            """
        )

    with col2:
        st.markdown(
            """
            ### Regresi
            Output berupa **nilai kontinu**

            | Input | Output |
            |-------|--------|
            | Luas rumah | Harga (Rp) |
            | Pengalaman | Gaji (Rp) |
            | Suhu | Penjualan es |
            """
        )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Fokus hari ini:</strong> Klasifikasi menggunakan Decision Tree & Naive Bayes
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_classification_intro():
    st.markdown("## Pengenalan Klasifikasi")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Klasifikasi</strong> adalah proses memprediksi <strong>kelas/kategori</strong>
            dari suatu data berdasarkan atribut-atribut yang dimilikinya.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Contoh Kasus Nyata

        | Domain | Problem | Kelas |
        |--------|---------|-------|
        | Kesehatan | Diagnosis penyakit jantung | Sakit / Sehat |
        | Email | Deteksi spam | Spam / Bukan Spam |
        | Perbankan | Deteksi fraud | Fraud / Legitimate |
        | Pendidikan | Prediksi kelulusan | Lulus / Tidak Lulus |
        | E-commerce | Prediksi pembelian | Beli / Tidak Beli |

        ### Algoritma Klasifikasi Populer
        - Decision Tree
        - Naive Bayes
        - Neural Network / Deep Learning
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Random Forest
        """
    )


def _slide_decision_tree_concept():
    st.markdown("## Decision Tree — Konsep")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Decision Tree</strong> adalah algoritma klasifikasi yang membuat keputusan
            dalam bentuk <strong>struktur pohon</strong>. Setiap node internal merepresentasikan
            sebuah <strong>tes pada atribut</strong>, cabang merepresentasikan <strong>hasil tes</strong>,
            dan daun merepresentasikan <strong>kelas/keputusan</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.markdown(
        """
        ### Analogi Sederhana

        ```
        Apakah cuaca cerah?
        |-- Ya --> Apakah angin kencang?
        |          |-- Ya  --> [Tidak bermain]
        |          +-- Tidak --> [Bermain]
        +-- Tidak --> Apakah hujan?
                   |-- Ya  --> [Tidak bermain]
                   +-- Tidak --> [Bermain]
        ```

        ### Komponen Decision Tree
        - **Root Node**: Node paling atas (pertanyaan pertama)
        - **Internal Node**: Node percabangan (pertanyaan lanjutan)
        - **Leaf Node**: Node akhir (keputusan/kelas)
        - **Branch**: Cabang yang menghubungkan node
        """
    )


def _slide_decision_tree_how():
    st.markdown("## Cara Kerja Decision Tree")
    st.markdown("---")

    st.markdown(
        """
        ### Proses Pembangunan Decision Tree

        1. **Pilih atribut terbaik** sebagai root node
           - Gunakan kriteria splitting (Information Gain / Gini Index)
        2. **Bagi data** berdasarkan nilai atribut terpilih
        3. **Ulangi proses** secara rekursif untuk setiap cabang
        4. **Hentikan** ketika:
           - Semua data di node sudah satu kelas (murni)
           - Tidak ada atribut tersisa
           - Mencapai kedalaman maksimum
        """
    )

    st.markdown(
        """
        ### Contoh: Prediksi Penyakit Jantung

        ```
        [Seluruh Data: 300 pasien]
                    |
            Chest Pain Type?
            /              \\
        Type 0-1          Type 2-3
        [180 pasien]      [120 pasien]
            |                  |
        Max Heart Rate?    Age > 55?
        /       \\          /       \\
      >150     <=150     Ya      Tidak
      [Sehat]  [Cek]    [Sakit]  [Cek]
        ```
        """
    )


def _slide_decision_tree_splitting():
    st.markdown("## Splitting Criteria")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Information Gain (ID3 / C4.5)

            Berbasis **Entropy** (ukuran ketidakteraturan):

            $$Entropy(S) = -\\sum_{i=1}^{c} p_i \\log_2(p_i)$$

            $$IG(S, A) = Entropy(S) - \\sum_{v} \\frac{|S_v|}{|S|} Entropy(S_v)$$

            - Entropy = 0 → Data **murni** (1 kelas)
            - Entropy = 1 → Data **sangat campuran**
            - Pilih atribut dengan **IG tertinggi**
            """
        )

    with col2:
        st.markdown(
            """
            ### Gini Index (CART)

            Mengukur **impurity** (ketidakmurnian):

            $$Gini(S) = 1 - \\sum_{i=1}^{c} p_i^2$$

            $$Gini_{split} = \\sum_{v} \\frac{|S_v|}{|S|} Gini(S_v)$$

            - Gini = 0 → Data **murni** (1 kelas)
            - Gini = 0.5 → Data **sangat campuran**
            - Pilih atribut dengan **Gini terendah**
            """
        )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Catatan:</strong> Scikit-learn menggunakan <strong>Gini Index</strong> sebagai default
            untuk DecisionTreeClassifier, tetapi mendukung juga <strong>Entropy</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_decision_tree_pros_cons():
    st.markdown("## Decision Tree — Kelebihan & Kekurangan")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Kelebihan
            - **Mudah dipahami** & divisualisasikan
            - **Tidak perlu normalisasi** data
            - Bisa menangani data **numerik & kategorikal**
            - Dapat menangkap **hubungan non-linear**
            - **Feature importance** otomatis
            - Cepat dalam prediksi
            """
        )

    with col2:
        st.markdown(
            """
            ### Kekurangan
            - Rentan terhadap **overfitting**
            - **Tidak stabil** — perubahan kecil pada data bisa mengubah tree
            - Cenderung bias pada fitur dengan banyak level
            - Bisa menjadi sangat **kompleks**
            - Tidak optimal untuk hubungan yang **sangat linear**
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Solusi Overfitting:</strong> Gunakan <em>pruning</em>
            (max_depth, min_samples_leaf) atau gunakan <strong>Random Forest</strong> (ensemble).
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_naive_bayes_concept():
    st.markdown("## Naive Bayes — Konsep")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Naive Bayes</strong> adalah algoritma klasifikasi berbasis
            <strong>Teorema Bayes</strong> dengan asumsi bahwa setiap fitur
            <strong>independen satu sama lain</strong> (asumsi "naive").
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Intuisi

        **Bayangkan seorang dokter** mendiagnosis pasien:
        - Dokter sudah tahu **probabilitas umum** penyakit jantung (prior)
        - Pasien datang dengan **gejala tertentu** (evidence)
        - Dokter **mengupdate keyakinannya** berdasarkan gejala (posterior)

        ### Konsep Kunci
        - **Prior**: Probabilitas awal sebelum melihat data ($P(C)$)
        - **Likelihood**: Probabilitas fitur mengingat kelas ($P(X|C)$)
        - **Evidence**: Probabilitas fitur ($P(X)$)
        - **Posterior**: Probabilitas kelas mengingat fitur ($P(C|X)$) — **yang kita cari!**
        """
    )


def _slide_naive_bayes_theorem():
    st.markdown("## Teorema Bayes")
    st.markdown("---")

    st.markdown(
        """
        ### Formula Teorema Bayes

        $$P(C|X) = \\frac{P(X|C) \\cdot P(C)}{P(X)}$$

        | Simbol | Nama | Penjelasan |
        |--------|------|------------|
        | $P(C|X)$ | **Posterior** | Probabilitas kelas C diberikan fitur X |
        | $P(X|C)$ | **Likelihood** | Probabilitas fitur X diberikan kelas C |
        | $P(C)$ | **Prior** | Probabilitas awal kelas C |
        | $P(X)$ | **Evidence** | Probabilitas fitur X (normalisasi) |

        ### Dengan Asumsi "Naive" (Independen)

        $$P(X|C) = P(x_1|C) \\cdot P(x_2|C) \\cdot ... \\cdot P(x_n|C) = \\prod_{i=1}^{n} P(x_i|C)$$

        ### Keputusan Klasifikasi

        $$\\hat{y} = \\arg\\max_{c} P(C=c) \\prod_{i=1}^{n} P(x_i|C=c)$$
        """
    )


def _slide_naive_bayes_types():
    st.markdown("## Tipe-Tipe Naive Bayes")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Gaussian NB
            - Fitur berdistribusi **normal**
            - Cocok untuk data **kontinu**
            - Estimasi mean dan std per kelas

            $$P(x_i|C) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(x_i - \\mu)^2}{2\\sigma^2}}$$

            **Paling umum digunakan**
            """
        )

    with col2:
        st.markdown(
            """
            ### Multinomial NB
            - Fitur berupa **frekuensi/count**
            - Cocok untuk **text classification**
            - Menggunakan distribusi multinomial

            **Aplikasi:**
            - Spam detection
            - Sentiment analysis
            - Document classification
            """
        )

    with col3:
        st.markdown(
            """
            ### Bernoulli NB
            - Fitur berupa **biner (0/1)**
            - Cocok untuk data **boolean**
            - Menggunakan distribusi Bernoulli

            **Aplikasi:**
            - Ada/tidak ada kata
            - Feature presence
            - Binary attributes
            """
        )


def _slide_naive_bayes_pros_cons():
    st.markdown("## Naive Bayes — Kelebihan & Kekurangan")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Kelebihan
            - **Sangat cepat** (training & prediksi)
            - Bekerja baik dengan **dataset kecil**
            - Efektif untuk **high-dimensional data**
            - **Tidak sensitif** terhadap fitur tidak relevan
            - Mudah diimplementasikan
            - Baik untuk **baseline model**
            """
        )

    with col2:
        st.markdown(
            """
            ### Kekurangan
            - Asumsi independensi **jarang terpenuhi**
            - Estimasi probabilitas bisa **kurang akurat**
            - Tidak menangkap **interaksi antar fitur**
            - Sensitif terhadap **distribusi data**
            - "Zero frequency problem" — perlu **Laplace smoothing**
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Tips:</strong> Meskipun asumsi independensi jarang terpenuhi sepenuhnya,
            Naive Bayes sering memberikan <strong>hasil yang surprisingly baik</strong> dalam praktik!
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_evaluation_metrics():
    st.markdown("## Metrik Evaluasi Model")
    st.markdown("---")

    st.markdown(
        """
        | Metrik | Formula | Penjelasan |
        |--------|---------|------------|
        | **Accuracy** | $\\frac{TP + TN}{TP + TN + FP + FN}$ | Proporsi prediksi yang benar secara keseluruhan |
        | **Precision** | $\\frac{TP}{TP + FP}$ | Dari yang diprediksi positif, berapa yang benar? |
        | **Recall** | $\\frac{TP}{TP + FN}$ | Dari yang sebenarnya positif, berapa yang terdeteksi? |
        | **F1-Score** | $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$ | Harmonik mean dari Precision & Recall |

        ### Kapan Menggunakan Metrik Apa?

        | Skenario | Metrik Utama | Alasan |
        |----------|-------------|--------|
        | Diagnosis penyakit | **Recall** | Jangan sampai ada pasien sakit yang terlewat |
        | Filter spam | **Precision** | Jangan sampai email penting masuk spam |
        | Balanced | **F1-Score** | Keseimbangan precision & recall |
        | Data seimbang | **Accuracy** | Proporsi kelas merata |
        """
    )


def _slide_confusion_matrix():
    st.markdown("## Confusion Matrix")
    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
            <strong>Confusion Matrix</strong> adalah tabel yang menunjukkan performa model
            klasifikasi dengan membandingkan nilai <strong>aktual vs prediksi</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        |  | **Prediksi: Positif** | **Prediksi: Negatif** |
        |--|----------------------|----------------------|
        | **Aktual: Positif** | True Positive (TP) | False Negative (FN) |
        | **Aktual: Negatif** | False Positive (FP) | True Negative (TN) |

        ### Penjelasan
        - **TP**: Model prediksi Positif, dan memang Positif (benar)
        - **TN**: Model prediksi Negatif, dan memang Negatif (benar)
        - **FP**: Model prediksi Positif, padahal Negatif (Type I Error)
        - **FN**: Model prediksi Negatif, padahal Positif (Type II Error)
        """
    )

    st.markdown(
        """
        <div class="warning-box">
            <strong>Contoh Medis:</strong><br>
            FN (False Negative) = Pasien <strong>sakit</strong> tapi didiagnosis <strong>sehat</strong> — <strong>Berbahaya!</strong><br>
            FP (False Positive) = Pasien <strong>sehat</strong> tapi didiagnosis <strong>sakit</strong> — Perlu tes lanjutan
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_workflow():
    st.markdown("## Workflow Machine Learning")
    st.markdown("---")

    st.markdown(
        """
        ```
        +-------------------+
        | 1. PENGUMPULAN    |    Kumpulkan data dari sumber yang relevan
        |    DATA           |
        +---------+---------+
                  |
        +---------v---------+
        | 2. EKSPLORASI     |    EDA: Pahami distribusi, korelasi, outlier
        |    DATA (EDA)     |
        +---------+---------+
                  |
        +---------v---------+
        | 3. PREPROCESSING  |    Cleaning, encoding, scaling, split data
        |                   |
        +---------+---------+
                  |
        +---------v---------+
        | 4. PEMILIHAN &    |    Pilih algoritma, latih model
        |    TRAINING MODEL |
        +---------+---------+
                  |
        +---------v---------+
        | 5. EVALUASI       |    Accuracy, Precision, Recall, F1, CM
        |    MODEL          |
        +---------+---------+
                  |
        +---------v---------+
        | 6. DEPLOYMENT     |    Deploy model ke production
        |    & PREDIKSI     |
        +-------------------+
        ```

        > **Hari ini kita akan mempraktikkan semua langkah ini secara interaktif!**
        """
    )


def _slide_comparison():
    st.markdown("## Decision Tree vs Naive Bayes")
    st.markdown("---")

    st.markdown(
        """
        | Aspek | Decision Tree | Naive Bayes |
        |-------|--------------|-------------|
        | **Pendekatan** | Rule-based (if-then) | Probabilistik |
        | **Interpretasi** | Sangat mudah (visual tree) | Cukup mudah (probabilitas) |
        | **Kecepatan Training** | Sedang | Sangat Cepat |
        | **Kecepatan Prediksi** | Cepat | Sangat Cepat |
        | **Data Kecil** | Kurang baik | Baik |
        | **Data Besar** | Baik | Baik |
        | **Overfitting** | Rentan (tanpa pruning) | Tahan |
        | **Asumsi** | Tidak ada | Independensi fitur |
        | **Missing Values** | Bisa ditangani | Perlu handling |
        | **Fitur Interaksi** | Ya (natural) | Tidak |
        | **Normalisasi** | Tidak perlu | Tergantung tipe |
        | **Best Use Case** | Data terstruktur, interpretasi penting | Text, high-dim, baseline |
        """
    )


def _slide_real_world():
    st.markdown("## Aplikasi Dunia Nyata")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### Decision Tree
            - **Diagnosis medis** — Sistem pendukung keputusan klinis
            - **Credit scoring** — Kelayakan kredit
            - **Quality control** — Deteksi produk cacat
            - **Customer churn** — Prediksi pelanggan keluar
            - **Pertanian** — Klasifikasi jenis tanah
            """
        )

    with col2:
        st.markdown(
            """
            ### Naive Bayes
            - **Email filtering** — Deteksi spam
            - **Sentiment analysis** — Analisis opini
            - **News classification** — Kategorisasi berita
            - **Medical diagnosis** — Screening awal
            - **Recommendation** — Sistem rekomendasi
            """
        )

    st.markdown(
        """
        <div class="success-box">
            <strong>Fun Fact:</strong> Google menggunakan variasi Naive Bayes untuk
            spam filtering di Gmail, dan Decision Tree (via Random Forest/XGBoost) banyak
            digunakan di kompetisi Kaggle!
        </div>
        """,
        unsafe_allow_html=True,
    )


def _slide_demo_intro():
    st.markdown("## Saatnya Demo!")
    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center; padding:30px;">
            <h3 style="color:#3a86ff;">Mari kita praktikkan teori yang sudah dipelajari!</h3>
            <p style="font-size:1.1rem; color:#555;">
                Gunakan <strong>menu navigasi di sidebar</strong> untuk mengakses setiap tahap:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Eksplorasi Data
            Pahami data sebelum modeling

            ### Preprocessing
            Siapkan data untuk training
            """
        )

    with col2:
        st.markdown(
            """
            ### Pemodelan
            Training Decision Tree & Naive Bayes

            ### Perbandingan
            Bandingkan performa kedua model
            """
        )

    with col3:
        st.markdown(
            """
            ### Prediksi
            Coba prediksi data baru secara live!

            ### Visualisasi
            Grafik & chart interaktif
            """
        )


def _slide_thank_you():
    st.markdown(
        """
        <div style="text-align:center; padding:50px 20px;">
            <h1 style="font-size:3rem; color:#1a1a2e;">Terima Kasih</h1>
            <h3 style="color:#555; font-weight:400; margin-top:1rem;">
                Semoga materi ini bermanfaat!
            </h3>
            <hr style="width:200px; margin:30px auto; border-color:#3a86ff;">
            <p style="font-size:1.1rem; color:#888;">
                <strong>Sesi Tanya Jawab</strong><br>
                Silakan ajukan pertanyaan
            </p>
            <br>
            <p style="font-size:0.9rem; color:#aaa;">
                Hubungi saya untuk diskusi lebih lanjut<br>
                Source code tersedia di GitHub
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
