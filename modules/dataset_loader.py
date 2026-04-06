"""
Dataset loader module
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer


def load_dataset(choice: str):
    """Load and return dataset based on user choice."""

    if choice == "Heart Disease":
        return _load_heart_disease()
    elif choice == "Breast Cancer":
        return _load_breast_cancer()
    elif choice == "Iris":
        return _load_iris()


def _load_heart_disease():
    """Generate a realistic Heart Disease dataset."""
    np.random.seed(42)
    n = 600

    age = np.random.normal(54, 9, n).astype(int)
    age = np.clip(age, 29, 77)

    sex = np.random.choice([0, 1], n, p=[0.32, 0.68])

    cp = np.random.choice([0, 1, 2, 3], n, p=[0.47, 0.17, 0.28, 0.08])

    trestbps = np.random.normal(131, 17, n).astype(int)
    trestbps = np.clip(trestbps, 94, 200)

    chol = np.random.normal(246, 52, n).astype(int)
    chol = np.clip(chol, 126, 564)

    fbs = np.random.choice([0, 1], n, p=[0.85, 0.15])

    restecg = np.random.choice([0, 1, 2], n, p=[0.48, 0.49, 0.03])

    thalach = np.random.normal(149, 23, n).astype(int)
    thalach = np.clip(thalach, 71, 202)

    exang = np.random.choice([0, 1], n, p=[0.67, 0.33])

    oldpeak = np.abs(np.random.normal(1.04, 1.16, n)).round(1)
    oldpeak = np.clip(oldpeak, 0, 6.2)

    slope = np.random.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])

    ca = np.random.choice([0, 1, 2, 3], n, p=[0.58, 0.22, 0.13, 0.07])

    thal = np.random.choice([0, 1, 2, 3], n, p=[0.03, 0.06, 0.55, 0.36])

    # Target with realistic correlation
    risk_score = (
        0.02 * age
        + 0.3 * sex
        - 0.2 * (cp == 0).astype(int)
        + 0.01 * trestbps
        + 0.003 * chol
        + 0.3 * fbs
        - 0.01 * thalach
        + 0.5 * exang
        + 0.3 * oldpeak
        + 0.2 * ca
        + 0.3 * (thal == 3).astype(int)
    )
    prob = 1 / (1 + np.exp(-(risk_score - np.median(risk_score))))
    target = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'chest_pain_type': cp,
        'resting_bp': trestbps,
        'cholesterol': chol,
        'fasting_blood_sugar': fbs,
        'rest_ecg': restecg,
        'max_heart_rate': thalach,
        'exercise_angina': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'num_major_vessels': ca,
        'thalassemia': thal,
        'target': target,
    })

    feature_names = list(df.columns[:-1])
    target_name = 'target'
    target_labels = {0: 'No Disease', 1: 'Heart Disease'}
    dataset_info = {
        'name': 'Heart Disease Dataset',
        'description': 'Dataset untuk prediksi penyakit jantung berdasarkan atribut klinis pasien.',
        'n_samples': n,
        'n_features': len(feature_names),
        'n_classes': 2,
        'source': 'Inspired by UCI Heart Disease Dataset',
        'features_desc': {
            'age': 'Umur pasien (tahun)',
            'sex': 'Jenis kelamin (0=Perempuan, 1=Laki-laki)',
            'chest_pain_type': 'Tipe nyeri dada (0-3)',
            'resting_bp': 'Tekanan darah istirahat (mm Hg)',
            'cholesterol': 'Kolesterol serum (mg/dl)',
            'fasting_blood_sugar': 'Gula darah puasa > 120 mg/dl (0/1)',
            'rest_ecg': 'Hasil elektrokardiografi istirahat (0-2)',
            'max_heart_rate': 'Detak jantung maksimum',
            'exercise_angina': 'Angina akibat olahraga (0/1)',
            'oldpeak': 'Depresi ST akibat olahraga',
            'slope': 'Slope segmen ST puncak (0-2)',
            'num_major_vessels': 'Jumlah pembuluh darah utama (0-3)',
            'thalassemia': 'Thalassemia (0-3)',
        },
    }
    return df, feature_names, target_name, target_labels, dataset_info


def _load_breast_cancer():
    """Load sklearn Breast Cancer dataset."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    feature_names = list(data.feature_names)
    target_name = 'target'
    target_labels = {0: 'Malignant', 1: 'Benign'}
    dataset_info = {
        'name': 'Breast Cancer Wisconsin Dataset',
        'description': 'Dataset untuk diagnosis kanker payudara (ganas vs jinak) berdasarkan fitur sel tumor.',
        'n_samples': len(df),
        'n_features': len(feature_names),
        'n_classes': 2,
        'source': 'sklearn.datasets.load_breast_cancer (UCI ML Repository)',
        'features_desc': {f: f for f in feature_names},
    }
    return df, feature_names, target_name, target_labels, dataset_info


def _load_iris():
    """Load sklearn Iris dataset."""
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    feature_names = list(data.feature_names)
    target_name = 'target'
    target_labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    dataset_info = {
        'name': 'Iris Flower Dataset',
        'description': 'Dataset klasik untuk klasifikasi spesies bunga iris berdasarkan ukuran kelopak dan mahkota.',
        'n_samples': len(df),
        'n_features': len(feature_names),
        'n_classes': 3,
        'source': 'sklearn.datasets.load_iris (UCI ML Repository)',
        'features_desc': {f: f for f in feature_names},
    }
    return df, feature_names, target_name, target_labels, dataset_info
