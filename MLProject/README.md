# MLProject - Bank Marketing Classification

MLflow Project untuk training model Bank Marketing Classification dengan GitHub Actions CI/CD.

## 📋 Deskripsi

Project ini berisi pipeline lengkap untuk training model Random Forest Classifier menggunakan dataset Bank Marketing dengan fitur:

- **Data Preprocessing**: StandardScaler untuk feature scaling dan Label Encoding
- **Model Training**: RandomForestClassifier untuk memprediksi apakah nasabah akan berlangganan deposito berjangka
- **MLflow Integration**: Autologging untuk params, metrics, dan model artifacts
- **CI/CD**: GitHub Actions workflow untuk automated re-training

## 📁 Struktur File

```
MLProject/
├── MLProject              # MLflow Project configuration
├── conda.yaml            # Environment dependencies
├── modelling.py          # Training script dengan argparse
├── bank_preprocessed/    # Dataset hasil preprocessing
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── scaler.pkl
│   └── feature_columns.pkl
└── README.md             # Dokumentasi ini
```

## 🚀 Cara Menjalankan

### Prerequisites

**Installasi:**
- Python 3.12
- Conda (untuk environment)
- MLflow 2.19.0

---

### 1. Lokal (Development)

#### PowerShell
```powershell
# Install dependencies
conda env create -f conda.yaml
conda activate bank-marketing-classification

# Jalankan training
mlflow run . --experiment-name bank_marketing_classification

# Atau jalankan manual
python modelling.py --n_estimators=100 --max_depth=10 --random_state=42
```

---

### 2. MLflow UI

#### PowerShell
```powershell
mlflow ui --host 0.0.0.0 --port 8080
```

**Akses di:** `http://127.0.0.1:8080`

---

### 3. GitHub Actions (CI)

Workflow otomatis berjalan saat push ke branch `main` atau `master`.

**Workflow steps:**
1. Checkout repository
2. Setup Python 3.12
3. Install dependencies (mlflow, scikit-learn, pandas, dll.)
4. Run MLflow Project (`mlflow run .`)
5. Upload artifacts (mlruns/)

---

## ⚙️ Konfigurasi

### MLProject (Entry Point)

```yaml
name: bank-marketing-classification
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
      random_state: {type: int, default: 42}
    command: "python modelling.py --n_estimators={n_estimators} --max_depth={max_depth} --random_state={random_state}"
```

### Environment Dependencies (conda.yaml)

```yaml
name: bank-marketing-classification
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pip
  - pip:
    - mlflow==2.19.0
    - scikit-learn==1.5.2
    - pandas==2.2.3
    - numpy==1.26.4
    - matplotlib==3.9.0
    - seaborn==0.13.2
```

---

## 📊 Model Details

- **Algorithm**: RandomForestClassifier
- **Dataset**: Bank Marketing (UCI)
- **Target**: `y` (1: Yes, 0: No)
- **Features**: 20 fitur (telemarketing data)
- **Metrics**: Accuracy, F1-score, Precision, Recall
- **Artifacts**: Model (joblib/mlflow), Classification Report, Metrics

---

## 🔧 Hyperparameters

| Parameter | Default | Type |
|-----------|---------|------|
| n_estimators | 100 | int |
| max_depth | 10 | int |
| random_state | 42 | int |

---

## 📈 Metrics

- **accuracy**: Model accuracy pada test set
- **test_accuracy**: Manual log accuracy
- **test_precision**: Manual log precision
- **test_f1_score**: Manual log F1-score

---

## 🔄 CI/CD Workflow

**Trigger:**
- Push ke `main` atau `master`
- Pull request ke `main` atau `master`
- Manual trigger (workflow_dispatch)

**Output:**
- MLflow experiment: `bank_marketing_classification`
- Run name: `ci_bank_marketing_model`
- Artifacts: `mlruns/` (upload ke GitHub Actions)

---

## 📝 Catatan

1. **Data**: Pastikan `bank_preprocessed/` sudah ada sebelum training.
2. **MLflow**: Autologging aktif (`mlflow.sklearn.autolog()`).
3. **Artifacts**: Model disimpan di folder `model/` dalam run MLflow.

---

## 📞 Kontak

- **Author**: Muhammad Muharram Ash shiddiqie
- **GitHub**: [diqie123](https://github.com/diqie123)

---

**Happy Training! 🚀**
