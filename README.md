# Workflow-CI_Muhammad_Muharram_Ash_shiddiqie

Repository ini berisi workflow CI untuk training model machine learning menggunakan MLflow Project.

## Struktur Folder

```
Workflow-CI_Muhammad_Muharram_Ash_shiddiqie/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml       # GitHub Actions workflow
├── MLProject/
│   ├── MLProject               # Konfigurasi MLflow Project
│   ├── conda.yaml              # Environment dependencies
│   ├── modelling.py            # Script training model
│   └── wine_preprocessed/      # Dataset hasil preprocessing
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── scaler.pkl
│       └── feature_columns.pkl
└── README.md
```

## Cara Menjalankan

1. **Lokal:**
   ```bash
   cd MLProject
   mlflow run .
   ```

2. **GitHub Actions:**
   - Workflow akan otomatis berjalan saat push ke branch main
   - Atau trigger manual via workflow_dispatch

## Kriteria
- Basic: Workflow CI yang menjalankan training model
- Skilled: Menyimpan artefak ke GitHub Actions
- Advanced: Build Docker image dan push ke Docker Hub