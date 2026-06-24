from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("data/dataset.csv")
RESULTS_DIR = Path("results")

LABEL_COLUMNS = ["Label", "label", "target", "Class"]


def normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def find_label_column(df: pd.DataFrame) -> str:
    normalized_candidates = {normalize_column_name(col) for col in LABEL_COLUMNS}
    for col in df.columns:
        if normalize_column_name(col) in normalized_candidates:
            return col
    raise ValueError(
        f"No se encontro columna de etiqueta. Busque: {LABEL_COLUMNS}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def convert_to_binary_label(value: object) -> int:
    value = str(value).strip().lower()
    if value == "benign":
        return 0
    return 1


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {DATASET_PATH}. "
            "Guarda el CSV como data/dataset.csv"
        )

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Cargando dataset...")
    df = pd.read_csv(DATASET_PATH)

    # Algunos datasets IDS traen espacios extra en los headers.
    df.columns = [str(col).strip() for col in df.columns]

    print("Limpiando datos...")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    label_col = find_label_column(df)

    y = df[label_col].apply(convert_to_binary_label)
    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=[np.number])

    print(f"Columna objetivo detectada: {label_col}")
    print(f"Features numericas utilizadas: {X.shape[1]}")
    print(f"Cantidad de registros: {X.shape[0]}")

    print("Separando train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Escalando datos...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Aplicando PCA...")
    pca_components = 4
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_pca, y_train)

    print("Evaluando modelo...")
    y_pred = model.predict(X_test_pca)

    metrics = {
        "model_name": "Random Forest",
        "environment": "Classical",
        "pca_components": pca_components,
        "metrics": {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    joblib.dump(model, RESULTS_DIR / "random_forest_model.joblib")
    joblib.dump(scaler, RESULTS_DIR / "scaler.joblib")
    joblib.dump(pca, RESULTS_DIR / "pca.joblib")

    with open(RESULTS_DIR / "classical_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("Entrenamiento finalizado.")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
