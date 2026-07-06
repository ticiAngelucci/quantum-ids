from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.classical.train_model import convert_to_binary_label, find_label_column
from src.live_detection.feature_engineering import curate_live_windows, enrich_live_feature_frame


LIVE_DATASET_PATH = Path("results/live_training_dataset.csv")
RESULTS_PATH = Path("results/classical_live_metrics.json")


def train_classical_live_baseline(
    dataset_path: Path = LIVE_DATASET_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset live en {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.columns = [str(column).strip() for column in df.columns]
    label_column = find_label_column(df)
    y = df[label_column].apply(convert_to_binary_label).to_numpy()
    metadata_columns = [column for column in ("Scenario", "SimulatorVersion") if column in df.columns]
    metadata_df = df[metadata_columns].copy() if metadata_columns else None
    X = df.drop(columns=[label_column]).select_dtypes(include=["number"])
    X = enrich_live_feature_frame(X)
    X, y, curation_report = curate_live_windows(
        X.reset_index(drop=True),
        y,
        metadata=metadata_df,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca_components = min(4, X.shape[1], X_train.shape[0])
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    payload = {
        "model_name": "Random Forest Live Baseline",
        "environment": "Classical Live",
        "dataset_source": "live",
        "dataset_path": str(dataset_path),
        "pca_components": pca_components,
        "sample_size": int(len(X)),
        "test_size": test_size,
        "live_curation_report": curation_report.__dict__,
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    RESULTS_PATH.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return payload


def main() -> None:
    payload = train_classical_live_baseline()
    print(json.dumps(payload, indent=4))


if __name__ == "__main__":
    main()
