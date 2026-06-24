from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.classical.train_model import find_label_column


INCOMPATIBLE_MESSAGE = (
    "El modelo actual fue entrenado con features CICIDS2017. "
    "Para usar live_capture.csv se debe entrenar un modelo nuevo con estas mismas features."
)


def load_expected_classical_features(dataset_path: Path) -> list[str]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontro {dataset_path}. No se puede validar compatibilidad con el modelo clasico."
        )

    df = pd.read_csv(dataset_path, nrows=512)
    df.columns = [str(col).strip() for col in df.columns]
    label_column = find_label_column(df)
    numeric_columns = df.drop(columns=[label_column]).select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        raise ValueError("No se detectaron columnas numericas en el dataset clasico.")

    return numeric_columns


def compare_feature_sets(live_columns: list[str], expected_columns: list[str]) -> dict[str, list[str] | bool]:
    live_set = set(live_columns)
    expected_set = set(expected_columns)
    return {
        "compatible": live_columns == expected_columns,
        "missing": sorted(expected_set - live_set),
        "extra": sorted(live_set - expected_set),
    }
