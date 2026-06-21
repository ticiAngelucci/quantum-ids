from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.classical.train_model import find_label_column


LOGGER = logging.getLogger("live_detection.predict_live")
DEFAULT_INPUT_PATH = Path("results/live_capture.csv")
DATASET_PATH = Path("data/dataset.csv")
MODEL_PATH = Path("results/random_forest_model.joblib")
SCALER_PATH = Path("results/scaler.joblib")
PCA_PATH = Path("results/pca.joblib")
INCOMPATIBLE_MESSAGE = (
    "El modelo actual fue entrenado con features CICIDS2017. "
    "Para usar live_capture.csv se debe entrenar un modelo nuevo con estas mismas features."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida si las features capturadas en vivo son compatibles con el modelo clasico actual."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="CSV con features en vivo.")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
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
        raise ValueError("No se detectaron columnas numericas en el dataset de entrenamiento.")

    return numeric_columns


def compare_feature_sets(live_columns: list[str], expected_columns: list[str]) -> dict[str, list[str] | bool]:
    live_set = set(live_columns)
    expected_set = set(expected_columns)
    return {
        "compatible": live_columns == expected_columns,
        "missing": sorted(expected_set - live_set),
        "extra": sorted(live_set - expected_set),
    }


def predict_if_compatible(live_df: pd.DataFrame) -> list[int]:
    try:
        import joblib
    except ImportError as error:
        raise ImportError(
            "No se pudo importar joblib. Instala las dependencias del proyecto para usar prediccion clasica."
        ) from error

    missing_artifacts = [path for path in [MODEL_PATH, SCALER_PATH, PCA_PATH] if not path.exists()]
    if missing_artifacts:
        missing_paths = ", ".join(str(path) for path in missing_artifacts)
        raise FileNotFoundError(f"Faltan artefactos del modelo clasico: {missing_paths}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)

    transformed = scaler.transform(live_df)
    projected = pca.transform(transformed)
    predictions = model.predict(projected)
    return predictions.tolist()


def main() -> None:
    args = parse_args()
    configure_logging()

    if not args.input.exists():
        raise FileNotFoundError(f"No se encontro el archivo de entrada {args.input}")

    live_df = pd.read_csv(args.input)
    live_df.columns = [str(col).strip() for col in live_df.columns]
    LOGGER.info("Archivo cargado: %s", args.input)
    LOGGER.info("Columnas detectadas en live capture: %s", live_df.columns.tolist())

    expected_features = load_expected_classical_features(DATASET_PATH)
    comparison = compare_feature_sets(live_df.columns.tolist(), expected_features)

    if not comparison["compatible"]:
        LOGGER.warning("Las features en vivo no son compatibles con el modelo clasico actual.")
        LOGGER.warning("Columnas faltantes respecto al entrenamiento: %s", comparison["missing"])
        LOGGER.warning("Columnas extra en live capture: %s", comparison["extra"])
        LOGGER.warning(INCOMPATIBLE_MESSAGE)
        LOGGER.info(
            "Compatibilidad VQC: el modelo cuantico tambien debe reentrenarse con exactamente estas features en vivo."
        )
        return

    LOGGER.info("Las columnas son compatibles con el modelo clasico actual. Ejecutando inferencia experimental.")
    predictions = predict_if_compatible(live_df[expected_features])
    LOGGER.info("Predicciones del modelo clasico: %s", predictions)
    LOGGER.info(
        "Nota VQC: la inferencia cuantica solo seria valida si el VQC fue entrenado con este mismo conjunto de features."
    )


if __name__ == "__main__":
    main()
