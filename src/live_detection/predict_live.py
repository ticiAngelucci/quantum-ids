from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from src.live_detection.compatibility import (
    INCOMPATIBLE_MESSAGE,
    compare_feature_sets,
    load_expected_classical_features,
)

LOGGER = logging.getLogger("live_detection.predict_live")
DEFAULT_INPUT_PATH = Path("results/live_capture.csv")
DATASET_PATH = Path("data/dataset.csv")

# Artefactos del modelo clasico
MODEL_PATH = Path("results/random_forest_model.joblib")
SCALER_PATH = Path("results/scaler.joblib")
PCA_PATH = Path("results/pca.joblib")

# Artefactos del modelo cuantico
QUANTUM_SCALER_PATH = Path("results/quantum_scaler.joblib")
QUANTUM_SELECTOR_PATH = Path("results/quantum_selector.joblib")
QUANTUM_MODEL_PATH = Path("results/vqc_model.model")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida si las features capturadas en vivo son compatibles con el modelo actual y ejecuta inferencia."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="CSV con features en vivo.")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["classical", "quantum"], 
        default="classical", 
        help="Elige si usar el pipeline de inferencia clasico o el cuantico."
    )
    return parser.parse_args()

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def predict_if_compatible(live_df: pd.DataFrame, mode: str = "classical") -> list[int]:
    try:
        import joblib
    except ImportError as error:
        raise ImportError(
            "No se pudo importar joblib. Instala las dependencias del proyecto para usar prediccion."
        ) from error

    if mode == "classical":
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

    elif mode == "quantum":
        missing_artifacts = [path for path in [QUANTUM_SELECTOR_PATH, QUANTUM_SCALER_PATH, QUANTUM_MODEL_PATH] if not path.exists()]
        if missing_artifacts:
            missing_paths = ", ".join(str(path) for path in missing_artifacts)
            LOGGER.warning("Faltan artefactos cuanticos: %s. Ejecuta el preprocesamiento y entrenamiento VQC primero.", missing_paths)
            return []
        
        # Importamos la clase de Qiskit acá adentro para no hacer lento el modo clásico
        from qiskit_machine_learning.algorithms import VQC
        
        # 1. Cargamos el selector y el escalador (que sí son de Scikit-Learn y usan joblib)
        quantum_selector = joblib.load(QUANTUM_SELECTOR_PATH)
        quantum_scaler = joblib.load(QUANTUM_SCALER_PATH)
        
        # 2. Cargamos el modelo cuántico con el método nativo de Qiskit
        vqc_model = VQC.load(str(QUANTUM_MODEL_PATH))
        
        # 3. Transformamos los datos en vivo
        selected_features = quantum_selector.transform(live_df)
        quantum_ready_data = quantum_scaler.transform(selected_features)
        
        LOGGER.info("Datos listos y escalados estrictamente para las compuertas del VQC:\n%s", quantum_ready_data)
        
        # 4. Predicción real
        predictions = vqc_model.predict(quantum_ready_data)
        return predictions.tolist()

    return []

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
        LOGGER.warning("Las features en vivo no son compatibles con el entrenamiento.")
        LOGGER.warning("Columnas faltantes respecto al entrenamiento: %s", comparison["missing"])
        LOGGER.warning("Columnas extra en live capture: %s", comparison["extra"])
        LOGGER.warning(INCOMPATIBLE_MESSAGE)
        return

    LOGGER.info("Las columnas son compatibles. Ejecutando inferencia en modo: %s", args.mode)
    predictions = predict_if_compatible(live_df[expected_features], mode=args.mode)
    
    if predictions:
        LOGGER.info("Predicciones (%s): %s", args.mode, predictions)
    else:
        LOGGER.info("No se generaron predicciones (faltan artefactos o el modelo no devolvió resultados).")

if __name__ == "__main__":
    main()