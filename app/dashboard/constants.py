from __future__ import annotations

from pathlib import Path

import numpy as np


PRIMARY_BLUE = "#0A1F44"
SECONDARY_BLUE = "#123A78"
ACCENT_YELLOW = "#F7B500"
BACKGROUND = "#07152E"
TEXT = "#F5F8FF"
MUTED_TEXT = "#A9B8D6"
SUCCESS = "#4DA3FF"
DANGER = "#F7B500"

RESULTS_DIR = Path("results")
DATASET_PATH = Path("data/dataset.csv")
CLASSICAL_RESULTS_PATH = RESULTS_DIR / "classical_metrics.json"
CLASSICAL_LIVE_RESULTS_PATH = RESULTS_DIR / "classical_live_metrics.json"
QUANTUM_SIMULATED_RESULTS_PATH = RESULTS_DIR / "quantum_simulated_metrics.json"
QUANTUM_LIVE_RESULTS_PATH = RESULTS_DIR / "quantum_live_simulated_metrics.json"
QUANTUM_HARDWARE_RESULTS_PATH = RESULTS_DIR / "quantum_hardware_metrics.json"
QUANTUM_LIVE_HARDWARE_RESULTS_PATH = RESULTS_DIR / "quantum_live_hardware_metrics.json"
LIVE_TRAINING_DATASET_PATH = RESULTS_DIR / "live_training_dataset.csv"
LIVE_CAPTURE_PATH = RESULTS_DIR / "live_capture.csv"
UPLOADED_QUANTUM_DATASET_PATH = RESULTS_DIR / "uploaded_quantum_dataset.csv"
CLASSICAL_MODEL_PATH = RESULTS_DIR / "random_forest_model.joblib"
SCALER_PATH = RESULTS_DIR / "scaler.joblib"
PCA_PATH = RESULTS_DIR / "pca.joblib"
SUPPORTED_QUANTUM_QUBITS = (2, 3, 4, 6, 8)
SUPPORTED_QUANTUM_DATASET_SOURCES = ("cicids", "live")
ENABLED_MODEL_OPTIONS = ("Modelo clasico", "Modelo cuantico")

LIVE_CLASSICAL_INCOMPATIBLE_MESSAGE = (
    "El modelo actual fue entrenado con features CICIDS2017. "
    "Para usar live_capture.csv se debe entrenar un modelo nuevo con estas mismas features."
)

MODEL_DATA = {
    "Modelo clasico": {
        "label": "Modelo clasico",
        "short_label": "Clasico",
        "description": "Baseline supervisado sobre caracteristicas numericas del trafico.",
        "accuracy": 0.942,
        "precision": 0.931,
        "recall": 0.956,
        "f1_score": 0.943,
        "execution_time": 38.4,
        "confusion_matrix": np.array([[912, 44], [31, 883]]),
        "color": PRIMARY_BLUE,
    },
    "Modelo cuantico": {
        "label": "Modelo cuantico",
        "short_label": "QML",
        "description": "Circuito variacional ejecutado en entorno cuantico controlado.",
        "accuracy": 0.918,
        "precision": 0.904,
        "recall": 0.929,
        "f1_score": 0.916,
        "execution_time": 72.8,
        "confusion_matrix": np.array([[884, 72], [65, 849]]),
        "color": SECONDARY_BLUE,
    },
    "Hardware cuantico real": {
        "label": "Hardware cuantico real",
        "short_label": "Hardware real",
        "description": "Ejecucion NISQ afectada por ruido, decoherencia y errores de lectura.",
        "accuracy": 0.861,
        "precision": 0.844,
        "recall": 0.872,
        "f1_score": 0.858,
        "execution_time": 214.5,
        "confusion_matrix": np.array([[819, 137], [117, 797]]),
        "color": ACCENT_YELLOW,
    },
}
