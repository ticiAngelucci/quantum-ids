from __future__ import annotations

import json

import joblib
import numpy as np

from dashboard.constants import (
    CLASSICAL_LIVE_RESULTS_PATH,
    CLASSICAL_MODEL_PATH,
    CLASSICAL_RESULTS_PATH,
    LIVE_TRAINING_DATASET_PATH,
    MODEL_DATA,
    PCA_PATH,
    QUANTUM_HARDWARE_RESULTS_PATH,
    QUANTUM_LIVE_HARDWARE_RESULTS_PATH,
    QUANTUM_LIVE_RESULTS_PATH,
    QUANTUM_SIMULATED_RESULTS_PATH,
    RESULTS_DIR,
    SCALER_PATH,
)
from dashboard.types import ClassicalArtifacts, ModelData, QuantumDatasetSource


def load_classical_results() -> dict | None:
    if not CLASSICAL_RESULTS_PATH.exists():
        return None
    with open(CLASSICAL_RESULTS_PATH, "r", encoding="utf-8") as results_file:
        payload = json.load(results_file)

    metrics = payload.get("metrics")
    confusion = payload.get("confusion_matrix")
    if not metrics or confusion is None:
        return None

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "confusion_matrix": np.array(confusion),
        "model_name": payload.get("model_name", "Random Forest"),
        "pca_components": payload.get("pca_components"),
    }


def load_classical_live_results() -> dict | None:
    if not CLASSICAL_LIVE_RESULTS_PATH.exists():
        return None
    with open(CLASSICAL_LIVE_RESULTS_PATH, "r", encoding="utf-8") as results_file:
        payload = json.load(results_file)

    metrics = payload.get("metrics")
    confusion = payload.get("confusion_matrix")
    if not metrics or confusion is None:
        return None

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "confusion_matrix": np.array(confusion),
        "model_name": payload.get("model_name", "Random Forest Live Baseline"),
        "sample_size": payload.get("sample_size"),
        "test_size": payload.get("test_size"),
        "live_curation_report": payload.get("live_curation_report"),
    }


def get_quantum_results_path(qubits: int, dataset_source: QuantumDatasetSource = "cicids"):
    if dataset_source == "live":
        return RESULTS_DIR / f"quantum_live_simulated_metrics_{qubits}q.json"
    return RESULTS_DIR / f"quantum_simulated_metrics_{qubits}q.json"


def get_quantum_hardware_results_path(qubits: int, dataset_source: QuantumDatasetSource = "cicids"):
    if dataset_source == "live":
        return RESULTS_DIR / f"quantum_live_hardware_metrics_{qubits}q.json"
    return RESULTS_DIR / f"quantum_hardware_metrics_{qubits}q.json"


def load_quantum_simulated_results(
    qubits: int | None = None,
    dataset_source: QuantumDatasetSource = "cicids",
) -> dict | None:
    if qubits is None:
        results_path = QUANTUM_LIVE_RESULTS_PATH if dataset_source == "live" else QUANTUM_SIMULATED_RESULTS_PATH
    else:
        results_path = get_quantum_results_path(qubits, dataset_source=dataset_source)
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as results_file:
        payload = json.load(results_file)

    metrics = payload.get("metrics")
    confusion = payload.get("confusion_matrix")
    if not metrics or confusion is None:
        return None

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "confusion_matrix": np.array(confusion),
        "model_name": payload.get("model_name", "Variational Quantum Classifier"),
        "pca_components": payload.get("pca_components"),
        "num_qubits": payload.get("num_qubits"),
        "sample_size": payload.get("sample_size"),
        "execution_time_seconds": payload.get("execution_time_seconds"),
        "dataset_source": payload.get("dataset_source", dataset_source),
        "dataset_path": payload.get("dataset_path"),
        "results_path": str(results_path),
    }


def load_quantum_hardware_results(
    qubits: int | None = None,
    dataset_source: QuantumDatasetSource = "cicids",
) -> dict | None:
    if qubits is None:
        results_path = QUANTUM_LIVE_HARDWARE_RESULTS_PATH if dataset_source == "live" else QUANTUM_HARDWARE_RESULTS_PATH
    else:
        results_path = get_quantum_hardware_results_path(qubits, dataset_source=dataset_source)
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as results_file:
        payload = json.load(results_file)

    metrics = payload.get("metrics")
    confusion = payload.get("confusion_matrix")
    if not metrics or confusion is None:
        return None

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "confusion_matrix": np.array(confusion),
        "model_name": payload.get("model_name", "Variational Quantum Classifier"),
        "pca_components": payload.get("pca_components"),
        "num_qubits": payload.get("num_qubits"),
        "sample_size": payload.get("sample_size"),
        "execution_time_seconds": payload.get("execution_time_seconds"),
        "dataset_source": payload.get("dataset_source", dataset_source),
        "dataset_path": payload.get("dataset_path"),
        "results_path": str(results_path),
        "ibm_backend_name": payload.get("ibm_backend_name"),
        "hardware_diagnostics": payload.get("hardware_diagnostics", {}),
        "hardware_gap_vs_simulator": payload.get("hardware_gap_vs_simulator", {}),
        "hardware_gap_vs_local_subset": payload.get("hardware_gap_vs_local_subset", {}),
        "validation_strategy": payload.get("validation_strategy"),
        "local_reference_metrics_subset": payload.get("local_reference_metrics_subset", {}),
        "local_reference_metrics_full": payload.get("local_reference_metrics_full", {}),
    }


def get_model_data(
    selected_quantum_qubits: int = 4,
    selected_quantum_dataset_source: QuantumDatasetSource = "cicids",
) -> ModelData:
    model_data: ModelData = {
        name: {
            **values,
            "source": "mock",
            "source_label": "Mock",
        }
        for name, values in MODEL_DATA.items()
    }

    classical_results = load_classical_results()
    if classical_results is not None:
        model_data["Modelo clasico"].update(
            {
                "accuracy": classical_results["accuracy"],
                "precision": classical_results["precision"],
                "recall": classical_results["recall"],
                "f1_score": classical_results["f1_score"],
                "confusion_matrix": classical_results["confusion_matrix"],
                "source": "real",
                "source_label": "Resultado real",
                "description": "Random Forest entrenado y evaluado con resultados cargados desde results/classical_metrics.json.",
                "trained_model_name": classical_results["model_name"],
                "pca_components": classical_results["pca_components"],
            }
        )

    quantum_simulated_results = load_quantum_simulated_results(
        selected_quantum_qubits,
        dataset_source=selected_quantum_dataset_source,
    )
    if quantum_simulated_results is not None:
        source_label = "Live" if selected_quantum_dataset_source == "live" else "CICIDS2017"
        source_description = (
            f"VQC entrenado con features agregadas capturadas del simulador. Archivo: {quantum_simulated_results['results_path']}."
            if selected_quantum_dataset_source == "live"
            else f"VQC cargado desde results/quantum_simulated_metrics_{selected_quantum_qubits}q.json. Permite comparar QML frente al baseline clasico."
        )
        model_data["Modelo cuantico"].update(
            {
                "accuracy": quantum_simulated_results["accuracy"],
                "precision": quantum_simulated_results["precision"],
                "recall": quantum_simulated_results["recall"],
                "f1_score": quantum_simulated_results["f1_score"],
                "confusion_matrix": quantum_simulated_results["confusion_matrix"],
                "source": "real",
                "source_label": "Resultado real",
                "description": source_description,
                "trained_model_name": quantum_simulated_results["model_name"],
                "pca_components": quantum_simulated_results["pca_components"],
                "num_qubits": quantum_simulated_results["num_qubits"],
                "sample_size": quantum_simulated_results["sample_size"],
                "selected_qubits": selected_quantum_qubits,
                "selected_dataset_source": selected_quantum_dataset_source,
                "dataset_source_label": source_label,
                "dataset_path": quantum_simulated_results["dataset_path"],
                "results_path": quantum_simulated_results["results_path"],
                "execution_time": quantum_simulated_results["execution_time_seconds"]
                if quantum_simulated_results["execution_time_seconds"] is not None
                else model_data["Modelo cuantico"]["execution_time"],
            }
        )
    else:
        command = (
            f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {selected_quantum_qubits}"
            if selected_quantum_dataset_source == "live"
            else f"python -m src.quantum.train_vqc_simulator --qubits {selected_quantum_qubits}"
        )
        description = (
            f"Todavia no se entreno el VQC live con {selected_quantum_qubits} qubits. Genera {LIVE_TRAINING_DATASET_PATH.as_posix()} y ejecuta: {command}"
            if selected_quantum_dataset_source == "live"
            else f"Todavia no se entreno el VQC con {selected_quantum_qubits} qubits. Ejecutar: {command}"
        )
        model_data["Modelo cuantico"].update(
            {
                "source": "missing",
                "source_label": "Pendiente",
                "selected_qubits": selected_quantum_qubits,
                "selected_dataset_source": selected_quantum_dataset_source,
                "dataset_source_label": "Live" if selected_quantum_dataset_source == "live" else "CICIDS2017",
                "description": description,
            }
        )

    hardware_results = load_quantum_hardware_results(
        selected_quantum_qubits,
        dataset_source=selected_quantum_dataset_source,
    )
    if hardware_results is not None:
        model_data["Hardware cuantico real"].update(
            {
                "accuracy": hardware_results["accuracy"],
                "precision": hardware_results["precision"],
                "recall": hardware_results["recall"],
                "f1_score": hardware_results["f1_score"],
                "confusion_matrix": hardware_results["confusion_matrix"],
                "source": "real",
                "source_label": "Resultado real",
                "description": (
                    f"IBM Quantum backend {hardware_results.get('ibm_backend_name') or 'desconocido'} "
                    f"cargado desde {hardware_results['results_path']}."
                ),
                "trained_model_name": hardware_results["model_name"],
                "pca_components": hardware_results["pca_components"],
                "num_qubits": hardware_results["num_qubits"],
                "sample_size": hardware_results["sample_size"],
                "selected_qubits": selected_quantum_qubits,
                "selected_dataset_source": selected_quantum_dataset_source,
                "dataset_source_label": "Live" if selected_quantum_dataset_source == "live" else "CICIDS2017",
                "dataset_path": hardware_results["dataset_path"],
                "results_path": hardware_results["results_path"],
                "ibm_backend_name": hardware_results.get("ibm_backend_name"),
                "hardware_diagnostics": hardware_results.get("hardware_diagnostics", {}),
                "hardware_gap_vs_simulator": hardware_results.get("hardware_gap_vs_simulator", {}),
                "execution_time": hardware_results["execution_time_seconds"]
                if hardware_results["execution_time_seconds"] is not None
                else model_data["Hardware cuantico real"]["execution_time"],
            }
        )
    return model_data


def load_classical_artifacts() -> ClassicalArtifacts | None:
    if not (CLASSICAL_MODEL_PATH.exists() and SCALER_PATH.exists() and PCA_PATH.exists()):
        return None
    return {
        "model": joblib.load(CLASSICAL_MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "pca": joblib.load(PCA_PATH),
    }
