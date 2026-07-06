from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.quantum.config import HARDWARE_RESULTS_PATH, LATEST_RESULTS_PATH, RANDOM_STATE, RESULTS_DIR


def get_latest_results_path(dataset_source: str, execution_target: str = "simulator") -> Path:
    if execution_target in {"ibm_quantum", "ibm_validate"}:
        if dataset_source == "live":
            return RESULTS_DIR / "quantum_live_hardware_metrics.json"
        return HARDWARE_RESULTS_PATH
    if dataset_source == "cicids":
        return LATEST_RESULTS_PATH
    return RESULTS_DIR / "quantum_live_simulated_metrics.json"


def get_results_path_for_qubits(num_qubits: int, dataset_source: str, execution_target: str = "simulator") -> Path:
    if execution_target in {"ibm_quantum", "ibm_validate"}:
        if dataset_source == "live":
            return RESULTS_DIR / f"quantum_live_hardware_metrics_{num_qubits}q.json"
        return RESULTS_DIR / f"quantum_hardware_metrics_{num_qubits}q.json"
    if dataset_source == "cicids":
        return RESULTS_DIR / f"quantum_simulated_metrics_{num_qubits}q.json"
    return RESULTS_DIR / f"quantum_live_simulated_metrics_{num_qubits}q.json"


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def build_base_metrics(
    *,
    dataset_source: str,
    resolved_dataset_path: Path,
    num_qubits: int,
    sample_size: int,
    test_size: float,
    execution_target: str,
    environment_label: str,
    feature_map_reps: int,
    ansatz_reps: int,
    maxiter: int,
) -> dict:
    return {
        "model_name": "Variational Quantum Classifier",
        "environment": environment_label,
        "execution_target": execution_target,
        "dataset_source": dataset_source,
        "dataset_path": str(resolved_dataset_path),
        "pca_components": num_qubits,
        "num_qubits": num_qubits,
        "sample_size": sample_size,
        "test_size": test_size,
        "feature_map_reps": feature_map_reps,
        "ansatz_reps": ansatz_reps,
        "optimizer": "COBYLA",
        "optimizer_maxiter": maxiter,
    }


def select_hardware_validation_subset(
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(y_test) <= max_samples:
        return X_test, y_test

    desired_samples = max(min(len(y_test), max_samples), len(np.unique(y_test)))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=desired_samples, random_state=RANDOM_STATE)
    _, selected_idx = next(splitter.split(X_test, y_test))
    return X_test[selected_idx], y_test[selected_idx]


def clone_trained_state(source_vqc, target_vqc) -> None:
    target_vqc._fit_result = copy.deepcopy(source_vqc.fit_result)
    target_vqc._target_encoder = copy.deepcopy(source_vqc._target_encoder)
    target_vqc._num_classes = source_vqc._num_classes
    target_vqc._one_hot = source_vqc._one_hot


def load_simulator_baseline_metrics(num_qubits: int, dataset_source: str) -> dict | None:
    baseline_path = get_results_path_for_qubits(
        num_qubits,
        dataset_source=dataset_source,
        execution_target="simulator",
    )
    if not baseline_path.exists():
        return None

    with open(baseline_path, "r", encoding="utf-8") as baseline_file:
        baseline_payload = json.load(baseline_file)
    return baseline_payload.get("metrics")


def add_gap_vs_simulator(metrics: dict, num_qubits: int, dataset_source: str) -> None:
    baseline_metrics = load_simulator_baseline_metrics(num_qubits=num_qubits, dataset_source=dataset_source)
    if baseline_metrics:
        metrics["hardware_gap_vs_simulator"] = {
            "accuracy_drop": round(float(baseline_metrics.get("accuracy", 0)) - metrics["metrics"]["accuracy"], 4),
            "f1_drop": round(float(baseline_metrics.get("f1_score", 0)) - metrics["metrics"]["f1_score"], 4),
        }
