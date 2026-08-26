from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Iterable

import numpy as np
from qiskit.primitives import BaseSamplerV2
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

from src.quantum.config import DEFAULT_IBM_SHOTS, RESULTS_DIR
from src.quantum.qsvm_feature_map import build_qiskit_qsvm_feature_map
from src.quantum.runtime import (
    create_ibm_runtime_service,
    extract_hardware_diagnostics,
    import_ibm_runtime_dependencies,
    select_ibm_backend,
)
from src.utils.save_results import save_results


class RecordingSampler(BaseSamplerV2):
    """Delegates to IBM SamplerV2 and keeps the runtime jobs for audit metadata."""

    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.jobs: list[object] = []

    @property
    def options(self):
        return self._delegate.options

    def run(self, pubs: Iterable, *, shots: int | None = None):
        job = self._delegate.run(pubs, shots=shots)
        self.jobs.append(job)
        return job


def _job_identifier(job: object) -> str | None:
    identifier = getattr(job, "job_id", None)
    try:
        value = identifier() if callable(identifier) else identifier
    except Exception:
        return None
    return str(value) if value else None


def _job_usage_seconds(job: object) -> float | None:
    usage_method = getattr(job, "usage", None)
    if not callable(usage_method):
        return None
    try:
        usage = usage_method()
    except Exception:
        return None
    return float(usage) if usage is not None else None


def _validate_hardware_cohort(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_qubits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Las matrices de entrenamiento y test deben tener dos dimensiones.")
    if X_train.shape[1] != num_qubits or X_test.shape[1] != num_qubits:
        raise ValueError(
            "La cantidad de features debe coincidir con los qubits: "
            f"train={X_train.shape[1]}, test={X_test.shape[1]}, qubits={num_qubits}."
        )
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Las features y etiquetas no tienen la misma cantidad de filas.")
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError("La cohorte IBM debe contener las dos clases en train y test.")
    if len(y_train) != 4 or len(y_test) != 4:
        raise ValueError(
            "La evaluación IBM reducida requiere exactamente 4 muestras de train "
            "y 4 de test para ejecutar 26 circuitos."
        )
    class_counts_are_invalid = any(
        np.sum(labels == class_label) != 2
        for labels in (y_train, y_test)
        for class_label in (0, 1)
    )
    if class_counts_are_invalid:
        raise ValueError(
            "La evaluación IBM requiere 2 muestras benignas y 2 de ataque "
            "tanto en train como en test."
        )
    return X_train, y_train, X_test, y_test


def _evaluate_local_reference(
    feature_map,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    local_kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        enforce_psd=False,
        evaluate_duplicates="all",
    )
    return (
        np.asarray(local_kernel.evaluate(x_vec=X_train), dtype=float),
        np.asarray(local_kernel.evaluate(x_vec=X_test, y_vec=X_train), dtype=float),
    )


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_ibm_qsvm_hardware_evaluation(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_qubits: int,
    dataset_source: str,
    dataset_path: Path,
    backend_name: str | None = None,
    shots: int = DEFAULT_IBM_SHOTS,
    optimization_level: int = 1,
    max_execution_time_seconds: int = 60,
    logger=print,
) -> dict:
    """Evaluate the same reduced fidelity QSVM cohort on IBM Quantum hardware."""

    X_train, y_train, X_test, y_test = _validate_hardware_cohort(
        X_train,
        y_train,
        X_test,
        y_test,
        num_qubits,
    )
    if shots <= 0:
        raise ValueError("La cantidad de shots para IBM debe ser mayor a cero.")
    if optimization_level not in (0, 1, 2, 3):
        raise ValueError("optimization_level debe estar entre 0 y 3.")
    if not 1 <= max_execution_time_seconds <= 60:
        raise ValueError(
            "max_execution_time_seconds debe estar entre 1 y 60 segundos."
        )

    feature_map = build_qiskit_qsvm_feature_map(num_qubits)
    logger("Calculando referencia local con el mismo feature map reducido...")
    local_train, local_test = _evaluate_local_reference(feature_map, X_train, X_test)
    local_qsvm = SVC(kernel="precomputed", class_weight="balanced")
    local_qsvm.fit(local_train, y_train)
    local_prediction = np.asarray(local_qsvm.predict(local_test), dtype=int)
    local_metrics = _classification_metrics(y_test, local_prediction)
    if not np.any((y_test == 1) & (local_prediction == 1)):
        raise RuntimeError(
            "La prevalidación local no separó ambas clases. La ejecución IBM "
            "fue cancelada para no gastar 26 circuitos en una cohorte no informativa."
        )

    started_at = time.perf_counter()
    started_at_utc = datetime.now(timezone.utc)
    _, Sampler, generate_preset_pass_manager = import_ibm_runtime_dependencies()
    logger("Inicializando IBM Quantum Runtime...")
    service = create_ibm_runtime_service()
    backend = select_ibm_backend(
        service=service,
        num_qubits=num_qubits,
        backend_name=backend_name or None,
    )
    resolved_backend_name = str(
        getattr(backend, "name", backend_name or "desconocido")
    )
    logger(f"Backend IBM seleccionado: {resolved_backend_name}")
    logger(f"Configurando {shots} shots y optimization_level={optimization_level}...")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = int(shots)
    sampler.options.max_execution_time = int(max_execution_time_seconds)
    recording_sampler = RecordingSampler(sampler)
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
    )
    fidelity = ComputeUncompute(
        sampler=recording_sampler,
        pass_manager=pass_manager,
    )
    hardware_kernel = FidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        enforce_psd=False,
        evaluate_duplicates="all",
    )

    train_circuits = len(X_train) * (len(X_train) + 1) // 2
    test_circuits = len(X_test) * len(X_train)
    logger(f"Enviando kernel de entrenamiento a IBM ({train_circuits} circuitos)...")
    train_kernel = np.asarray(hardware_kernel.evaluate(x_vec=X_train), dtype=float)
    logger(f"Enviando kernel de test a IBM ({test_circuits} circuitos)...")
    test_kernel = np.asarray(
        hardware_kernel.evaluate(x_vec=X_test, y_vec=X_train),
        dtype=float,
    )

    qsvm = SVC(kernel="precomputed", class_weight="balanced")
    qsvm.fit(train_kernel, y_train)
    y_pred = np.asarray(qsvm.predict(test_kernel), dtype=int)
    elapsed_seconds = time.perf_counter() - started_at

    kernel_deviations = np.concatenate(
        (
            np.abs(train_kernel - local_train).ravel(),
            np.abs(test_kernel - local_test).ravel(),
        )
    )
    job_ids = [
        identifier
        for identifier in (_job_identifier(job) for job in recording_sampler.jobs)
        if identifier is not None
    ]
    job_usage_seconds = [
        usage
        for usage in (_job_usage_seconds(job) for job in recording_sampler.jobs)
        if usage is not None
    ]
    hardware_metrics = _classification_metrics(y_test, y_pred)

    return {
        "model_name": "Quantum Kernel (QSVM)",
        "metrics": hardware_metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "prediction_counts": {
            "normal": int(np.sum(y_pred == 0)),
            "intrusion": int(np.sum(y_pred == 1)),
        },
        "train_sample_size": int(len(y_train)),
        "sample_size": int(len(y_test)),
        "rows": int(len(y_test)),
        "num_features": int(X_train.shape[1]),
        "num_qubits": int(num_qubits),
        "dataset_source": dataset_source,
        "dataset_path": str(dataset_path),
        "execution_target": "ibm_quantum",
        "pipeline_version": "qsvm_fidelity_v2",
        "validation_strategy": "balanced_reduced_fidelity_kernel",
        "feature_map": "custom_h_rz_linear_zz",
        "feature_map_reps": 1,
        "ibm_backend_name": resolved_backend_name,
        "ibm_shots": int(shots),
        "ibm_optimization_level": int(optimization_level),
        "ibm_max_execution_time_seconds_per_job": int(
            max_execution_time_seconds
        ),
        "ibm_job_ids": job_ids,
        "ibm_job_usage_seconds": job_usage_seconds,
        "ibm_total_usage_seconds": (
            float(sum(job_usage_seconds)) if job_usage_seconds else None
        ),
        "train_circuit_count": int(train_circuits),
        "test_circuit_count": int(test_circuits),
        "circuit_count": int(train_circuits + test_circuits),
        "runtime_job_count": int(len(recording_sampler.jobs)),
        "execution_time_seconds": round(float(elapsed_seconds), 4),
        "started_at_utc": started_at_utc.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_kernel_matrix": train_kernel.tolist(),
        "test_kernel_matrix": test_kernel.tolist(),
        "local_train_kernel_matrix": local_train.tolist(),
        "local_test_kernel_matrix": local_test.tolist(),
        "local_reference_metrics_subset": local_metrics,
        "hardware_gap_vs_local_subset": {
            "accuracy_drop": float(
                local_metrics["accuracy"] - hardware_metrics["accuracy"]
            ),
            "precision_drop": float(
                local_metrics["precision"] - hardware_metrics["precision"]
            ),
            "recall_drop": float(
                local_metrics["recall"] - hardware_metrics["recall"]
            ),
            "f1_drop": float(
                local_metrics["f1_score"] - hardware_metrics["f1_score"]
            ),
        },
        "quantum_noise": {
            "mean_absolute_deviation": float(np.mean(kernel_deviations)),
            "max_absolute_deviation": float(np.max(kernel_deviations)),
            "comparison_points": int(kernel_deviations.size),
        },
        "hardware_diagnostics": extract_hardware_diagnostics(backend),
    }


def get_ibm_results_paths(num_qubits: int, dataset_source: str) -> tuple[Path, Path]:
    if dataset_source == "live":
        latest = RESULTS_DIR / "quantum_live_ibm_hardware_metrics.json"
        specific = RESULTS_DIR / f"quantum_live_ibm_hardware_metrics_{num_qubits}q.json"
    else:
        latest = RESULTS_DIR / "quantum_ibm_hardware_metrics.json"
        specific = RESULTS_DIR / f"quantum_ibm_hardware_metrics_{num_qubits}q.json"
    return latest, specific


def persist_ibm_qsvm_results(payload: dict) -> tuple[Path, Path]:
    latest, specific = get_ibm_results_paths(
        int(payload["num_qubits"]),
        str(payload["dataset_source"]),
    )
    save_results(payload, latest)
    save_results(payload, specific)
    return latest, specific
