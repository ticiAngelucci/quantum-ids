import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset
from src.quantum.config import (
    ATTACK_SAMPLES,
    BENIGN_SAMPLES,
    COBYLA_MAXITER,
    DATASET_PATH,
    DEFAULT_ANSATZ_REPS,
    DEFAULT_FEATURE_MAP_REPS,
    DEFAULT_IBM_SHOTS,
    DEFAULT_IBM_VALIDATION_SAMPLES,
    DEFAULT_QUBITS,
    LIVE_DATASET_PATH,
    RESULTS_DIR,
    RANDOM_STATE,
    SUPPORTED_DATASET_SOURCES,
    SUPPORTED_EXECUTION_TARGETS,
    SUPPORTED_QUBITS,
    TEST_SIZE,
)
from src.quantum.results import (
    add_gap_vs_simulator,
    build_base_metrics,
    clone_trained_state,
    compute_classification_metrics,
    get_latest_results_path,
    get_results_path_for_qubits,
    select_hardware_validation_subset,
)
from src.quantum.runtime import build_vqc, extract_hardware_diagnostics
from src.utils.save_results import save_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un VQC simulado para quantum-ids.")
    parser.add_argument(
        "--qubits",
        type=int,
        default=DEFAULT_QUBITS,
        choices=SUPPORTED_QUBITS,
        help="Cantidad de qubits y componentes PCA a utilizar.",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="cicids",
        choices=SUPPORTED_DATASET_SOURCES,
        help="Fuente de datos para el VQC: dataset CICIDS actual o dataset live generado con el simulador.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Ruta opcional al dataset CSV a usar. Si no se informa, usa la ruta por defecto segun dataset-source.",
    )
    parser.add_argument(
        "--benign-samples",
        type=int,
        default=BENIGN_SAMPLES,
        help="Cantidad maxima de muestras benignas para el muestreo balanceado.",
    )
    parser.add_argument(
        "--attack-samples",
        type=int,
        default=ATTACK_SAMPLES,
        help="Cantidad maxima de muestras de ataque para el muestreo balanceado.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Proporcion del dataset reservada para test. Ejemplo: 0.2",
    )
    parser.add_argument(
        "--execution-target",
        type=str,
        default="simulator",
        choices=SUPPORTED_EXECUTION_TARGETS,
        help="Entorno de ejecucion del VQC: simulador local o IBM Quantum remoto.",
    )
    parser.add_argument(
        "--ibm-backend",
        type=str,
        default=None,
        help="Nombre opcional del backend IBM Quantum a usar.",
    )
    parser.add_argument(
        "--ibm-shots",
        type=int,
        default=DEFAULT_IBM_SHOTS,
        help="Cantidad de shots para ejecucion en IBM Quantum.",
    )
    parser.add_argument(
        "--ibm-validation-samples",
        type=int,
        default=DEFAULT_IBM_VALIDATION_SAMPLES,
        help="Cantidad maxima de muestras del test a validar en IBM cuando execution-target=ibm_validate.",
    )
    parser.add_argument(
        "--feature-map-reps",
        type=int,
        default=DEFAULT_FEATURE_MAP_REPS,
        help="Cantidad de repeticiones del feature map.",
    )
    parser.add_argument(
        "--ansatz-reps",
        type=int,
        default=DEFAULT_ANSATZ_REPS,
        help="Cantidad de repeticiones del ansatz.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=COBYLA_MAXITER,
        help="Cantidad maxima de iteraciones del optimizador COBYLA.",
    )
    return parser.parse_args()


def resolve_dataset_path(dataset_source: str, dataset_path: Path | None) -> Path:
    if dataset_path is not None:
        return dataset_path
    if dataset_source == "cicids":
        return DATASET_PATH
    return LIVE_DATASET_PATH


def train_quantum_simulator(
    num_qubits: int = DEFAULT_QUBITS,
    dataset_source: str = "cicids",
    dataset_path: Path | None = None,
    benign_samples: int = BENIGN_SAMPLES,
    attack_samples: int = ATTACK_SAMPLES,
    test_size: float = TEST_SIZE,
    execution_target: str = "simulator",
    ibm_backend_name: str | None = None,
    ibm_shots: int = DEFAULT_IBM_SHOTS,
    ibm_validation_samples: int = DEFAULT_IBM_VALIDATION_SAMPLES,
    feature_map_reps: int = DEFAULT_FEATURE_MAP_REPS,
    ansatz_reps: int = DEFAULT_ANSATZ_REPS,
    maxiter: int = COBYLA_MAXITER,
    logger=print,
) -> dict:
    if num_qubits not in SUPPORTED_QUBITS:
        raise ValueError(f"Cantidad de qubits no soportada: {num_qubits}. Opciones: {SUPPORTED_QUBITS}")

    if dataset_source not in SUPPORTED_DATASET_SOURCES:
        raise ValueError(
            f"Fuente de dataset no soportada: {dataset_source}. Opciones: {SUPPORTED_DATASET_SOURCES}"
        )
    if execution_target not in SUPPORTED_EXECUTION_TARGETS:
        raise ValueError(
            f"Execution target no soportado: {execution_target}. Opciones: {SUPPORTED_EXECUTION_TARGETS}"
        )

    resolved_dataset_path = resolve_dataset_path(dataset_source=dataset_source, dataset_path=dataset_path)

    if not resolved_dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {resolved_dataset_path}. "
            "Verifica la ruta o genera el CSV correspondiente antes de entrenar."
        )

    logger(f"Cargando dataset desde {resolved_dataset_path}...")
    dataset_bundle = prepare_quantum_dataset(
        dataset_path=resolved_dataset_path,
        benign_samples=benign_samples,
        attack_samples=attack_samples,
        pca_components=num_qubits,
        test_size=test_size,
        random_state=RANDOM_STATE,
        dataset_source=dataset_source,
    )

    logger(f"Columna objetivo detectada: {dataset_bundle.label_column}")
    logger(f"Cantidad de registros usados: {dataset_bundle.sample_size}")
    logger(f"Cantidad de features numericas: {dataset_bundle.feature_count}")
    logger(
        f"Configuracion del circuito: feature_map_reps={feature_map_reps}, "
        f"ansatz_reps={ansatz_reps}, maxiter={maxiter}"
    )
    if dataset_bundle.live_curation_report:
        logger(f"Curacion live aplicada: {dataset_bundle.live_curation_report}")
    if dataset_bundle.live_proxy_baseline_metrics:
        logger(f"Baseline proxy live previo al VQC: {dataset_bundle.live_proxy_baseline_metrics}")
    logger(f"Aplicando PCA a {dataset_bundle.pca_components} componentes...")

    num_qubits = dataset_bundle.pca_components
    logger(f"Numero de qubits: {num_qubits}")
    logger("Construccion del circuito cuantico...")
    metrics = run_quantum_experiment(
        dataset_bundle=dataset_bundle,
        dataset_source=dataset_source,
        resolved_dataset_path=resolved_dataset_path,
        num_qubits=num_qubits,
        test_size=test_size,
        execution_target=execution_target,
        ibm_backend_name=ibm_backend_name,
        ibm_shots=ibm_shots,
        ibm_validation_samples=ibm_validation_samples,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        maxiter=maxiter,
        logger=logger,
    )

    specific_results_path = get_results_path_for_qubits(
        num_qubits,
        dataset_source=dataset_source,
        execution_target=execution_target,
    )
    latest_results_path = get_latest_results_path(
        dataset_source=dataset_source,
        execution_target=execution_target,
    )
    save_results(metrics, specific_results_path)
    save_results(metrics, latest_results_path)

    logger(f"Resultados guardados en {specific_results_path}")
    logger(f"Copia actualizada en {latest_results_path}")
    logger(str(metrics))
    return metrics


def main():
    args = parse_args()
    train_quantum_simulator(
        num_qubits=args.qubits,
        dataset_source=args.dataset_source,
        dataset_path=args.dataset_path,
        benign_samples=args.benign_samples,
        attack_samples=args.attack_samples,
        test_size=args.test_size,
        execution_target=args.execution_target,
        ibm_backend_name=args.ibm_backend,
        ibm_shots=args.ibm_shots,
        ibm_validation_samples=args.ibm_validation_samples,
        feature_map_reps=args.feature_map_reps,
        ansatz_reps=args.ansatz_reps,
        maxiter=args.maxiter,
        logger=print,
    )


def run_quantum_experiment(
    *,
    dataset_bundle,
    dataset_source: str,
    resolved_dataset_path: Path,
    num_qubits: int,
    test_size: float,
    execution_target: str,
    ibm_backend_name: str | None,
    ibm_shots: int,
    ibm_validation_samples: int,
    feature_map_reps: int,
    ansatz_reps: int,
    maxiter: int,
    logger,
) -> dict:
    if execution_target == "ibm_validate":
        return run_ibm_validation_experiment(
            dataset_bundle=dataset_bundle,
            dataset_source=dataset_source,
            resolved_dataset_path=resolved_dataset_path,
            num_qubits=num_qubits,
            test_size=test_size,
            ibm_backend_name=ibm_backend_name,
            ibm_shots=ibm_shots,
            ibm_validation_samples=ibm_validation_samples,
            feature_map_reps=feature_map_reps,
            ansatz_reps=ansatz_reps,
            maxiter=maxiter,
            logger=logger,
        )

    vqc, backend = build_vqc(
        num_qubits=num_qubits,
        execution_target=execution_target,
        ibm_backend_name=ibm_backend_name,
        ibm_shots=ibm_shots,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        maxiter=maxiter,
        logger=logger,
    )

    logger("Inicio de entrenamiento VQC...")
    start_time = time.perf_counter()
    vqc.fit(dataset_bundle.X_train, dataset_bundle.y_train)
    elapsed_time = time.perf_counter() - start_time

    logger("Evaluacion final...")
    y_pred = np.asarray(vqc.predict(dataset_bundle.X_test)).astype(int)
    environment_label = "Quantum Hardware" if execution_target == "ibm_quantum" else "Quantum Simulator"
    metrics = build_base_metrics(
        dataset_source=dataset_source,
        resolved_dataset_path=resolved_dataset_path,
        num_qubits=num_qubits,
        sample_size=dataset_bundle.sample_size,
        test_size=test_size,
        execution_target=execution_target,
        environment_label=environment_label,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        maxiter=maxiter,
    )
    metrics.update(
        {
            "ibm_backend_name": getattr(backend, "name", None) if backend is not None else None,
            "ibm_shots": ibm_shots if execution_target == "ibm_quantum" else None,
            "execution_time_seconds": round(elapsed_time, 4),
            "metrics": compute_classification_metrics(dataset_bundle.y_test, y_pred),
            "confusion_matrix": confusion_matrix(dataset_bundle.y_test, y_pred).tolist(),
        }
    )
    if dataset_bundle.live_curation_report is not None:
        metrics["live_curation_report"] = dataset_bundle.live_curation_report
    if dataset_bundle.live_proxy_baseline_metrics is not None:
        metrics["live_proxy_baseline_metrics"] = dataset_bundle.live_proxy_baseline_metrics
    if backend is not None:
        metrics["hardware_diagnostics"] = extract_hardware_diagnostics(backend)
        add_gap_vs_simulator(metrics, num_qubits=num_qubits, dataset_source=dataset_source)
    return metrics


def run_ibm_validation_experiment(
    *,
    dataset_bundle,
    dataset_source: str,
    resolved_dataset_path: Path,
    num_qubits: int,
    test_size: float,
    ibm_backend_name: str | None,
    ibm_shots: int,
    ibm_validation_samples: int,
    feature_map_reps: int,
    ansatz_reps: int,
    maxiter: int,
    logger,
) -> dict:
    logger("Entrenando VQC en simulador local para obtener pesos base...")
    simulator_vqc, _ = build_vqc(
        num_qubits=num_qubits,
        execution_target="simulator",
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        maxiter=maxiter,
        logger=logger,
    )
    local_train_start = time.perf_counter()
    simulator_vqc.fit(dataset_bundle.X_train, dataset_bundle.y_train)
    local_training_time = time.perf_counter() - local_train_start

    logger("Calculando referencia local sobre el test...")
    local_full_pred = np.asarray(simulator_vqc.predict(dataset_bundle.X_test)).astype(int)
    X_ibm, y_ibm = select_hardware_validation_subset(
        dataset_bundle.X_test,
        dataset_bundle.y_test,
        ibm_validation_samples,
    )
    local_subset_pred = np.asarray(simulator_vqc.predict(X_ibm)).astype(int)

    logger(f"Validando en IBM Quantum con {len(y_ibm)} muestras del test...")
    hardware_vqc, backend = build_vqc(
        num_qubits=num_qubits,
        execution_target="ibm_quantum",
        ibm_backend_name=ibm_backend_name,
        ibm_shots=ibm_shots,
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        maxiter=maxiter,
        logger=logger,
    )
    clone_trained_state(simulator_vqc, hardware_vqc)
    hardware_validation_start = time.perf_counter()
    hardware_pred = np.asarray(hardware_vqc.predict(X_ibm)).astype(int)
    hardware_validation_time = time.perf_counter() - hardware_validation_start

    metrics = build_base_metrics(
        dataset_source=dataset_source,
        resolved_dataset_path=resolved_dataset_path,
        num_qubits=num_qubits,
        sample_size=dataset_bundle.sample_size,
        test_size=test_size,
        execution_target="ibm_validate",
        environment_label="Quantum Hardware Validation",
        feature_map_reps=feature_map_reps,
        ansatz_reps=ansatz_reps,
        maxiter=maxiter,
    )
    metrics.update(
        {
            "validation_strategy": "train_local_validate_ibm",
            "ibm_backend_name": getattr(backend, "name", None),
            "ibm_shots": ibm_shots,
            "ibm_validation_samples": int(len(y_ibm)),
            "local_training_time_seconds": round(local_training_time, 4),
            "hardware_validation_time_seconds": round(hardware_validation_time, 4),
            "execution_time_seconds": round(local_training_time + hardware_validation_time, 4),
            "metrics": compute_classification_metrics(y_ibm, hardware_pred),
            "confusion_matrix": confusion_matrix(y_ibm, hardware_pred).tolist(),
            "local_reference_metrics_full": compute_classification_metrics(dataset_bundle.y_test, local_full_pred),
            "local_reference_metrics_subset": compute_classification_metrics(y_ibm, local_subset_pred),
            "hardware_gap_vs_local_subset": {
                "accuracy_drop": round(
                    accuracy_score(y_ibm, local_subset_pred) - accuracy_score(y_ibm, hardware_pred),
                    4,
                ),
                "f1_drop": round(
                    f1_score(y_ibm, local_subset_pred, zero_division=0)
                    - f1_score(y_ibm, hardware_pred, zero_division=0),
                    4,
                ),
            },
            "hardware_diagnostics": extract_hardware_diagnostics(backend),
        }
    )
    if dataset_bundle.live_curation_report is not None:
        metrics["live_curation_report"] = dataset_bundle.live_curation_report
    if dataset_bundle.live_proxy_baseline_metrics is not None:
        metrics["live_proxy_baseline_metrics"] = dataset_bundle.live_proxy_baseline_metrics
    add_gap_vs_simulator(metrics, num_qubits=num_qubits, dataset_source=dataset_source)
    return metrics


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        print(f"Error de dependencias: {error}")
        sys.exit(1)
