import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset
from src.utils.save_results import save_results


DATASET_PATH = Path("data/dataset.csv")
RESULTS_DIR = Path("results")
LATEST_RESULTS_PATH = RESULTS_DIR / "quantum_simulated_metrics.json"
HARDWARE_RESULTS_PATH = RESULTS_DIR / "quantum_hardware_metrics.json"
LIVE_DATASET_PATH = RESULTS_DIR / "live_training_dataset.csv"
BENIGN_SAMPLES = 200
ATTACK_SAMPLES = 200
DEFAULT_QUBITS = 4
SUPPORTED_QUBITS = (2, 4, 6, 8)
SUPPORTED_DATASET_SOURCES = ("cicids", "live")
SUPPORTED_EXECUTION_TARGETS = ("simulator", "ibm_quantum", "ibm_validate")
TEST_SIZE = 0.2
RANDOM_STATE = 42
COBYLA_MAXITER = 50
DEFAULT_IBM_SHOTS = 1024
DEFAULT_IBM_VALIDATION_SAMPLES = 16


def import_quantum_dependencies():
    missing_packages = []

    try:
        from qiskit.circuit.library import zz_feature_map, real_amplitudes
    except ImportError as error:
        raise ImportError("No se pudo importar qiskit. Verifica que `qiskit` este instalado.") from error

    try:
        from qiskit.primitives import StatevectorSampler
    except ImportError as error:
        raise ImportError(
            "No se pudo importar StatevectorSampler desde qiskit.primitives. "
            "Revisa la version de qiskit instalada."
        ) from error

    try:
        from qiskit_machine_learning.algorithms import VQC
    except ImportError:
        missing_packages.append("qiskit-machine-learning")
        VQC = None

    optimizer_error = None
    COBYLA = None
    try:
        from qiskit_algorithms.optimizers import COBYLA as imported_cobyla

        COBYLA = imported_cobyla
    except ImportError as error:
        optimizer_error = error
        try:
            from qiskit.algorithms.optimizers import COBYLA as imported_cobyla

            COBYLA = imported_cobyla
        except ImportError:
            try:
                from qiskit_machine_learning.optimizers import COBYLA as imported_cobyla

                COBYLA = imported_cobyla
            except ImportError:
                pass

    if VQC is None:
        install_hint = "python -m pip install qiskit-machine-learning qiskit-aer"
        raise ImportError(
            "Falta instalar la libreria necesaria para QML: "
            f"{', '.join(missing_packages)}. Ejecuta: {install_hint}"
        )

    if COBYLA is None:
        raise ImportError(
            "No se pudo importar COBYLA desde qiskit_algorithms, qiskit.algorithms ni qiskit_machine_learning.optimizers. "
            "Instala una version compatible de qiskit-machine-learning y qiskit."
        ) from optimizer_error

    return zz_feature_map, real_amplitudes, StatevectorSampler, VQC, COBYLA


def import_ibm_runtime_dependencies():
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    except ImportError as error:
        raise ImportError(
            "No se pudo importar qiskit_ibm_runtime. Instala `qiskit-ibm-runtime` para usar IBM Quantum."
        ) from error

    try:
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except ImportError as error:
        raise ImportError("No se pudo importar generate_preset_pass_manager desde qiskit.") from error

    return QiskitRuntimeService, Sampler, generate_preset_pass_manager


def build_vqc(
    num_qubits: int,
    execution_target: str = "simulator",
    ibm_backend_name: str | None = None,
    ibm_shots: int = DEFAULT_IBM_SHOTS,
    logger=print,
):
    zz_feature_map, real_amplitudes, StatevectorSampler, VQC, COBYLA = import_quantum_dependencies()

    logger("Construyendo feature map y ansatz...")
    feature_map = zz_feature_map(feature_dimension=num_qubits, reps=1)
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=1)
    optimizer = COBYLA(maxiter=COBYLA_MAXITER)
    pass_manager = None
    backend = None

    if execution_target == "ibm_quantum":
        QiskitRuntimeService, Sampler, generate_preset_pass_manager = import_ibm_runtime_dependencies()
        service = create_ibm_runtime_service()
        backend = select_ibm_backend(service=service, num_qubits=num_qubits, backend_name=ibm_backend_name)
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = ibm_shots
        pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
        logger(f"Backend IBM seleccionado: {backend.name}")
        logger(f"Shots configurados para IBM Quantum: {ibm_shots}")
    else:
        sampler = StatevectorSampler()

    logger("Construyendo clasificador VQC...")
    vqc = VQC(
        num_qubits=num_qubits,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        pass_manager=pass_manager,
    )
    return vqc, backend


def create_ibm_runtime_service():
    QiskitRuntimeService, _, _ = import_ibm_runtime_dependencies()
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")

    try:
        if token:
            return QiskitRuntimeService(
                channel="ibm_quantum_platform",
                token=token,
                instance=instance,
            )
        return QiskitRuntimeService(channel="ibm_quantum_platform", instance=instance)
    except Exception as error:
        raise RuntimeError(
            "No se pudo inicializar la cuenta de IBM Quantum. "
            "Configura IBM_QUANTUM_TOKEN o guarda la cuenta con QiskitRuntimeService.save_account(...)."
        ) from error


def select_ibm_backend(service, num_qubits: int, backend_name: str | None = None):
    if backend_name:
        backend = service.backend(backend_name)
        if getattr(backend, "num_qubits", 0) < num_qubits:
            raise ValueError(
                f"El backend {backend_name} no tiene qubits suficientes. "
                f"Disponibles: {getattr(backend, 'num_qubits', 'desconocido')}, requeridos: {num_qubits}"
            )
        return backend

    candidates = [
        backend
        for backend in service.backends(simulator=False, operational=True)
        if getattr(backend, "num_qubits", 0) >= num_qubits
    ]
    if not candidates:
        raise RuntimeError(f"No se encontraron backends IBM operativos con al menos {num_qubits} qubits.")

    return sorted(candidates, key=lambda backend: getattr(backend.status(), "pending_jobs", 10**9))[0]


def extract_hardware_diagnostics(backend) -> dict:
    diagnostics = {
        "backend_name": getattr(backend, "name", "desconocido"),
        "backend_version": getattr(backend, "backend_version", None),
        "num_qubits": getattr(backend, "num_qubits", None),
        "operational": None,
        "pending_jobs": None,
        "basis_gate_count": None,
        "coupling_edge_count": None,
        "avg_t1_us": None,
        "avg_t2_us": None,
        "avg_frequency_ghz": None,
        "limitation_flags": [],
    }

    try:
        status = backend.status()
        diagnostics["operational"] = getattr(status, "operational", None)
        diagnostics["pending_jobs"] = getattr(status, "pending_jobs", None)
    except Exception:
        pass

    target = getattr(backend, "target", None)
    if target is not None:
        operation_names = getattr(target, "operation_names", None)
        if operation_names is not None:
            diagnostics["basis_gate_count"] = len(list(operation_names))

        try:
            coupling_map = target.build_coupling_map()
            if coupling_map is not None:
                diagnostics["coupling_edge_count"] = len(coupling_map.get_edges())
        except Exception:
            pass

        qubit_properties = getattr(target, "qubit_properties", None)
        if qubit_properties:
            t1_values = [prop.t1 for prop in qubit_properties if prop is not None and getattr(prop, "t1", None) is not None]
            t2_values = [prop.t2 for prop in qubit_properties if prop is not None and getattr(prop, "t2", None) is not None]
            frequency_values = [
                prop.frequency for prop in qubit_properties if prop is not None and getattr(prop, "frequency", None) is not None
            ]
            if t1_values:
                diagnostics["avg_t1_us"] = round(float(np.mean(t1_values) * 1e6), 4)
            if t2_values:
                diagnostics["avg_t2_us"] = round(float(np.mean(t2_values) * 1e6), 4)
            if frequency_values:
                diagnostics["avg_frequency_ghz"] = round(float(np.mean(frequency_values) / 1e9), 4)

    pending_jobs = diagnostics["pending_jobs"]
    if pending_jobs is not None and pending_jobs > 10:
        diagnostics["limitation_flags"].append("queue_pressure")
    avg_t1_us = diagnostics["avg_t1_us"]
    avg_t2_us = diagnostics["avg_t2_us"]
    if avg_t1_us is not None and avg_t1_us < 100:
        diagnostics["limitation_flags"].append("low_t1")
    if avg_t2_us is not None and avg_t2_us < 100:
        diagnostics["limitation_flags"].append("low_t2")
    if diagnostics["coupling_edge_count"] is not None and diagnostics["num_qubits"] is not None:
        if diagnostics["coupling_edge_count"] < diagnostics["num_qubits"] - 1:
            diagnostics["limitation_flags"].append("sparse_connectivity")

    return diagnostics


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

    import json

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
    )

    logger(f"Columna objetivo detectada: {dataset_bundle.label_column}")
    logger(f"Cantidad de registros usados: {dataset_bundle.sample_size}")
    logger(f"Cantidad de features numericas: {dataset_bundle.feature_count}")
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
            logger=logger,
        )

    vqc, backend = build_vqc(
        num_qubits=num_qubits,
        execution_target=execution_target,
        ibm_backend_name=ibm_backend_name,
        ibm_shots=ibm_shots,
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
    logger,
) -> dict:
    logger("Entrenando VQC en simulador local para obtener pesos base...")
    simulator_vqc, _ = build_vqc(num_qubits=num_qubits, execution_target="simulator", logger=logger)
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
    add_gap_vs_simulator(metrics, num_qubits=num_qubits, dataset_source=dataset_source)
    return metrics


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        print(f"Error de dependencias: {error}")
        sys.exit(1)
