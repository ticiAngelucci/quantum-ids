import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset
from src.utils.save_results import save_results


DATASET_PATH = Path("data/dataset.csv")
RESULTS_DIR = Path("results")
LATEST_RESULTS_PATH = RESULTS_DIR / "quantum_simulated_metrics.json"
BENIGN_SAMPLES = 200
ATTACK_SAMPLES = 200
DEFAULT_QUBITS = 4
SUPPORTED_QUBITS = (2, 4, 6, 8)
TEST_SIZE = 0.2
RANDOM_STATE = 42
COBYLA_MAXITER = 50


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


def build_vqc(num_qubits: int, logger=print):
    zz_feature_map, real_amplitudes, StatevectorSampler, VQC, COBYLA = import_quantum_dependencies()

    logger("Construyendo feature map y ansatz...")
    feature_map = zz_feature_map(feature_dimension=num_qubits, reps=1)
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=1)
    optimizer = COBYLA(maxiter=COBYLA_MAXITER)
    sampler = StatevectorSampler()

    logger("Construyendo clasificador VQC...")
    return VQC(
        num_qubits=num_qubits,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
    )


def get_results_path_for_qubits(num_qubits: int) -> Path:
    return RESULTS_DIR / f"quantum_simulated_metrics_{num_qubits}q.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un VQC simulado para quantum-ids.")
    parser.add_argument(
        "--qubits",
        type=int,
        default=DEFAULT_QUBITS,
        choices=SUPPORTED_QUBITS,
        help="Cantidad de qubits y componentes PCA a utilizar.",
    )
    return parser.parse_args()


def train_quantum_simulator(num_qubits: int = DEFAULT_QUBITS, logger=print) -> dict:
    if num_qubits not in SUPPORTED_QUBITS:
        raise ValueError(f"Cantidad de qubits no soportada: {num_qubits}. Opciones: {SUPPORTED_QUBITS}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {DATASET_PATH}. "
            "Guarda el CSV como data/dataset.csv"
        )

    logger("Cargando dataset...")
    dataset_bundle = prepare_quantum_dataset(
        dataset_path=DATASET_PATH,
        benign_samples=BENIGN_SAMPLES,
        attack_samples=ATTACK_SAMPLES,
        pca_components=num_qubits,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    logger(f"Columna objetivo detectada: {dataset_bundle.label_column}")
    logger(f"Cantidad de registros usados: {dataset_bundle.sample_size}")
    logger(f"Cantidad de features numericas: {dataset_bundle.feature_count}")
    logger(f"Aplicando PCA a {dataset_bundle.pca_components} componentes...")

    num_qubits = dataset_bundle.pca_components
    logger(f"Numero de qubits: {num_qubits}")
    logger("Construccion del circuito cuantico...")
    vqc = build_vqc(num_qubits=num_qubits, logger=logger)

    logger("Inicio de entrenamiento VQC...")
    start_time = time.perf_counter()
    vqc.fit(dataset_bundle.X_train, dataset_bundle.y_train)
    elapsed_time = time.perf_counter() - start_time

    logger("Evaluacion final...")
    y_pred = np.asarray(vqc.predict(dataset_bundle.X_test)).astype(int)

    metrics = {
        "model_name": "Variational Quantum Classifier",
        "environment": "Quantum Simulator",
        "pca_components": dataset_bundle.pca_components,
        "num_qubits": num_qubits,
        "sample_size": dataset_bundle.sample_size,
        "execution_time_seconds": round(elapsed_time, 4),
        "metrics": {
            "accuracy": round(accuracy_score(dataset_bundle.y_test, y_pred), 4),
            "precision": round(precision_score(dataset_bundle.y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(dataset_bundle.y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(dataset_bundle.y_test, y_pred, zero_division=0), 4),
        },
        "confusion_matrix": confusion_matrix(dataset_bundle.y_test, y_pred).tolist(),
    }

    specific_results_path = get_results_path_for_qubits(num_qubits)
    save_results(metrics, specific_results_path)
    save_results(metrics, LATEST_RESULTS_PATH)

    logger(f"Resultados guardados en {specific_results_path}")
    logger(f"Copia actualizada en {LATEST_RESULTS_PATH}")
    logger(str(metrics))
    return metrics


def main():
    args = parse_args()
    train_quantum_simulator(num_qubits=args.qubits, logger=print)


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        print(f"Error de dependencias: {error}")
        sys.exit(1)
