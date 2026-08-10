from __future__ import annotations
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset
from src.quantum.config import DATASET_PATH, DEFAULT_QUBITS

def train_quantum_kernel_model(
    num_qubits: int = DEFAULT_QUBITS, 
    dataset_path=DATASET_PATH,
    execution_target: str = "simulator",
    ibm_validation_samples: int = 16,
    feature_map_reps: int = 2
):
    print(f"Cargando y preparando dataset para Quantum Kernel ({num_qubits} qubits)...")
    dataset_bundle = prepare_quantum_dataset(
        dataset_path=dataset_path,
        benign_samples=300,
        attack_samples=300,
        qubits=num_qubits
    )

    X_train = dataset_bundle.X_train
    X_test = dataset_bundle.X_test
    y_train = dataset_bundle.y_train
    y_test = dataset_bundle.y_test

    print(f"Construyendo el Feature Map (reps={feature_map_reps})...")
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=feature_map_reps, entanglement="linear")

    print("Calculando la matriz de Kernel Cuántica (FidelityQuantumKernel)...")
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # Si se selecciona validación en hardware real de IBM con un subconjunto acotado
    if execution_target == "ibm_validate":
        print(f"Modo IBM Runtime activado: recortando test a {ibm_validation_samples} muestras para validación física...")
        X_test = X_test[:ibm_validation_samples]
        y_test = y_test[:ibm_validation_samples]

    train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)
    test_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

    print("Entrenando el clasificador clásico SVM sobre la matriz cuántica...")
    qsvm = SVC(kernel="precomputed")
    qsvm.fit(train_kernel_matrix, y_train)

    print("Evaluando el modelo...")
    y_pred = qsvm.predict(test_kernel_matrix)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n--- RESULTADOS DEL QUANTUM KERNEL (QSVM) [{execution_target.upper()}] ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nReporte de clasificación detallado:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    
    return qsvm, acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Entrenar Quantum Kernel Classifier (QSVM)")
    parser.add_argument("--qubits", type=int, default=DEFAULT_QUBITS, help="Cantidad de qubits")
    parser.add_argument("--execution-target", type=str, default="simulator", choices=["simulator", "ibm_validate"], help="Entorno de ejecución")
    parser.add_argument("--ibm-validation-samples", type=int, default=16, help="Muestras para validar en IBM")
    parser.add_argument("--feature-map-reps", type=int, default=2, help="Repeticiones del feature map")
    args = parser.parse_args()

    train_quantum_kernel_model(
        num_qubits=args.qubits,
        execution_target=args.execution_target,
        ibm_validation_samples=args.ibm_validation_samples,
        feature_map_reps=args.feature_map_reps
    )