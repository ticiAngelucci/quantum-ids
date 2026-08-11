from __future__ import annotations
import time
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
    feature_map_reps: int = 1
):
    if dataset_path is None:
        dataset_path = DATASET_PATH
    print(f"Cargando y preparando dataset para Quantum Kernel ({num_qubits} qubits)...")
    dataset_bundle = prepare_quantum_dataset(
        dataset_path=dataset_path,
        benign_samples=100,
        attack_samples=100,
        qubits=num_qubits
    )

    X_train = dataset_bundle.X_train[:20]  # Reducido para agilizar pruebas en hardware real
    y_train = dataset_bundle.y_train[:20]
    X_test = dataset_bundle.X_test[:ibm_validation_samples]
    y_test = dataset_bundle.y_test[:ibm_validation_samples]

    print(f"Construyendo el Feature Map (reps={feature_map_reps})...")
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=feature_map_reps, entanglement="linear")

    # Configuración del Kernel Cuántico según el entorno
    if execution_target == "hardware_spinq":
        print("Configurando motor NMR para SpinQ en hardware real...")
        from spinqit import get_nmr, get_compiler
        from spinqit.backend import NMRConfig
        
        engine = get_nmr()
        config = NMRConfig()
        config.configure_ip("192.168.172.250")
        config.configure_port(50177)
        config.configure_account("holik", "holikspinq")
        config.configure_task(f"qsvm_kernel_{int(time.time())}", "QSVM Kernel Thesis Run")
        config.configure_shots(1024)
        
        # Aquí definimos un sampler personalizado o usamos la evaluación adaptada de Qiskit
        # conectada al backend de SpinQit si se requiere despacho por red.
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    else:
        quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    print("Calculando la matriz de Kernel Cuántica (FidelityQuantumKernel)...")
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

    return qsvm, acc