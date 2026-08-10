from __future__ import annotations
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset

def validate_quantum_kernel_for_hardware(num_qubits: int = 3, max_test_samples: int = 8):
    print("Preparando dataset optimizado para validación en hardware acotado...")
    
    # Usamos pocas muestras para evitar saturar el hardware físico (ej. SpinQ)
    dataset_bundle = prepare_quantum_dataset(
        benign_samples=100,
        attack_samples=100,
        qubits=num_qubits,
        test_size=0.2
    )

    # Limitamos aún más el set de test para la prueba rápida en QPU/Simulador físico
    X_train = dataset_bundle.X_train[:50]
    y_train = dataset_bundle.y_train[:50]
    X_test = dataset_bundle.X_test[:max_test_samples]
    y_test = dataset_bundle.y_test[:max_test_samples]

    print(f"Configurando Feature Map para {num_qubits} qubits (Entrelazamiento lineal)...")
    feature_map = ZZFeatureMap(
        feature_dimension=num_qubits, 
        reps=1,  # reps=1 es ideal para reducir profundidad y ruido en hardware real
        entanglement="linear"
    )

    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    print("Calculando matriz de Kernel reducida para evaluación...")
    # Matriz de entrenamiento
    train_kernel = quantum_kernel.evaluate(x_vec=X_train)
    # Matriz de test frente a entrenamiento (subset acotado)
    test_kernel = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

    from sklearn.svm import SVC
    qsvm = SVC(kernel="precomputed")
    qsvm.fit(train_kernel, y_train)
    
    predictions = qsvm.predict(test_kernel)
    print("¡Evaluación completada con éxito!")
    print(f"Predicciones sobre el subset físico/acotado: {predictions}")
    print(f"Etiquetas reales: {y_test}")

    return predictions, y_test

if __name__ == "__main__":
    validate_quantum_kernel_for_hardware()