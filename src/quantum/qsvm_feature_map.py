from __future__ import annotations


def build_qiskit_qsvm_feature_map(num_qubits: int = 3):
    """Feature map usado para prevalidar el circuito físico de SpinQ."""
    from qiskit.circuit import ParameterVector, QuantumCircuit

    parameters = ParameterVector("x", num_qubits)
    circuit = QuantumCircuit(num_qubits)

    for qubit in range(num_qubits):
        circuit.h(qubit)
        circuit.rz(parameters[qubit], qubit)

    for control, target in zip(range(num_qubits - 1), range(1, num_qubits)):
        circuit.cx(control, target)
        circuit.rz(2 * parameters[control] * parameters[target], target)
        circuit.cx(control, target)

    return circuit
