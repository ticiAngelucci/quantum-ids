from __future__ import annotations

import os

import numpy as np

from src.quantum.config import COBYLA_MAXITER, DEFAULT_ANSATZ_REPS, DEFAULT_FEATURE_MAP_REPS, DEFAULT_IBM_SHOTS, RANDOM_STATE


def import_quantum_dependencies():
    missing_packages = []

    try:
        from qiskit_machine_learning.utils import algorithm_globals

        algorithm_globals.random_seed = RANDOM_STATE
    except ImportError:
        algorithm_globals = None

    try:
        from qiskit.circuit.library import zz_feature_map, z_feature_map, pauli_feature_map, real_amplitudes
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


def build_vqc(
   num_qubits: int,
    execution_target: str = "simulator",
    ibm_backend_name: str | None = None,
    ibm_shots: int = DEFAULT_IBM_SHOTS,
    feature_map_reps: int = DEFAULT_FEATURE_MAP_REPS,
    ansatz_reps: int = DEFAULT_ANSATZ_REPS,
    maxiter: int = COBYLA_MAXITER,
    feature_map_type: str = "zz",
    logger=print,
):
    zz_feature_map, real_amplitudes, StatevectorSampler, VQC, COBYLA = import_quantum_dependencies()

    logger("Construyendo feature map y ansatz...")
    if feature_map_type == "z":
        # Más simple, sin entrelazamiento cruzado (evita barren plateaus)
        feature_map = z_feature_map(feature_dimension=num_qubits, reps=feature_map_reps)
    elif feature_map_type == "pauli":
        # Permite probar bases de Pauli con entrelazamiento lineal
        feature_map = pauli_feature_map(feature_dimension=num_qubits, reps=feature_map_reps, entanglement="linear")
    else:
        # El ZZFeatureMap clásico por defecto, pero probando entrelazamiento lineal en lugar de full
        feature_map = zz_feature_map(feature_dimension=num_qubits, reps=feature_map_reps, entanglement="linear")
    ansatz = real_amplitudes(num_qubits=num_qubits, reps=ansatz_reps)
    optimizer = COBYLA(maxiter=maxiter)
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
