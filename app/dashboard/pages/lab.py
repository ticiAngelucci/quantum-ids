from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.analytics import evaluate_classical_dataset, make_confusion_chart
from dashboard.constants import (
    CLASSICAL_MODEL_PATH,
    CLASSICAL_RESULTS_PATH,
    DATASET_PATH,
    PCA_PATH,
    QUANTUM_HARDWARE_RESULTS_PATH,
    QUANTUM_IBM_HARDWARE_RESULTS_PATH,
    SCALER_PATH,
)
from dashboard.types import ModelData, QuantumDatasetSource
from dashboard.ui import render_quantum_noise_card
from src.utils.save_results import save_results


def _load_saved_hardware_result(path, execution_target: str) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("execution_target") != execution_target
        or payload.get("pipeline_version") != "qsvm_fidelity_v2"
        or not isinstance(payload.get("metrics"), dict)
        or payload.get("confusion_matrix") is None
    ):
        return None
    return payload


def _sync_quantum_target() -> None:
    if st.session_state.get("quantum_execution_target_radio") in {
        "spinq",
        "ibm_quantum",
    }:
        st.session_state["selected_quantum_qubits"] = 3
        st.session_state["quantum_results_selectbox"] = 3
    st.session_state.pop("quantum_lab_results", None)
    st.session_state.pop("quantum_lab_results_qubits", None)
    st.session_state.pop("quantum_lab_results_source", None)

def _render_quantum_lab(model_data: ModelData, selected_quantum_qubits: int, selected_quantum_dataset_source: QuantumDatasetSource) -> None:
    if st.session_state.get("quantum_execution_target_radio") in {
        "spinq",
        "ibm_quantum",
    }:
        selected_quantum_qubits = 3

    selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
    selected_quantum_feature_map_reps = int(st.session_state.get("selected_quantum_feature_map_reps", 2))
    spinq_connectivity_only = True
    ibm_shots = 1024
    
    left, right = st.columns([1.2, 1], gap="large")
    
    with left:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.8rem; margin-bottom: 1.5rem;">
                <h3 style="color: #FFFFFF; font-size: 1.3rem; margin: 0.3rem 0 0.8rem 0;">Parámetros del Kernel Cuántico (QSVM)</h3>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.4; margin-bottom: 0.8rem;">
                    <b>¿Qué es este conjunto de datos?</b> Se procesa el dataset estándar <b>CICIDS2017</b>. Es un referente mundial que recopila flujos de tráfico de red reales combinados con simulaciones de ataques modernos frente a tráfico benigno.
                </p>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.4; margin-bottom: 1.2rem;">
                    <b>¿Cómo actúa el Kernel Cuántico (QSVM)?</b> Este modelo proyecta los datos de red hacia un <i>Espacio de Hilbert</i> de alta dimensión mediante un circuito cuántico (ZZFeatureMap). A diferencia del enfoque clásico, esta proyección busca encontrar patrones de correlación en la estructura probabilística de los datos.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown('<p style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.4rem;">Entorno de ejecución cuántica</p>', unsafe_allow_html=True)
        selected_quantum_execution_target = st.radio(
            "Entorno de ejecución cuántica",
            options=["simulator", "ibm_quantum", "spinq"],
            format_func=lambda value: {
                "simulator": "Simulador Local Qiskit",
                "ibm_quantum": "IBM Quantum Cloud (Hardware Real)",
                "spinq": "Validación Acotada Hardware Real (SpinQ)"
            }.get(value, value),
            horizontal=True,
            key="quantum_execution_target_radio",
            label_visibility="collapsed",
            on_change=_sync_quantum_target,
        )

        if (
            selected_quantum_execution_target == "spinq"
            and "quantum_lab_results" not in st.session_state
        ):
            saved_spinq_result = _load_saved_hardware_result(
                QUANTUM_HARDWARE_RESULTS_PATH,
                "spinq",
            )
            if (
                saved_spinq_result
                and saved_spinq_result.get("dataset_source")
                == selected_quantum_dataset_source
            ):
                st.session_state["quantum_lab_results"] = saved_spinq_result
                st.session_state["quantum_lab_results_qubits"] = 3
                st.session_state[
                    "quantum_lab_results_source"
                ] = selected_quantum_dataset_source

        if (
            selected_quantum_execution_target == "ibm_quantum"
            and "quantum_lab_results" not in st.session_state
        ):
            saved_ibm_result = _load_saved_hardware_result(
                QUANTUM_IBM_HARDWARE_RESULTS_PATH,
                "ibm_quantum",
            )
            if (
                saved_ibm_result
                and saved_ibm_result.get("dataset_source") == selected_quantum_dataset_source
                and saved_ibm_result.get("num_qubits") == selected_quantum_qubits
            ):
                st.session_state["quantum_lab_results"] = saved_ibm_result
                st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
                st.session_state["quantum_lab_results_source"] = selected_quantum_dataset_source

        if selected_quantum_execution_target == "spinq":
            spinq_connectivity_only = st.checkbox(
                "Prueba rápida de conexión (1 circuito)",
                value=True,
                help=(
                    "Ejecuta un único circuito de 3 qubits y muestra los counts "
                    "devueltos por SpinQuasar, sin entrenar el QSVM. "
                    "Desmarcala para ejecutar la evaluación QSVM balanceada de 26 circuitos."
                ),
                key="spinq_connectivity_only",
            )
        elif selected_quantum_execution_target == "ibm_quantum":
            with st.expander("Conexión IBM Quantum", expanded=True):
                ibm_shots = st.selectbox(
                    "Shots por circuito",
                    options=[1024, 2048, 4096],
                    index=0,
                    key="ibm_qsvm_shots",
                )
                st.caption(
                    "Selecciona automáticamente un backend compatible según tu acceso "
                    "y disponibilidad. El límite de seguridad es 60 s de QPU por job "
                    "(2 jobs por corrida)."
                )
        
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        
        st.markdown('<p style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.2rem;">Porción de datos para prueba (Test)</p>', unsafe_allow_html=True)
        selected_quantum_test_size = st.select_slider(
            "Porción de datos para prueba (Test)",
            options=[0.2, 0.25, 0.33, 0.5],
            value=selected_quantum_test_size,
            format_func=lambda value: f"{int(value * 100)}%",
            key="quantum_test_size_slider",
            label_visibility="collapsed",
        )
        st.session_state["selected_quantum_test_size"] = selected_quantum_test_size
        
        with st.expander("Ajustes Avanzados del Feature Map", expanded=False):
            st.markdown('<p style="color: #FFFFFF; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.2rem;">Repeticiones del circuito (reps)</p>', unsafe_allow_html=True)
            selected_quantum_feature_map_reps = st.select_slider(
                "Repeticiones del circuito (reps)",
                options=[1, 2, 3],
                value=selected_quantum_feature_map_reps,
                key="quantum_feature_map_reps_slider",
                label_visibility="collapsed",
            )
            
        st.session_state["selected_quantum_feature_map_reps"] = selected_quantum_feature_map_reps
        
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        quantum_button = st.button(
            (
                "Probar conexión SpinQ (1 circuito)"
                if selected_quantum_execution_target == "spinq"
                and spinq_connectivity_only
                else (
                    f"Ejecutar QSVM en IBM ({selected_quantum_qubits}q · 26 circuitos)"
                    if selected_quantum_execution_target == "ibm_quantum"
                    else f"Ejecutar Quantum Kernel ({selected_quantum_qubits}q)"
                )
            ),
            width="stretch",
            type="primary",
        )

    with right:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <span style="color: #FDB913; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Diagnóstico Técnico</span>
                <h4 style="color: #FFFFFF; font-size: 1.15rem; margin: 0.3rem 0 0.8rem 0;">Estado del Motor Cuántico</h4>
                <div style="font-size: 0.88rem; color: #FFFFFF; line-height: 1.6;">
                    • <b>Circuitos Qiskit (ZZFeatureMap):</b> OK <br><span style="color: #A0B3C6; font-size: 0.8rem;">Mapeador de características listo.</span><br><br>
                    • <b>Kernel Cuántico (Fidelity):</b> OK <br><span style="color: #A0B3C6; font-size: 0.8rem;">Calcula la matriz de similitud.</span><br><br>
                    • <b>Clasificador SVM:</b> OK <br><span style="color: #A0B3C6; font-size: 0.8rem;">Optimizado con kernel precomputado.</span><br><br>
                    • <b>Dataset Base:</b> OK <br><span style="color: #A0B3C6; font-size: 0.8rem;">CICIDS2017 preparado.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if quantum_button:
        status_box = st.empty()
        spinq_counter_box = st.empty()
        try:
            # ========================================================
            # SPINQ - HARDWARE REAL
            # ========================================================
            if selected_quantum_execution_target == "spinq":

                expected_circuits = 1 if spinq_connectivity_only else 26
                spinq_counter_box.info(
                    f"Circuitos completados: 0 de {expected_circuits} | "
                    "Preparando ejecución..."
                )

                spinq_status_container = st.empty()
                spinq_status_container.info(
                    "Ejecutando QSVM físicamente en SpinQ..."
                )
                with st.container():

                    st.write("Conectando con SpinQ Triangulum...")

                    # ------------------------------------------------
                    # Imports específicos de SpinQ
                    # ------------------------------------------------
                    from src.quantum.spinq_connector import connect_to_spinq
                    from src.preprocessing.quantum_preprocessing import (
                        prepare_quantum_dataset,
                        select_balanced_quantum_subset,
                    )
                    from src.quantum.qsvm_feature_map import build_qiskit_qsvm_feature_map

                    from sklearn.metrics import (
                        accuracy_score,
                        precision_score,
                        recall_score,
                        f1_score,
                        confusion_matrix,
                    )
                    from sklearn.svm import SVC

                    from spinqit import (
                        get_compiler,
                        Circuit,
                        H,
                        CX,
                    )

                    try:
                        from spinqit import Rz
                    except ImportError:
                        from spinqit.gate import Rz

                    # ------------------------------------------------
                    # 1. CONEXIÓN
                    # ------------------------------------------------
                    engine, config = connect_to_spinq(
                        task_name=f"lab_spinq_qsvm_{int(time.time())}"
                    )

                    if engine is None or config is None:
                        raise ConnectionError(
                            "No se pudo establecer conexión con SpinQ. "
                            "Verificá IP, puerto, SpinQuasar y SpinQit."
                        )

                    st.write("Conexión con SpinQ establecida.")

                    # ------------------------------------------------
                    # 2. DATASET REDUCIDO
                    # ------------------------------------------------
                    spinq_counter_box.info(
                        "Preparando dataset reducido para validación física..."
                    )

                    dataset_bundle = prepare_quantum_dataset(
                        dataset_path=DATASET_PATH,
                        benign_samples=10,
                        attack_samples=10,
                        qubits=3,
                        test_size=0.5,
                    )
                    spinq_counter_box.info(
                        "Dataset preparado. Seleccionando muestras balanceadas..."
                    )

                    X_train, y_train = select_balanced_quantum_subset(
                        dataset_bundle.X_train,
                        dataset_bundle.y_train,
                        samples_per_class=2,
                    )
                    X_test, y_test = select_balanced_quantum_subset(
                        dataset_bundle.X_test,
                        dataset_bundle.y_test,
                        samples_per_class=2,
                    )

                    from qiskit_machine_learning.kernels import FidelityQuantumKernel

                    preflight_kernel = FidelityQuantumKernel(
                        feature_map=build_qiskit_qsvm_feature_map(3)
                    )
                    preflight_train = preflight_kernel.evaluate(X_train)
                    preflight_test = preflight_kernel.evaluate(X_test, y_vec=X_train)
                    preflight_model = SVC(
                        kernel="precomputed",
                        class_weight="balanced",
                    )
                    preflight_model.fit(preflight_train, y_train)
                    preflight_prediction = preflight_model.predict(preflight_test)
                    if not np.any(
                        (np.asarray(y_test) == 1) & (preflight_prediction == 1)
                    ):
                        raise RuntimeError(
                            "La prevalidación local no separó ambas clases. "
                            "La ejecución física fue cancelada para no gastar "
                            "26 circuitos en una cohorte no informativa."
                        )

                    st.write(
                        f"Dataset físico: "
                        f"{len(X_train)} train + "
                        f"{len(X_test)} test"
                    )

                    # ------------------------------------------------
                    # 3. COMPILADOR
                    # ------------------------------------------------
                    st.write("Inicializando compilador SpinQit...")

                    compiler = get_compiler("native")
                    spinq_counter_box.info(
                        "Compilador preparado. Iniciando circuitos físicos..."
                    )

                    # ------------------------------------------------
                    # 4. CONSTRUCTOR DEL CIRCUITO
                    # ------------------------------------------------
                    def build_fidelity_circuit(x_a, x_b):
                        """
                        Construye el circuito:

                            U(x_a) -> U†(x_b)

                        La probabilidad de obtener |000>
                        se utiliza como estimación de:

                            |<psi(x_b)|psi(x_a)>|²
                        """

                        circuit = Circuit()
                        qubits = circuit.allocateQubits(3)

                        # ============================================
                        # U(x_a)
                        # ============================================

                        for q_idx in range(3):

                            value = float(
                                x_a[q_idx]
                            )

                            circuit << (
                                H,
                                qubits[q_idx],
                            )

                            circuit << (
                                Rz,
                                qubits[q_idx],
                                value,
                            )

                        # Entrelazamiento dependiente de los datos.
                        for control, target in ((0, 1), (1, 2)):
                            circuit << (CX, (qubits[control], qubits[target]))
                            circuit << (
                                Rz,
                                qubits[target],
                                2.0 * float(x_a[control]) * float(x_a[target]),
                            )
                            circuit << (CX, (qubits[control], qubits[target]))

                        # ============================================
                        # U†(x_b)
                        # ============================================

                        for control, target in ((1, 2), (0, 1)):
                            circuit << (CX, (qubits[control], qubits[target]))
                            circuit << (
                                Rz,
                                qubits[target],
                                -2.0 * float(x_b[control]) * float(x_b[target]),
                            )
                            circuit << (CX, (qubits[control], qubits[target]))

                        for q_idx in range(3):

                            value = float(
                                x_b[q_idx]
                            )

                            circuit << (
                                Rz,
                                qubits[q_idx],
                                -value,
                            )

                            circuit << (
                                H,
                                qubits[q_idx],
                            )

                        return circuit

                    # ------------------------------------------------
                    # 5. EJECUTOR DE CIRCUITO
                    # ------------------------------------------------
                    def execute_fidelity_circuit(
                        x_a,
                        x_b,
                        label_a="",
                        label_b="",
                    ):
                        circuit = build_fidelity_circuit(
                            x_a,
                            x_b,
                        )

                        # Compilación
                        compiled = compiler.compile(
                            circuit,
                            0,
                        )

                        # Ejecución física
                        result = engine.execute(
                            compiled,
                            config,
                        )

                        if result is None:
                            raise RuntimeError(
                                "SpinQ no devolvió ningún resultado."
                            )

                        # ------------------------------------------------
                        # Obtener counts
                        # ------------------------------------------------
                        if hasattr(result, "counts"):
                            counts = result.counts
                        else:
                            counts = {}

                        if not counts:
                            raise RuntimeError(
                                f"SpinQ devolvió counts vacíos: {counts}"
                            )

                        # ------------------------------------------------
                        # Normalizar keys
                        # ------------------------------------------------
                        normalized_counts = {}

                        for key, value in counts.items():
                            normalized_counts[str(key)] = value

                        total_shots = sum(
                            normalized_counts.values()
                        )

                        if total_shots <= 0:
                            raise RuntimeError(
                                "La cantidad de shots obtenida "
                                "es cero."
                            )

                        zero_hits = normalized_counts.get(
                            "000",
                            0,
                        )

                        # Algunas versiones podrían devolver "0"
                        if zero_hits == 0:
                            zero_hits = normalized_counts.get(
                                "0",
                                0,
                            )

                        fidelity = (
                            float(zero_hits)
                            / float(total_shots)
                        )

                        return {
                            "fidelity": fidelity,
                            "counts": normalized_counts,
                            "shots": total_shots,
                            "circuit": circuit,
                            "compiled": compiled,
                            "label_a": label_a,
                            "label_b": label_b,
                        }

                    if spinq_connectivity_only:
                        spinq_counter_box.info(
                            "Circuito 1 de 1 | Prueba de conexión"
                        )
                        connectivity_result = execute_fidelity_circuit(
                            X_train[0],
                            X_train[0],
                            label_a="connectivity_test",
                            label_b="connectivity_test",
                        )

                        st.success(
                            "SpinQuasar respondió correctamente: "
                            "se ejecutó 1 circuito físico."
                        )
                        st.metric(
                            "Fidelidad de control",
                            f"{connectivity_result['fidelity']:.4f}",
                        )
                        st.caption(
                            f"Shots recibidos: {connectivity_result['shots']}"
                        )
                        st.json(connectivity_result["counts"])
                        spinq_status_container.success(
                            "Prueba de conexión SpinQ completada"
                        )
                        return

                    # ------------------------------------------------
                    # 6. KERNEL DE TRAIN
                    # ------------------------------------------------
                    st.write(
                        "**1/3 — Calculando Kernel de entrenamiento "
                        "en hardware SpinQ...**"
                    )

                    n_train = len(X_train)

                    train_kernel_matrix_real = np.zeros(
                        (n_train, n_train),
                        dtype=float,
                    )

                    train_records = []

                    # El kernel es simétrico: sólo se ejecuta el triángulo
                    # superior y se refleja el resultado para ahorrar hardware.
                    train_total = n_train * (n_train + 1) // 2
                    train_operation = 0
                    n_test = len(X_test)
                    test_total = n_test * n_train
                    circuit_total = train_total + test_total
                    circuit_operation = 0

                    circuit_counter = spinq_counter_box
                    circuit_progress = st.progress(0)

                    for i, x_i in enumerate(X_train):

                        for j in range(i, n_train):
                            x_j = X_train[j]

                            train_operation += 1
                            circuit_operation += 1
                            circuit_counter.info(
                                f"Ejecutando circuito {circuit_operation} de {circuit_total} | "
                                f"Completados: {circuit_operation - 1} | "
                                "Entrenamiento "
                                f"({i + 1},{j + 1})"
                            )
                            circuit_progress.progress(
                                circuit_operation / circuit_total,
                                text=(
                                    f"Progreso total: {circuit_operation}/"
                                    f"{circuit_total} circuitos"
                                ),
                            )
                            time.sleep(0.15)

                            result = execute_fidelity_circuit(
                                x_i,
                                x_j,
                                label_a=f"train[{i}]",
                                label_b=f"train[{j}]",
                            )
                            circuit_counter.info(
                                f"Circuito {circuit_operation} de {circuit_total} completado | "
                                f"Entrenamiento ({i + 1},{j + 1})"
                            )

                            train_kernel_matrix_real[
                                i,
                                j,
                            ] = result["fidelity"]
                            train_kernel_matrix_real[
                                j,
                                i,
                            ] = result["fidelity"]

                            train_records.append(
                                {
                                    "i": i,
                                    "j": j,
                                    "fidelity": result["fidelity"],
                                    "counts": result["counts"],
                                    "shots": result["shots"],
                                }
                            )

                    # ------------------------------------------------
                    # 7. KERNEL DE TEST
                    # ------------------------------------------------
                    test_kernel_matrix_real = np.zeros(
                        (n_test, n_train),
                        dtype=float,
                    )

                    test_records = []

                    test_operation = 0

                    for i, x_test in enumerate(X_test):

                        for j, x_train in enumerate(X_train):

                            test_operation += 1
                            circuit_operation += 1
                            circuit_counter.info(
                                f"Ejecutando circuito {circuit_operation} de {circuit_total} | "
                                f"Completados: {circuit_operation - 1} | "
                                "Prueba "
                                f"({i + 1},{j + 1})"
                            )
                            circuit_progress.progress(
                                circuit_operation / circuit_total,
                                text=(
                                    f"Progreso total: {circuit_operation}/"
                                    f"{circuit_total} circuitos"
                                ),
                            )
                            time.sleep(0.15)

                            result = execute_fidelity_circuit(
                                x_train,
                                x_test,
                                label_a=f"train[{j}]",
                                label_b=f"test[{i}]",
                            )
                            circuit_counter.info(
                                f"Circuito {circuit_operation} de {circuit_total} completado | "
                                f"Prueba ({i + 1},{j + 1})"
                            )

                            test_kernel_matrix_real[
                                i,
                                j,
                            ] = result["fidelity"]

                            test_records.append(
                                {
                                    "test": i,
                                    "train": j,
                                    "fidelity": result["fidelity"],
                                    "counts": result["counts"],
                                    "shots": result["shots"],
                                }
                            )

                    circuit_counter.success(
                        f"{circuit_total} de {circuit_total} "
                        "circuitos completados."
                    )
                    circuit_progress.progress(
                        1.0,
                        text=f"Completado: {circuit_total}/{circuit_total} circuitos",
                    )

                    qsvm_model = SVC(
                        kernel="precomputed",
                        class_weight="balanced",
                    )

                    qsvm_model.fit(
                        train_kernel_matrix_real,
                        y_train,
                    )

                    y_pred = qsvm_model.predict(
                        test_kernel_matrix_real
                    )

                    y_true = np.asarray(
                        y_test
                    )

                    kernel_deviations = np.concatenate(
                        (
                            np.abs(
                                train_kernel_matrix_real
                                - np.asarray(preflight_train, dtype=float)
                            ).ravel(),
                            np.abs(
                                test_kernel_matrix_real
                                - np.asarray(preflight_test, dtype=float)
                            ).ravel(),
                        )
                    )
                    quantum_noise = {
                        "mean_absolute_deviation": float(
                            np.mean(kernel_deviations)
                        ),
                        "max_absolute_deviation": float(
                            np.max(kernel_deviations)
                        ),
                        "comparison_points": int(kernel_deviations.size),
                    }

                    # ------------------------------------------------
                    # 10. MÉTRICAS
                    # ------------------------------------------------
                    metrics_dict = {
                        "accuracy": float(
                            accuracy_score(
                                y_true,
                                y_pred,
                            )
                        ),
                        "precision": float(
                            precision_score(
                                y_true,
                                y_pred,
                                zero_division=0,
                            )
                        ),
                        "recall": float(
                            recall_score(
                                y_true,
                                y_pred,
                                zero_division=0,
                            )
                        ),
                        "f1_score": float(
                            f1_score(
                                y_true,
                                y_pred,
                                zero_division=0,
                            )
                        ),
                    }

                    # ------------------------------------------------
                    # 11. GUARDAR RESULTADOS
                    # ------------------------------------------------
                    spinq_results = {
                        "metrics": metrics_dict,
                        "confusion_matrix": confusion_matrix(
                            y_true,
                            y_pred,
                        ).tolist(),
                        "sample_size": len(X_test),
                        "rows": len(X_test),
                        "train_sample_size": len(X_train),
                        "train_circuit_count": train_total,
                        "test_circuit_count": test_total,
                        "circuit_count": circuit_total,
                        "execution_target": "spinq",
                        "pipeline_version": "qsvm_fidelity_v2",
                        "num_qubits": 3,
                        "dataset_source": selected_quantum_dataset_source,
                        "quantum_noise": quantum_noise,
                    }

                    st.session_state[
                        "quantum_lab_results"
                    ] = spinq_results

                    st.session_state[
                        "quantum_lab_results_qubits"
                    ] = 3

                    st.session_state[
                        "quantum_lab_results_source"
                    ] = selected_quantum_dataset_source

                    try:
                        save_results(spinq_results, QUANTUM_HARDWARE_RESULTS_PATH)
                    except OSError as persistence_error:
                        st.session_state["spinq_lab_persistence_warning"] = str(
                            persistence_error
                        )

                    spinq_status_container.success(
                        "QSVM físico completado en SpinQ"
                    )

                st.rerun()

            # ========================================================
            # IBM QUANTUM - HARDWARE REAL
            # ========================================================
            elif selected_quantum_execution_target == "ibm_quantum":
                from src.preprocessing.quantum_preprocessing import (
                    prepare_quantum_dataset,
                    select_balanced_quantum_subset,
                )
                from src.quantum.ibm_qsvm import (
                    persist_ibm_qsvm_results,
                    run_ibm_qsvm_hardware_evaluation,
                )

                status_box.info("[1/4] Preparando cohorte balanceada para IBM Quantum...")
                dataset_bundle = prepare_quantum_dataset(
                    dataset_path=DATASET_PATH,
                    benign_samples=10,
                    attack_samples=10,
                    qubits=selected_quantum_qubits,
                    test_size=selected_quantum_test_size,
                )
                X_train, y_train = select_balanced_quantum_subset(
                    dataset_bundle.X_train,
                    dataset_bundle.y_train,
                    samples_per_class=2,
                )
                X_test, y_test = select_balanced_quantum_subset(
                    dataset_bundle.X_test,
                    dataset_bundle.y_test,
                    samples_per_class=2,
                )

                def ibm_logger(message: str) -> None:
                    status_box.info(message)

                status_box.info("[2/4] Conectando con IBM Quantum Runtime...")
                quantum_results = run_ibm_qsvm_hardware_evaluation(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    num_qubits=selected_quantum_qubits,
                    dataset_source=selected_quantum_dataset_source,
                    dataset_path=DATASET_PATH,
                    backend_name=None,
                    shots=int(ibm_shots),
                    logger=ibm_logger,
                )
                st.session_state["quantum_lab_results"] = quantum_results
                st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
                st.session_state["quantum_lab_results_source"] = selected_quantum_dataset_source
                status_box.info("[3/4] Guardando resultados auditables de IBM...")
                try:
                    latest_path, specific_path = persist_ibm_qsvm_results(
                        quantum_results
                    )
                    quantum_results["results_path"] = str(specific_path)
                    status_box.success(
                        "[4/4] QSVM ejecutado en "
                        f"{quantum_results['ibm_backend_name']} y guardado en {latest_path}."
                    )
                except OSError as persistence_error:
                    status_box.warning(
                        "El QSVM terminó correctamente en IBM, pero no pude guardar "
                        f"el JSON local: {persistence_error}"
                    )

            # ========================================================
            # SIMULADOR LOCAL
            # ========================================================
            else:

                status_box.info(
                    "[1/5] Cargando y preprocesando "
                    "dataset CICIDS2017..."
                )

                time.sleep(0.6)

                from src.preprocessing.quantum_preprocessing import (
                    prepare_quantum_dataset
                )

                from qiskit.circuit.library import (
                    ZZFeatureMap
                )

                from qiskit_machine_learning.kernels import (
                    FidelityQuantumKernel
                )

                from sklearn.svm import SVC

                from sklearn.metrics import (
                    accuracy_score,
                    confusion_matrix,
                    f1_score,
                    precision_score,
                    recall_score,
                )

                dataset_bundle = prepare_quantum_dataset(
                    dataset_path=DATASET_PATH,
                    benign_samples=300,
                    attack_samples=300,
                    qubits=selected_quantum_qubits,
                    test_size=selected_quantum_test_size,
                )

                X_train = dataset_bundle.X_train
                X_test = dataset_bundle.X_test
                y_train = dataset_bundle.y_train
                y_test = dataset_bundle.y_test

                status_box.info(
                    f"[2/5] Construyendo ZZFeatureMap "
                    f"con {selected_quantum_qubits} qubits..."
                )

                feature_map = ZZFeatureMap(
                    feature_dimension=selected_quantum_qubits,
                    reps=selected_quantum_feature_map_reps,
                    entanglement="linear",
                )

                quantum_kernel = FidelityQuantumKernel(
                    feature_map=feature_map
                )

                status_box.info(
                    "[3/5] Calculando Kernel de entrenamiento..."
                )

                train_kernel_matrix = quantum_kernel.evaluate(
                    x_vec=X_train
                )

                status_box.info(
                    "[4/5] Calculando Kernel de test..."
                )

                test_kernel_matrix = quantum_kernel.evaluate(
                    x_vec=X_test,
                    y_vec=X_train,
                )

                status_box.info(
                    "[5/5] Entrenando SVM..."
                )

                qsvm = SVC(
                    kernel="precomputed"
                )

                qsvm.fit(
                    train_kernel_matrix,
                    y_train,
                )

                y_pred = qsvm.predict(
                    test_kernel_matrix
                )

                metrics_dict = {
                    "accuracy": accuracy_score(
                        y_test,
                        y_pred,
                    ),
                    "precision": precision_score(
                        y_test,
                        y_pred,
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        y_test,
                        y_pred,
                        zero_division=0,
                    ),
                    "f1_score": f1_score(
                        y_test,
                        y_pred,
                        zero_division=0,
                    ),
                }

                quantum_results = {
                    "metrics": metrics_dict,
                    "confusion_matrix": confusion_matrix(
                        y_test,
                        y_pred,
                    ).tolist(),
                    "sample_size": len(X_test),
                    "rows": len(X_test),
                    "train_sample_size": len(X_train),
                    "train_circuit_count": len(X_train) * (len(X_train) - 1) // 2,
                    "test_circuit_count": len(X_test) * len(X_train),
                    "circuit_count": (
                        len(X_train) * (len(X_train) - 1) // 2
                        + len(X_test) * len(X_train)
                    ),
                    "execution_target": selected_quantum_execution_target,
                }

                status_box.empty()

                st.session_state[
                    "quantum_lab_results"
                ] = quantum_results

                st.session_state[
                    "quantum_lab_results_qubits"
                ] = selected_quantum_qubits

                st.session_state[
                    "quantum_lab_results_source"
                ] = selected_quantum_dataset_source

                target_label = "Simulador Local"

                st.success(
                    f"¡Prueba de Quantum Kernel finalizada "
                    f"en {target_label}!"
                )

        except Exception as error:

            status_box.empty()

            st.error(
                f"Error crítico al ejecutar "
                f"el Quantum Kernel: {error}"
            )

            # Para desarrollo: muestra traceback completo
            with st.expander(
                "Detalle técnico del error",
                expanded=False,
            ):
                import traceback

                st.code(
                    traceback.format_exc()
                )
                

    quantum_lab_results = st.session_state.get("quantum_lab_results")
    quantum_lab_results_qubits = st.session_state.get("quantum_lab_results_qubits")
    
    if quantum_lab_results and quantum_lab_results_qubits == selected_quantum_qubits:
        result_target = quantum_lab_results.get(
            "execution_target",
            selected_quantum_execution_target,
        )
        target_label = {
            "spinq": "SpinQ Triangulum",
            "ibm_quantum": "IBM Quantum Cloud",
            "simulator": "Simulador Local",
        }.get(result_target, str(result_target))
        evaluated_rows = int(
            quantum_lab_results.get(
                "rows",
                quantum_lab_results.get("sample_size", 0),
            )
        )
        metrics = quantum_lab_results["metrics"]

        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader(
            f"Resultados del QSVM · {target_label} "
            f"({evaluated_rows:,} registros evaluados)"
        )

        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        metric_cols[1].metric("Precision", f"{metrics['precision'] * 100:.2f}%")
        metric_cols[2].metric("Recall", f"{metrics['recall'] * 100:.2f}%")
        metric_cols[3].metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")

        if result_target in {"spinq", "ibm_quantum"}:
            render_quantum_noise_card(
                quantum_lab_results.get("quantum_noise")
            )
        if result_target == "ibm_quantum":
            ibm_usage = quantum_lab_results.get("ibm_total_usage_seconds")
            usage_label = f"{float(ibm_usage):.2f} s" if ibm_usage is not None else "n/d"
            st.caption(
                f"Backend: {quantum_lab_results.get('ibm_backend_name', 'n/d')} · "
                f"Shots: {quantum_lab_results.get('ibm_shots', 'n/d')} · "
                f"Uso QPU: {usage_label} · "
                f"Jobs: {', '.join(quantum_lab_results.get('ibm_job_ids', [])) or 'n/d'}"
            )

        st.markdown("#### Matriz de Confusión")
        st.plotly_chart(
            make_confusion_chart(np.array(quantum_lab_results["confusion_matrix"]), height=300),
            width="stretch",
            key=(
                f"lab_quantum_confusion_chart_"
                f"{result_target}_{selected_quantum_qubits}q"
            ),
        )

def _render_classical_lab(model_data: ModelData) -> None:
    left, right = st.columns([1.2, 1], gap="large")
    
    with left:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.8rem; margin-bottom: 1.5rem;">
                <h3 style="color: #FFFFFF; font-size: 1.3rem; margin: 0.3rem 0 0.8rem 0;">Parámetros del Baseline Clásico</h3>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.4; margin-bottom: 0.8rem;">
                    <b>¿Qué es este conjunto de datos?</b> Se procesa el dataset estándar <b>CICIDS2017</b>, diseñado por el <i>Canadian Institute for Cybersecurity</i> (Canadá).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        use_holdout = st.checkbox("Usar el mismo conjunto de prueba del entrenamiento", value=True, key="classical_holdout_checkbox")
        run_button = st.button("Ejecutar prueba clásica", width="stretch", type="primary", key="run_classical_button")

    with right:
        model_status = "OK" if CLASSICAL_MODEL_PATH.exists() else "Faltante"
        scaler_status = "OK" if SCALER_PATH.exists() else "Faltante"
        pca_status = "OK" if PCA_PATH.exists() else "Faltante"
        metrics_status = "OK" if CLASSICAL_RESULTS_PATH.exists() else "Faltante"

        st.markdown(
            f"""
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <span style="color: #FDB913; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Diagnóstico Técnico</span>
                <h4 style="color: #FFFFFF; font-size: 1.15rem; margin: 0.3rem 0 0.8rem 0;">Estado de Artefactos del Pipeline</h4>
                <div style="font-size: 0.88rem; color: #FFFFFF; line-height: 1.6;">
                    • <b>Modelo (Random Forest):</b> {model_status} <br>
                    • <b>Escalador (Scaler):</b> {scaler_status} <br>
                    • <b>Reducción (PCA):</b> {pca_status} <br>
                    • <b>Archivo de Métricas:</b> {metrics_status} <br>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_button:
        status_container = st.empty()
        
        try:
            status_container.info("Verificando artefactos y cargando modelo...")
            time.sleep(0.4)
            
            if not (CLASSICAL_MODEL_PATH.exists() and SCALER_PATH.exists() and PCA_PATH.exists() and CLASSICAL_RESULTS_PATH.exists()):
                import subprocess
                import sys
                result = subprocess.run([sys.executable, "-m", "src.classical.train_model"], capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Error al entrenar: {result.stderr}")

            if not DATASET_PATH.exists():
                raise FileNotFoundError("No se encontró data/dataset.csv.")
            df = pd.read_csv(DATASET_PATH)

            results = evaluate_classical_dataset(df, use_holdout_split=use_holdout)
            st.session_state["lab_results"] = results
            
            status_container.success("¡Prueba del modelo clásico ejecutada correctamente!")
            
        except Exception as error:
            status_container.error(f"Error: {error}")

    lab_results = st.session_state.get("lab_results")
    if lab_results:
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader(f"Resultados del Baseline Clásico ({lab_results['rows']:,} registros evaluados)")
        
        c1, c2, c3, c4 = st.columns(4)
        m = lab_results['metrics']
        
        c1.metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
        c2.metric("Precision", f"{m['precision'] * 100:.2f}%")
        c3.metric("Recall", f"{m['recall'] * 100:.2f}%")
        c4.metric("F1-Score", f"{m['f1_score'] * 100:.2f}%")

        st.markdown("#### Matriz de Confusión")
        st.plotly_chart(
            make_confusion_chart(lab_results["confusion_matrix"], height=300),
            use_container_width=True
        )

def render_lab_tab(
    model_data: ModelData,
    selected_model: str,
    selected_quantum_qubits: int,
    selected_quantum_dataset_source: QuantumDatasetSource,
) -> None:
    if selected_model == "Modelo cuantico":
        subtitulo = "Kernel Cuántico (QSVM)"
        descripcion = "evaluar el modelo de <b>Kernel Cuántico (QSVM)</b> mediante proyecciones en espacios de Hilbert."
    else:
        subtitulo = "Baseline Clásico (Random Forest)"
        descripcion = "evaluar el baseline <b>clásico (Random Forest)</b>, nuestro modelo de referencia determinista."

    st.markdown(
        f"""
        <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 2px solid #FDB913; margin-bottom: 2rem;">
            <span style="color: #FDB913; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;">Tesina de Licenciatura en Sistemas</span>
            <h1 style="margin: 0.3rem 0; font-size: 2.6rem; color: #FFFFFF; font-weight: 900;">Laboratorio: {subtitulo}</h1>
            <p style="color: #A0B3C6; margin: 0; font-size: 1.1rem; line-height: 1.5;">
                Espacio para correr pruebas controladas sobre el dataset estándar de referencia. Permite {descripcion}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if selected_model == "Modelo cuantico":
        if selected_quantum_dataset_source == "live":
            st.info("El origen de datos `Live` se gestiona directamente desde la solapa `Live`.")
            return
        _render_quantum_lab(model_data, selected_quantum_qubits, selected_quantum_dataset_source)
        return

    _render_classical_lab(model_data)
