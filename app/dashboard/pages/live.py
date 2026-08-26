from __future__ import annotations

import json
import numpy as np
import pandas as pd
import streamlit as st

from dashboard.analytics import (
    capture_live_monitoring_batch,
    capture_live_multivector_batch,
    inspect_live_quantum_dataset,
    make_confusion_chart,
)
from dashboard.constants import (
    LIVE_CAPTURE_PATH,
    LIVE_TRAINING_DATASET_PATH,
    QUANTUM_LIVE_IBM_HARDWARE_RESULTS_PATH,
    QUANTUM_LIVE_HARDWARE_RESULTS_PATH,
)
from dashboard.types import ModelData
from dashboard.ui import render_info_card, render_quantum_noise_card
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


SIMULATOR_CONFIGS = {
    "Simulador v2": {
        "label": "Simulador v2",
        "script": "scripts/01_attack-scrapy_v2.py",
        "summary": "Versión avanzada con variación de tasa, ataque híbrido y tráfico de fondo.",
    },
    "Simulador v3 (Multivectorial)": {
        "label": "Simulador v3 Multivectorial",
        "script": "scripts/01_attack-scrapy_v3.py",  
        "summary": "Versión avanzada con múltiples vectores concurrentes y sincronización por hilos.",
    },
}
LIVE_DEFAULT_FEATURE_MAP_REPS = 2


def _sync_live_quantum_target() -> None:
    st.session_state.pop("quantum_lab_results", None)
    st.session_state.pop("quantum_lab_results_qubits", None)
    st.session_state.pop("quantum_lab_results_source", None)
    if st.session_state.get("live_quantum_execution_target") in {
        "spinq",
        "ibm_quantum",
    }:
        st.session_state["selected_quantum_qubits"] = 3
        st.session_state["quantum_results_selectbox"] = 3

SIMULATOR_SCENARIOS = {
    "Sin escenario": {
        "suggested_label": None,
        "category": "manual",
        "summary": "Modo libre sin metadatos de escenario.",
    },
    "TCP SYN Flood Avanzado": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Ataque TCP con variación de tasa y spoofing de IPs.",
    },
    "UDP Flood con payload variable": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Inundación UDP con payload variable.",
    },
    "ICMP Flood": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Flood ICMP de alta tasa.",
    },
    "Ataque Hibrido": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Secuencia multi-vector TCP/UDP/ICMP.",
    },
    "Ataques Paralelos": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Prueba de estrés con múltiples ataques concurrentes.",
    },
    "Generar Trafico de Fondo": {
        "suggested_label": "benign",
        "category": "benign",
        "summary": "Tráfico de fondo HTTP/DNS benigno.",
    },
    "Configuracion avanzada": {
        "suggested_label": None,
        "category": "manual",
        "summary": "Modo manual avanzado.",
    },
}

LIVE_CAPTURE_PRESETS = {
    "rapida": {
        "label": "Prueba rápida",
        "duration": 2,
        "windows": 10,
        "summary": "Validación veloz del flujo de red.",
    },
    "recomendada": {
        "label": "Dataset recomendado",
        "duration": 2,
        "windows": 40,
        "summary": "Punto de partida óptimo para separar tráfico.",
    },
    "robusta": {
        "label": "Dataset robusto",
        "duration": 2,
        "windows": 80,
        "summary": "Máxima estabilidad para el Quantum Kernel.",
    },
}


def _resolve_suggested_label(selected_mode: str, selected_scenario: str) -> str | None:
    if selected_mode == "Sin etiqueta":
        return None
    if selected_mode == "Etiqueta sugerida por escenario":
        return SIMULATOR_SCENARIOS.get(selected_scenario, {}).get("suggested_label")
    return selected_mode


def _render_live_monitoring(selected_quantum_qubits: int) -> None:
    st.markdown("### Captura de Tráfico en Vivo")
    st.caption(
        "Generá ventanas etiquetadas de tráfico benigno o ataques para "
        "construir el dataset experimental."
    )

    scenario_options = list(SIMULATOR_SCENARIOS.keys())
    monitor_left, monitor_right = st.columns([1.35, 0.85], gap="large")
    with monitor_left:
        selected_sim_version_key = st.selectbox(
            "Simulador de tráfico",
            options=list(SIMULATOR_CONFIGS.keys()),
            index=1,
            key="live_simulator_version_selector",
        )
        active_sim_config = SIMULATOR_CONFIGS[selected_sim_version_key]

        selected_preset = st.selectbox(
            "Tamaño del experimento",
            options=list(LIVE_CAPTURE_PRESETS.keys()),
            index=1,
            format_func=lambda key: LIVE_CAPTURE_PRESETS[key]["label"],
            key="live_capture_preset",
        )
        preset_config = LIVE_CAPTURE_PRESETS[selected_preset]
        last_applied_preset = st.session_state.get("live_capture_preset_applied")
        if last_applied_preset != selected_preset:
            st.session_state["live_monitor_duration"] = preset_config["duration"]
            st.session_state["live_monitor_windows"] = preset_config["windows"]
            st.session_state["live_capture_preset_applied"] = selected_preset

        live_scenario = st.selectbox(
            "Escenario de laboratorio",
            options=scenario_options,
            index=0,
            key="live_monitor_scenario",
        )
        scenario_metadata = SIMULATOR_SCENARIOS[live_scenario]

        duration_col, windows_col = st.columns(2)
        with duration_col:
            live_duration = st.number_input(
                "Segundos por ventana",
                min_value=1,
                max_value=60,
                step=1,
                key="live_monitor_duration",
            )
        with windows_col:
            live_windows = st.number_input(
                "Cantidad de ventanas",
                min_value=1,
                max_value=200,
                step=1,
                key="live_monitor_windows",
            )

        with st.expander("Ajustes avanzados de captura", expanded=False):
            live_iface = st.text_input(
                "Interfaz de red (opcional)",
                value="",
                placeholder="Ejemplo: Ethernet o Wi-Fi",
                key="live_monitor_iface",
            )
            live_count = st.number_input(
                "Límite de paquetes por ventana",
                min_value=0,
                max_value=100000,
                value=0,
                step=10,
                key="live_monitor_count",
                help="Usá 0 para capturar sin un límite fijo de paquetes.",
            )
            live_label_mode = st.selectbox(
                "Etiqueta del lote",
                options=["Etiqueta sugerida por escenario", "Sin etiqueta", "benign", "attack"],
                index=0,
                key="live_monitor_label",
            )

        live_label = _resolve_suggested_label(live_label_mode, live_scenario)
        append_to_training = live_label in {"benign", "attack"}
        run_live_monitoring = st.button(
            "Iniciar captura Live",
            width="stretch",
            type="primary",
            key="live_monitor_run",
        )
        
    with monitor_right:
        st.markdown(
            f"""
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.35); border-radius: 14px; padding: 1.5rem;">
                <span style="color: #FDB913; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Configuración activa</span>
                <h4 style="color: #FFFFFF; margin: 0.4rem 0 1rem 0;">{preset_config['label']}</h4>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.5;"><b>Simulador:</b> {active_sim_config['label']}</p>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.5;"><b>Escenario:</b> {live_scenario}</p>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.5;"><b>Volumen:</b> {int(live_windows)} ventanas de {int(live_duration)} s</p>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.5; margin-bottom: 0;"><b>Etiqueta:</b> {live_label or 'Sin etiqueta'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_live_monitoring:
        progress_placeholder = st.empty()
        try:
            live_monitor_label_value = live_label
            live_monitor_scenario_value = None if live_scenario == "Sin escenario" else live_scenario

            def live_logger(message: str) -> None:
                progress_placeholder.info(message)

            with st.spinner(f"Capturando tráfico con {active_sim_config['label']}..."):
                if not LIVE_CAPTURE_PATH.parent.exists():
                    LIVE_CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

                if selected_sim_version_key == "Simulador v3 (Multivectorial)":
                    live_batch_df = capture_live_multivector_batch(
                        duration=int(live_duration),
                        windows=int(live_windows),
                        iface=live_iface or None,
                        count=int(live_count),
                        label=live_monitor_label_value,
                        scenario=live_monitor_scenario_value,
                        simulator_version=active_sim_config["label"],
                        append_to_training=append_to_training,
                        logger=live_logger,
                    )
                else:
                    live_batch_df = capture_live_monitoring_batch(
                        duration=int(live_duration),
                        windows=int(live_windows),
                        iface=live_iface or None,
                        count=int(live_count),
                        label=live_monitor_label_value,
                        scenario=live_monitor_scenario_value,
                        simulator_version=active_sim_config["label"],
                        append_to_training=append_to_training,
                        logger=live_logger,
                    )

            monitor_result = {
                "batch_df": live_batch_df.to_dict(orient="records"),
                "rows": int(len(live_batch_df)),
                "label": live_monitor_label_value,
                "scenario": live_monitor_scenario_value,
                "simulator_version": active_sim_config["label"],
                "saved_to_training": append_to_training,
                "output_path": LIVE_CAPTURE_PATH.as_posix(),
            }

            st.session_state["live_monitor_results"] = monitor_result
            progress_placeholder.empty()
        except Exception as error:
            progress_placeholder.empty()
            st.error(f"No pude ejecutar el monitoreo live: {error}")

    live_monitor_results = st.session_state.get("live_monitor_results")
    if live_monitor_results:
        st.success(f"Lote capturado con éxito: {live_monitor_results['rows']} ventanas guardadas.")
        preview_df = pd.DataFrame(live_monitor_results["batch_df"])
        with st.expander("Ver una muestra del lote capturado", expanded=False):
            st.caption("Se muestran como máximo las primeras 10 ventanas.")
            st.dataframe(preview_df.head(10), width="stretch")


def render_live_tab(model_data: ModelData, selected_quantum_qubits: int) -> None:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 2px solid #FDB913; margin-bottom: 2rem;">
            <span style="color: #FDB913; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;">Tesina de Licenciatura en Sistemas</span>
            <h1 style="margin: 0.3rem 0; font-size: 2.6rem; color: #FFFFFF; font-weight: 900;">Laboratorio Live</h1>
            <p style="color: #A0B3C6; margin: 0; font-size: 1.1rem; line-height: 1.5;">
                Captura tráfico de red por ventanas, construye un dataset experimental y evalúa un
                <b>Quantum Kernel (QSVM)</b> en simulación local, IBM Quantum o la <b>SpinQ Triangulum</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    _render_live_monitoring(selected_quantum_qubits)

    st.write("")
    st.markdown("---")
    st.markdown("### Estado del Dataset Live")
    st.caption("Disponibilidad y balance de las muestras acumuladas para entrenar el QSVM.")
    selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
    selected_quantum_feature_map_reps = int(
        st.session_state.get("selected_quantum_feature_map_reps_live", LIVE_DEFAULT_FEATURE_MAP_REPS)
    )
    live_dataset_summary = inspect_live_quantum_dataset(test_size=selected_quantum_test_size)
    
    status_cols = st.columns(4)
    with status_cols[0]:
        render_info_card("Estado", "Listo" if live_dataset_summary["ready"] else "Incompleto", "Validación de clases.")
    with status_cols[1]:
        render_info_card("Ventanas", str(live_dataset_summary["total_rows"]), "Total de registros.")
    with status_cols[2]:
        render_info_card("Clases", f"{live_dataset_summary['benign_count']} B / {live_dataset_summary['attack_count']} A", "Balance Benign / Attack.")
    with status_cols[3]:
        render_info_card("Qubits Máx.", str(live_dataset_summary["max_supported_qubits"]), "Límite dimensional.")

    if live_dataset_summary["ready"]:
        st.success(live_dataset_summary["message"])
    else:
        st.warning(live_dataset_summary["message"])

    st.write("")
    st.markdown("---")
    execute_left, execute_right = st.columns([1.2, 1], gap="large")
    ibm_shots = 1024
    with execute_left:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.8rem; margin-bottom: 1.5rem;">
                <span style="color: #FDB913; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Control de Ejecución</span>
                <h3 style="color: #FFFFFF; font-size: 1.3rem; margin: 0.3rem 0 0.8rem 0;">Parámetros del QSVM Live</h3>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.5; margin: 0;">
                    Entrena y evalúa el Quantum Kernel con las ventanas etiquetadas capturadas en esta sección.
                    Podés comparar el simulador ideal con ejecuciones físicas acotadas en IBM o SpinQ.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="color: #FFFFFF; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.4rem;">Entorno de ejecución cuántica</p>',
            unsafe_allow_html=True,
        )
        live_execution_target = st.radio(
            "Entorno de ejecución cuántica Live",
            options=["simulator", "ibm_quantum", "spinq"],
            format_func=lambda value: {
                "simulator": "Simulador Local Qiskit",
                "ibm_quantum": "Hardware Real IBM Quantum",
                "spinq": "Hardware Real SpinQ",
            }[value],
            horizontal=True,
            key="live_quantum_execution_target",
            label_visibility="collapsed",
            on_change=_sync_live_quantum_target,
        )

        if (
            live_execution_target == "spinq"
            and "spinq_live_results" not in st.session_state
        ):
            saved_spinq_result = _load_saved_hardware_result(
                QUANTUM_LIVE_HARDWARE_RESULTS_PATH,
                "spinq",
            )
            if saved_spinq_result:
                st.session_state["spinq_live_results"] = saved_spinq_result
        if live_execution_target in {"spinq", "ibm_quantum"}:
            selected_quantum_qubits = 3

        if live_execution_target == "ibm_quantum":
            saved_ibm_result = _load_saved_hardware_result(
                QUANTUM_LIVE_IBM_HARDWARE_RESULTS_PATH,
                "ibm_quantum",
            )
            if (
                saved_ibm_result
                and int(saved_ibm_result.get("num_qubits", 0)) == selected_quantum_qubits
                and saved_ibm_result.get("dataset_source") == "live"
            ):
                st.session_state["quantum_lab_results"] = saved_ibm_result
                st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
                st.session_state["quantum_lab_results_source"] = "live"

            with st.expander("Configuración de IBM Quantum", expanded=False):
                ibm_shots = st.select_slider(
                    "Shots por circuito",
                    options=[1024, 2048, 4096],
                    value=1024,
                    key="live_ibm_shots",
                )
                st.caption(
                    "Selecciona automáticamente un backend compatible según tu acceso "
                    "y disponibilidad. El límite de seguridad es 60 s de QPU por job "
                    "(2 jobs por corrida)."
                )

        selected_quantum_test_size = st.select_slider(
            "Porción de datos para prueba (Test)",
            options=[0.2, 0.25, 0.33, 0.5],
            value=selected_quantum_test_size,
            format_func=lambda value: f"{int(value * 100)}%",
            key="live_quantum_test_size_slider",
        )
        st.session_state["selected_quantum_test_size"] = selected_quantum_test_size

        can_run_live = live_dataset_summary["ready"] and (
            live_dataset_summary["max_supported_qubits"] >= 3
            if live_execution_target == "spinq"
            else selected_quantum_qubits <= live_dataset_summary["max_supported_qubits"]
        )
        run_live_action = st.button(
            (
                "Ejecutar QSVM Live en SpinQ (26 circuitos)"
                if live_execution_target == "spinq"
                else (
                    f"Ejecutar QSVM Live en IBM ({selected_quantum_qubits}q, 26 circuitos)"
                    if live_execution_target == "ibm_quantum"
                    else f"Ejecutar Quantum Kernel Live ({selected_quantum_qubits}q)"
                )
            ),
            width="stretch",
            type="primary",
            key="run_live_quantum_button",
            disabled=not can_run_live,
        )
        run_live_quantum = run_live_action and live_execution_target == "simulator"
        run_ibm_live_btn = run_live_action and live_execution_target == "ibm_quantum"
        run_spinq_live_btn = run_live_action and live_execution_target == "spinq"

    with execute_right:
        target_description = (
            "Evaluación física balanceada de 26 circuitos y 3 qubits."
            if live_execution_target == "spinq"
            else (
                f"Evaluación física balanceada de 26 circuitos ({selected_quantum_qubits}q)."
                if live_execution_target == "ibm_quantum"
                else f"Kernel ideal de {selected_quantum_qubits} qubits."
            )
        )
        dataset_status = "Listo" if can_run_live else "Dataset incompleto"
        st.markdown(
            f"""
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <span style="color: #FDB913; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Diagnóstico Técnico</span>
                <h4 style="color: #FFFFFF; font-size: 1.15rem; margin: 0.3rem 0 0.8rem 0;">Estado del QSVM Live</h4>
                <div style="font-size: 0.88rem; color: #FFFFFF; line-height: 1.7;">
                    <b>Modelo:</b> Quantum Kernel + SVC<br>
                    <b>Entorno:</b> {target_description}<br>
                    <b>Dataset:</b> {dataset_status}<br>
                    <b>Balance:</b> {live_dataset_summary['benign_count']} benignas / {live_dataset_summary['attack_count']} ataques
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_live_quantum:
        progress_placeholder = st.empty()
        try:
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
            from qiskit_machine_learning.kernels import FidelityQuantumKernel
            from qiskit.circuit.library import ZZFeatureMap
            from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset

            with st.spinner("Calculando Kernel Cuántico sobre tráfico live..."):
                dataset_bundle = prepare_quantum_dataset(
                    dataset_path=LIVE_TRAINING_DATASET_PATH,
                    benign_samples=min(250, live_dataset_summary["benign_count"]),
                    attack_samples=min(250, live_dataset_summary["attack_count"]),
                    qubits=selected_quantum_qubits,
                    test_size=selected_quantum_test_size,
                    dataset_source="live"
                )

                feature_map = ZZFeatureMap(
                    feature_dimension=selected_quantum_qubits,
                    reps=selected_quantum_feature_map_reps,
                    entanglement="linear"
                )
                quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

                X_test = dataset_bundle.X_test
                y_test = dataset_bundle.y_test

                train_kernel_matrix = quantum_kernel.evaluate(x_vec=dataset_bundle.X_train)
                test_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test, y_vec=dataset_bundle.X_train)

                qsvm = SVC(kernel="precomputed")
                qsvm.fit(train_kernel_matrix, dataset_bundle.y_train)
                y_pred = qsvm.predict(test_kernel_matrix)

                metrics_dict = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, zero_division=0),
                }

                quantum_results = {
                    "metrics": metrics_dict,
                    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                    "sample_size": len(y_test),
                    "rows": len(y_test),
                    "train_sample_size": len(dataset_bundle.y_train),
                    "train_circuit_count": (
                        len(dataset_bundle.y_train)
                        * (len(dataset_bundle.y_train) - 1)
                        // 2
                    ),
                    "test_circuit_count": len(y_test) * len(dataset_bundle.y_train),
                    "circuit_count": (
                        len(dataset_bundle.y_train)
                        * (len(dataset_bundle.y_train) - 1)
                        // 2
                        + len(y_test) * len(dataset_bundle.y_train)
                    ),
                    "source": "real",
                    "execution_target": live_execution_target,
                }

            progress_placeholder.empty()
            st.session_state["quantum_lab_results"] = quantum_results
            st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
            st.session_state["quantum_lab_results_source"] = "live"
            st.success("¡Prueba de QSVM live finalizada con éxito!")
        except Exception as error:
            progress_placeholder.empty()
            st.error(f"No pude ejecutar la prueba cuántica live: {error}")

    # ==========================================
    # IBM Quantum Hardware desde Live
    # ==========================================
    if run_ibm_live_btn:
        ibm_status_placeholder = st.empty()
        try:
            from src.preprocessing.quantum_preprocessing import (
                prepare_quantum_dataset,
                select_balanced_quantum_subset,
            )
            from src.quantum.ibm_qsvm import (
                persist_ibm_qsvm_results,
                run_ibm_qsvm_hardware_evaluation,
            )

            ibm_status_placeholder.info(
                "[1/4] Preparando cohorte balanceada para IBM Quantum..."
            )
            dataset_bundle = prepare_quantum_dataset(
                dataset_path=LIVE_TRAINING_DATASET_PATH,
                benign_samples=min(100, live_dataset_summary["benign_count"]),
                attack_samples=min(100, live_dataset_summary["attack_count"]),
                qubits=selected_quantum_qubits,
                test_size=selected_quantum_test_size,
                dataset_source="live",
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
                ibm_status_placeholder.info(message)

            ibm_status_placeholder.info(
                "[2/4] Conectando con IBM Quantum Runtime..."
            )
            quantum_results = run_ibm_qsvm_hardware_evaluation(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                num_qubits=selected_quantum_qubits,
                dataset_source="live",
                dataset_path=LIVE_TRAINING_DATASET_PATH,
                backend_name=None,
                shots=int(ibm_shots),
                logger=ibm_logger,
            )
            st.session_state["quantum_lab_results"] = quantum_results
            st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
            st.session_state["quantum_lab_results_source"] = "live"
            ibm_status_placeholder.info(
                "[3/4] Guardando resultados auditables de IBM..."
            )
            try:
                latest_path, specific_path = persist_ibm_qsvm_results(
                    quantum_results
                )
                quantum_results["results_path"] = str(specific_path)
                ibm_status_placeholder.success(
                    "[4/4] QSVM ejecutado en "
                    f"{quantum_results['ibm_backend_name']} y guardado en {latest_path}."
                )
            except OSError as persistence_error:
                ibm_status_placeholder.warning(
                    "El QSVM terminó correctamente en IBM, pero no pude guardar "
                    f"el JSON local: {persistence_error}"
                )
        except Exception as error:
            ibm_status_placeholder.error(
                f"No pude ejecutar el QSVM en IBM Quantum: {error}"
            )

    # ==========================================
    # Validación Física en SpinQ desde Live
    # ==========================================
    with st.container():
        if run_spinq_live_btn:
            st.session_state.pop("spinq_live_results", None)
            spinq_status_placeholder = st.empty()
            spinq_status_placeholder.info(
                "Preparando y prevalidando la cohorte balanceada..."
            )
            try:
                from src.preprocessing.quantum_preprocessing import (
                    prepare_quantum_dataset,
                    select_balanced_quantum_subset,
                )
                from src.quantum.qsvm_feature_map import build_qiskit_qsvm_feature_map
                from src.quantum.spinq_connector import connect_to_spinq
                from spinqit import get_compiler, Circuit, H, CX
                try:
                    from spinqit import Rz
                except ImportError:
                    from spinqit.gate import Rz
                import time
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                from sklearn.svm import SVC

                bundle = prepare_quantum_dataset(
                    dataset_path=LIVE_TRAINING_DATASET_PATH,
                    benign_samples=min(100, live_dataset_summary["benign_count"]),
                    attack_samples=min(100, live_dataset_summary["attack_count"]),
                    qubits=3,
                    test_size=selected_quantum_test_size,
                    dataset_source="live"
                )
                
                X_train, y_train = select_balanced_quantum_subset(
                    bundle.X_train,
                    bundle.y_train,
                    samples_per_class=2,
                )
                X_test, y_test = select_balanced_quantum_subset(
                    bundle.X_test,
                    bundle.y_test,
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
                
                engine, config = connect_to_spinq(task_name=f"live_spinq_{int(time.time())}")
                if engine is None or config is None:
                    raise RuntimeError("No se pudo establecer conexión con el servidor SpinQ.")
                
                comp = get_compiler("native")

                def build_fidelity_circuit(x_a, x_b):
                    circuit = Circuit()
                    qubits = circuit.allocateQubits(3)

                    for q_idx in range(3):
                        circuit << (H, qubits[q_idx])
                        circuit << (Rz, qubits[q_idx], float(x_a[q_idx]))

                    for control, target in ((0, 1), (1, 2)):
                        circuit << (CX, (qubits[control], qubits[target]))
                        circuit << (
                            Rz,
                            qubits[target],
                            2.0 * float(x_a[control]) * float(x_a[target]),
                        )
                        circuit << (CX, (qubits[control], qubits[target]))

                    for control, target in ((1, 2), (0, 1)):
                        circuit << (CX, (qubits[control], qubits[target]))
                        circuit << (
                            Rz,
                            qubits[target],
                            -2.0 * float(x_b[control]) * float(x_b[target]),
                        )
                        circuit << (CX, (qubits[control], qubits[target]))

                    for q_idx in range(3):
                        circuit << (Rz, qubits[q_idx], -float(x_b[q_idx]))
                        circuit << (H, qubits[q_idx])

                    return circuit

                def evaluate_fidelity(x_a, x_b):
                    executable = comp.compile(
                        build_fidelity_circuit(x_a, x_b),
                        0,
                    )
                    result = engine.execute(executable, config)
                    counts = getattr(result, "counts", None) if result else None
                    if not counts:
                        raise RuntimeError("SpinQ no devolvió counts para el kernel Live.")

                    normalized_counts = {
                        str(state): int(count)
                        for state, count in counts.items()
                    }
                    shots = sum(normalized_counts.values())
                    if shots <= 0:
                        raise RuntimeError("SpinQ devolvió cero shots.")
                    zero_hits = normalized_counts.get(
                        "000",
                        normalized_counts.get("0", 0),
                    )
                    return float(zero_hits) / float(shots)

                train_size = len(X_train)
                test_size = len(X_test)
                circuit_total = train_size * (train_size + 1) // 2 + test_size * train_size
                train_kernel = np.zeros((train_size, train_size), dtype=float)
                operation = 0
                spinq_progress = st.progress(0)
                for i in range(train_size):
                    for j in range(i, train_size):
                        operation += 1
                        spinq_status_placeholder.info(
                            f"Ejecutando circuito {operation} de {circuit_total} | "
                            f"Completados: {operation - 1} | "
                            f"Entrenamiento ({i + 1},{j + 1})"
                        )
                        spinq_progress.progress(
                            operation / circuit_total,
                            text=f"Progreso total: {operation}/{circuit_total} circuitos",
                        )
                        time.sleep(0.15)
                        fidelity = evaluate_fidelity(
                            X_train[i],
                            X_train[j],
                        )
                        spinq_status_placeholder.info(
                            f"Circuito {operation} de {circuit_total} completado | "
                            f"Entrenamiento ({i + 1},{j + 1})"
                        )
                        train_kernel[i, j] = fidelity
                        train_kernel[j, i] = fidelity

                test_kernel = np.zeros((test_size, train_size), dtype=float)
                for i in range(test_size):
                    for j in range(train_size):
                        operation += 1
                        spinq_status_placeholder.info(
                            f"Ejecutando circuito {operation} de {circuit_total} | "
                            f"Completados: {operation - 1} | "
                            f"Prueba ({i + 1},{j + 1})"
                        )
                        spinq_progress.progress(
                            operation / circuit_total,
                            text=f"Progreso total: {operation}/{circuit_total} circuitos",
                        )
                        time.sleep(0.15)
                        test_kernel[i, j] = evaluate_fidelity(
                            X_train[j],
                            X_test[i],
                        )
                        spinq_status_placeholder.info(
                            f"Circuito {operation} de {circuit_total} completado | "
                            f"Prueba ({i + 1},{j + 1})"
                        )

                qsvm = SVC(kernel="precomputed", class_weight="balanced")
                qsvm.fit(train_kernel, y_train)
                y_pred = qsvm.predict(test_kernel)
                y_true = np.asarray(y_test)
                kernel_deviations = np.concatenate(
                    (
                        np.abs(
                            train_kernel
                            - np.asarray(preflight_train, dtype=float)
                        ).ravel(),
                        np.abs(
                            test_kernel
                            - np.asarray(preflight_test, dtype=float)
                        ).ravel(),
                    )
                )
                quantum_noise = {
                    "mean_absolute_deviation": float(np.mean(kernel_deviations)),
                    "max_absolute_deviation": float(np.max(kernel_deviations)),
                    "comparison_points": int(kernel_deviations.size),
                }
                spinq_status_placeholder.success(
                    f"{circuit_total} de {circuit_total} circuitos completados. Entrenando la SVM..."
                )
                spinq_progress.progress(
                    1.0,
                    text=f"Completado: {circuit_total}/{circuit_total} circuitos",
                )
                
                spinq_live_results = {
                    "metrics": {
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                    },
                    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
                    "prediction_counts": {
                        "normal": int(np.sum(y_pred == 0)),
                        "intrusion": int(np.sum(y_pred == 1))
                    },
                    "rows": len(y_pred),
                    "sample_size": len(y_pred),
                    "train_sample_size": train_size,
                    "train_circuit_count": train_size * (train_size + 1) // 2,
                    "test_circuit_count": test_size * train_size,
                    "circuit_count": circuit_total,
                    "execution_target": "spinq",
                    "pipeline_version": "qsvm_fidelity_v2",
                    "num_qubits": 3,
                    "dataset_source": "live",
                    "train_kernel_matrix": train_kernel.tolist(),
                    "test_kernel_matrix": test_kernel.tolist(),
                    "quantum_noise": quantum_noise,
                }
                st.session_state["spinq_live_results"] = spinq_live_results
                try:
                    save_results(
                        spinq_live_results,
                        QUANTUM_LIVE_HARDWARE_RESULTS_PATH,
                    )
                except OSError as persistence_error:
                    st.session_state["spinq_live_persistence_warning"] = str(
                        persistence_error
                    )
                st.rerun()
            except Exception as e:
                spinq_status_placeholder.error(f"Error al ejecutar en la SpinQ: {e}")

    # ==========================================
    # RENDERIZADO DE RESULTADOS (SPINQ & QSVM)
    # ==========================================
    spinq_live_res = st.session_state.get("spinq_live_results")
    if (
        live_execution_target == "spinq"
        and spinq_live_res
        and "metrics" in spinq_live_res
        and spinq_live_res.get("pipeline_version") == "qsvm_fidelity_v2"
    ):
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader(
            f"Resultados del QSVM Live · SpinQ Triangulum "
            f"({spinq_live_res.get('rows', 0):,} registros evaluados)"
        )
        sq_cols = st.columns(4)
        spinq_metrics = spinq_live_res["metrics"]
        sq_cols[0].metric("Accuracy", f"{spinq_metrics['accuracy'] * 100:.2f}%")
        sq_cols[1].metric("Precision", f"{spinq_metrics['precision'] * 100:.2f}%")
        sq_cols[2].metric("Recall", f"{spinq_metrics['recall'] * 100:.2f}%")
        sq_cols[3].metric("F1-Score", f"{spinq_metrics['f1_score'] * 100:.2f}%")
        render_quantum_noise_card(spinq_live_res.get("quantum_noise"))
        st.markdown("#### Matriz de Confusión")
        st.plotly_chart(
            make_confusion_chart(np.array(spinq_live_res["confusion_matrix"]), height=280),
            width="stretch",
            key="live_spinq_confusion_chart"
        )

    live_runtime_results = st.session_state.get("quantum_lab_results")
    live_runtime_qubits = st.session_state.get("quantum_lab_results_qubits")
    live_runtime_source = st.session_state.get("quantum_lab_results_source")
    
    display_results = (
        live_runtime_results
        if live_runtime_results and live_runtime_qubits == selected_quantum_qubits and live_runtime_source == "live"
        else None
    )

    if (
        live_execution_target != "spinq"
        and display_results
        and ("metrics" in display_results or "accuracy" in display_results)
        and display_results.get("execution_target", "simulator")
        == live_execution_target
    ):
        metrics_payload = display_results["metrics"] if "metrics" in display_results else display_results
        confusion_payload = display_results["confusion_matrix"]
        
        evaluated_rows = int(
            display_results.get(
                "rows",
                display_results.get("sample_size", 0),
            )
        )
        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        runtime_label = (
            "IBM Quantum Cloud"
            if live_execution_target == "ibm_quantum"
            else "Simulador Local"
        )
        st.subheader(
            f"Resultados del QSVM Live · {runtime_label} "
            f"({evaluated_rows:,} registros evaluados)"
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("Accuracy", f"{metrics_payload['accuracy'] * 100:.2f}%")
        metric_cols[1].metric("Precision", f"{metrics_payload['precision'] * 100:.2f}%")
        metric_cols[2].metric("Recall", f"{metrics_payload['recall'] * 100:.2f}%")
        metric_cols[3].metric("F1-Score", f"{metrics_payload['f1_score'] * 100:.2f}%")
        if live_execution_target == "ibm_quantum":
            render_quantum_noise_card(display_results.get("quantum_noise"))
            ibm_usage = display_results.get("ibm_total_usage_seconds")
            usage_label = f"{float(ibm_usage):.2f} s" if ibm_usage is not None else "n/d"
            st.caption(
                f"Backend: {display_results.get('ibm_backend_name', 'n/d')} · "
                f"Shots: {display_results.get('ibm_shots', 'n/d')} · "
                f"Uso QPU: {usage_label} · "
                f"Jobs: {', '.join(display_results.get('ibm_job_ids', [])) or 'n/d'}"
            )
        st.markdown("#### Matriz de Confusión")
        st.plotly_chart(
            make_confusion_chart(np.array(confusion_payload), height=280),
            width="stretch",
            key=(
                f"live_confusion_chart_{live_execution_target}_"
                f"{selected_quantum_qubits}q"
            ),
        )
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0; color: #A0B3C6; font-size: 0.9rem;">
            <p style="margin: 0; color: #FFFFFF; font-weight: 700;">Quantum IDS · Tesina de Licenciatura en Sistemas</p>
            <p style="margin: 0.3rem 0 0 0;">Autor: <b>Ticiana Angelucci</b> | Universidad Champagnat | 2026</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
