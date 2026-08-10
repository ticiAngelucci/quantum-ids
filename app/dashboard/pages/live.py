from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.analytics import (
    capture_live_monitoring_batch,
    capture_live_multivector_batch,
    inspect_live_quantum_dataset,
    make_confusion_chart,
)
from dashboard.constants import LIVE_CAPTURE_PATH, LIVE_TRAINING_DATASET_PATH
from dashboard.data import get_quantum_hardware_results_path, get_quantum_results_path, load_classical_live_results
from dashboard.types import ModelData
from dashboard.ui import render_info_card, render_metric_card, section_header


SIMULATOR_CONFIGS = {
    "Simulador v2": {
        "label": "Simulador v2",
        "script": "../01_attack-scrapy_v2.py",
        "summary": "Versión avanzada con variación de tasa, ataque híbrido y tráfico de fondo.",
    },
    "Simulador v3 (Multivectorial)": {
        "label": "Simulador v3 Multivectorial",
        "script": "../01_attack-scrapy_v3.py",  
        "summary": "Versión avanzada con múltiples vectores concurrentes y sincronización por hilos.",
    },
}
LIVE_DEFAULT_FEATURE_MAP_REPS = 2

SIMULATOR_SCENARIOS = {
    "Sin escenario": {
        "suggested_label": None,
        "category": "manual",
        "summary": "Usalo solo si no querés documentar el lote o si vas a etiquetarlo manualmente.",
    },
    "TCP SYN Flood Avanzado": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Ataque TCP con variación de tasa y spoofing de IPs. Sirve para entrenar patrones de conexiones half-open.",
    },
    "UDP Flood con payload variable": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Inundación UDP con payload variable y puertos comunes de servicios expuestos.",
    },
    "ICMP Flood": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Flood ICMP de alta tasa para saturación básica del objetivo.",
    },
    "Ataque Hibrido": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Secuencia multi-vector TCP/UDP/ICMP pensada para simular campañas más realistas.",
    },
    "Ataques Paralelos": {
        "suggested_label": "attack",
        "category": "attack",
        "summary": "Prueba de estrés con múltiples ataques concurrentes. Genera la firma más agresiva del simulador.",
    },
    "Generar Trafico de Fondo": {
        "suggested_label": "benign",
        "category": "benign",
        "summary": "Tráfico de fondo HTTP/DNS para construir ventanas benignas más realistas.",
    },
    "Configuracion avanzada": {
        "suggested_label": None,
        "category": "manual",
        "summary": "Modo abierto: elegí la etiqueta manualmente según el experimento que realmente ejecutes.",
    },
}

LIVE_CAPTURE_PRESETS = {
    "rapida": {
        "label": "Prueba rapida",
        "duration": 2,
        "windows": 10,
        "summary": "Sirve para validar que el flujo funciona, pero normalmente requiere más volumen para el Quantum Kernel.",
    },
    "recomendada": {
        "label": "Dataset recomendado",
        "duration": 2,
        "windows": 40,
        "summary": "Punto de partida razonable para empezar a separar mejor benign y attack en entorno live.",
    },
    "robusta": {
        "label": "Dataset robusto",
        "duration": 2,
        "windows": 80,
        "summary": "Conviene cuando queres acercarte a un dataset live mas estable y reducir sensibilidad.",
    },
}


def _resolve_suggested_label(selected_mode: str, selected_scenario: str) -> str | None:
    if selected_mode == "Sin etiqueta":
        return None
    if selected_mode == "Etiqueta sugerida por escenario":
        return SIMULATOR_SCENARIOS.get(selected_scenario, {}).get("suggested_label")
    return selected_mode


def _render_live_monitoring(selected_quantum_qubits: int) -> None:
    st.markdown("#### Captura live")
    st.caption(
        "Este bloque automatiza la captura por ventanas. Permite alternar entre el simulador de laboratorio v2 y la nueva versión v3 multivectorial."
    )
    
    selected_sim_version_key = st.selectbox(
        "Versión del Simulador de Laboratorio",
        options=list(SIMULATOR_CONFIGS.keys()),
        index=1,
        key="live_simulator_version_selector",
    )
    active_sim_config = SIMULATOR_CONFIGS[selected_sim_version_key]

    scenario_options = list(SIMULATOR_SCENARIOS.keys())
    monitor_left, monitor_right = st.columns([1.25, 1])
    with monitor_left:
        selected_preset = st.selectbox(
            "Preset experimental",
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
        st.caption(preset_config["summary"])
        live_duration = st.number_input(
            "Duracion por ventana (segundos)",
            min_value=1,
            max_value=60,
            value=int(st.session_state.get("live_monitor_duration", preset_config["duration"])),
            step=1,
            key="live_monitor_duration",
        )
        live_windows = st.number_input(
            "Cantidad de ventanas",
            min_value=1,
            max_value=200,
            value=int(st.session_state.get("live_monitor_windows", preset_config["windows"])),
            step=1,
            key="live_monitor_windows",
        )
        live_iface = st.text_input("Interfaz de red (ej: lo, wlo1, enp3s0)", value="", placeholder="Ej: lo", key="live_monitor_iface")
        live_count = st.number_input("Limite de paquetes por ventana", min_value=0, max_value=100000, value=0, step=10, key="live_monitor_count")
        live_scenario = st.selectbox("Escenario de laboratorio", options=scenario_options, index=0, key="live_monitor_scenario")
        scenario_metadata = SIMULATOR_SCENARIOS[live_scenario]
        live_label_mode = st.selectbox(
            "Guardar este lote como",
            options=["Etiqueta sugerida por escenario", "Sin etiqueta", "benign", "attack"],
            index=0,
            key="live_monitor_label",
        )
        live_label = _resolve_suggested_label(live_label_mode, live_scenario)
        append_to_training = live_label in {"benign", "attack"}
        run_live_monitoring = st.button("Capturar lote live", width="stretch", type="primary", key="live_monitor_run")
        if live_label_mode == "Etiqueta sugerida por escenario" and live_label is None:
            st.caption("Este escenario no tiene etiqueta automática. Elegí `benign` o `attack` manualmente si querés sumar el lote al dataset.")
        elif live_label_mode in {"benign", "attack"} and scenario_metadata["suggested_label"] and live_label != scenario_metadata["suggested_label"]:
            st.warning(
                f"El escenario `{live_scenario}` suele etiquetarse como `{scenario_metadata['suggested_label']}`. "
                f"Vas a guardarlo manualmente como `{live_label}`."
            )
    with monitor_right:
        render_info_card("Preset activo", preset_config["label"], preset_config["summary"])
        st.write("")
        render_info_card("Version activa", active_sim_config["label"], active_sim_config["summary"])
        st.write("")
        render_info_card("Archivo de salida", LIVE_CAPTURE_PATH.as_posix(), "Cada lote capturado desde el front se guarda aca.")
        st.write("")
        render_info_card("Interfaz de red", "No es una URL", "Este campo espera una interfaz del sistema como lo, wlo1 o enp3s0.")
        st.write("")
        render_info_card(
            "Escenario tecnico",
            live_scenario,
            scenario_metadata["summary"],
        )
        st.write("")
        render_info_card(
            "Etiqueta sugerida",
            live_label or "manual",
            f"Script asociado: {active_sim_config['script']}.",
        )

    if run_live_monitoring:
        progress_placeholder = st.empty()
        try:
            live_monitor_label_value = live_label
            live_monitor_scenario_value = None if live_scenario == "Sin escenario" else live_scenario

            def live_logger(message: str) -> None:
                progress_placeholder.info(message)

            with st.spinner(f"Capturando ventanas con {active_sim_config['label']} y procesando features..."):
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
        st.success(f"Lote live capturado: {live_monitor_results['rows']} ventanas guardadas en {live_monitor_results['output_path']}.")
        if live_monitor_results.get("simulator_version"):
            st.caption(f"Version de simulador asociada al lote: {live_monitor_results['simulator_version']}.")
        if live_monitor_results.get("scenario"):
            st.caption(f"Escenario asociado al lote: {live_monitor_results['scenario']}.")
        if live_monitor_results.get("saved_to_training"):
            st.caption(
                f"Este lote tambien se agrego a {LIVE_TRAINING_DATASET_PATH.as_posix()} con etiqueta {live_monitor_results['label']}."
            )
        preview_df = pd.DataFrame(live_monitor_results["batch_df"])
        st.dataframe(preview_df, width="stretch")


def render_live_tab(model_data: ModelData, selected_quantum_qubits: int) -> None:
    section_header(
        "Live",
        "Espacio exclusivo del laboratorio cuántico con tráfico capturado. Acá se construye el dataset live y se evalúa el Quantum Kernel sobre ese entorno.",
    )
    st.markdown("#### Qué pasa en esta sección")
    explain_cols = st.columns(3)
    with explain_cols[0]:
        render_info_card("1. Capturar", "Ventanas de trafico", "El sistema escucha trafico durante varios segundos y lo resume en features agregadas.")
    with explain_cols[1]:
        render_info_card("2. Etiquetar", "benign o attack", "Vos decidis si ese lote se agrega como trafico normal o ataque al dataset live.")
    with explain_cols[2]:
        render_info_card("3. Evaluar", "Quantum Kernel", "Con suficientes ventanas, se computa la matriz de fidelidad y se evalúa el clasificador.")

    st.write("")
    st.info(
        "Guía de laboratorio: esta sección no ejecuta el script de ataque directamente. El generador de tráfico corre por separado "
        "y la UI se encarga de capturar, resumir y almacenar lotes para el entrenamiento del Quantum Kernel."
    )
    st.write("")
    _render_live_monitoring(selected_quantum_qubits)

    st.write("")
    st.markdown("#### Estado del dataset live")
    selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
    selected_quantum_feature_map_reps = int(
        st.session_state.get("selected_quantum_feature_map_reps_live", LIVE_DEFAULT_FEATURE_MAP_REPS)
    )
    live_dataset_summary = inspect_live_quantum_dataset(test_size=selected_quantum_test_size)
    status_cols = st.columns(4)
    with status_cols[0]:
        render_info_card("CSV live", "Listo para entrenar" if live_dataset_summary["ready"] else "Falta completar", LIVE_TRAINING_DATASET_PATH.as_posix())
    with status_cols[1]:
        render_info_card("Ventanas", str(live_dataset_summary["total_rows"]), "Cantidad total de filas capturadas para el experimento live.")
    with status_cols[2]:
        render_info_card("Clases", f"{live_dataset_summary['benign_count']} benign / {live_dataset_summary['attack_count']} attack", "Balance actual del dataset live.")
    with status_cols[3]:
        render_info_card("Qubits maximos", str(live_dataset_summary["max_supported_qubits"]), "Limite actual segun filas disponibles.")

    if live_dataset_summary["ready"]:
        st.success(live_dataset_summary["message"])
    else:
        st.warning(live_dataset_summary["message"])

    st.write("")
    st.markdown("#### Ejecutar Quantum Kernel live")
    execute_left, execute_right = st.columns([1.2, 1])
    with execute_left:
        selected_quantum_test_size = st.select_slider(
            "Porcion reservada para test",
            options=[0.2, 0.25, 0.33, 0.5],
            value=selected_quantum_test_size,
            format_func=lambda value: f"{int(value * 100)}%",
            key="live_quantum_test_size_slider",
        )
        st.session_state["selected_quantum_test_size"] = selected_quantum_test_size
        
        with st.expander("Ajustes del Feature Map Cuántico live", expanded=False):
            selected_quantum_feature_map_reps = st.select_slider(
                "Repeticiones del feature map",
                options=[1, 2, 3],
                value=selected_quantum_feature_map_reps,
                key="live_quantum_feature_map_reps_slider",
            )
        st.session_state["selected_quantum_feature_map_reps_live"] = selected_quantum_feature_map_reps

        can_run_live = live_dataset_summary["ready"] and selected_quantum_qubits <= live_dataset_summary["max_supported_qubits"]
        run_live_quantum = st.button(
            f"Entrenar y evaluar Quantum Kernel live ({selected_quantum_qubits}q)",
            width="stretch",
            type="primary",
            key="run_live_quantum_button",
            disabled=not can_run_live,
        )
        if not live_dataset_summary["ready"]:
            st.caption("Primero completá el dataset live con suficientes ventanas benign y attack.")
        elif selected_quantum_qubits > live_dataset_summary["max_supported_qubits"]:
            st.caption(
                f"Con el dataset actual solo podés correr hasta {live_dataset_summary['max_supported_qubits']} qubits para este split."
            )

    with execute_right:
        render_info_card("Modelo activo", "Quantum Kernel (QSVM)", "Clasificación basada en matrices de fidelidad cuántica.")
        st.write("")
        render_info_card("Qubits elegidos", str(selected_quantum_qubits), "Dimensión actual del espacio de Hilbert.")
        st.write("")
        render_info_card(
            "Configuración",
            f"ZZFeatureMap (reps={selected_quantum_feature_map_reps})",
            "Entrelazamiento lineal para estabilidad en entorno live.",
        )

    if run_live_quantum:
        progress_placeholder = st.empty()
        try:
            import numpy as np
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
            from qiskit_machine_learning.kernels import FidelityQuantumKernel
            from qiskit.circuit.library import ZZFeatureMap
            from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset

            with st.spinner("Procesando dataset live y calculando matriz de Kernel Cuántico..."):
                progress_placeholder.info("Preparando datos y extrayendo features...")
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

                progress_placeholder.info("Calculando matriz de fidelidad cuántica (entrenamiento y test)...")
                train_kernel_matrix = quantum_kernel.evaluate(x_vec=dataset_bundle.X_train)
                test_kernel_matrix = quantum_kernel.evaluate(x_vec=dataset_bundle.X_test, y_vec=dataset_bundle.X_train)

                qsvm = SVC(kernel="precomputed")
                qsvm.fit(train_kernel_matrix, dataset_bundle.y_train)
                y_pred = qsvm.predict(test_kernel_matrix)

                metrics_dict = {
                    "accuracy": accuracy_score(dataset_bundle.y_test, y_pred),
                    "precision": precision_score(dataset_bundle.y_test, y_pred, zero_division=0),
                    "recall": recall_score(dataset_bundle.y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(dataset_bundle.y_test, y_pred, zero_division=0),
                }

                quantum_results = {
                    "metrics": metrics_dict,
                    "confusion_matrix": confusion_matrix(dataset_bundle.y_test, y_pred).tolist(),
                    "sample_size": dataset_bundle.sample_size,
                    "source": "real"
                }

            progress_placeholder.empty()
            st.session_state["quantum_lab_results"] = quantum_results
            st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
            st.session_state["quantum_lab_results_source"] = "live"
        except Exception as error:
            progress_placeholder.empty()
            st.error(f"No pude ejecutar la prueba cuantica live: {error}")

    st.write("")
    st.markdown("#### Resultado cuántico live actual")
    
    classical_live_results = load_classical_live_results()
    if classical_live_results is not None:
        st.write("")
        st.markdown("#### Comparacion contra baseline live clásico")
        compare_cols = st.columns(4)
        with compare_cols[0]:
            render_metric_card("RF live Accuracy", classical_live_results["accuracy"], "Baseline clásico sobre live")
        with compare_cols[1]:
            render_metric_card("RF live Precision", classical_live_results["precision"], "Misma fuente")
        with compare_cols[2]:
            render_metric_card("RF live Recall", classical_live_results["recall"], "Misma fuente")
        with compare_cols[3]:
            render_metric_card("RF live F1", classical_live_results["f1_score"], "Misma fuente")

    live_runtime_results = st.session_state.get("quantum_lab_results")
    live_runtime_qubits = st.session_state.get("quantum_lab_results_qubits")
    live_runtime_source = st.session_state.get("quantum_lab_results_source")
    
    display_results = (
        live_runtime_results
        if live_runtime_results and live_runtime_qubits == selected_quantum_qubits and live_runtime_source == "live"
        else None
    )

    if display_results and ("metrics" in display_results or "accuracy" in display_results):
        metrics_payload = display_results["metrics"] if "metrics" in display_results else display_results
        confusion_payload = display_results["confusion_matrix"]
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric_card("Accuracy", metrics_payload["accuracy"], "Resultado Quantum Kernel live")
        with metric_cols[1]:
            render_metric_card("Precision", metrics_payload["precision"], "Resultado Quantum Kernel live")
        with metric_cols[2]:
            render_metric_card("Recall", metrics_payload["recall"], "Resultado Quantum Kernel live")
        with metric_cols[3]:
            render_metric_card("F1-Score", metrics_payload["f1_score"], "Resultado Quantum Kernel live")
        st.write("")
        st.plotly_chart(
            make_confusion_chart(np.array(confusion_payload), height=300),
            width="stretch",
            key=f"live_confusion_chart_{selected_quantum_qubits}q",
        )
    else:
        if live_dataset_summary["ready"]:
            st.info(
                f"Todavía no hay una corrida de Quantum Kernel live ejecutada para {selected_quantum_qubits} qubits en esta sesión. "
                "Podés dispararla con el botón de arriba."
            )
        else:
            st.warning("Completá los requisitos del dataset live para habilitar el entrenamiento y evaluación.")