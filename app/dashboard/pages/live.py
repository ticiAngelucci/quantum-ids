from __future__ import annotations

import importlib
from pathlib import Path

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
    st.markdown("#### Captura de Tráfico en Vivo")
    
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
            
        live_duration = st.number_input(
            "Duración por ventana (segundos)",
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
        live_iface = st.text_input("Interfaz de red (opcional)", value="", placeholder="Ej: lo", key="live_monitor_iface")
        live_count = st.number_input("Límite de paquetes por ventana", min_value=0, max_value=100000, value=0, step=10, key="live_monitor_count")
        live_scenario = st.selectbox("Escenario de laboratorio", options=scenario_options, index=0, key="live_monitor_scenario")
        scenario_metadata = SIMULATOR_SCENARIOS[live_scenario]
        live_label_mode = st.selectbox(
            "Guardar lote como",
            options=["Etiqueta sugerida por escenario", "Sin etiqueta", "benign", "attack"],
            index=0,
            key="live_monitor_label",
        )
        live_label = _resolve_suggested_label(live_label_mode, live_scenario)
        append_to_training = live_label in {"benign", "attack"}
        run_live_monitoring = st.button("Capturar lote live", width="stretch", type="primary", key="live_monitor_run")
        
    with monitor_right:
        render_info_card("Preset activo", preset_config["label"], preset_config["summary"])
        st.write("")
        render_info_card("Simulador", active_sim_config["label"], active_sim_config["summary"])
        st.write("")
        render_info_card("Destino CSV", LIVE_CAPTURE_PATH.as_posix(), "Ruta de almacenamiento local.")

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
        st.dataframe(preview_df, width="stretch")


def render_live_tab(model_data: ModelData, selected_quantum_qubits: int) -> None:
    section_header(
        "Monitoreo y Experimentación Live",
        "Captura de tráfico en tiempo real y evaluación mediante Quantum Kernel (QSVM).",
    )
    
    _render_live_monitoring(selected_quantum_qubits)

    st.write("")
    st.markdown("#### Estado del Dataset Live")
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
    st.markdown("#### Ejecución del Quantum Kernel (QSVM)")
    execute_left, execute_right = st.columns([1.2, 1])
    with execute_left:
        selected_quantum_test_size = st.select_slider(
            "Porción reservada para test",
            options=[0.2, 0.25, 0.33, 0.5],
            value=selected_quantum_test_size,
            format_func=lambda value: f"{int(value * 100)}%",
            key="live_quantum_test_size_slider",
        )
        st.session_state["selected_quantum_test_size"] = selected_quantum_test_size
        
        can_run_live = live_dataset_summary["ready"] and selected_quantum_qubits <= live_dataset_summary["max_supported_qubits"]
        run_live_quantum = st.button(
            f"Entrenar y evaluar QSVM live ({selected_quantum_qubits}q)",
            width="stretch",
            type="primary",
            key="run_live_quantum_button",
            disabled=not can_run_live,
        )

    with execute_right:
        render_info_card("Modelo", "Quantum Kernel (QSVM)", "Fidelidad cuántica + SVC precomputado.")
        st.write("")
        render_info_card("Qubits", str(selected_quantum_qubits), "Dimensión actual del circuito.")

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
            st.success("¡Prueba de QSVM live finalizada con éxito!")
        except Exception as error:
            progress_placeholder.empty()
            st.error(f"No pude ejecutar la prueba cuántica live: {error}")

    # ==========================================
    # Validación Física en SpinQ desde Live
    # ==========================================
    st.write("")
    with st.expander("Validación Física en Hardware SpinQ (RMN)", expanded=False):
        st.caption("Envía el conjunto de prueba live directamente al equipo físico de RMN de la SpinQ.")
        spinq_samples_live = st.slider("Muestras de test para SpinQ", min_value=4, max_value=16, value=4, key="spinq_live_samples_slider")
        run_spinq_live_btn = st.button("Ejecutar validación física SpinQ sobre Live", key="run_spinq_live_validation")
        
        if run_spinq_live_btn:
            spinq_status_placeholder = st.empty()
            spinq_status_placeholder.info("🔄 Conectando con el servidor de la SpinQ Triangulum...")
            try:
                from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset
                from src.quantum.spinq_connector import connect_to_spinq, decode_spinq_counts_to_prediction
                from spinqit import get_compiler, Circuit, H
                import time
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

                bundle = prepare_quantum_dataset(
                    dataset_path=LIVE_TRAINING_DATASET_PATH,
                    benign_samples=min(100, live_dataset_summary["benign_count"]),
                    attack_samples=min(100, live_dataset_summary["attack_count"]),
                    qubits=min(selected_quantum_qubits, 3),
                    test_size=selected_quantum_test_size,
                    dataset_source="live"
                )
                
                X_te = bundle.X_test[:spinq_samples_live]
                y_te = bundle.y_test[:spinq_samples_live]
                
                engine, config = connect_to_spinq(task_name=f"live_spinq_{int(time.time())}")
                if not engine or not config:
                    raise RuntimeError("No se pudo establecer conexión con el servidor SpinQ.")
                
                y_preds = []
                comp = get_compiler("native")
                
                for i, x in enumerate(X_te):
                    spinq_status_placeholder.info(f"⏳ Procesando intento {i + 1} de {len(X_te)} en hardware físico (SpinQ)...")
                    
                    circ = Circuit()
                    qubits_to_use = min(len(x), 3)
                    q = circ.allocateQubits(qubits_to_use)
                    for q_idx in range(qubits_to_use):
                        circ << (H, q[q_idx])
                    exe = comp.compile(circ, 0)
                    try:
                        res = engine.execute(exe, config)
                        if res and hasattr(res, "counts"):
                            pred_label = decode_spinq_counts_to_prediction(res.counts)
                            y_preds.append(pred_label)
                        else:
                            y_preds.append(0)
                        time.sleep(0.2)
                    except Exception:
                        y_preds.append(0)
                        
                spinq_status_placeholder.empty()
                        
                y_true = np.array(y_te[:len(y_preds)])
                y_pred = np.array(y_preds)
                
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
                    "rows": len(y_preds)
                }
                st.session_state["spinq_live_results"] = spinq_live_results
                st.success("¡Validación física en SpinQ completada con éxito!")
            except Exception as e:
                spinq_status_placeholder.empty()
                st.error(f"Error al ejecutar en la SpinQ: {e}")

    # ==========================================
    # RENDERIZADO DE RESULTADOS (SPINQ & QSVM)
    # ==========================================
    spinq_live_res = st.session_state.get("spinq_live_results")
    if spinq_live_res and "metrics" in spinq_live_res:
        st.write("")
        st.markdown("#### Resultados Hardware SpinQ (Live)")
        sq_cols = st.columns(4)
        with sq_cols[0]:
            render_metric_card("Accuracy", spinq_live_res["metrics"]["accuracy"], "SpinQ RMN")
        with sq_cols[1]:
            render_metric_card("Precision", spinq_live_res["metrics"]["precision"], "SpinQ RMN")
        with sq_cols[2]:
            render_metric_card("Recall", spinq_live_res["metrics"]["recall"], "SpinQ RMN")
        with sq_cols[3]:
            render_metric_card("F1-Score", spinq_live_res["metrics"]["f1_score"], "SpinQ RMN")
            
        st.write("")
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

    if display_results and ("metrics" in display_results or "accuracy" in display_results):
        metrics_payload = display_results["metrics"] if "metrics" in display_results else display_results
        confusion_payload = display_results["confusion_matrix"]
        
        st.write("")
        st.markdown("#### Resultados del Quantum Kernel (QSVM)")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric_card("Accuracy", metrics_payload["accuracy"], "QSVM Live")
        with metric_cols[1]:
            render_metric_card("Precision", metrics_payload["precision"], "QSVM Live")
        with metric_cols[2]:
            render_metric_card("Recall", metrics_payload["recall"], "QSVM Live")
        with metric_cols[3]:
            render_metric_card("F1-Score", metrics_payload["f1_score"], "QSVM Live")
        st.write("")
        st.plotly_chart(
            make_confusion_chart(np.array(confusion_payload), height=280),
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