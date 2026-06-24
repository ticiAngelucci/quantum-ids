from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.analytics import (
    capture_live_monitoring_batch,
    inspect_live_quantum_dataset,
    make_confusion_chart,
    predict_quantum_live_batch,
)
from dashboard.constants import LIVE_CAPTURE_PATH, LIVE_TRAINING_DATASET_PATH
from dashboard.data import get_quantum_hardware_results_path, get_quantum_results_path
from dashboard.types import ModelData
from dashboard.ui import render_info_card, render_metric_card, section_header


def _render_live_monitoring(selected_quantum_qubits: int) -> None:
    scenario_options = [
        "Sin escenario",
        "TCP SYN Flood",
        "UDP Flood",
        "ICMP Flood",
        "Ataque Multi-Fuente",
        "Modo Experto",
    ]
    st.markdown("#### Captura live")
    st.caption(
        "Este bloque automatiza la captura por ventanas. No lanza ataques: solo escucha trafico, resume features y guarda el lote para el experimento cuantico live."
    )
    monitor_left, monitor_right = st.columns([1.25, 1])
    with monitor_left:
        live_duration = st.number_input("Duracion por ventana (segundos)", min_value=1, max_value=60, value=2, step=1, key="live_monitor_duration")
        live_windows = st.number_input("Cantidad de ventanas", min_value=1, max_value=100, value=5, step=1, key="live_monitor_windows")
        live_iface = st.text_input("Interfaz de red (ej: lo, wlo1, enp3s0)", value="", placeholder="Ej: lo", key="live_monitor_iface")
        live_count = st.number_input("Limite de paquetes por ventana", min_value=0, max_value=100000, value=0, step=10, key="live_monitor_count")
        live_label = st.selectbox("Guardar este lote como", options=["Sin etiqueta", "benign", "attack"], index=0, key="live_monitor_label")
        live_scenario = st.selectbox("Escenario de laboratorio", options=scenario_options, index=0, key="live_monitor_scenario")
        append_to_training = live_label in {"benign", "attack"}
        run_live_monitoring = st.button("Capturar lote live", width="stretch", type="primary", key="live_monitor_run")
    with monitor_right:
        render_info_card("Archivo de salida", LIVE_CAPTURE_PATH.as_posix(), "Cada lote capturado desde el front se guarda aca.")
        st.write("")
        render_info_card("Interfaz de red", "No es una URL", "Este campo espera una interfaz del sistema como lo, wlo1 o enp3s0. No se pone localhost:8501.")
        st.write("")
        render_info_card("Escenario elegido", live_scenario, "Sirve para documentar el lote capturado. No ejecuta el script de ataque desde el dashboard.")

    if run_live_monitoring:
        progress_placeholder = st.empty()
        try:
            live_monitor_label_value = None if live_label == "Sin etiqueta" else live_label
            live_monitor_scenario_value = None if live_scenario == "Sin escenario" else live_scenario

            def live_logger(message: str) -> None:
                progress_placeholder.info(message)

            with st.spinner("Capturando ventanas y procesando features live..."):
                live_batch_df = capture_live_monitoring_batch(
                    duration=int(live_duration),
                    windows=int(live_windows),
                    iface=live_iface or None,
                    count=int(live_count),
                    label=live_monitor_label_value,
                    scenario=live_monitor_scenario_value,
                    append_to_training=append_to_training,
                    logger=live_logger,
                )

            monitor_result = {
                "batch_df": live_batch_df.to_dict(orient="records"),
                "rows": int(len(live_batch_df)),
                "label": live_monitor_label_value,
                "scenario": live_monitor_scenario_value,
                "saved_to_training": append_to_training,
                "output_path": LIVE_CAPTURE_PATH.as_posix(),
            }

            selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
            quantum_live_summary = inspect_live_quantum_dataset(test_size=selected_quantum_test_size)
            if quantum_live_summary["ready"] and selected_quantum_qubits <= quantum_live_summary["max_supported_qubits"]:
                prediction_result = predict_quantum_live_batch(
                    live_df=live_batch_df.select_dtypes(include=[np.number]),
                    num_qubits=selected_quantum_qubits,
                    test_size=selected_quantum_test_size,
                    logger=live_logger,
                )
                monitor_result["prediction_result"] = prediction_result
            else:
                monitor_result["prediction_result"] = {
                    "compatible": False,
                    "message": (
                        "Todavia no hay dataset live suficiente para inferencia cuantica directa desde el front. "
                        "Captura mas ventanas benign y attack o baja los qubits."
                    ),
                }

            st.session_state["live_monitor_results"] = monitor_result
            progress_placeholder.empty()
        except Exception as error:
            progress_placeholder.empty()
            st.error(f"No pude ejecutar el monitoreo live: {error}")

    live_monitor_results = st.session_state.get("live_monitor_results")
    if live_monitor_results:
        st.success(f"Lote live capturado: {live_monitor_results['rows']} ventanas guardadas en {live_monitor_results['output_path']}.")
        if live_monitor_results.get("scenario"):
            st.caption(f"Escenario asociado al lote: {live_monitor_results['scenario']}.")
        if live_monitor_results.get("saved_to_training"):
            st.caption(
                f"Este lote tambien se agrego a {LIVE_TRAINING_DATASET_PATH.as_posix()} con etiqueta {live_monitor_results['label']}."
            )
        preview_df = pd.DataFrame(live_monitor_results["batch_df"])
        st.dataframe(preview_df, width="stretch")
        prediction_result = live_monitor_results.get("prediction_result", {})
        if "prediction_counts" in prediction_result:
            pred_cols = st.columns(2)
            with pred_cols[0]:
                render_info_card("Ventanas benignas", str(prediction_result["prediction_counts"]["normal"]), "Resultado del VQC live para este lote.")
            with pred_cols[1]:
                render_info_card("Ventanas attack", str(prediction_result["prediction_counts"]["intrusion"]), "Ventanas marcadas como anomalias por el VQC live.")
        else:
            st.info(prediction_result.get("message", "El monitoreo live ya capturo el lote; ahora puedes usarlo para entrenar el VQC."))


def render_live_tab(model_data: ModelData, selected_quantum_qubits: int) -> None:
    from src.quantum.train_vqc_simulator import train_quantum_simulator

    section_header(
        "Live",
        "Espacio exclusivo del laboratorio cuantico con trafico capturado. Aca se construye el dataset live y se prueba el VQC sobre ese entorno experimental.",
    )
    st.markdown("#### Que pasa en esta seccion")
    explain_cols = st.columns(3)
    with explain_cols[0]:
        render_info_card("1. Capturar", "Ventanas de trafico", "El sistema escucha trafico durante varios segundos y lo resume en features agregadas.")
    with explain_cols[1]:
        render_info_card("2. Etiquetar", "benign o attack", "Vos decidis si ese lote se agrega como trafico normal o como trafico de ataque al dataset live.")
    with explain_cols[2]:
        render_info_card("3. Evaluar", "VQC live", "Con suficientes ventanas, el VQC puede entrenarse y evaluarse usando este mismo esquema de features.")

    st.write("")
    st.markdown(
        """
        <div class="compact-card">
            <div class="card-label">Guia de laboratorio</div>
            <div class="card-help">
                Esta seccion no ejecuta el script de ataque. El generador de trafico sigue corriendo por separado en laboratorio.
                La UI se ocupa de capturar, resumir, guardar el lote y mostrar resultados del flujo cuantico live.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    _render_live_monitoring(selected_quantum_qubits)

    st.write("")
    st.markdown("#### Estado del dataset live")
    selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
    live_dataset_summary = inspect_live_quantum_dataset(test_size=selected_quantum_test_size)
    status_cols = st.columns(4)
    with status_cols[0]:
        render_info_card("CSV live", "Listo para entrenar" if live_dataset_summary["ready"] else "Falta completar", LIVE_TRAINING_DATASET_PATH.as_posix())
    with status_cols[1]:
        render_info_card("Ventanas", str(live_dataset_summary["total_rows"]), "Cantidad total de filas capturadas para el experimento live.")
    with status_cols[2]:
        render_info_card("Clases", f"{live_dataset_summary['benign_count']} benign / {live_dataset_summary['attack_count']} attack", "Balance actual del dataset live.")
    with status_cols[3]:
        render_info_card("Qubits maximos", str(live_dataset_summary["max_supported_qubits"]), "Limite actual segun filas de entrenamiento disponibles.")

    if live_dataset_summary["ready"]:
        st.success(live_dataset_summary["message"])
    else:
        st.warning(live_dataset_summary["message"])

    st.write("")
    st.markdown("#### Ejecutar VQC live")
    execute_left, execute_right = st.columns([1.2, 1])
    selected_quantum_execution_target = st.session_state.get("selected_quantum_execution_target", "simulator")
    with execute_left:
        selected_quantum_execution_target = st.radio(
            "Modo de ejecucion cuantica",
            options=["simulator", "ibm_validate"],
            index=0 if selected_quantum_execution_target == "simulator" else 1,
            format_func=lambda value: "Simulador local" if value == "simulator" else "Entrenamiento local + validacion IBM",
            horizontal=True,
            key="live_quantum_execution_target_radio",
        )
        st.session_state["selected_quantum_execution_target"] = selected_quantum_execution_target
        selected_quantum_test_size = st.select_slider(
            "Porcion reservada para test",
            options=[0.2, 0.25, 0.33, 0.5],
            value=selected_quantum_test_size,
            format_func=lambda value: f"{int(value * 100)}%",
            key="live_quantum_test_size_slider",
        )
        st.session_state["selected_quantum_test_size"] = selected_quantum_test_size
        selected_ibm_validation_samples = int(st.session_state.get("selected_ibm_validation_samples", 16))
        if selected_quantum_execution_target == "ibm_validate":
            selected_ibm_validation_samples = st.select_slider(
                "Muestras del test a validar en IBM",
                options=[4, 8, 12, 16, 24, 32],
                value=selected_ibm_validation_samples,
                key="live_ibm_validation_samples_slider",
            )
            st.session_state["selected_ibm_validation_samples"] = selected_ibm_validation_samples
        else:
            selected_ibm_validation_samples = int(st.session_state.get("selected_ibm_validation_samples", 16))

        can_run_live = live_dataset_summary["ready"] and selected_quantum_qubits <= live_dataset_summary["max_supported_qubits"]
        run_live_quantum = st.button(
            f"Entrenar y evaluar VQC live ({selected_quantum_qubits}q)",
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
        live_results_path = (
            get_quantum_hardware_results_path(selected_quantum_qubits, dataset_source="live")
            if selected_quantum_execution_target == "ibm_validate"
            else get_quantum_results_path(selected_quantum_qubits, dataset_source="live")
        )
        render_info_card("Corrida esperada", live_results_path.name, "Archivo que se actualiza cuando termina la prueba live.")
        st.write("")
        render_info_card("Qubits elegidos", str(selected_quantum_qubits), "Cantidad actual de qubits para este experimento live.")
        st.write("")
        render_info_card(
            "Capacidad del dataset",
            f"hasta {live_dataset_summary['max_supported_qubits']}q",
            "El dataset live limita cuántos qubits pueden entrenarse según las filas de entrenamiento disponibles.",
        )
        if selected_quantum_execution_target == "ibm_validate":
            st.write("")
            render_info_card("Subset IBM", str(selected_ibm_validation_samples), "Cuántas muestras del test se envían a IBM para la validación corta.")

    if run_live_quantum:
        progress_placeholder = st.empty()
        try:
            def ui_logger(message: str) -> None:
                progress_placeholder.info(message)

            with st.spinner("Entrenando VQC live y evaluando resultados..."):
                quantum_results = train_quantum_simulator(
                    num_qubits=selected_quantum_qubits,
                    dataset_source="live",
                    test_size=selected_quantum_test_size,
                    execution_target=selected_quantum_execution_target,
                    ibm_validation_samples=selected_ibm_validation_samples,
                    logger=ui_logger,
                )

            progress_placeholder.empty()
            st.session_state["quantum_lab_results"] = quantum_results
            st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
            st.session_state["quantum_lab_results_source"] = "live"
        except Exception as error:
            progress_placeholder.empty()
            st.error(f"No pude ejecutar la prueba cuantica live: {error}")

    st.write("")
    st.markdown("#### Resultado cuantico live actual")
    quantum_results = model_data["Modelo cuantico"]
    if quantum_results.get("selected_dataset_source") != "live":
        st.info("Para ver y ejecutar el flujo live, elegi `Modelo cuantico` y `Live simulador` en la barra lateral.")
        return

    info_cols = st.columns(3)
    with info_cols[0]:
        render_info_card("Modo activo", "Live simulador", "Este flujo usa el dataset construido desde capturas del laboratorio.")
    with info_cols[1]:
        render_info_card("Qubits elegidos", str(selected_quantum_qubits), "Cantidad actual de qubits configurada para el experimento live.")
    with info_cols[2]:
        render_info_card("IBM validate", "Disponible", "Si queres medir hardware real, esa opcion sigue en la seccion Experimentar del VQC.")

    live_runtime_results = st.session_state.get("quantum_lab_results")
    live_runtime_qubits = st.session_state.get("quantum_lab_results_qubits")
    live_runtime_source = st.session_state.get("quantum_lab_results_source")
    display_results = (
        live_runtime_results
        if live_runtime_results and live_runtime_qubits == selected_quantum_qubits and live_runtime_source == "live"
        else quantum_results
    )

    if display_results.get("source") == "real" or "metrics" in display_results:
        metrics_payload = display_results["metrics"] if "metrics" in display_results else display_results
        confusion_payload = display_results["confusion_matrix"]
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric_card("Accuracy", metrics_payload["accuracy"], "Resultado del VQC live")
        with metric_cols[1]:
            render_metric_card("Precision", metrics_payload["precision"], "Resultado del VQC live")
        with metric_cols[2]:
            render_metric_card("Recall", metrics_payload["recall"], "Resultado del VQC live")
        with metric_cols[3]:
            render_metric_card("F1-Score", metrics_payload["f1_score"], "Resultado del VQC live")
        st.write("")
        st.plotly_chart(
            make_confusion_chart(np.array(confusion_payload), height=300),
            width="stretch",
            key=f"live_confusion_chart_{selected_quantum_qubits}q",
        )
    else:
        command = f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size}"
        if selected_quantum_qubits > live_dataset_summary["max_supported_qubits"]:
            st.warning(
                f"El dataset live actual solo soporta hasta {live_dataset_summary['max_supported_qubits']} qubits con test {int(selected_quantum_test_size * 100)}%. "
                f"Agregá más ventanas o bajá el número de qubits."
            )
        else:
            st.info(
                f"Todavia no hay una corrida VQC live guardada para {selected_quantum_qubits} qubits. "
                f"Podés ejecutarla desde este frente con el boton de arriba o por terminal con: {command}"
            )
