from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from app.dashboard.analytics import (
    build_metrics_dataframe,
    build_quantum_runs_dataframe,
    capture_live_monitoring_batch,
    classify_mock_connection,
    evaluate_classical_dataset,
    inspect_live_quantum_dataset,
    make_confusion_chart,
    make_global_comparison_chart,
    make_noise_chart,
    make_time_chart,
    predict_quantum_live_batch,
)
from app.dashboard.constants import (
    CLASSICAL_MODEL_PATH,
    CLASSICAL_RESULTS_PATH,
    DATASET_PATH,
    LIVE_CAPTURE_PATH,
    LIVE_TRAINING_DATASET_PATH,
    PCA_PATH,
    SCALER_PATH,
    SUPPORTED_QUANTUM_DATASET_SOURCES,
    UPLOADED_QUANTUM_DATASET_PATH,
)
from app.dashboard.data import get_quantum_hardware_results_path, get_quantum_results_path
from app.dashboard.ui import render_info_card, render_metric_card, section_header


def render_overview_tab(model_data: dict, selected_model: str) -> None:
    section_header(
        "Vision general",
        "Resumen rapido para entender que modelo estas viendo, que tan bien funciona y de donde salen los datos.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card("Base principal", "data/dataset.csv", "Dataset base del experimento clasico y del escenario cuantico de referencia.")
    with col2:
        render_info_card("Estado clasico", model_data["Modelo clasico"]["source_label"], model_data["Modelo clasico"]["description"])
    with col3:
        render_info_card(
            "Panorama",
            "2 enfoques comparados",
            (
                f"Clasico: {model_data['Modelo clasico']['source_label']} | "
                f"Cuantico: {model_data['Modelo cuantico']['source_label']}"
            ),
        )

    if model_data["Modelo cuantico"]["source"] != "real":
        quantum_command = (
            f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {model_data['Modelo cuantico']['selected_qubits']}"
            if model_data["Modelo cuantico"].get("selected_dataset_source") == "live"
            else f"python -m src.quantum.train_vqc_simulator --qubits {model_data['Modelo cuantico']['selected_qubits']}"
        )
        st.warning(
            f"Todavia no hay una corrida cuantica disponible para {model_data['Modelo cuantico'].get('dataset_source_label', 'CICIDS2017')} con {model_data['Modelo cuantico']['selected_qubits']} qubits. "
            f"Ejecutar: {quantum_command}"
        )

    st.write("")
    st.plotly_chart(make_global_comparison_chart(model_data), width="stretch", key="overview_global_comparison_chart")

    model = model_data[selected_model]
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Accuracy", model["accuracy"], "Porcentaje total de aciertos")
    with metric_cols[1]:
        render_metric_card("Precision", model["precision"], "Que tan confiables son las alertas")
    with metric_cols[2]:
        render_metric_card("Recall", model["recall"], "Ataques reales detectados")
    with metric_cols[3]:
        render_metric_card("F1-Score", model["f1_score"], "Equilibrio general del modelo")

    st.write("")
    chart_col, info_col = st.columns([1.3, 1])
    with chart_col:
        st.plotly_chart(
            make_confusion_chart(model["confusion_matrix"], height=300),
            width="stretch",
            key=f"overview_confusion_chart_{selected_model}",
        )
        st.caption("La matriz de confusión resume cómo se reparten aciertos y errores entre tráfico benigno e intrusiones detectadas.")
    with info_col:
        render_info_card("Origen de metricas", model["source_label"], "Te dice si los numeros vienen de una corrida real o de una demo.")
        st.write("")
        render_info_card("Tiempo estimado", f"{model['execution_time']:.2f}s", "Tiempo total aproximado del enfoque seleccionado.")


def render_lab_tab(
    model_data: dict,
    selected_model: str,
    selected_quantum_qubits: int,
    selected_quantum_dataset_source: str,
) -> None:
    section_header(
        "Laboratorio de prueba",
        "Espacio de experimentacion guiada para ejecutar pruebas sin salir del dashboard.",
    )
    if selected_model == "Modelo cuantico" and selected_quantum_dataset_source == "live":
        st.markdown("#### Monitoreo live automatizado")
        st.caption(
            "Captura varias ventanas seguidas desde la UI, guarda el lote en results/live_capture.csv y, cuando corresponde, lo agrega tambien al dataset live de entrenamiento."
        )
        monitor_left, monitor_right = st.columns([1.25, 1])
        with monitor_left:
            live_duration = st.number_input("Duracion por ventana (segundos)", min_value=1, max_value=60, value=2, step=1, key="live_monitor_duration")
            live_windows = st.number_input("Cantidad de ventanas", min_value=1, max_value=100, value=5, step=1, key="live_monitor_windows")
            live_iface = st.text_input("Interfaz de red opcional", value="", placeholder="Ej: wlan0", key="live_monitor_iface")
            live_count = st.number_input("Limite de paquetes por ventana", min_value=0, max_value=100000, value=0, step=10, key="live_monitor_count")
            live_label = st.selectbox("Guardar este lote como", options=["Sin etiqueta", "benign", "attack"], index=0, key="live_monitor_label")
            append_to_training = live_label in {"benign", "attack"}
            run_live_monitoring = st.button("Capturar lote live desde el front", width="stretch", type="primary", key="live_monitor_run")
        with monitor_right:
            render_info_card("Archivo de salida", LIVE_CAPTURE_PATH.as_posix(), "Cada lote capturado desde el front se guarda aca.")
            st.write("")
            render_info_card("Uso recomendado", "1 boton, varias ventanas", "Sirve para dejar de abrir capturas una por una. Si asignas etiqueta, el lote tambien se agrega al dataset live.")
            st.write("")
            render_info_card("Inferencia cuantica live", "Experimental", "Si ya existe un dataset live suficiente, el dashboard puede entrenar un VQC en simulador y predecir este lote capturado.")

        if run_live_monitoring:
            progress_placeholder = st.empty()
            try:
                live_monitor_label_value = None if live_label == "Sin etiqueta" else live_label

                def live_logger(message: str) -> None:
                    progress_placeholder.info(message)

                with st.spinner("Capturando ventanas y procesando features live..."):
                    live_batch_df = capture_live_monitoring_batch(
                        duration=int(live_duration),
                        windows=int(live_windows),
                        iface=live_iface or None,
                        count=int(live_count),
                        label=live_monitor_label_value,
                        append_to_training=append_to_training,
                        logger=live_logger,
                    )

                monitor_result = {
                    "batch_df": live_batch_df.to_dict(orient="records"),
                    "rows": int(len(live_batch_df)),
                    "label": live_monitor_label_value,
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
            st.write("")
    elif selected_model == "Modelo cuantico":
        st.session_state.pop("live_monitor_results", None)

    if selected_model == "Modelo cuantico":
        from src.quantum.train_vqc_simulator import train_quantum_simulator

        selected_quantum_execution_target = st.session_state.get("selected_quantum_execution_target", "simulator")
        selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
        selected_quantum_data_source = st.session_state.get("quantum_cicids_data_source", "Usar data/dataset.csv")
        live_dataset_summary = inspect_live_quantum_dataset(test_size=selected_quantum_test_size) if selected_quantum_dataset_source == "live" else None
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("#### VQC")
            st.caption("Aca se ejecuta el experimento cuantico. El sistema entrena y evalua un clasificador variacional sobre una muestra controlada del dataset elegido.")
            uploaded_quantum_file = None
            if selected_quantum_dataset_source == "cicids":
                selected_quantum_data_source = st.radio(
                    "Origen de datos",
                    ["Usar data/dataset.csv", "Subir CSV propio"],
                    horizontal=True,
                    key="quantum_cicids_data_source",
                )
                if selected_quantum_data_source == "Subir CSV propio":
                    uploaded_quantum_file = st.file_uploader(
                        "CSV para entrenar y evaluar el VQC",
                        type=["csv"],
                        key="quantum_cicids_csv_uploader",
                    )
            selected_quantum_execution_target = st.radio(
                "Modo de ejecucion cuantica",
                options=["simulator", "ibm_validate"],
                index=0 if selected_quantum_execution_target == "simulator" else 1,
                format_func=lambda value: "Simulador local" if value == "simulator" else "Entrenamiento local + validacion IBM",
                horizontal=True,
                key="quantum_execution_target_radio",
            )
            st.session_state["selected_quantum_execution_target"] = selected_quantum_execution_target
            selected_quantum_test_size = st.select_slider(
                "Porcion reservada para test",
                options=[0.2, 0.25, 0.33, 0.5],
                value=selected_quantum_test_size,
                format_func=lambda value: f"{int(value * 100)}%",
                key="quantum_test_size_slider",
            )
            st.session_state["selected_quantum_test_size"] = selected_quantum_test_size
            selected_ibm_validation_samples = int(st.session_state.get("selected_ibm_validation_samples", 16))
            if selected_quantum_execution_target == "ibm_validate":
                selected_ibm_validation_samples = st.select_slider(
                    "Muestras del test a validar en IBM",
                    options=[4, 8, 12, 16, 24, 32],
                    value=selected_ibm_validation_samples,
                    key="ibm_validation_samples_slider",
                )
                st.session_state["selected_ibm_validation_samples"] = selected_ibm_validation_samples
            if selected_quantum_dataset_source == "live":
                live_dataset_summary = inspect_live_quantum_dataset(test_size=selected_quantum_test_size)
            quantum_button = st.button(
                f"Ejecutar prueba cuantica ({selected_quantum_qubits}q)",
                width="stretch",
                type="primary",
                disabled=(
                    (
                        selected_quantum_dataset_source == "live"
                        and live_dataset_summary is not None
                        and (not live_dataset_summary["ready"] or selected_quantum_qubits > live_dataset_summary["max_supported_qubits"])
                    )
                    or (
                        selected_quantum_dataset_source == "cicids"
                        and selected_quantum_data_source == "Subir CSV propio"
                        and uploaded_quantum_file is None
                    )
                ),
            )
            if selected_quantum_dataset_source == "live":
                st.caption(
                    f"Modo live exclusivo de la metodologia cuantica: usa {LIVE_TRAINING_DATASET_PATH.as_posix()} con capturas benign y attack construidas en laboratorio."
                )
                st.caption(
                    f"Con test {int(selected_quantum_test_size * 100)}% necesitas al menos 2 capturas benign, 2 attack y {live_dataset_summary['minimum_total_samples']} filas totales. Para una conclusion seria conviene usar muchas mas."
                )
                st.markdown(
                    """
                    <div class="compact-card">
                        <div class="card-label">Guia de laboratorio live</div>
                        <div class="card-help">
                            1. Abri otra terminal en la raiz del proyecto.<br>
                            2. Activa el entorno: <code>source venv/bin/activate</code>.<br>
                            3. Para capturas benign, deja el simulador apagado y corré:
                            <code>sudo "$(which python3)" -m src.live_detection.capture --duration 2 --windows 20 --output results/live_training_dataset.csv --label benign --append</code><br>
                            4. Para capturas attack, ejecuta manualmente <code>01_attack-scrapy.py</code> en otra terminal y, mientras corre, capturá:
                            <code>sudo "$(which python3)" -m src.live_detection.capture --duration 2 --windows 20 --output results/live_training_dataset.csv --label attack --append</code><br>
                            5. Volve al dashboard, verifica el estado del CSV live y recien ahi ejecuta la prueba cuantica.<br>
                            6. Si usas IBM validate, IBM solo toma una parte chica del test para ahorrar cuota.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if live_dataset_summary is not None:
                    if live_dataset_summary["ready"]:
                        st.success(live_dataset_summary["message"])
                    else:
                        st.warning(live_dataset_summary["message"])
                    if selected_quantum_qubits > live_dataset_summary["max_supported_qubits"]:
                        st.warning(
                            f"Con tu dataset actual y test {int(selected_quantum_test_size * 100)}%, solo podes probar hasta "
                            f"{live_dataset_summary['max_supported_qubits']} qubits. Baja el selector de qubits o agrega mas capturas."
                        )
            else:
                st.caption(
                    f"Modo base: usa una muestra balanceada del dataset CICIDS2017 y la reduce a {selected_quantum_qubits} dimensiones para representar {selected_quantum_qubits} qubits."
                )
            if selected_quantum_execution_target == "ibm_validate":
                st.info("Metodo recomendado: primero se entrena en simulador local y despues IBM valida una parte chica del test. Asi se mide ruido real sin gastar tanta cuota.")
            st.markdown(
                """
                <div class="compact-card">
                    <div class="card-label">Por que IBM valida y no entrena todo</div>
                    <div class="card-help">
                        Entrenar todo en hardware real consume mucha cuota y tarda mas por la naturaleza iterativa del optimizador.
                        El simulador local funciona como referencia ideal y repetible.
                        IBM Quantum se usa para validar una parte chica del test con los mismos pesos ya entrenados y asi medir ruido, latencia, cola y perdida de rendimiento real.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            quantum_results_path = (
                get_quantum_hardware_results_path(selected_quantum_qubits, dataset_source=selected_quantum_dataset_source)
                if selected_quantum_execution_target == "ibm_validate"
                else get_quantum_results_path(selected_quantum_qubits, dataset_source=selected_quantum_dataset_source)
            )
            status_label = "Resultado real" if quantum_results_path.exists() else "Pendiente"
            render_info_card("Estado del experimento", status_label, f"Se actualiza cuando se genera el archivo {quantum_results_path.as_posix()}.")
            st.write("")
            render_info_card("Origen de datos", "Live simulador" if selected_quantum_dataset_source == "live" else "CICIDS2017", "Esto afecta solo al experimento cuantico. El modelo clasico no usa el simulador de ataques.")
            st.write("")
            if selected_quantum_dataset_source == "cicids":
                render_info_card("Dataset elegido", selected_quantum_data_source.replace("Usar ", ""), "Para CICIDS podes usar el dataset local o subir un CSV propio solo para este experimento cuantico.")
                st.write("")
            render_info_card(
                "Comando en terminal",
                (
                    (
                        f"python -m src.quantum.train_vqc_simulator --execution-target ibm_validate --dataset-source live --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size} --ibm-validation-samples {selected_ibm_validation_samples}"
                        if selected_quantum_execution_target == "ibm_validate" and selected_quantum_dataset_source == "live"
                        else f"python -m src.quantum.train_vqc_simulator --execution-target ibm_validate --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size} --ibm-validation-samples {selected_ibm_validation_samples}"
                    )
                    if selected_quantum_execution_target == "ibm_validate"
                    else (
                        f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size}"
                        if selected_quantum_dataset_source == "live"
                        else f"python -m src.quantum.train_vqc_simulator --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size}"
                    )
                ),
                "La misma prueba que tambien puede ejecutarse fuera del dashboard.",
            )
            if selected_quantum_dataset_source == "live":
                st.write("")
                dataset_status = "Listo para entrenar" if live_dataset_summary is not None and live_dataset_summary["ready"] else "Falta completar"
                render_info_card("CSV live", dataset_status, f"Archivo esperado: {LIVE_TRAINING_DATASET_PATH.as_posix()}")
                st.write("")
                render_info_card(
                    "Capturas live",
                    f"{live_dataset_summary['benign_count']} benign / {live_dataset_summary['attack_count']} attack" if live_dataset_summary is not None else "Sin datos",
                    "Cantidad de ventanas etiquetadas detectadas en el dataset live.",
                )
                st.write("")
                render_info_card("Qubits maximos", str(live_dataset_summary["max_supported_qubits"]) if live_dataset_summary is not None else "0", "Limite actual segun las muestras disponibles para entrenar.")
            if selected_quantum_execution_target == "ibm_validate":
                st.write("")
                render_info_card("Subset IBM", str(selected_ibm_validation_samples), "Cuantas muestras del test se envian a IBM para la validacion corta.")

        if quantum_button:
            progress_placeholder = st.empty()
            try:
                log_messages = []

                def ui_logger(message: str) -> None:
                    log_messages.append(message)
                    progress_placeholder.info(message)

                dataset_path_for_run = None
                if selected_quantum_dataset_source == "cicids":
                    if selected_quantum_data_source == "Subir CSV propio":
                        if uploaded_quantum_file is None:
                            raise ValueError("Subi un CSV antes de ejecutar la prueba cuantica.")
                        UPLOADED_QUANTUM_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
                        UPLOADED_QUANTUM_DATASET_PATH.write_bytes(uploaded_quantum_file.getvalue())
                        dataset_path_for_run = UPLOADED_QUANTUM_DATASET_PATH
                    else:
                        dataset_path_for_run = DATASET_PATH

                with st.spinner("Entrenando VQC y evaluando resultados..."):
                    quantum_results = train_quantum_simulator(
                        num_qubits=selected_quantum_qubits,
                        dataset_source=selected_quantum_dataset_source,
                        dataset_path=dataset_path_for_run,
                        test_size=selected_quantum_test_size,
                        execution_target=selected_quantum_execution_target,
                        ibm_validation_samples=selected_ibm_validation_samples,
                        logger=ui_logger,
                    )

                progress_placeholder.empty()
                st.session_state["quantum_lab_results"] = quantum_results
                st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
                st.session_state["quantum_lab_results_source"] = selected_quantum_dataset_source
                st.session_state["selected_quantum_qubits"] = selected_quantum_qubits
            except Exception as error:
                progress_placeholder.empty()
                st.error(f"No pude ejecutar la prueba cuantica: {error}")

        quantum_lab_results = st.session_state.get("quantum_lab_results")
        quantum_lab_results_qubits = st.session_state.get("quantum_lab_results_qubits")
        quantum_lab_results_source = st.session_state.get("quantum_lab_results_source")
        if (
            quantum_lab_results
            and quantum_lab_results_qubits == selected_quantum_qubits
            and quantum_lab_results_source == selected_quantum_dataset_source
        ):
            st.success("Prueba cuantica finalizada.")
            if quantum_lab_results.get("validation_strategy") == "train_local_validate_ibm":
                st.info(
                    "Resultado IBM de bajo costo: el modelo se entreno localmente y IBM solo valido una parte del test. Esto sirve para medir impacto del hardware real, no para reemplazar el entrenamiento completo."
                )
            metric_cols = st.columns(4)
            with metric_cols[0]:
                render_metric_card("Accuracy", quantum_lab_results["metrics"]["accuracy"], "Resultado del VQC")
            with metric_cols[1]:
                render_metric_card("Precision", quantum_lab_results["metrics"]["precision"], "Resultado del VQC")
            with metric_cols[2]:
                render_metric_card("Recall", quantum_lab_results["metrics"]["recall"], "Resultado del VQC")
            with metric_cols[3]:
                render_metric_card("F1-Score", quantum_lab_results["metrics"]["f1_score"], "Resultado del VQC")
            st.write("")
            st.plotly_chart(
                make_confusion_chart(np.array(quantum_lab_results["confusion_matrix"]), height=300),
                width="stretch",
                key=f"lab_quantum_confusion_chart_{selected_quantum_dataset_source}_{selected_quantum_qubits}q",
            )
            st.caption(
                (
                    (
                        f"Este boton entrena localmente y valida en IBM con {selected_quantum_qubits} qubits, actualizando {quantum_results_path.as_posix()}."
                        if selected_quantum_execution_target == "ibm_validate"
                        else (
                            f"Este boton entrena el VQC live con {selected_quantum_qubits} qubits y actualiza results/quantum_live_simulated_metrics_{selected_quantum_qubits}q.json y results/quantum_live_simulated_metrics.json."
                            if selected_quantum_dataset_source == "live"
                            else f"Este boton entrena el VQC con {selected_quantum_qubits} qubits y actualiza results/quantum_simulated_metrics_{selected_quantum_qubits}q.json y results/quantum_simulated_metrics.json."
                        )
                    )
                )
            )
        elif quantum_lab_results and quantum_lab_results_qubits is not None:
            st.info(
                f"Los ultimos resultados visibles del laboratorio corresponden a {quantum_lab_results_qubits} qubits en fuente {str(quantum_lab_results_source).upper()}. "
                f"Si queres ver {selected_quantum_qubits} qubits en {selected_quantum_dataset_source.upper()}, ejecuta esa configuracion."
            )
        return

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Baseline clasico")
        st.caption("Aca se prueba el modelo clasico ya entrenado. Sirve como referencia principal porque hoy es el enfoque mas estable del sistema.")
        source = st.radio("Origen de datos", ["Usar data/dataset.csv", "Subir CSV propio"], horizontal=True, key="classical_data_source")
        uploaded_file = None
        if source == "Subir CSV propio":
            uploaded_file = st.file_uploader("CSV para evaluar", type=["csv"], key="classical_csv_uploader")

        use_holdout = st.checkbox(
            "Reproducir holdout 80/20 del entrenamiento",
            value=(source == "Usar data/dataset.csv"),
            help="Si esta activo, recrea el split del pipeline clasico y evalua sobre el 20% de test.",
            key="classical_holdout_checkbox",
        )
        run_button = st.button("Ejecutar prueba clasica", width="stretch", type="primary", key="run_classical_button")

    with right:
        st.markdown(
            f"""
            <div class="compact-card">
                <div class="card-label">Estado de artefactos</div>
                <div class="card-help">
                    <span class="status-pill {'real' if CLASSICAL_MODEL_PATH.exists() else 'mock'}">Modelo {'ok' if CLASSICAL_MODEL_PATH.exists() else 'faltante'}</span>
                    <span class="status-pill {'real' if SCALER_PATH.exists() else 'mock'}">Scaler {'ok' if SCALER_PATH.exists() else 'faltante'}</span>
                    <span class="status-pill {'real' if PCA_PATH.exists() else 'mock'}">PCA {'ok' if PCA_PATH.exists() else 'faltante'}</span>
                    <span class="status-pill {'real' if CLASSICAL_RESULTS_PATH.exists() else 'mock'}">Metricas {'ok' if CLASSICAL_RESULTS_PATH.exists() else 'faltante'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        render_info_card("Estado clasico", model_data["Modelo clasico"]["source_label"], "El dashboard usa metricas reales del clasico si encuentra results/classical_metrics.json.")

    if run_button:
        progress_placeholder = st.empty()
        try:
            with st.spinner("Ejecutando prueba clasica y calculando metricas..."):
                progress_placeholder.info("Preparando dataset para evaluacion...")
                if source == "Usar data/dataset.csv":
                    if not DATASET_PATH.exists():
                        raise FileNotFoundError("No existe data/dataset.csv.")
                    df = pd.read_csv(DATASET_PATH)
                else:
                    if uploaded_file is None:
                        raise ValueError("Subi un CSV antes de ejecutar la prueba.")
                    uploaded_bytes = BytesIO(uploaded_file.getvalue())
                    df = pd.read_csv(uploaded_bytes)

                progress_placeholder.info("Aplicando scaler, PCA y modelo clasico...")
                results = evaluate_classical_dataset(df, use_holdout_split=use_holdout)
                st.session_state["lab_results"] = results
                st.session_state["lab_source"] = source
            progress_placeholder.empty()
        except Exception as error:
            progress_placeholder.empty()
            st.error(f"No pude ejecutar la prueba: {error}")

    lab_results = st.session_state.get("lab_results")
    if lab_results:
        st.success(f"Prueba ejecutada sobre {lab_results['rows']} registros.")
        metric_cols = st.columns(4)
        if "metrics" in lab_results:
            with metric_cols[0]:
                render_metric_card("Accuracy", lab_results["metrics"]["accuracy"], "Resultado de la prueba")
            with metric_cols[1]:
                render_metric_card("Precision", lab_results["metrics"]["precision"], "Resultado de la prueba")
            with metric_cols[2]:
                render_metric_card("Recall", lab_results["metrics"]["recall"], "Resultado de la prueba")
            with metric_cols[3]:
                render_metric_card("F1-Score", lab_results["metrics"]["f1_score"], "Resultado de la prueba")
            st.caption(
                "Si ves valores muy altos, no significa que el problema sea trivial: este modelo ya viene muy ajustado al dataset de referencia. Por eso mostramos tambien precision y recall."
            )
        else:
            with metric_cols[0]:
                render_info_card("Registros", str(lab_results["rows"]), "Muestras evaluadas")
            with metric_cols[1]:
                render_info_card("Normal", str(lab_results["prediction_counts"]["normal"]), "Predicciones benignas")
            with metric_cols[2]:
                render_info_card("Intrusion", str(lab_results["prediction_counts"]["intrusion"]), "Predicciones positivas")
            with metric_cols[3]:
                render_info_card("Fuente", "Sin etiqueta", "Solo se muestran predicciones")

        st.write("")
        count_cols = st.columns(2)
        with count_cols[0]:
            render_info_card("Predicciones normales", str(lab_results["prediction_counts"]["normal"]), "Cantidad predicha como trafico benigno.")
        with count_cols[1]:
            render_info_card("Predicciones de intrusion", str(lab_results["prediction_counts"]["intrusion"]), "Cantidad predicha como trafico malicioso.")

        if "confusion_matrix" in lab_results:
            st.write("")
            st.plotly_chart(make_confusion_chart(lab_results["confusion_matrix"]), width="stretch", key="lab_classical_confusion_chart")


def render_analysis_tab(model_data: dict, selected_model: str, selected_quantum_dataset_source: str) -> None:
    section_header(
        "Comparacion y analisis",
        "Lectura guiada de rendimiento, ruido, tiempos y diferencias entre los enfoques.",
    )
    st.markdown("#### Resumen general")
    if model_data["Modelo cuantico"]["source"] != "real":
        command = (
            f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {model_data['Modelo cuantico']['selected_qubits']}"
            if selected_quantum_dataset_source == "live"
            else f"python -m src.quantum.train_vqc_simulator --qubits {model_data['Modelo cuantico']['selected_qubits']}"
        )
        st.info(
            f"Todavia no se entreno el VQC {'live' if selected_quantum_dataset_source == 'live' else 'CICIDS'} con {model_data['Modelo cuantico']['selected_qubits']} qubits. "
            f"Ejecutar: {command}"
        )
    st.plotly_chart(make_global_comparison_chart(model_data, height=380), width="stretch", key="analysis_global_comparison_chart")
    table_df = build_metrics_dataframe(model_data).pivot(index="Modelo", columns="Metrica", values="Valor")
    table_df = table_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
    st.dataframe(table_df.style.format("{:.1%}"), width="stretch")
    st.caption(
        f"El clasico muestra su referencia real. El bloque cuantico refleja la corrida {model_data['Modelo cuantico'].get('dataset_source_label', 'CICIDS2017')} de {model_data['Modelo cuantico']['selected_qubits']} qubits si existe un resultado guardado."
    )

    st.write("")
    st.markdown("#### Corridas VQC disponibles")
    quantum_runs_df = build_quantum_runs_dataframe(dataset_source=selected_quantum_dataset_source)
    st.dataframe(
        quantum_runs_df.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "Tiempo (s)": "{:.2f}",
                "Sample": "{:.0f}",
            },
            na_rep="Sin correr",
        ),
        width="stretch",
    )
    trained_runs = quantum_runs_df[quantum_runs_df["Estado"] == "Entrenado"]
    if not trained_runs.empty:
        best_row = trained_runs.sort_values(["F1-Score", "Accuracy"], ascending=False).iloc[0]
        st.caption(
            f"Mejor corrida VQC disponible en {best_row['Fuente']}: {int(best_row['Qubits'])} qubits "
            f"con F1-score {best_row['F1-Score']:.2%} y tiempo {best_row['Tiempo (s)']:.2f}s."
        )
    else:
        st.info("Todavia no hay corridas VQC disponibles para comparar.")

    st.write("")
    st.markdown("#### Ruido y limites del hardware")
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.plotly_chart(make_noise_chart(model_data, height=350), width="stretch", key="analysis_noise_chart")
    with col2:
        simulated = model_data["Modelo cuantico"]
        hardware = model_data["Hardware cuantico real"]
        render_info_card("Caida de Accuracy", f"{(simulated['accuracy'] - hardware['accuracy']):.1%}", "Perdida al pasar del ideal al hardware real.")
        st.write("")
        render_info_card("Caida de F1-Score", f"{(simulated['f1_score'] - hardware['f1_score']):.1%}", "Perdida general al salir del simulador ideal.")
        st.write("")
        diagnostics = hardware.get("hardware_diagnostics", {})
        limitation_flags = diagnostics.get("limitation_flags") or []
        render_info_card("Backend IBM", str(hardware.get("ibm_backend_name", "Pendiente")), "Procesador cuantico real usado en la validacion IBM.")
        st.write("")
        render_info_card("Alertas del hardware", ", ".join(limitation_flags) if limitation_flags else "Sin flags", "Senales resumidas de cola, ruido o conectividad limitada detectadas en el backend.")
        st.write("")
        hardware_gap_local = hardware.get("hardware_gap_vs_local_subset", {})
        if hardware_gap_local:
            render_info_card(
                "Caida vs local",
                f"Acc {hardware_gap_local.get('accuracy_drop', 0):.1%} | F1 {hardware_gap_local.get('f1_drop', 0):.1%}",
                "Diferencia entre IBM y la misma muestra evaluada localmente con los mismos pesos.",
            )
            st.write("")
        render_info_card("Modelo destacado", model_data[selected_model]["short_label"], "Enfoque activo en la lectura actual del dashboard.")
        if diagnostics:
            st.caption(
                f"T1 medio: {diagnostics.get('avg_t1_us', 'n/d')} us | "
                f"T2 medio: {diagnostics.get('avg_t2_us', 'n/d')} us | "
                f"Pending jobs: {diagnostics.get('pending_jobs', 'n/d')}"
            )

    st.write("")
    st.markdown("#### Costos de tiempo")
    st.plotly_chart(make_time_chart(model_data, height=350), width="stretch", key="analysis_time_chart")
    st.caption("El clasico sigue siendo el mas eficiente; el hardware real conserva el mayor costo temporal.")


def render_demo_tab(model_data: dict, selected_model: str) -> None:
    section_header(
        "Demo rapida de conexion",
        "Una simulacion sencilla para explicar como cambia la lectura del sistema sin cargar datasets reales.",
    )
    col1, col2 = st.columns([1.2, 1])
    with col1:
        packet_rate = st.slider("Paquetes por segundo", 50, 1000, 380, 10)
        failed_logins = st.slider("Intentos fallidos", 0, 20, 4)
        protocol_risk = st.slider("Riesgo del protocolo", 0, 10, 3)
        run_demo = st.button("Simular conexion", width="stretch")

    with col2:
        if run_demo:
            label, risk_score = classify_mock_connection(packet_rate, failed_logins, protocol_risk, selected_model)
            card_class = "attack" if label == "Intrusion detectada" else "normal"
            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="result-title">{label}</div>
                    <div class="card-help">Score de riesgo estimado: {risk_score:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_info_card("Estado", "Esperando simulacion", "Mové los controles y ejecutá la demo para ver una lectura rapida.")


def render_conclusion_tab(model_data: dict, selected_model: str) -> None:
    section_header(
        "Conclusiones visuales",
        "Cierre rapido para entender que aporta cada enfoque y por que esta comparacion importa.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card("Clasico", "Referencia principal", "Hoy es el camino mas estable, rapido y facil de interpretar.")
    with col2:
        render_info_card("QML", "Laboratorio experimental", "Sirve para estudiar si un enfoque cuantico puede aprender patrones utiles.")
    with col3:
        render_info_card("Hardware real", "Validacion fisica", "Permite mostrar que pasa cuando el modelo sale del simulador ideal.")

    st.write("")
    st.markdown(
        """
        <div class="compact-card">
            <div class="card-label">Lectura preliminar</div>
            <div class="card-help">
                El modelo clasico ofrece hoy la referencia mas solida para deteccion de anomalias en este entorno.
                El valor de QML aparece como linea experimental para medir potencial, limites y costo del enfoque cuantico.
                El hardware real se usa para validar que ocurre fuera del simulador ideal y entender mejor las restricciones actuales.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Enfoque activo al momento de lectura: {model_data[selected_model]['short_label']}.")
