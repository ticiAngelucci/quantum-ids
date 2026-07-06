from __future__ import annotations

import importlib
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.analytics import evaluate_classical_dataset, make_confusion_chart
from dashboard.constants import CLASSICAL_MODEL_PATH, CLASSICAL_RESULTS_PATH, DATASET_PATH, PCA_PATH, SCALER_PATH, UPLOADED_QUANTUM_DATASET_PATH
from dashboard.data import get_quantum_hardware_results_path, get_quantum_results_path
from dashboard.types import ModelData, QuantumDatasetSource
from dashboard.ui import render_info_card, render_metric_card, section_header


def _load_train_quantum_simulator():
    module = importlib.import_module("src.quantum.train_vqc_simulator")
    module = importlib.reload(module)
    return module.train_quantum_simulator


def _render_quantum_lab(
    model_data: ModelData,
    selected_quantum_qubits: int,
    selected_quantum_dataset_source: QuantumDatasetSource,
) -> None:
    train_quantum_simulator = _load_train_quantum_simulator()

    selected_quantum_execution_target = st.session_state.get("selected_quantum_execution_target", "simulator")
    selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
    selected_quantum_data_source = st.session_state.get("quantum_cicids_data_source", "Usar data/dataset.csv")
    selected_quantum_feature_map_reps = int(st.session_state.get("selected_quantum_feature_map_reps", 1))
    selected_quantum_ansatz_reps = int(st.session_state.get("selected_quantum_ansatz_reps", 1))
    selected_quantum_maxiter = int(st.session_state.get("selected_quantum_maxiter", 50))
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### VQC")
        st.caption("Aca se ejecutan pruebas controladas sobre datasets ya preparados. Esta seccion sirve para comparar el baseline clasico y el flujo cuantico en un entorno mas estable.")
        uploaded_quantum_file = None
        selected_quantum_data_source = st.radio(
            "Dataset de entrada",
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
        with st.expander("Ajustes del circuito cuantico", expanded=False):
            selected_quantum_feature_map_reps = st.select_slider(
                "Repeticiones del feature map",
                options=[1, 2, 3],
                value=selected_quantum_feature_map_reps,
                key="quantum_feature_map_reps_slider",
            )
            selected_quantum_ansatz_reps = st.select_slider(
                "Repeticiones del ansatz",
                options=[1, 2, 3],
                value=selected_quantum_ansatz_reps,
                key="quantum_ansatz_reps_slider",
            )
            selected_quantum_maxiter = st.select_slider(
                "Iteraciones maximas de COBYLA",
                options=[25, 50, 75, 100, 150],
                value=selected_quantum_maxiter,
                key="quantum_maxiter_slider",
            )
            st.caption(
                "Mas repeticiones o mas iteraciones pueden mejorar el ajuste, pero tambien aumentar el tiempo y el riesgo de sobreajuste."
            )
        st.session_state["selected_quantum_feature_map_reps"] = selected_quantum_feature_map_reps
        st.session_state["selected_quantum_ansatz_reps"] = selected_quantum_ansatz_reps
        st.session_state["selected_quantum_maxiter"] = selected_quantum_maxiter
        selected_ibm_validation_samples = int(st.session_state.get("selected_ibm_validation_samples", 16))
        if selected_quantum_execution_target == "ibm_validate":
            selected_ibm_validation_samples = st.select_slider(
                "Muestras del test a validar en IBM",
                options=[4, 8, 12, 16, 24, 32],
                value=selected_ibm_validation_samples,
                key="ibm_validation_samples_slider",
            )
            st.session_state["selected_ibm_validation_samples"] = selected_ibm_validation_samples
        quantum_button = st.button(
            f"Ejecutar prueba cuantica ({selected_quantum_qubits}q)",
            width="stretch",
            type="primary",
            disabled=(
                selected_quantum_data_source == "Subir CSV propio"
                and uploaded_quantum_file is None
            ),
        )
        st.caption(
            f"Esta prueba usa una muestra del dataset elegido y la reduce a {selected_quantum_qubits} dimensiones para representar {selected_quantum_qubits} qubits."
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
        render_info_card("Entrada del experimento", "CICIDS2017 o CSV propio", "Esta seccion trabaja sobre datasets ya preparados. El flujo live vive separado en la seccion Live.")
        st.write("")
        render_info_card("Dataset elegido", selected_quantum_data_source.replace("Usar ", ""), "Podes usar el dataset local o subir un CSV propio solo para este experimento cuantico.")
        st.write("")
        render_info_card(
            "Circuito actual",
            f"FM x{selected_quantum_feature_map_reps} · Ansatz x{selected_quantum_ansatz_reps}",
            f"Optimizador COBYLA con maxiter={selected_quantum_maxiter}.",
        )
        st.write("")
        render_info_card(
            "Comando en terminal",
            (
                (
                    f"python -m src.quantum.train_vqc_simulator --execution-target ibm_validate --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size} --ibm-validation-samples {selected_ibm_validation_samples} --feature-map-reps {selected_quantum_feature_map_reps} --ansatz-reps {selected_quantum_ansatz_reps} --maxiter {selected_quantum_maxiter}"
                )
                if selected_quantum_execution_target == "ibm_validate"
                else f"python -m src.quantum.train_vqc_simulator --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size} --feature-map-reps {selected_quantum_feature_map_reps} --ansatz-reps {selected_quantum_ansatz_reps} --maxiter {selected_quantum_maxiter}"
            ),
            "La misma prueba que tambien puede ejecutarse fuera del dashboard.",
        )
        if selected_quantum_execution_target == "ibm_validate":
            st.write("")
            render_info_card("Subset IBM", str(selected_ibm_validation_samples), "Cuantas muestras del test se envian a IBM para la validacion corta.")

    if quantum_button:
        progress_placeholder = st.empty()
        try:
            def ui_logger(message: str) -> None:
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
                    feature_map_reps=selected_quantum_feature_map_reps,
                    ansatz_reps=selected_quantum_ansatz_reps,
                    maxiter=selected_quantum_maxiter,
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
                "Resultado IBM de bajo costo: el modelo se entreno localmente y IBM solo valido una parte del test. "
                "Esto sirve para medir impacto del hardware real, no para reemplazar el entrenamiento completo."
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
                    else f"Este boton entrena el VQC con {selected_quantum_qubits} qubits y actualiza results/quantum_simulated_metrics_{selected_quantum_qubits}q.json y results/quantum_simulated_metrics.json."
                )
            )
        )
    elif quantum_lab_results and quantum_lab_results_qubits is not None:
        st.info(
            f"Los ultimos resultados visibles del laboratorio corresponden a {quantum_lab_results_qubits} qubits en fuente {str(quantum_lab_results_source).upper()}. "
            f"Si queres ver {selected_quantum_qubits} qubits en {selected_quantum_dataset_source.upper()}, ejecuta esa configuracion."
        )


def _render_classical_lab(model_data: ModelData) -> None:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Baseline clasico")
        st.caption("Aca se prueba el modelo clasico ya entrenado. Sirve como referencia principal porque hoy es el enfoque mas estable del sistema.")
        source = st.radio("Dataset de entrada", ["Usar data/dataset.csv", "Subir CSV propio"], horizontal=True, key="classical_data_source")
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
                    df = pd.read_csv(BytesIO(uploaded_file.getvalue()))

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


def render_lab_tab(
    model_data: ModelData,
    selected_model: str,
    selected_quantum_qubits: int,
    selected_quantum_dataset_source: QuantumDatasetSource,
) -> None:
    section_header(
        "Experimentar",
        "Espacio para correr pruebas controladas sobre datasets ya preparados. Aca vive el baseline clasico y el experimento cuantico sobre CICIDS o un CSV tabular propio.",
    )
    if selected_model == "Modelo cuantico":
        if selected_quantum_dataset_source == "live":
            st.info("El origen `Live simulador` se opera desde la seccion `Live`. Esta pantalla queda reservada para pruebas con datasets ya preparados.")
            return
        _render_quantum_lab(model_data, selected_quantum_qubits, selected_quantum_dataset_source)
        return

    _render_classical_lab(model_data)
