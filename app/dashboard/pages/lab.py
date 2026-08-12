from __future__ import annotations

import importlib
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.analytics import evaluate_classical_dataset, make_confusion_chart
from dashboard.constants import (
    CLASSICAL_MODEL_PATH, 
    CLASSICAL_RESULTS_PATH, 
    DATASET_PATH, 
    PCA_PATH, 
    SCALER_PATH, 
    UPLOADED_QUANTUM_DATASET_PATH
)
from dashboard.data import get_quantum_hardware_results_path, get_quantum_results_path
from dashboard.types import ModelData, QuantumDatasetSource
from dashboard.ui import render_info_card, render_metric_card, section_header


def _load_train_quantum_kernel():
    module = importlib.import_module("src.quantum.train_qkernel")
    module = importlib.reload(module)
    return module.train_quantum_kernel_model


def _render_quantum_lab(
    model_data: ModelData,
    selected_quantum_qubits: int,
    selected_quantum_dataset_source: QuantumDatasetSource,
) -> None:
    selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
    selected_quantum_data_source = st.session_state.get("quantum_cicids_data_source", "Usar data/dataset.csv")
    selected_quantum_feature_map_reps = int(st.session_state.get("selected_quantum_feature_map_reps", 2))
    
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Quantum Kernel (QSVM)")
        st.caption("Espacio de experimentación controlado usando Quantum Kernels y SVM clásico. Este enfoque evita las mesetas estériles del VQC tradicional.")
        
        uploaded_quantum_file = None
        selected_quantum_data_source = st.radio(
            "Dataset de entrada",
            ["Usar data/dataset.csv", "Subir CSV propio"],
            horizontal=True,
            key="quantum_cicids_data_source",
        )
        if selected_quantum_data_source == "Subir CSV propio":
            uploaded_quantum_file = st.file_uploader(
                "CSV para entrenar y evaluar el Quantum Kernel",
                type=["csv"],
                key="quantum_cicids_csv_uploader",
            )
            
        selected_quantum_execution_target = st.radio(
            "Modo de ejecucion cuantica",
            options=["simulator", "ibm_validate"],
            format_func=lambda value: "Simulador local" if value == "simulator" else "Validación corta hardware real (IBM)",
            horizontal=True,
            key="quantum_execution_target_radio",
        )
            
        selected_quantum_test_size = st.select_slider(
            "Porción reservada para test",
            options=[0.2, 0.25, 0.33, 0.5],
            value=selected_quantum_test_size,
            format_func=lambda value: f"{int(value * 100)}%",
            key="quantum_test_size_slider",
        )
        st.session_state["selected_quantum_test_size"] = selected_quantum_test_size
        
        with st.expander("Ajustes del Feature Map Cuántico", expanded=False):
            selected_quantum_feature_map_reps = st.select_slider(
                "Repeticiones del feature map",
                options=[1, 2, 3],
                value=selected_quantum_feature_map_reps,
                key="quantum_feature_map_reps_slider",
            )
            st.caption("Mayor repetición incrementa la expresividad del espacio de Hilbert proyectado.")
            
        st.session_state["selected_quantum_feature_map_reps"] = selected_quantum_feature_map_reps
        
        quantum_button = st.button(
            f"Ejecutar Quantum Kernel ({selected_quantum_qubits}q)",
            width="stretch",
            type="primary",
            disabled=(
                selected_quantum_data_source == "Subir CSV propio"
                and uploaded_quantum_file is None
            ),
        )
        st.caption(
            f"Esta prueba calcula la matriz de similitud cuántica para {selected_quantum_qubits} qubits."
        )

    # ==========================================
        # BLOQUE: Validación física acotada SpinQ
        # ==========================================
        with st.expander("Validación física acotada (SpinQ / RMN)", expanded=False):
            st.caption(
                "Permite correr un subconjunto de test utilizando circuitos nativos "
                "conectados directamente al hardware de la SpinQ Triangulum."
            )
            
            spinq_qubits = st.slider("Qubits para SpinQ", min_value=2, max_value=3, value=3, key="spinq_qubits_slider")
            spinq_samples = st.slider("Muestras de test acotadas", min_value=4, max_value=16, value=4, key="spinq_samples_slider")
            
            spinq_button = st.button("Ejecutar validación acotada para SpinQ", key="run_spinq_validation")
            
            if spinq_button:
                with st.spinner("Conectando al servidor SpinQuasar y ejecutando en hardware físico..."):
                    try:
                        from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset
                        from dashboard.constants import DATASET_PATH
                        import time
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                        from spinqit import get_compiler, Circuit, H
                        from src.quantum.spinq_connector import connect_to_spinq

                        bundle = prepare_quantum_dataset(
                            dataset_path=DATASET_PATH,
                            benign_samples=20,
                            attack_samples=20,
                            qubits=spinq_qubits,
                            test_size=0.2
                        )
                        
                        X_te = bundle.X_test[:spinq_samples]
                        y_te = bundle.y_test[:spinq_samples]
                        
                        engine, config = connect_to_spinq(task_name=f"spinq_eval_{int(time.time())}")
                        if not engine or not config:
                            raise RuntimeError("No se pudo establecer la conexión con el servidor SpinQ.")
                        
                        # Función interna de decodificación de conteos físicos
                        def _decode_counts(counts: dict) -> int:
                            if not counts or "error" in counts or "warning" in counts:
                                return 0
                            dominant_state = max(counts, key=counts.get)
                            if dominant_state[-1] == '1':
                                return 1
                            return 0

                        y_preds = []
                        comp = get_compiler("native")
                        
                        for i, x in enumerate(X_te):
                            circ = Circuit()
                            qubits_to_use = min(len(x), 3)
                            q = circ.allocateQubits(qubits_to_use)
                            for q_idx in range(qubits_to_use):
                                circ << (H, q[q_idx])
                            exe = comp.compile(circ, 0)
                            try:
                                res = engine.execute(exe, config)
                                if res and hasattr(res, "counts"):
                                    pred_label = _decode_counts(res.counts)
                                    y_preds.append(pred_label)
                                else:
                                    y_preds.append(0)
                                time.sleep(0.2)
                            except Exception:
                                y_preds.append(0)
                                
                        y_true = np.array(y_te[:len(y_preds)])
                        y_pred = np.array(y_preds)
                        
                        spinq_results = {
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
                        
                        st.session_state["spinq_lab_results"] = spinq_results
                        st.success("¡Validación física en SpinQ completada con éxito!")
                            
                    except Exception as e:
                        st.error(f"Error al conectar con la SpinQ: {e}")

        # Renderizado visual de las métricas y matriz de confusión de la SpinQ
        spinq_lab_results = st.session_state.get("spinq_lab_results")
        if spinq_lab_results and "metrics" in spinq_lab_results:
            st.write("")
            st.markdown("### Resultados de la Validación Física (SpinQ)")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                render_metric_card("Accuracy", spinq_lab_results["metrics"]["accuracy"], "Hardware SpinQ")
            with metric_cols[1]:
                render_metric_card("Precision", spinq_lab_results["metrics"]["precision"], "Hardware SpinQ")
            with metric_cols[2]:
                render_metric_card("Recall", spinq_lab_results["metrics"]["recall"], "Hardware SpinQ")
            with metric_cols[3]:
                render_metric_card("F1-Score", spinq_lab_results["metrics"]["f1_score"], "Hardware SpinQ")

            st.write("")
            count_cols = st.columns(2)
            with count_cols[0]:
                render_info_card("Predicciones normales", str(spinq_lab_results["prediction_counts"]["normal"]), "Cantidad predicha como tráfico benigno.")
            with count_cols[1]:
                render_info_card("Predicciones de intrusión", str(spinq_lab_results["prediction_counts"]["intrusion"]), "Cantidad predicha como tráfico malicioso.")

            if "confusion_matrix" in spinq_lab_results:
                st.write("")
                st.plotly_chart(
                    make_confusion_chart(np.array(spinq_lab_results["confusion_matrix"]), height=320),
                    width="stretch",
                    key="lab_spinq_confusion_chart"
                )
    with right:
        render_info_card("Estado del modelo", "Quantum Kernel (QSVM)", "Modelo activo basado en matrices de fidelidad cuántica + SVC precomputado.")
        st.write("")
        render_info_card("Dataset elegido", selected_quantum_data_source.replace("Usar ", ""), "Trabaja sobre los datos tabulares limpios y seleccionados.")
        st.write("")
        render_info_card(
            "Configuración actual",
            f"ZZFeatureMap (reps={selected_quantum_feature_map_reps})",
            "Entrelazamiento lineal optimizado para estabilidad.",
        )
        st.write("")
        render_info_card(
            "Comando en terminal equivalente",
            f"python -m src.quantum.train_qkernel --qubits {selected_quantum_qubits} --execution-target {selected_quantum_execution_target}",
            "Ejecución directa recomendada para control por consola.",
        )

    if quantum_button:
        progress_placeholder = st.empty()
        try:
            from dashboard.constants import DATASET_PATH as DASHBOARD_DATASET_PATH, UPLOADED_QUANTUM_DATASET_PATH
            
            dataset_path_for_run = DASHBOARD_DATASET_PATH
            if selected_quantum_data_source == "Subir CSV propio" and uploaded_quantum_file is not None:
                UPLOADED_QUANTUM_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
                UPLOADED_QUANTUM_DATASET_PATH.write_bytes(uploaded_quantum_file.getvalue())
                dataset_path_for_run = UPLOADED_QUANTUM_DATASET_PATH

            print("--> [DEBUG] Iniciando preparación del dataset...")
            with st.spinner("Calculando matriz de Kernel Cuántico y entrenando SVM clásico..."):
                from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset
                from qiskit.circuit.library import ZZFeatureMap
                from qiskit_machine_learning.kernels import FidelityQuantumKernel
                from sklearn.svm import SVC
                from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

                dataset_bundle = prepare_quantum_dataset(
                    dataset_path=dataset_path_for_run,
                    benign_samples=300,
                    attack_samples=300,
                    qubits=selected_quantum_qubits,
                    test_size=selected_quantum_test_size
                )
                print("--> [DEBUG] Dataset preparado con éxito. Extrayendo splits...")

                X_train = dataset_bundle.X_train
                X_test = dataset_bundle.X_test
                y_train = dataset_bundle.y_train
                y_test = dataset_bundle.y_test

                print(f"--> [DEBUG] Construyendo Feature Map con {selected_quantum_qubits} qubits y reps={selected_quantum_feature_map_reps}...")
                feature_map = ZZFeatureMap(
                    feature_dimension=selected_quantum_qubits, 
                    reps=selected_quantum_feature_map_reps, 
                    entanglement="linear"
                )
                
                quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

                if selected_quantum_execution_target == "ibm_validate":
                    X_test = X_test[:16]
                    y_test = y_test[:16]

                print("--> [DEBUG] Calculando matriz de kernel para TRAIN (¡Este es el paso pesado)...")
                train_kernel_matrix = quantum_kernel.evaluate(x_vec=X_train)
                
                print("--> [DEBUG] Calculando matriz de kernel para TEST...")
                test_kernel_matrix = quantum_kernel.evaluate(x_vec=X_test, y_vec=X_train)

                print("--> [DEBUG] Entrenando modelo SVM clásico con kernel precomputado...")
                qsvm = SVC(kernel="precomputed")
                qsvm.fit(train_kernel_matrix, y_train)
                
                print("--> [DEBUG] Prediciendo y calculando métricas...")
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
                    "sample_size": dataset_bundle.sample_size
                }
                print("--> [DEBUG] ¡Ejecución finalizada correctamente!")

            progress_placeholder.empty()
            st.session_state["quantum_lab_results"] = quantum_results
            st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
            st.session_state["quantum_lab_results_source"] = selected_quantum_dataset_source
            
        except Exception as error:
            progress_placeholder.empty()
            print(f"--> [ERROR] {error}")
            st.error(f"No pude ejecutar el Quantum Kernel: {error}")
    quantum_lab_results = st.session_state.get("quantum_lab_results")
    quantum_lab_results_qubits = st.session_state.get("quantum_lab_results_qubits")

    quantum_lab_results = st.session_state.get("quantum_lab_results")
    quantum_lab_results_qubits = st.session_state.get("quantum_lab_results_qubits")
    
    if quantum_lab_results and quantum_lab_results_qubits == selected_quantum_qubits:
        st.success("¡Prueba de Quantum Kernel finalizada con éxito!")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            render_metric_card("Accuracy", quantum_lab_results["metrics"]["accuracy"], "QSVM Evaluado")
        with metric_cols[1]:
            render_metric_card("Precision", quantum_lab_results["metrics"]["precision"], "QSVM Evaluado")
        with metric_cols[2]:
            render_metric_card("Recall", quantum_lab_results["metrics"]["recall"], "QSVM Evaluado")
        with metric_cols[3]:
            render_metric_card("F1-Score", quantum_lab_results["metrics"]["f1_score"], "QSVM Evaluado")
        st.write("")
        st.plotly_chart(
            make_confusion_chart(np.array(quantum_lab_results["confusion_matrix"]), height=300),
            width="stretch",
            key=f"lab_quantum_confusion_chart_{selected_quantum_dataset_source}_{selected_quantum_qubits}q",
        )


def _render_classical_lab(model_data: ModelData) -> None:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Baseline clásico")
        st.caption("Acá se evalúa el modelo clásico. Si los artefactos no existen, se entrenan y guardan automáticamente al presionar el botón.")
        source = st.radio("Dataset de entrada", ["Usar data/dataset.csv", "Subir CSV propio"], horizontal=True, key="classical_data_source")
        uploaded_file = None
        if source == "Subir CSV propio":
            uploaded_file = st.file_uploader("CSV para evaluar", type=["csv"], key="classical_csv_uploader")

        use_holdout = st.checkbox(
            "Reproducir holdout 80/20 del entrenamiento",
            value=(source == "Usar data/dataset.csv"),
            help="Si está activo, recrea el split del pipeline clásico y evalúa sobre el 20% de test.",
            key="classical_holdout_checkbox",
        )
        run_button = st.button("Ejecutar prueba clásica", width="stretch", type="primary", key="run_classical_button")

    with right:
        st.caption("Estado de artefactos")
        st.write(
            f"Modelo: {'ok' if CLASSICAL_MODEL_PATH.exists() else 'faltante'}\n\n"
            f"Scaler: {'ok' if SCALER_PATH.exists() else 'faltante'}\n\n"
            f"PCA: {'ok' if PCA_PATH.exists() else 'faltante'}\n\n"
            f"Métricas: {'ok' if CLASSICAL_RESULTS_PATH.exists() else 'faltante'}"
        )
        st.write("")
        render_info_card("Estado clásico", model_data["Modelo clasico"]["source_label"], "El dashboard usa métricas reales del clásico si encuentra results/classical_metrics.json.")

    if run_button:
        progress_placeholder = st.empty()
        try:
            with st.spinner("Procesando pipeline clásico..."):
                # 1. Verificamos si faltan los artefactos y los entrenamos de forma transparente al usuario
                if not (CLASSICAL_MODEL_PATH.exists() and SCALER_PATH.exists() and PCA_PATH.exists() and CLASSICAL_RESULTS_PATH.exists()):
                    progress_placeholder.info("No se encontraron los artefactos guardados. Entrenando modelo clásico automáticamente...")
                    import subprocess
                    import sys
                    result = subprocess.run([sys.executable, "-m", "src.classical.train_model"], capture_output=True, text=True)
                    if result.returncode != 0:
                        raise RuntimeError(f"Error al entrenar automáticamente: {result.stderr}")

                # 2. Preparamos el dataset para la evaluación
                progress_placeholder.info("Preparando dataset para evaluación...")
                if source == "Usar data/dataset.csv":
                    if not DATASET_PATH.exists():
                        raise FileNotFoundError("No existe data/dataset.csv.")
                    df = pd.read_csv(DATASET_PATH)
                else:
                    if uploaded_file is None:
                        raise ValueError("Subí un CSV antes de ejecutar la prueba.")
                    df = pd.read_csv(BytesIO(uploaded_file.getvalue()))

                # 3. Ejecutamos la evaluación
                progress_placeholder.info("Aplicando scaler, PCA y modelo clásico...")
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
        else:
            with metric_cols[0]:
                render_info_card("Registros", str(lab_results["rows"]), "Muestras evaluadas")
            with metric_cols[1]:
                render_info_card("Normal", str(lab_results["prediction_counts"]["normal"]), "Predicciones benignas")
            with metric_cols[2]:
                render_info_card("Intrusión", str(lab_results["prediction_counts"]["intrusion"]), "Predicciones positivas")
            with metric_cols[3]:
                render_info_card("Fuente", "Sin etiqueta", "Solo se muestran predicciones")

        st.write("")
        count_cols = st.columns(2)
        with count_cols[0]:
            render_info_card("Predicciones normales", str(lab_results["prediction_counts"]["normal"]), "Cantidad predicha como tráfico benigno.")
        with count_cols[1]:
            render_info_card("Predicciones de intrusión", str(lab_results["prediction_counts"]["intrusion"]), "Cantidad predicha como tráfico malicioso.")

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