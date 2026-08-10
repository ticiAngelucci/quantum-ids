from __future__ import annotations

import streamlit as st

from dashboard.analytics import build_quantum_runs_dataframe, make_confusion_chart, make_global_comparison_chart
from dashboard.pages.demo import render_demo_panel
from dashboard.types import ModelData
from dashboard.ui import render_info_card, render_metric_card, render_spotlight_panel, section_header


def _build_active_model_summary(model_data: ModelData, selected_model: str) -> tuple[str, str]:
    if selected_model == "Modelo clasico":
        return (
            "Modelo clasico",
            "Baseline Random Forest entrenado sobre datos tabulares. Hoy funciona como referencia principal por estabilidad, velocidad e interpretabilidad.",
        )

    if model_data["Modelo cuantico"].get("selected_dataset_source") == "live":
        return (
            "Modelo cuantico · Live simulador",
            "Corrida cuantica sobre trafico capturado en laboratorio y resumido por ventanas. Es el escenario experimental mas cercano al uso real.",
        )

    return (
        "Modelo cuantico · CICIDS2017",
        "Corrida cuantica sobre el dataset de referencia CICIDS2017. Sirve como entorno controlado antes de pasar al laboratorio live.",
    )


def render_overview_tab(model_data: ModelData, selected_model: str) -> None:
    section_header(
        "Resumen",
        "Panel principal para leer estado del experimento, metricas y comparaciones sin recorrer toda la aplicacion.",
    )

    active_title, active_copy = _build_active_model_summary(model_data, selected_model)
    model = model_data[selected_model]
    classical = model_data["Modelo clasico"]
    quantum = model_data["Modelo cuantico"]
    hardware = model_data["Hardware cuantico real"]
    leader = classical if classical["accuracy"] >= quantum["accuracy"] else quantum
    accuracy_gap = abs(classical["accuracy"] - quantum["accuracy"])
    f1_gap = abs(classical["f1_score"] - quantum["f1_score"])
    selected_dataset_source = model_data["Modelo cuantico"].get("selected_dataset_source", "cicids")
    selected_qubits = model_data["Modelo cuantico"]["selected_qubits"]
    execution_target = st.session_state.get("selected_quantum_execution_target", "simulator")
    platform_label = "IBM validate" if execution_target == "ibm_validate" else "Simulador local"
    dataset_label = "Live simulador" if selected_dataset_source == "live" else "CICIDS2017"
    runs_df = build_quantum_runs_dataframe(dataset_source=selected_dataset_source)
    trained_runs = runs_df[runs_df["Estado"] == "Entrenado"]
    latest_run = "Sin corrida guardada"
    if not trained_runs.empty:
        latest = trained_runs.sort_values(["Qubits"], ascending=False).iloc[0]
        latest_run = f"{int(latest['Qubits'])}q · {latest['Fuente']} · F1 {latest['F1-Score']:.2%}"

    row1_left, row1_right = st.columns([1.35, 1])
    with row1_left:
        render_spotlight_panel(
            "Estado del experimento",
            active_title,
            "Vista tecnica del experimento activo. Este bloque resume la corrida actual y permite identificar rapidamente la configuracion visible.",
            meta=[
                ("Dataset", dataset_label),
                ("Plataforma", platform_label),
                ("Qubits", str(selected_qubits)),
            ],
        )
    with row1_right:
        render_info_card("Modelo", model["short_label"], active_copy)
        st.write("")
        render_info_card("Optimizer", "COBYLA", f"Estado visible: {model['source_label']}. Tiempo de referencia: {model['execution_time']:.2f}s.")
        st.write("")
        render_info_card("Ultima corrida", latest_run, "La consola prioriza la ultima evidencia guardada para esta fuente cuantica.")

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_metric_card("Accuracy", model["accuracy"], "Resultado principal del experimento activo")
    with kpi_cols[1]:
        render_metric_card("Precision", model["precision"], "Confiabilidad de las alertas")
    with kpi_cols[2]:
        render_metric_card("Recall", model["recall"], "Ataques reales detectados")
    with kpi_cols[3]:
        render_metric_card("F1-Score", model["f1_score"], "Equilibrio general")

    st.write("")
    compare_left, compare_right = st.columns([1.1, 1])
    with compare_left:
        render_spotlight_panel(
            "Comparacion inmediata",
            f"{classical['short_label']} vs {quantum['short_label']}",
            "Comparacion directa entre baseline clasico y corrida cuantica visible. Sirve para responder rapido quien lidera y cuanta distancia queda por cerrar.",
            meta=[
                ("Lider actual", leader["short_label"]),
                ("Brecha acc", f"{accuracy_gap:.1%}"),
                ("Brecha F1", f"{f1_gap:.1%}"),
            ],
        )
    with compare_right:
        render_info_card("Clasico", f"{classical['accuracy']:.1%} acc", f"F1 {classical['f1_score']:.1%} · {classical['source_label']}")
        st.write("")
        render_info_card("Cuantico", f"{quantum['accuracy']:.1%} acc", f"F1 {quantum['f1_score']:.1%} · {quantum.get('dataset_source_label', 'CICIDS2017')}")
        st.write("")
        render_info_card("Hardware real", f"{hardware['accuracy']:.1%} acc", f"F1 {hardware['f1_score']:.1%} · {hardware['source_label']}")

    if model_data["Modelo cuantico"]["source"] != "real":
        quantum_command = (
            f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {selected_qubits}"
            if selected_dataset_source == "live"
            else f"python -m src.quantum.train_vqc_simulator --qubits {selected_qubits}"
        )
        st.warning(
            f"Todavia no hay una corrida cuantica disponible para {model_data['Modelo cuantico'].get('dataset_source_label', 'CICIDS2017')} "
            f"con {selected_qubits} qubits. Ejecutar: {quantum_command}"
        )

    st.write("")
    st.markdown("#### Comparacion global")
    st.plotly_chart(make_global_comparison_chart(model_data), width="stretch", key="overview_global_comparison_chart")

    st.write("")
    st.markdown("#### Resultados por qubits")
    qubit_cols = st.columns(min(4, len(runs_df)))
    for col, (_, row) in zip(qubit_cols, runs_df.iterrows()):
        with col:
            if row["Estado"] == "Entrenado":
                render_info_card(
                    f"{int(row['Qubits'])}q",
                    f"{row['Accuracy']:.1%} acc",
                    f"F1 {row['F1-Score']:.1%} · {row['Tiempo (s)']:.1f}s · {row['Estado']}",
                )
            else:
                render_info_card(
                    f"{int(row['Qubits'])}q",
                    "Pendiente",
                    "Todavia no existe un JSON guardado para esta configuracion.",
                )

    st.write("")
    matrix_left, matrix_right = st.columns([1.35, 0.95])
    with matrix_left:
        st.markdown("#### Matriz de confusion")
        st.plotly_chart(
            make_confusion_chart(model["confusion_matrix"], height=320),
            width="stretch",
            key=f"overview_confusion_chart_{selected_model}",
        )
    with matrix_right:
        render_spotlight_panel(
            "Lectura tecnica",
            active_title,
            active_copy,
            meta=[
                ("Metricas", model["source_label"]),
                ("Tiempo", f"{model['execution_time']:.2f}s"),
                ("Estado", model["short_label"]),
            ],
        )

    st.write("")
    with st.expander("Configuracion experimental y detalles tecnicos", expanded=False):
        cfg_cols = st.columns(3)
        with cfg_cols[0]:
            render_info_card("Dataset", dataset_label, "Fuente de datos usada por la corrida cuantica visible.")
        with cfg_cols[1]:
            render_info_card("Plataforma", platform_label, "Entorno donde se ejecuta la corrida cuantica actual.")
        with cfg_cols[2]:
            render_info_card("Qubits", str(selected_qubits), "Cantidad de qubits actualmente visible en la consola.")
        st.write("")
        render_info_card("Modelo activo", model["short_label"], model["description"])

    with st.expander("Contexto de investigacion", expanded=False):
        render_spotlight_panel(
            "Pregunta de investigacion",
            "¿Puede Quantum Machine Learning detectar anomalias de red de forma competitiva frente a un baseline clasico?",
            "La consola compara un baseline Random Forest con un VQC experimental. La lectura correcta no es solo mirar accuracy: tambien importa precision, recall, F1, costo temporal y estabilidad entre simulacion y hardware real.",
            meta=[
                ("Lider actual", leader["short_label"]),
                ("Brecha acc", f"{accuracy_gap:.1%}"),
                ("Brecha F1", f"{f1_gap:.1%}"),
            ],
        )

    with st.expander("Acerca del proyecto", expanded=False):
        info_cols = st.columns(3)
        with info_cols[0]:
            render_info_card("Clasico", "Baseline", "Random Forest entrenado con datos tabulares.")
        with info_cols[1]:
            render_info_card("Cuantico", "VQC", "Clasificador variacional de Qiskit usado como linea experimental.")
        with info_cols[2]:
            render_info_card("IBM validate", "Hardware", "Validacion corta en hardware real con entrenamiento local.")

    with st.expander("Demo rapida de lectura", expanded=False):
        render_demo_panel(selected_model)
