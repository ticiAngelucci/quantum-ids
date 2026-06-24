from __future__ import annotations

import streamlit as st

from app.dashboard.analytics import make_confusion_chart, make_global_comparison_chart
from app.dashboard.types import ModelData
from app.dashboard.ui import render_info_card, render_metric_card, section_header


def render_overview_tab(model_data: ModelData, selected_model: str) -> None:
    section_header(
        "Vision general",
        "Resumen rapido para entender que modelo estas viendo, que tan bien funciona y de donde salen los datos.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card(
            "Base principal",
            "data/dataset.csv",
            "Dataset base del experimento clasico y del escenario cuantico de referencia.",
        )
    with col2:
        render_info_card(
            "Estado clasico",
            model_data["Modelo clasico"]["source_label"],
            model_data["Modelo clasico"]["description"],
        )
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
            f"Todavia no hay una corrida cuantica disponible para {model_data['Modelo cuantico'].get('dataset_source_label', 'CICIDS2017')} "
            f"con {model_data['Modelo cuantico']['selected_qubits']} qubits. Ejecutar: {quantum_command}"
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
        st.caption(
            "La matriz de confusión resume cómo se reparten aciertos y errores entre tráfico benigno e intrusiones detectadas."
        )
    with info_col:
        render_info_card("Origen de metricas", model["source_label"], "Te dice si los numeros vienen de una corrida real o de una demo.")
        st.write("")
        render_info_card("Tiempo estimado", f"{model['execution_time']:.2f}s", "Tiempo total aproximado del enfoque seleccionado.")
