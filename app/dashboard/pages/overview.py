from __future__ import annotations

import streamlit as st

from dashboard.analytics import make_confusion_chart, make_global_comparison_chart
from dashboard.pages.demo import render_demo_panel
from dashboard.types import ModelData
from dashboard.ui import render_metric_card, section_header


def _build_active_model_summary(model_data: ModelData, selected_model: str) -> tuple[str, str]:
    if selected_model == "Modelo clasico":
        return (
            "Modelo clasico",
            "Estas viendo el baseline principal del proyecto: un Random Forest entrenado sobre datos tabulares. "
            "Sirve como referencia porque hoy es el enfoque mas estable, rapido y facil de interpretar.",
        )

    if model_data["Modelo cuantico"].get("selected_dataset_source") == "live":
        return (
            "Modelo cuantico · Live simulador",
            "Estas viendo el experimento cuantico sobre trafico capturado en laboratorio. "
            "Aca el VQC no trabaja con el dataset CICIDS2017, sino con ventanas reales resumidas en features agregadas.",
        )

    return (
        "Modelo cuantico · CICIDS2017",
        "Estas viendo el experimento cuantico sobre el dataset de referencia CICIDS2017. "
        "Este modo sirve para comparar el enfoque QML en un entorno controlado antes de pasar al laboratorio live.",
    )


def render_overview_tab(model_data: ModelData, selected_model: str) -> None:
    section_header(
        "Resumen",
        "Vista general para entender que hace el sistema, que significa cada enfoque y como leer los resultados sin conocimientos previos de cuantica.",
    )
    active_title, active_copy = _build_active_model_summary(model_data, selected_model)
    st.markdown(
        f"""
        <div class="compact-card">
            <div class="card-label">Enfoque activo</div>
            <div class="card-value">{active_title}</div>
            <div class="card-help">{active_copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("#### Que significa cada termino")
    term_left, term_right = st.columns(2)
    with term_left:
        st.markdown(
            """
            **Modelo clasico**
            Random Forest entrenado con datos tabulares. Es la referencia principal del proyecto.

            **Modelo cuantico**
            VQC experimental de Qiskit. Se usa para estudiar si un enfoque de Quantum Machine Learning puede aprender patrones utiles.

            **CICIDS2017**
            Dataset de referencia, curado y estable. Sirve para comparar modelos en un entorno controlado.
            """
        )
    with term_right:
        st.markdown(
            """
            **Live simulador**
            Dataset armado en laboratorio a partir de trafico capturado por ventanas. Sirve para probar el enfoque cuantico en un entorno mas real.

            **IBM validate**
            Entrenamiento local mas validacion corta en hardware real. Se usa para medir ruido, cola y limitaciones fisicas sin gastar tanta cuota.

            **Matriz de confusion**
            Resume cuantos aciertos y errores tuvo el modelo al distinguir trafico benigno e intrusiones.
            """
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
    st.markdown("#### Comparacion general")
    st.caption("Este grafico muestra, de forma resumida, como rinden el baseline clasico, el experimento cuantico y el hardware real si ya existe una corrida guardada.")
    st.plotly_chart(make_global_comparison_chart(model_data), width="stretch", key="overview_global_comparison_chart")

    model = model_data[selected_model]
    st.write("")
    st.markdown(f"#### Resultados actuales: {active_title}")
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
        st.markdown("**Como leer este bloque**")
        st.markdown(
            f"""
            - **Origen de metricas:** {model["source_label"]}
            - **Tiempo estimado:** {model["execution_time"]:.2f}s
            - **Descripcion:** {model["description"]}
            """
        )

    st.write("")
    st.markdown("#### Ejemplo simple")
    st.caption("Mini simulacion pedagogica para ver como cambia la lectura del sistema cuando sube el riesgo de una conexion.")
    render_demo_panel(selected_model)
