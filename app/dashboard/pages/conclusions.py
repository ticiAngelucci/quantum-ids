from __future__ import annotations

import streamlit as st

from dashboard.types import ModelData
from dashboard.ui import render_info_card, section_header


def render_conclusion_tab(model_data: ModelData, selected_model: str) -> None:
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
