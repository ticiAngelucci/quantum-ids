from __future__ import annotations

import streamlit as st

from app.dashboard.analytics import classify_mock_connection
from app.dashboard.types import ModelData
from app.dashboard.ui import render_info_card, section_header


def render_demo_tab(model_data: ModelData, selected_model: str) -> None:
    _ = model_data
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
