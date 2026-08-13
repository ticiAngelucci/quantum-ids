from __future__ import annotations

import streamlit as st

from dashboard.constants import (
    ACCENT_YELLOW,
    BACKGROUND,
    DANGER,
    MUTED_TEXT,
    PRIMARY_BLUE,
    SECONDARY_BLUE,
    SUCCESS,
    TEXT,
)


def configure_page() -> None:
    st.set_page_config(
        page_title="Quantum IDS · Panel de Control",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --boca-blue: {PRIMARY_BLUE};
                --boca-blue-dark: {BACKGROUND};
                --boca-gold: {ACCENT_YELLOW};
                --text-main: {TEXT};
                --text-muted: {MUTED_TEXT};
            }}

            .stApp {{
                background-color: var(--boca-blue-dark);
                color: var(--text-main);
                font-family: "Segoe UI", Roboto, Helvetica, sans-serif;
            }}

            header[data-testid="stHeader"] {{
                background-color: transparent !important;
            }}

            /* Barra lateral (Azul Boca y acentos Oro) */
            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #051021 0%, #0A1E40 60%, #10316B 100%);
                border-right: 2px solid var(--boca-gold);
            }}

            [data-testid="stSidebar"] * {{
                color: #FFFFFF !important;
            }}

            /* Selectbox e inputs oscuros legibles */
            div[data-baseweb="select"] > div, 
            div[data-baseweb="input"] > div,
            .stSelectbox div[data-baseweb="select"] > div {{
                background-color: #0A1E40 !important;
                color: #FFFFFF !important;
                border: 1px solid rgba(253, 185, 19, 0.4) !important;
                border-radius: 8px !important;
            }}

            div[data-baseweb="select"] span, 
            div[data-baseweb="select"] div,
            input {{
                color: #FFFFFF !important;
            }}

            /* Tarjetas y contenedores limpios */
            div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {{
                background-color: rgba(10, 30, 64, 0.7);
                border: 1px solid rgba(253, 185, 19, 0.2);
                border-radius: 12px;
                padding: 1rem;
            }}

            /* Botones principales con estilo Azul y Oro */
            div[data-testid="stButton"] > button {{
                background-color: var(--boca-gold);
                color: #051021 !important;
                font-weight: 700;
                border: none;
                border-radius: 8px;
                padding: 0.5rem 1rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                transition: all 0.2s ease-in-out;
            }}

            div[data-testid="stButton"] > button:hover {{
                background-color: #e5a60c;
                border-color: var(--boca-gold);
                color: #051021 !important;
            }}

            /* Títulos claros */
            h1, h2, h3 {{
                color: var(--text-main) !important;
                font-weight: 800;
            }}
            
            /* CORRECCIÓN VISIBILIDAD DE MÉTRICAS */
            [data-testid="stMetricLabel"] {{
                color: #FFFFFF !important;
                font-weight: 700 !important;
            }}

            [data-testid="stMetricValue"] {{
                color: var(--boca-gold) !important;
            }}

            [data-testid="stMetricDelta"] svg {{
                fill: var(--boca-gold) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )