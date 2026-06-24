from __future__ import annotations

import streamlit as st

from app.dashboard.constants import (
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
        page_title="Quantum IDS Dashboard",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --primary-blue: {PRIMARY_BLUE};
                --secondary-blue: {SECONDARY_BLUE};
                --accent-yellow: {ACCENT_YELLOW};
                --background: {BACKGROUND};
                --text: {TEXT};
                --muted-text: {MUTED_TEXT};
            }}

            .stApp {{
                background:
                    radial-gradient(circle at top right, rgba(255, 204, 0, 0.14), transparent 20%),
                    linear-gradient(180deg, #EAF0FB 0%, var(--background) 220px);
                color: var(--text);
                font-family: "Trebuchet MS", "Segoe UI", sans-serif;
            }}

            input[type="radio"] {{
                accent-color: {ACCENT_YELLOW};
            }}

            [data-testid="stWidgetLabel"] p,
            .stRadio label,
            .stRadio label p,
            .stRadio label span,
            .stCheckbox label,
            .stCheckbox label p,
            .stCheckbox label span,
            [role="radiogroup"] label,
            [role="radiogroup"] label p,
            [role="radiogroup"] label span,
            [data-baseweb="checkbox"] label,
            [data-baseweb="checkbox"] label p,
            [data-baseweb="checkbox"] label span {{
                color: var(--text) !important;
            }}

            [data-testid="stHeader"] {{
                background: rgba(10, 42, 102, 0.94);
                backdrop-filter: blur(8px);
            }}

            [data-testid="stHeader"] * {{
                color: #F8FBFF !important;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #082458 0%, #0A2A66 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
                min-width: 280px !important;
                max-width: 280px !important;
            }}

            [data-testid="stSidebar"] * {{
                color: #F8FBFF;
            }}

            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
            [data-testid="stSidebar"] .stRadio label,
            [data-testid="stSidebar"] .stRadio label p,
            [data-testid="stSidebar"] .stRadio label span,
            [data-testid="stSidebar"] [role="radiogroup"] label,
            [data-testid="stSidebar"] [role="radiogroup"] label p,
            [data-testid="stSidebar"] [role="radiogroup"] label span {{
                color: #F8FBFF !important;
            }}

            [data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {{
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: none;
            }}

            .block-container {{
                max-width: 100%;
                padding-top: 2.8rem;
                padding-bottom: 2.4rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }}

            h1, h2, h3 {{
                color: var(--primary-blue);
                letter-spacing: -0.02em;
            }}

            .hero {{
                padding: 0.4rem 0 0.2rem 0;
                margin-bottom: 0.8rem;
            }}

            .hero h1 {{
                margin: 0 0 0.3rem 0;
                font-size: 2rem;
                color: var(--primary-blue);
            }}

            .badge-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.75rem;
            }}

            .badge {{
                background: rgba(21, 66, 132, 0.08);
                border: 1px solid rgba(10, 42, 102, 0.12);
                border-radius: 999px;
                color: var(--primary-blue);
                display: inline-flex;
                font-size: 0.78rem;
                font-weight: 800;
                padding: 0.28rem 0.68rem;
            }}

            .badge.accent {{
                background: var(--accent-yellow);
                border-color: #EAB800;
                color: #0A2A66;
            }}

            .section-intro {{
                color: var(--muted-text);
                margin-top: -0.1rem;
                margin-bottom: 0.9rem;
                max-width: 900px;
                line-height: 1.46;
            }}

            .info-card, .metric-card, .result-card, .compact-card, .journey-shell {{
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(10, 42, 102, 0.08);
                border-radius: 16px;
                box-shadow: 0 16px 36px rgba(10, 42, 102, 0.08);
            }}

            .info-card, .compact-card {{
                padding: 0.95rem 1rem;
            }}

            .metric-card {{
                border-top: 5px solid var(--accent-yellow);
                padding: 0.95rem 1rem;
                min-height: 122px;
            }}

            .card-label {{
                color: var(--muted-text);
                font-size: 0.74rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 0.28rem;
            }}

            .card-value {{
                color: var(--primary-blue);
                font-size: 1.18rem;
                font-weight: 850;
                margin-bottom: 0.18rem;
            }}

            .card-help {{
                color: var(--muted-text);
                font-size: 0.86rem;
                line-height: 1.42;
            }}

            .metric-value {{
                color: var(--primary-blue);
                font-size: 1.8rem;
                font-weight: 850;
                margin-top: 0.05rem;
            }}

            .metric-caption {{
                color: var(--muted-text);
                font-size: 0.8rem;
                line-height: 1.35;
            }}

            .result-card {{
                padding: 1rem 1.05rem;
            }}

            .result-card.normal {{
                border-left: 6px solid {SUCCESS};
            }}

            .result-card.attack {{
                border-left: 6px solid {DANGER};
            }}

            .result-title {{
                font-size: 1.25rem;
                font-weight: 850;
                margin-bottom: 0.15rem;
            }}

            .normal .result-title {{
                color: {SUCCESS};
            }}

            .attack .result-title {{
                color: {DANGER};
            }}

            .status-pill {{
                display: inline-flex;
                border-radius: 999px;
                padding: 0.22rem 0.6rem;
                font-size: 0.8rem;
                font-weight: 800;
                background: #E7EEFF;
                color: var(--primary-blue);
                margin-right: 0.4rem;
            }}

            .status-pill.real {{
                background: #E8FFF7;
                color: {SUCCESS};
            }}

            .status-pill.mock {{
                background: #FFF6D5;
                color: #8A6700;
            }}

            .sidebar-card {{
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 0.9rem 0.95rem;
                margin: 0.2rem 0 0.9rem 0;
            }}

            .sidebar-title {{
                color: #FFCC00;
                font-size: 0.78rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.28rem;
            }}

            .sidebar-copy {{
                color: rgba(248, 251, 255, 0.92);
                font-size: 0.88rem;
                line-height: 1.45;
            }}

            .stRadio [role="radiogroup"] {{
                gap: 0.45rem;
            }}

            .stRadio [role="radiogroup"] > label {{
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(10, 42, 102, 0.08);
                border-radius: 999px;
                padding: 0.42rem 0.74rem;
                box-shadow: 0 6px 16px rgba(10, 42, 102, 0.05);
            }}

            div[data-testid="stButton"] > button {{
                background: linear-gradient(180deg, #154284, #0A2A66);
                color: white;
                border: 1px solid rgba(10, 42, 102, 0.14);
                border-radius: 12px;
                box-shadow: 0 14px 28px rgba(10, 42, 102, 0.18);
            }}

            div[data-testid="stButton"] > button p,
            div[data-testid="stButton"] > button span {{
                color: white !important;
            }}

            div[data-testid="stButton"] > button:hover {{
                background: linear-gradient(180deg, #1A4E9D, #123775);
                color: white;
                border-color: var(--accent-yellow);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
