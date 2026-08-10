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
                    radial-gradient(circle at 12% 12%, rgba(247, 181, 0, 0.08), transparent 18%),
                    radial-gradient(circle at 88% 14%, rgba(18, 58, 120, 0.18), transparent 24%),
                    radial-gradient(circle at 52% 100%, rgba(247, 181, 0, 0.06), transparent 28%),
                    linear-gradient(180deg, #081326 0%, #0A1730 34%, #0B1A35 68%, var(--background) 100%);
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
                background: rgba(7, 21, 46, 0.92);
                backdrop-filter: blur(8px);
            }}

            [data-testid="stHeader"] * {{
                color: #F8FBFF !important;
            }}

            [data-testid="stSidebar"] {{
                background: linear-gradient(180deg, #061227 0%, #0A1F44 45%, #123A78 100%);
                border-right: 1px solid rgba(247, 181, 0, 0.12);
                min-width: 248px !important;
                max-width: 248px !important;
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
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(247, 181, 0, 0.12);
                box-shadow: none;
            }}

            .block-container {{
                max-width: 100%;
                padding-top: 2.2rem;
                padding-bottom: 2.4rem;
                padding-left: 1.6rem;
                padding-right: 1.6rem;
            }}

            h1, h2, h3 {{
                color: var(--text);
                letter-spacing: -0.02em;
            }}

            .hero {{
                padding: 0.2rem 0 0.35rem 0;
                margin-bottom: 1rem;
            }}

            .hero-shell {{
                display: grid;
                grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.9fr);
                gap: 1rem;
                align-items: stretch;
                background: linear-gradient(145deg, rgba(12, 31, 66, 0.94), rgba(8, 22, 46, 0.98));
                border: 1px solid rgba(247, 181, 0, 0.18);
                border-radius: 24px;
                padding: 1.2rem 1.25rem;
                box-shadow: 0 24px 54px rgba(0, 0, 0, 0.28);
                overflow: hidden;
                position: relative;
            }}

            .hero-shell::after {{
                content: "";
                position: absolute;
                inset: auto -5% -45% auto;
                width: 320px;
                height: 320px;
                background: radial-gradient(circle, rgba(247, 181, 0, 0.2), transparent 70%);
                pointer-events: none;
            }}

            .hero-eyebrow {{
                color: var(--accent-yellow);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.78rem;
                font-weight: 900;
                margin-bottom: 0.35rem;
            }}

            .hero h1 {{
                margin: 0 0 0.3rem 0;
                font-size: 2.3rem;
                color: var(--text);
            }}

            .hero p {{
                color: var(--muted-text);
                max-width: 60ch;
                line-height: 1.5;
                margin: 0;
            }}

            .hero-status {{
                background: linear-gradient(180deg, rgba(18, 58, 120, 0.48), rgba(12, 31, 66, 0.82));
                border: 1px solid rgba(247, 181, 0, 0.16);
                border-radius: 20px;
                padding: 1rem 1rem 1.1rem;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 100%;
            }}

            .hero-status-label {{
                color: var(--accent-yellow);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.72rem;
                font-weight: 900;
                margin-bottom: 0.45rem;
            }}

            .hero-status-value {{
                color: var(--text);
                font-size: 1rem;
                line-height: 1.5;
                font-weight: 700;
            }}

            .badge-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.75rem;
            }}

            .badge {{
                background: rgba(18, 58, 120, 0.32);
                border: 1px solid rgba(247, 181, 0, 0.16);
                border-radius: 999px;
                color: var(--text);
                display: inline-flex;
                font-size: 0.78rem;
                font-weight: 800;
                padding: 0.28rem 0.68rem;
            }}

            .badge.accent {{
                background: var(--accent-yellow);
                border-color: #DFA300;
                color: #07152E;
            }}

            .section-intro {{
                color: var(--muted-text);
                margin-top: -0.1rem;
                margin-bottom: 0.9rem;
                max-width: 900px;
                line-height: 1.46;
            }}

            .info-card, .metric-card, .result-card, .compact-card, .journey-shell {{
                background: linear-gradient(180deg, rgba(13, 33, 69, 0.95), rgba(8, 21, 46, 0.98));
                border: 1px solid rgba(247, 181, 0, 0.14);
                border-radius: 16px;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.32);
            }}

            .spotlight-panel {{
                background:
                    radial-gradient(circle at top right, rgba(247, 181, 0, 0.12), transparent 26%),
                    linear-gradient(180deg, rgba(14, 37, 77, 0.97), rgba(8, 21, 46, 0.98));
                border: 1px solid rgba(247, 181, 0, 0.16);
                border-radius: 22px;
                padding: 1.1rem 1.15rem 1.15rem;
                box-shadow: 0 20px 44px rgba(0, 0, 0, 0.26);
                margin-bottom: 0.95rem;
            }}

            .spotlight-eyebrow {{
                color: var(--accent-yellow);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.74rem;
                font-weight: 900;
                margin-bottom: 0.3rem;
            }}

            .spotlight-panel h2 {{
                margin: 0 0 0.35rem 0;
                font-size: 1.55rem;
            }}

            .spotlight-panel p {{
                margin: 0;
                color: var(--muted-text);
                line-height: 1.48;
                max-width: 72ch;
            }}

            .spotlight-meta-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin-top: 0.95rem;
            }}

            .spotlight-meta-item {{
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(247, 181, 0, 0.12);
                border-radius: 14px;
                padding: 0.75rem 0.8rem;
            }}

            .spotlight-meta-label {{
                display: block;
                color: var(--muted-text);
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 800;
                margin-bottom: 0.2rem;
            }}

            .spotlight-meta-value {{
                color: var(--text);
                font-size: 1rem;
                font-weight: 800;
            }}

            .story-card {{
                background: linear-gradient(180deg, rgba(10, 28, 59, 0.92), rgba(7, 19, 40, 0.98));
                border: 1px solid rgba(247, 181, 0, 0.12);
                border-radius: 18px;
                padding: 0.95rem 1rem;
                min-height: 100%;
            }}

            .story-step {{
                color: var(--accent-yellow);
                font-size: 0.74rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.3rem;
            }}

            .story-title {{
                color: var(--text);
                font-size: 1.02rem;
                font-weight: 800;
                margin-bottom: 0.28rem;
            }}

            .story-body {{
                color: var(--muted-text);
                line-height: 1.45;
                font-size: 0.88rem;
            }}

            .info-card, .compact-card {{
                padding: 0.95rem 1rem;
            }}

            .metric-card {{
                border-top: 4px solid var(--accent-yellow);
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
                color: var(--text);
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
                color: var(--text);
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
                background: rgba(18, 58, 120, 0.34);
                color: var(--text);
                margin-right: 0.4rem;
            }}

            .status-pill.real {{
                background: rgba(77, 163, 255, 0.22);
                color: {SUCCESS};
            }}

            .status-pill.mock {{
                background: rgba(247, 181, 0, 0.22);
                color: {ACCENT_YELLOW};
            }}

            .sidebar-card {{
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(247, 181, 0, 0.1);
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
                background: rgba(10, 31, 68, 0.92);
                border: 1px solid rgba(247, 181, 0, 0.12);
                border-radius: 999px;
                padding: 0.42rem 0.74rem;
                box-shadow: 0 10px 24px rgba(0, 0, 0, 0.18);
            }}

            div[data-testid="stButton"] > button {{
                background: linear-gradient(180deg, #143C7A, #0A1F44);
                color: white;
                border: 1px solid rgba(247, 181, 0, 0.2);
                border-radius: 12px;
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.28);
            }}

            div[data-testid="stButton"] > button p,
            div[data-testid="stButton"] > button span {{
                color: white !important;
            }}

            div[data-testid="stButton"] > button:hover {{
                background: linear-gradient(180deg, #1B509F, #143C7A);
                color: white;
                border-color: var(--accent-yellow);
            }}

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            .stSelectbox div[data-baseweb="select"] > div {{
                background: rgba(10, 31, 68, 0.92);
                border-color: rgba(247, 181, 0, 0.12);
                color: var(--text);
            }}

            .stSlider [data-baseweb="slider"] * {{
                color: var(--text);
            }}

            .stDataFrame, .stTable, [data-testid="stMarkdownContainer"] code {{
                color: var(--text);
            }}

            [data-testid="stDataFrame"] {{
                border: 1px solid rgba(247, 181, 0, 0.12);
                border-radius: 18px;
                overflow: hidden;
            }}

            [data-testid="stDataFrame"] [role="grid"] {{
                background: rgba(8, 21, 46, 0.9);
            }}

            [data-testid="stExpander"] {{
                border: 1px solid rgba(247, 181, 0, 0.12);
                border-radius: 16px;
                overflow: hidden;
            }}

            [data-testid="stAlert"] {{
                background: rgba(10, 31, 68, 0.92);
                border: 1px solid rgba(247, 181, 0, 0.18);
                border-radius: 14px;
                color: var(--text);
            }}

            [data-testid="stAlert"] * {{
                color: var(--text) !important;
            }}

            [data-testid="stAlert"] svg {{
                fill: var(--accent-yellow) !important;
            }}

            [data-testid="stAlert"][kind="error"],
            [data-testid="stAlert"][kind="warning"],
            [data-testid="stAlert"][kind="success"],
            [data-testid="stAlert"][kind="info"] {{
                border-left: 4px solid var(--accent-yellow);
            }}

            [data-testid="stNotification"],
            [data-testid="toastContainer"] * {{
                color: var(--text) !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
