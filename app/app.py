import sys
from math import ceil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import random
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split

from src.classical.train_model import (
    convert_to_binary_label,
    find_label_column,
)
import json
import random
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.classical.train_model import convert_to_binary_label, find_label_column


PRIMARY_BLUE = "#0A2A66"
SECONDARY_BLUE = "#154284"
ACCENT_YELLOW = "#FFCC00"
BACKGROUND = "#F2F6FF"
TEXT = "#13233F"
MUTED_TEXT = "#5C6E91"
SUCCESS = "#0F766E"
DANGER = "#B91C1C"

RESULTS_DIR = Path("results")
DATASET_PATH = Path("data/dataset.csv")
CLASSICAL_RESULTS_PATH = RESULTS_DIR / "classical_metrics.json"
QUANTUM_SIMULATED_RESULTS_PATH = RESULTS_DIR / "quantum_simulated_metrics.json"
QUANTUM_LIVE_RESULTS_PATH = RESULTS_DIR / "quantum_live_simulated_metrics.json"
QUANTUM_HARDWARE_RESULTS_PATH = RESULTS_DIR / "quantum_hardware_metrics.json"
QUANTUM_LIVE_HARDWARE_RESULTS_PATH = RESULTS_DIR / "quantum_live_hardware_metrics.json"
LIVE_TRAINING_DATASET_PATH = RESULTS_DIR / "live_training_dataset.csv"
CLASSICAL_MODEL_PATH = RESULTS_DIR / "random_forest_model.joblib"
SCALER_PATH = RESULTS_DIR / "scaler.joblib"
PCA_PATH = RESULTS_DIR / "pca.joblib"
SUPPORTED_QUANTUM_QUBITS = (2, 4, 6, 8)
SUPPORTED_QUANTUM_DATASET_SOURCES = ("cicids", "live")

MODEL_DATA = {
    "Modelo clasico": {
        "label": "Modelo clasico",
        "short_label": "Clasico",
        "description": "Baseline supervisado sobre caracteristicas numericas del trafico.",
        "accuracy": 0.942,
        "precision": 0.931,
        "recall": 0.956,
        "f1_score": 0.943,
        "execution_time": 38.4,
        "confusion_matrix": np.array([[912, 44], [31, 883]]),
        "color": PRIMARY_BLUE,
    },
    "Modelo cuantico": {
        "label": "Modelo cuantico",
        "short_label": "QML",
        "description": "Circuito variacional ejecutado en entorno cuantico controlado.",
        "accuracy": 0.918,
        "precision": 0.904,
        "recall": 0.929,
        "f1_score": 0.916,
        "execution_time": 72.8,
        "confusion_matrix": np.array([[884, 72], [65, 849]]),
        "color": SECONDARY_BLUE,
    },
    "Hardware cuantico real": {
        "label": "Hardware cuantico real",
        "short_label": "Hardware real",
        "description": "Ejecucion NISQ afectada por ruido, decoherencia y errores de lectura.",
        "accuracy": 0.861,
        "precision": 0.844,
        "recall": 0.872,
        "f1_score": 0.858,
        "execution_time": 214.5,
        "confusion_matrix": np.array([[819, 137], [117, 797]]),
        "color": ACCENT_YELLOW,
    },
}

ENABLED_MODEL_OPTIONS = ("Modelo clasico", "Modelo cuantico")

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

            .journey-shell {{
                padding: 0.85rem 0.95rem 0.68rem 0.95rem;
                margin: 0.85rem 0 1rem 0;
            }}

            .journey-title {{
                color: var(--primary-blue);
                font-size: 0.82rem;
                font-weight: 900;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.38rem;
            }}

            .journey-copy {{
                color: var(--muted-text);
                font-size: 0.9rem;
                line-height: 1.45;
                margin-bottom: 0.1rem;
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


def load_classical_results() -> dict | None:
    if not CLASSICAL_RESULTS_PATH.exists():
        return None
    with open(CLASSICAL_RESULTS_PATH, "r", encoding="utf-8") as results_file:
        payload = json.load(results_file)

    metrics = payload.get("metrics")
    confusion = payload.get("confusion_matrix")
    if not metrics or confusion is None:
        return None

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "confusion_matrix": np.array(confusion),
        "model_name": payload.get("model_name", "Random Forest"),
        "pca_components": payload.get("pca_components"),
    }


def get_quantum_results_path(qubits: int, dataset_source: str = "cicids") -> Path:
    if dataset_source == "live":
        return RESULTS_DIR / f"quantum_live_simulated_metrics_{qubits}q.json"
    return RESULTS_DIR / f"quantum_simulated_metrics_{qubits}q.json"


def get_quantum_hardware_results_path(qubits: int, dataset_source: str = "cicids") -> Path:
    if dataset_source == "live":
        return RESULTS_DIR / f"quantum_live_hardware_metrics_{qubits}q.json"
    return RESULTS_DIR / f"quantum_hardware_metrics_{qubits}q.json"


def load_quantum_simulated_results(qubits: int | None = None, dataset_source: str = "cicids") -> dict | None:
    if qubits is None:
        results_path = QUANTUM_LIVE_RESULTS_PATH if dataset_source == "live" else QUANTUM_SIMULATED_RESULTS_PATH
    else:
        results_path = get_quantum_results_path(qubits, dataset_source=dataset_source)
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as results_file:
        payload = json.load(results_file)

    metrics = payload.get("metrics")
    confusion = payload.get("confusion_matrix")
    if not metrics or confusion is None:
        return None

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "confusion_matrix": np.array(confusion),
        "model_name": payload.get("model_name", "Variational Quantum Classifier"),
        "pca_components": payload.get("pca_components"),
        "num_qubits": payload.get("num_qubits"),
        "sample_size": payload.get("sample_size"),
        "execution_time_seconds": payload.get("execution_time_seconds"),
        "dataset_source": payload.get("dataset_source", dataset_source),
        "dataset_path": payload.get("dataset_path"),
        "results_path": str(results_path),
    }


def load_quantum_hardware_results(qubits: int | None = None, dataset_source: str = "cicids") -> dict | None:
    if qubits is None:
        results_path = QUANTUM_LIVE_HARDWARE_RESULTS_PATH if dataset_source == "live" else QUANTUM_HARDWARE_RESULTS_PATH
    else:
        results_path = get_quantum_hardware_results_path(qubits, dataset_source=dataset_source)
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as results_file:
        payload = json.load(results_file)

    metrics = payload.get("metrics")
    confusion = payload.get("confusion_matrix")
    if not metrics or confusion is None:
        return None

    return {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "confusion_matrix": np.array(confusion),
        "model_name": payload.get("model_name", "Variational Quantum Classifier"),
        "pca_components": payload.get("pca_components"),
        "num_qubits": payload.get("num_qubits"),
        "sample_size": payload.get("sample_size"),
        "execution_time_seconds": payload.get("execution_time_seconds"),
        "dataset_source": payload.get("dataset_source", dataset_source),
        "dataset_path": payload.get("dataset_path"),
        "results_path": str(results_path),
        "ibm_backend_name": payload.get("ibm_backend_name"),
        "hardware_diagnostics": payload.get("hardware_diagnostics", {}),
        "hardware_gap_vs_simulator": payload.get("hardware_gap_vs_simulator", {}),
        "hardware_gap_vs_local_subset": payload.get("hardware_gap_vs_local_subset", {}),
        "validation_strategy": payload.get("validation_strategy"),
        "local_reference_metrics_subset": payload.get("local_reference_metrics_subset", {}),
        "local_reference_metrics_full": payload.get("local_reference_metrics_full", {}),
    }


def get_model_data(selected_quantum_qubits: int = 4, selected_quantum_dataset_source: str = "cicids") -> dict:
    model_data = {
        name: {
            **values,
            "source": "mock",
            "source_label": "Mock",
        }
        for name, values in MODEL_DATA.items()
    }

    classical_results = load_classical_results()
    if classical_results is not None:
        model_data["Modelo clasico"].update(
            {
                "accuracy": classical_results["accuracy"],
                "precision": classical_results["precision"],
                "recall": classical_results["recall"],
                "f1_score": classical_results["f1_score"],
                "confusion_matrix": classical_results["confusion_matrix"],
                "source": "real",
                "source_label": "Resultado real",
                "description": "Random Forest entrenado y evaluado con resultados cargados desde results/classical_metrics.json.",
                "trained_model_name": classical_results["model_name"],
                "pca_components": classical_results["pca_components"],
            }
        )

    quantum_simulated_results = load_quantum_simulated_results(
        selected_quantum_qubits,
        dataset_source=selected_quantum_dataset_source,
    )
    if quantum_simulated_results is not None:
        source_label = "Live" if selected_quantum_dataset_source == "live" else "CICIDS2017"
        source_description = (
            f"VQC entrenado con features agregadas capturadas del simulador. Archivo: {quantum_simulated_results['results_path']}."
            if selected_quantum_dataset_source == "live"
            else f"VQC cargado desde results/quantum_simulated_metrics_{selected_quantum_qubits}q.json. Permite comparar QML frente al baseline clasico."
        )
        model_data["Modelo cuantico"].update(
            {
                "accuracy": quantum_simulated_results["accuracy"],
                "precision": quantum_simulated_results["precision"],
                "recall": quantum_simulated_results["recall"],
                "f1_score": quantum_simulated_results["f1_score"],
                "confusion_matrix": quantum_simulated_results["confusion_matrix"],
                "source": "real",
                "source_label": "Resultado real",
                "description": source_description,
                "trained_model_name": quantum_simulated_results["model_name"],
                "pca_components": quantum_simulated_results["pca_components"],
                "num_qubits": quantum_simulated_results["num_qubits"],
                "sample_size": quantum_simulated_results["sample_size"],
                "selected_qubits": selected_quantum_qubits,
                "selected_dataset_source": selected_quantum_dataset_source,
                "dataset_source_label": source_label,
                "dataset_path": quantum_simulated_results["dataset_path"],
                "results_path": quantum_simulated_results["results_path"],
                "execution_time": quantum_simulated_results["execution_time_seconds"]
                if quantum_simulated_results["execution_time_seconds"] is not None
                else model_data["Modelo cuantico"]["execution_time"],
            }
        )
    else:
        command = (
            f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {selected_quantum_qubits}"
            if selected_quantum_dataset_source == "live"
            else f"python -m src.quantum.train_vqc_simulator --qubits {selected_quantum_qubits}"
        )
        description = (
            f"Todavia no se entreno el VQC live con {selected_quantum_qubits} qubits. Genera {LIVE_TRAINING_DATASET_PATH.as_posix()} y ejecuta: {command}"
            if selected_quantum_dataset_source == "live"
            else f"Todavia no se entreno el VQC con {selected_quantum_qubits} qubits. Ejecutar: {command}"
        )
        model_data["Modelo cuantico"].update(
            {
                "source": "missing",
                "source_label": "Pendiente",
                "selected_qubits": selected_quantum_qubits,
                "selected_dataset_source": selected_quantum_dataset_source,
                "dataset_source_label": "Live" if selected_quantum_dataset_source == "live" else "CICIDS2017",
                "description": description,
            }
        )

    hardware_results = load_quantum_hardware_results(
        selected_quantum_qubits,
        dataset_source=selected_quantum_dataset_source,
    )
    if hardware_results is not None:
        model_data["Hardware cuantico real"].update(
            {
                "accuracy": hardware_results["accuracy"],
                "precision": hardware_results["precision"],
                "recall": hardware_results["recall"],
                "f1_score": hardware_results["f1_score"],
                "confusion_matrix": hardware_results["confusion_matrix"],
                "source": "real",
                "source_label": "Resultado real",
                "description": (
                    f"IBM Quantum backend {hardware_results.get('ibm_backend_name') or 'desconocido'} "
                    f"cargado desde {hardware_results['results_path']}."
                ),
                "trained_model_name": hardware_results["model_name"],
                "pca_components": hardware_results["pca_components"],
                "num_qubits": hardware_results["num_qubits"],
                "sample_size": hardware_results["sample_size"],
                "selected_qubits": selected_quantum_qubits,
                "selected_dataset_source": selected_quantum_dataset_source,
                "dataset_source_label": "Live" if selected_quantum_dataset_source == "live" else "CICIDS2017",
                "dataset_path": hardware_results["dataset_path"],
                "results_path": hardware_results["results_path"],
                "ibm_backend_name": hardware_results.get("ibm_backend_name"),
                "hardware_diagnostics": hardware_results.get("hardware_diagnostics", {}),
                "hardware_gap_vs_simulator": hardware_results.get("hardware_gap_vs_simulator", {}),
                "execution_time": hardware_results["execution_time_seconds"]
                if hardware_results["execution_time_seconds"] is not None
                else model_data["Hardware cuantico real"]["execution_time"],
            }
        )
    return model_data


@st.cache_resource(show_spinner=False)
def load_classical_artifacts():
    if not (CLASSICAL_MODEL_PATH.exists() and SCALER_PATH.exists() and PCA_PATH.exists()):
        return None
    return {
        "model": joblib.load(CLASSICAL_MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
        "pca": joblib.load(PCA_PATH),
    }


def section_header(title: str, description: str) -> None:
    st.markdown(f"### {title}")
    st.markdown(f"<p class='section-intro'>{description}</p>", unsafe_allow_html=True)


def render_info_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: float, caption: str) -> None:
    formatted_value = f"{value * 100:.2f}%"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="card-label">{label}</div>
            <div class="metric-value">{formatted_value}</div>
            <div class="metric-caption">{caption} · valor crudo: {value:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(model_data: dict) -> None:
    classical_badge = "Clasico con datos reales" if model_data["Modelo clasico"]["source"] == "real" else "Clasico en modo demo"
    st.markdown(
        f"""
        <section class="hero">
            <h1>Quantum IDS Dashboard</h1>
            <div class="badge-row">
                <span class="badge accent">Tesis</span>
                <span class="badge">IDS</span>
                <span class="badge">QML</span>
                <span class="badge">NISQ</span>
                <span class="badge">{classical_badge}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="compact-card">
            <div class="card-label">Como leer este panel</div>
            <div class="card-help">
                <strong>Accuracy</strong> es el porcentaje total de aciertos.
                <strong> Precision</strong> dice que tan confiables son las alertas.
                <strong> Recall</strong> muestra cuantos ataques reales detecta el sistema.
                <strong> F1-Score</strong> resume el equilibrio entre precision y recall.
                <strong> Live simulador</strong> usa trafico capturado en laboratorio.
                <strong> IBM validate</strong> entrena local y valida una porcion chica en hardware real.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls(
    model_data: dict,
    selected_quantum_qubits: int,
    selected_quantum_dataset_source: str,
) -> tuple[str, int, str, str]:
    with st.sidebar:
        st.markdown("## Quantum IDS")
        st.caption(
            "Esta app compara dos caminos para detectar trafico anomalo en red: uno clasico y otro cuantico. "
            "La idea es mostrar resultados, limites y valor experimental de cada enfoque en un lenguaje claro."
        )

        st.markdown("---")
        st.markdown("### Configuracion")
        available_models = [model_name for model_name in ENABLED_MODEL_OPTIONS if model_name in model_data]
        default_model = st.session_state.get("selected_model", "Modelo clasico")
        if default_model not in available_models:
            default_model = "Modelo clasico"
        selected_model = st.radio(
            "Modelo",
            options=available_models,
            index=available_models.index(default_model),
            key="model_switcher_radio",
        )
        st.session_state["selected_model"] = selected_model

        if selected_model == "Modelo cuantico":
            dataset_source_options = {"cicids": "CICIDS2017", "live": "Live simulador"}
            quantum_dataset_source = st.radio(
                "Origen de datos cuanticos",
                options=list(dataset_source_options.keys()),
                format_func=lambda key: dataset_source_options[key],
                index=list(SUPPORTED_QUANTUM_DATASET_SOURCES).index(selected_quantum_dataset_source),
                key="quantum_dataset_source_radio",
            )
            if quantum_dataset_source != selected_quantum_dataset_source:
                st.session_state["selected_quantum_dataset_source"] = quantum_dataset_source
                st.session_state.pop("quantum_lab_results", None)
                st.session_state.pop("quantum_lab_results_qubits", None)
                st.session_state.pop("quantum_lab_results_source", None)
                st.rerun()
            selected_quantum_dataset_source = quantum_dataset_source
            st.session_state["selected_quantum_dataset_source"] = quantum_dataset_source

            chosen_qubits = st.selectbox(
                "Cantidad de qubits",
                options=list(SUPPORTED_QUANTUM_QUBITS),
                index=list(SUPPORTED_QUANTUM_QUBITS).index(selected_quantum_qubits),
                key="quantum_results_selectbox",
            )
            if chosen_qubits != selected_quantum_qubits:
                st.session_state["selected_quantum_qubits"] = chosen_qubits
                st.session_state.pop("quantum_lab_results", None)
                st.session_state.pop("quantum_lab_results_qubits", None)
                st.session_state.pop("quantum_lab_results_source", None)
                st.rerun()
            selected_quantum_qubits = chosen_qubits
            st.session_state["selected_quantum_qubits"] = chosen_qubits

        st.markdown("---")
        st.markdown("### Seccion")
        current_step = st.radio(
            "Seccion",
            options=[
                "1. Vision general",
                "2. Probar modelo",
                "3. Analisis",
                "4. Simulacion",
                "5. Conclusiones",
            ],
            key="journey_radio",
            label_visibility="collapsed",
        )

        model = model_data[selected_model]
        source_class = "real" if model["source"] == "real" else "mock"
        st.markdown("---")
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-title">{model["short_label"]}</div>
                <div class="sidebar-copy">
                    <span class="status-pill {source_class}">{model["source_label"]}</span>
                    {model["description"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-title">Glosario rapido</div>
                <div class="sidebar-copy">
                    Accuracy: aciertos totales.<br>
                    Precision: confianza de una alerta.<br>
                    Recall: ataques reales detectados.<br>
                    Live: trafico del laboratorio.<br>
                    IBM validate: local + validacion corta en IBM.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_model, selected_quantum_qubits, selected_quantum_dataset_source, current_step


def build_metrics_dataframe(model_data: dict) -> pd.DataFrame:
    rows = []
    for model in model_data.values():
        rows.extend(
            [
                {"Modelo": model["short_label"], "Metrica": "Accuracy", "Valor": model["accuracy"]},
                {"Modelo": model["short_label"], "Metrica": "Precision", "Valor": model["precision"]},
                {"Modelo": model["short_label"], "Metrica": "Recall", "Valor": model["recall"]},
                {"Modelo": model["short_label"], "Metrica": "F1-Score", "Valor": model["f1_score"]},
            ]
        )
    return pd.DataFrame(rows)


def build_time_dataframe(model_data: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Modelo": [model["short_label"] for model in model_data.values()],
            "Tiempo de ejecucion (s)": [model["execution_time"] for model in model_data.values()],
        }
    )


def build_quantum_runs_dataframe(dataset_source: str = "cicids") -> pd.DataFrame:
    rows = []
    for qubits in SUPPORTED_QUANTUM_QUBITS:
        quantum_results = load_quantum_simulated_results(qubits, dataset_source=dataset_source)
        if quantum_results is None:
            rows.append(
                {
                    "Qubits": qubits,
                    "Fuente": "Live" if dataset_source == "live" else "CICIDS2017",
                    "Estado": "Pendiente",
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1-Score": np.nan,
                    "Tiempo (s)": np.nan,
                    "Sample": np.nan,
                }
            )
            continue

        rows.append(
            {
                "Qubits": qubits,
                "Fuente": "Live" if dataset_source == "live" else "CICIDS2017",
                "Estado": "Entrenado",
                "Accuracy": quantum_results["accuracy"],
                "Precision": quantum_results["precision"],
                "Recall": quantum_results["recall"],
                "F1-Score": quantum_results["f1_score"],
                "Tiempo (s)": quantum_results["execution_time_seconds"],
                "Sample": quantum_results["sample_size"],
            }
        )

    return pd.DataFrame(rows)


def make_global_comparison_chart(model_data: dict, height: int = 320) -> go.Figure:
    metrics_df = build_metrics_dataframe(model_data)
    fig = px.bar(
        metrics_df,
        x="Metrica",
        y="Valor",
        color="Modelo",
        barmode="group",
        text=metrics_df["Valor"].map(lambda value: f"{value:.1%}"),
        color_discrete_map={
            "Clasico": PRIMARY_BLUE,
            "QML": SECONDARY_BLUE,
            "Hardware real": ACCENT_YELLOW,
        },
    )
    fig.update_layout(
        height=height,
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="",
        margin=dict(l=12, r=12, t=14, b=10),
        font=dict(color=TEXT),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    return fig


def make_confusion_chart(matrix: np.ndarray, height: int = 320) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicho benigno", "Predicho intrusion"],
            y=["Real benigno", "Real intrusion"],
            colorscale=[[0, "#EEF4FB"], [0.55, SECONDARY_BLUE], [1, PRIMARY_BLUE]],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=12, r=12, t=14, b=10),
        font=dict(color=TEXT),
    )
    return fig


def make_time_chart(model_data: dict, height: int = 320) -> go.Figure:
    time_df = build_time_dataframe(model_data)
    fig = px.bar(
        time_df,
        x="Modelo",
        y="Tiempo de ejecucion (s)",
        text="Tiempo de ejecucion (s)",
        color="Modelo",
        color_discrete_map={
            "Clasico": PRIMARY_BLUE,
            "QML": SECONDARY_BLUE,
            "Hardware real": ACCENT_YELLOW,
        },
    )
    fig.update_traces(texttemplate="%{text:.1f}s", textposition="outside")
    fig.update_layout(
        height=height,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=12, r=12, t=14, b=10),
        font=dict(color=TEXT),
    )
    return fig


def make_noise_chart(model_data: dict, height: int = 320) -> go.Figure:
    simulated = model_data["Modelo cuantico"]
    hardware = model_data["Hardware cuantico real"]
    noise_df = pd.DataFrame(
        {
            "Entorno": ["Simulador ideal", "Hardware real"],
            "Accuracy": [simulated["accuracy"], hardware["accuracy"]],
            "F1-Score": [simulated["f1_score"], hardware["f1_score"]],
        }
    )
    fig = px.line(
        noise_df,
        x="Entorno",
        y=["Accuracy", "F1-Score"],
        markers=True,
        color_discrete_sequence=[PRIMARY_BLUE, ACCENT_YELLOW],
    )
    fig.update_layout(
        height=height,
        yaxis_tickformat=".0%",
        yaxis_range=[0.78, 0.95],
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="",
        margin=dict(l=12, r=12, t=14, b=10),
        font=dict(color=TEXT),
    )
    fig.update_traces(line=dict(width=4), marker=dict(size=9))
    return fig


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df.columns = [str(col).strip() for col in clean_df.columns]
    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.dropna()
    return clean_df


def split_features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    label_col = None
    try:
        label_col = find_label_column(df)
    except ValueError:
        label_col = None

    if label_col is None:
        return df.select_dtypes(include=[np.number]), None

    y = df[label_col].apply(convert_to_binary_label)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    return X, y


def evaluate_classical_dataset(df: pd.DataFrame, use_holdout_split: bool) -> dict:
    artifacts = load_classical_artifacts()
    if artifacts is None:
        raise FileNotFoundError("No se encontraron modelo, scaler o PCA en results/.")

    clean_df = sanitize_dataframe(df)
    X, y = split_features_and_labels(clean_df)
    if X.empty:
        raise ValueError("No quedaron features numericas validas luego de la limpieza.")

    if use_holdout_split:
        if y is None:
            raise ValueError("La evaluacion holdout requiere un dataset con columna objetivo.")
        X_train, X_eval, _, y_eval = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        _ = X_train  # Se mantiene por claridad respecto al pipeline de entrenamiento.
    else:
        X_eval = X
        y_eval = y

    X_eval_scaled = artifacts["scaler"].transform(X_eval)
    X_eval_pca = artifacts["pca"].transform(X_eval_scaled)
    predictions = artifacts["model"].predict(X_eval_pca)

    result = {
        "rows": int(len(X_eval)),
        "prediction_counts": {
            "normal": int((predictions == 0).sum()),
            "intrusion": int((predictions == 1).sum()),
        },
    }

    if y_eval is not None:
        result["metrics"] = {
            "accuracy": accuracy_score(y_eval, predictions),
            "precision": precision_score(y_eval, predictions),
            "recall": recall_score(y_eval, predictions),
            "f1_score": f1_score(y_eval, predictions),
        }
        result["confusion_matrix"] = confusion_matrix(y_eval, predictions)

    return result


def inspect_live_quantum_dataset(
    dataset_path: Path = LIVE_TRAINING_DATASET_PATH,
    test_size: float = 0.2,
) -> dict:
    minimum_test_samples = 2
    minimum_total_samples = ceil(minimum_test_samples / test_size)
    summary = {
        "exists": dataset_path.exists(),
        "path": str(dataset_path),
        "total_rows": 0,
        "benign_count": 0,
        "attack_count": 0,
        "feature_count": 0,
        "train_rows": 0,
        "max_supported_qubits": 0,
        "minimum_total_samples": minimum_total_samples,
        "ready": False,
        "message": "",
    }

    if not dataset_path.exists():
        summary["message"] = "Todavia no existe el CSV live. Genera capturas benign y attack antes de entrenar."
        return summary

    df = pd.read_csv(dataset_path)
    df.columns = [str(col).strip() for col in df.columns]

    if df.empty:
        summary["message"] = "El CSV live existe pero esta vacio. Genera capturas benign y attack antes de entrenar."
        return summary

    label_col = find_label_column(df)
    label_series = df[label_col].apply(convert_to_binary_label)
    class_counts = label_series.value_counts().to_dict()
    feature_count = int(df.drop(columns=[label_col]).select_dtypes(include=[np.number]).shape[1])

    benign_count = int(class_counts.get(0, 0))
    attack_count = int(class_counts.get(1, 0))
    total_rows = int(len(df))
    test_rows = ceil(test_size * total_rows)
    train_rows = total_rows - test_rows
    max_supported_qubits = min(train_rows, feature_count)

    summary["total_rows"] = total_rows
    summary["benign_count"] = benign_count
    summary["attack_count"] = attack_count
    summary["feature_count"] = feature_count
    summary["train_rows"] = train_rows
    summary["max_supported_qubits"] = max_supported_qubits
    summary["ready"] = benign_count >= 2 and attack_count >= 2 and total_rows >= minimum_total_samples and max_supported_qubits >= 2

    if summary["ready"]:
        summary["message"] = (
            f"Dataset live disponible: {summary['total_rows']} filas "
            f"({benign_count} benign, {attack_count} attack). "
            f"Con este split el maximo soportado es {max_supported_qubits} qubits."
        )
    else:
        summary["message"] = (
            f"Dataset live insuficiente: {summary['total_rows']} filas "
            f"({benign_count} benign, {attack_count} attack). "
            f"Necesitas al menos 2 capturas por clase, {minimum_total_samples} filas totales y soporte para al menos 2 qubits. "
            f"Con este split el maximo actual es {max_supported_qubits} qubits; en la practica conviene 10 o mas por clase."
        )

    return summary


def classify_mock_connection(packet_rate: int, failed_logins: int, protocol_risk: int, selected_model: str) -> tuple[str, float]:
    model_bias = {
        "Modelo clasico": 0.02,
        "Modelo cuantico": 0.05,
        "Hardware cuantico real": 0.09,
    }
    risk_score = (packet_rate / 1000) * 0.42 + (failed_logins / 20) * 0.38 + (protocol_risk / 10) * 0.20
    risk_score += model_bias[selected_model] + random.uniform(-0.04, 0.04)
    risk_score = min(max(risk_score, 0), 1)
    label = "Intrusion detectada" if risk_score >= 0.56 else "Trafico normal"
    return label, risk_score


def render_overview_tab(model_data: dict, selected_model: str) -> None:
    section_header(
        "Vision general",
        "Resumen rapido para entender que modelo estas viendo, que tan bien funciona y de donde salen los datos.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card("Base principal", "data/dataset.csv", "Dataset base del experimento clasico y del escenario cuantico de referencia.")
    with col2:
        render_info_card("Estado clasico", model_data["Modelo clasico"]["source_label"], model_data["Modelo clasico"]["description"])
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
            f"Todavia no hay una corrida cuantica disponible para {model_data['Modelo cuantico'].get('dataset_source_label', 'CICIDS2017')} con {model_data['Modelo cuantico']['selected_qubits']} qubits. "
            f"Ejecutar: {quantum_command}"
        )

    st.write("")
    st.plotly_chart(
        make_global_comparison_chart(model_data),
        width="stretch",
        key="overview_global_comparison_chart",
    )

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
    with info_col:
        render_info_card("Origen de metricas", model["source_label"], "Te dice si los numeros vienen de una corrida real o de una demo.")
        st.write("")
        render_info_card("Tiempo estimado", f"{model['execution_time']:.2f}s", "Tiempo total aproximado del enfoque seleccionado.")


def render_lab_tab(
    model_data: dict,
    selected_model: str,
    selected_quantum_qubits: int,
    selected_quantum_dataset_source: str,
) -> None:
    section_header(
        "Laboratorio de prueba",
        "Espacio de experimentacion guiada para ejecutar pruebas sin salir del dashboard.",
    )
    if selected_model == "Modelo cuantico":
        selected_quantum_execution_target = st.session_state.get("selected_quantum_execution_target", "simulator")
        selected_quantum_test_size = float(st.session_state.get("selected_quantum_test_size", 0.2))
        live_dataset_summary = (
            inspect_live_quantum_dataset(test_size=selected_quantum_test_size)
            if selected_quantum_dataset_source == "live"
            else None
        )
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("#### VQC")
            st.caption(
                "Aca se ejecuta el experimento cuantico. El sistema entrena y evalua un clasificador variacional sobre una muestra controlada del dataset elegido."
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
            selected_ibm_validation_samples = int(st.session_state.get("selected_ibm_validation_samples", 16))
            if selected_quantum_execution_target == "ibm_validate":
                selected_ibm_validation_samples = st.select_slider(
                    "Muestras del test a validar en IBM",
                    options=[4, 8, 12, 16, 24, 32],
                    value=selected_ibm_validation_samples,
                    key="ibm_validation_samples_slider",
                )
                st.session_state["selected_ibm_validation_samples"] = selected_ibm_validation_samples
            if selected_quantum_dataset_source == "live":
                live_dataset_summary = inspect_live_quantum_dataset(test_size=selected_quantum_test_size)
            quantum_button = st.button(
                f"Ejecutar prueba cuantica ({selected_quantum_qubits}q)",
                width="stretch",
                type="primary",
                disabled=(
                    selected_quantum_dataset_source == "live"
                    and live_dataset_summary is not None
                    and (
                        not live_dataset_summary["ready"]
                        or selected_quantum_qubits > live_dataset_summary["max_supported_qubits"]
                    )
                ),
            )
            if selected_quantum_dataset_source == "live":
                st.caption(
                    f"Modo live exclusivo de la metodologia cuantica: usa {LIVE_TRAINING_DATASET_PATH.as_posix()} con capturas benign y attack construidas en laboratorio."
                )
                st.caption(
                    f"Con test {int(selected_quantum_test_size * 100)}% necesitas al menos 2 capturas benign, 2 attack y {live_dataset_summary['minimum_total_samples']} filas totales. Para una conclusion seria conviene usar muchas mas."
                )
                st.markdown(
                    """
                    <div class="compact-card">
                        <div class="card-label">Guia de laboratorio live</div>
                        <div class="card-help">
                            1. Abri otra terminal en la raiz del proyecto.<br>
                            2. Activa el entorno: <code>source venv/bin/activate</code>.<br>
                            3. Para capturas benign, deja el simulador apagado y corré:
                            <code>sudo "$(which python3)" -m src.live_detection.capture --duration 2 --windows 20 --output results/live_training_dataset.csv --label benign --append</code><br>
                            4. Para capturas attack, ejecuta manualmente <code>01_attack-scrapy.py</code> en otra terminal y, mientras corre, capturá:
                            <code>sudo "$(which python3)" -m src.live_detection.capture --duration 2 --windows 20 --output results/live_training_dataset.csv --label attack --append</code><br>
                            5. Volve al dashboard, verifica el estado del CSV live y recien ahi ejecuta la prueba cuantica.<br>
                            6. Si usas IBM validate, IBM solo toma una parte chica del test para ahorrar cuota.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if live_dataset_summary is not None:
                    if live_dataset_summary["ready"]:
                        st.success(live_dataset_summary["message"])
                    else:
                        st.warning(live_dataset_summary["message"])
                    if selected_quantum_qubits > live_dataset_summary["max_supported_qubits"]:
                        st.warning(
                            f"Con tu dataset actual y test {int(selected_quantum_test_size * 100)}%, solo podes probar hasta "
                            f"{live_dataset_summary['max_supported_qubits']} qubits. Baja el selector de qubits o agrega mas capturas."
                        )
            else:
                st.caption(
                    f"Modo base: usa una muestra balanceada del dataset CICIDS2017 y la reduce a {selected_quantum_qubits} dimensiones para representar {selected_quantum_qubits} qubits."
                )
            if selected_quantum_execution_target == "ibm_validate":
                st.info(
                    "Metodo recomendado: primero se entrena en simulador local y despues IBM valida una parte chica del test. Asi se mide ruido real sin gastar tanta cuota."
                )
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
                get_quantum_hardware_results_path(
                    selected_quantum_qubits,
                    dataset_source=selected_quantum_dataset_source,
                )
                if selected_quantum_execution_target == "ibm_validate"
                else get_quantum_results_path(
                    selected_quantum_qubits,
                    dataset_source=selected_quantum_dataset_source,
                )
            )
            status_label = "Resultado real" if quantum_results_path.exists() else "Pendiente"
            render_info_card(
                "Estado del experimento",
                status_label,
                f"Se actualiza cuando se genera el archivo {quantum_results_path.as_posix()}.",
            )
            st.write("")
            render_info_card(
                "Origen de datos",
                "Live simulador" if selected_quantum_dataset_source == "live" else "CICIDS2017",
                "Esto afecta solo al experimento cuantico. El modelo clasico no usa el simulador de ataques.",
            )
            st.write("")
            render_info_card(
                "Comando en terminal",
                (
                    (
                        f"python -m src.quantum.train_vqc_simulator --execution-target ibm_validate --dataset-source live --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size} --ibm-validation-samples {selected_ibm_validation_samples}"
                        if selected_quantum_execution_target == "ibm_validate" and selected_quantum_dataset_source == "live"
                        else f"python -m src.quantum.train_vqc_simulator --execution-target ibm_validate --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size} --ibm-validation-samples {selected_ibm_validation_samples}"
                    )
                    if selected_quantum_execution_target == "ibm_validate"
                    else (
                        f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size}"
                        if selected_quantum_dataset_source == "live"
                        else f"python -m src.quantum.train_vqc_simulator --qubits {selected_quantum_qubits} --test-size {selected_quantum_test_size}"
                    )
                ),
                "La misma prueba que tambien puede ejecutarse fuera del dashboard.",
            )
            if selected_quantum_dataset_source == "live":
                st.write("")
                dataset_status = (
                    "Listo para entrenar"
                    if live_dataset_summary is not None and live_dataset_summary["ready"]
                    else "Falta completar"
                )
                render_info_card(
                    "CSV live",
                    dataset_status,
                    f"Archivo esperado: {LIVE_TRAINING_DATASET_PATH.as_posix()}",
                )
                st.write("")
                render_info_card(
                    "Capturas live",
                    (
                        f"{live_dataset_summary['benign_count']} benign / {live_dataset_summary['attack_count']} attack"
                        if live_dataset_summary is not None
                        else "Sin datos"
                    ),
                    "Cantidad de ventanas etiquetadas detectadas en el dataset live.",
                )
                st.write("")
                render_info_card(
                    "Qubits maximos",
                    str(live_dataset_summary["max_supported_qubits"]) if live_dataset_summary is not None else "0",
                    "Limite actual segun las muestras disponibles para entrenar.",
                )
            if selected_quantum_execution_target == "ibm_validate":
                st.write("")
                render_info_card(
                    "Subset IBM",
                    str(selected_ibm_validation_samples),
                    "Cuantas muestras del test se envian a IBM para la validacion corta.",
                )

        if quantum_button:
            progress_placeholder = st.empty()
            try:
                from src.quantum.train_vqc_simulator import train_quantum_simulator

                log_messages = []

                def ui_logger(message: str) -> None:
                    log_messages.append(message)
                    progress_placeholder.info(message)

                with st.spinner("Entrenando VQC y evaluando resultados..."):
                    quantum_results = train_quantum_simulator(
                        num_qubits=selected_quantum_qubits,
                        dataset_source=selected_quantum_dataset_source,
                        test_size=selected_quantum_test_size,
                        execution_target=selected_quantum_execution_target,
                        ibm_validation_samples=selected_ibm_validation_samples,
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
                    "Resultado IBM de bajo costo: el modelo se entreno localmente y IBM solo valido una parte del test. Esto sirve para medir impacto del hardware real, no para reemplazar el entrenamiento completo."
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
                        else (
                            f"Este boton entrena el VQC live con {selected_quantum_qubits} qubits y actualiza results/quantum_live_simulated_metrics_{selected_quantum_qubits}q.json y results/quantum_live_simulated_metrics.json."
                            if selected_quantum_dataset_source == "live"
                            else f"Este boton entrena el VQC con {selected_quantum_qubits} qubits y actualiza results/quantum_simulated_metrics_{selected_quantum_qubits}q.json y results/quantum_simulated_metrics.json."
                        )
                    )
                )
            )
        elif quantum_lab_results and quantum_lab_results_qubits is not None:
            st.info(
                f"Los ultimos resultados visibles del laboratorio corresponden a {quantum_lab_results_qubits} qubits en fuente {str(quantum_lab_results_source).upper()}. "
                f"Si queres ver {selected_quantum_qubits} qubits en {selected_quantum_dataset_source.upper()}, ejecuta esa configuracion."
            )
        return

    # Si el enfoque activo es clasico, solo se muestra el flujo clasico.
    if selected_model == "Modelo clasico":
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("#### Baseline clasico")
            st.caption("Aca se prueba el modelo clasico ya entrenado. Sirve como referencia principal porque hoy es el enfoque mas estable del sistema.")
            source = st.radio(
                "Origen de datos",
                ["Usar data/dataset.csv", "Subir CSV propio"],
                horizontal=True,
                key="classical_data_source",
            )
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
            render_info_card(
                "Estado clasico",
                model_data["Modelo clasico"]["source_label"],
                "El dashboard usa metricas reales del clasico si encuentra results/classical_metrics.json.",
            )

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
                        uploaded_bytes = BytesIO(uploaded_file.getvalue())
                        df = pd.read_csv(uploaded_bytes)

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
                st.plotly_chart(
                    make_confusion_chart(lab_results["confusion_matrix"]),
                    width="stretch",
                    key="lab_classical_confusion_chart",
                )


def render_analysis_tab(model_data: dict, selected_model: str, selected_quantum_dataset_source: str) -> None:
    section_header(
        "Comparacion y analisis",
        "Lectura guiada de rendimiento, ruido, tiempos y diferencias entre los enfoques.",
    )
    st.markdown("#### Resumen general")
    if model_data["Modelo cuantico"]["source"] != "real":
        command = (
            f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {model_data['Modelo cuantico']['selected_qubits']}"
            if selected_quantum_dataset_source == "live"
            else f"python -m src.quantum.train_vqc_simulator --qubits {model_data['Modelo cuantico']['selected_qubits']}"
        )
        st.info(
            f"Todavia no se entreno el VQC {'live' if selected_quantum_dataset_source == 'live' else 'CICIDS'} con {model_data['Modelo cuantico']['selected_qubits']} qubits. "
            f"Ejecutar: {command}"
        )
    st.plotly_chart(
        make_global_comparison_chart(model_data, height=380),
        width="stretch",
        key="analysis_global_comparison_chart",
    )
    table_df = build_metrics_dataframe(model_data).pivot(index="Modelo", columns="Metrica", values="Valor")
    table_df = table_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
    st.dataframe(table_df.style.format("{:.1%}"), width="stretch")
    st.caption(
        f"El clasico muestra su referencia real. El bloque cuantico refleja la corrida {model_data['Modelo cuantico'].get('dataset_source_label', 'CICIDS2017')} de {model_data['Modelo cuantico']['selected_qubits']} qubits si existe un resultado guardado."
    )

    st.write("")
    st.markdown("#### Corridas VQC disponibles")
    quantum_runs_df = build_quantum_runs_dataframe(dataset_source=selected_quantum_dataset_source)
    st.dataframe(
        quantum_runs_df.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "Tiempo (s)": "{:.2f}",
                "Sample": "{:.0f}",
            },
            na_rep="Sin correr",
        ),
        width="stretch",
    )
    trained_runs = quantum_runs_df[quantum_runs_df["Estado"] == "Entrenado"]
    if not trained_runs.empty:
        best_row = trained_runs.sort_values(["F1-Score", "Accuracy"], ascending=False).iloc[0]
        st.caption(
            f"Mejor corrida VQC disponible en {best_row['Fuente']}: {int(best_row['Qubits'])} qubits "
            f"con F1-score {best_row['F1-Score']:.2%} y tiempo {best_row['Tiempo (s)']:.2f}s."
        )
    else:
        st.info("Todavia no hay corridas VQC disponibles para comparar.")

    st.write("")
    st.markdown("#### Ruido y limites del hardware")
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.plotly_chart(
            make_noise_chart(model_data, height=350),
            width="stretch",
            key="analysis_noise_chart",
        )
    with col2:
        simulated = model_data["Modelo cuantico"]
        hardware = model_data["Hardware cuantico real"]
        render_info_card("Caida de Accuracy", f"{(simulated['accuracy'] - hardware['accuracy']):.1%}", "Perdida al pasar del ideal al hardware real.")
        st.write("")
        render_info_card("Caida de F1-Score", f"{(simulated['f1_score'] - hardware['f1_score']):.1%}", "Perdida general al salir del simulador ideal.")
        st.write("")
        diagnostics = hardware.get("hardware_diagnostics", {})
        limitation_flags = diagnostics.get("limitation_flags") or []
        render_info_card(
            "Backend IBM",
            str(hardware.get("ibm_backend_name", "Pendiente")),
            "Procesador cuantico real usado en la validacion IBM.",
        )
        st.write("")
        render_info_card(
            "Alertas del hardware",
            ", ".join(limitation_flags) if limitation_flags else "Sin flags",
            "Senales resumidas de cola, ruido o conectividad limitada detectadas en el backend.",
        )
        st.write("")
        hardware_gap_local = hardware.get("hardware_gap_vs_local_subset", {})
        if hardware_gap_local:
            render_info_card(
                "Caida vs local",
                f"Acc {hardware_gap_local.get('accuracy_drop', 0):.1%} | F1 {hardware_gap_local.get('f1_drop', 0):.1%}",
                "Diferencia entre IBM y la misma muestra evaluada localmente con los mismos pesos.",
            )
            st.write("")
        render_info_card("Modelo destacado", model_data[selected_model]["short_label"], "Enfoque activo en la lectura actual del dashboard.")
        if diagnostics:
            st.caption(
                f"T1 medio: {diagnostics.get('avg_t1_us', 'n/d')} us | "
                f"T2 medio: {diagnostics.get('avg_t2_us', 'n/d')} us | "
                f"Pending jobs: {diagnostics.get('pending_jobs', 'n/d')}"
            )

    st.write("")
    st.markdown("#### Costos de tiempo")
    st.plotly_chart(
        make_time_chart(model_data, height=350),
        width="stretch",
        key="analysis_time_chart",
    )
    st.caption("El clasico sigue siendo el mas eficiente; el hardware real conserva el mayor costo temporal.")


def render_demo_tab(model_data: dict, selected_model: str) -> None:
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


def render_conclusion_tab(model_data: dict, selected_model: str) -> None:
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


def main() -> None:
    configure_page()
    inject_css()
    selected_quantum_qubits = st.session_state.get("selected_quantum_qubits", 4)
    selected_quantum_dataset_source = st.session_state.get("selected_quantum_dataset_source", "cicids")
    model_data = get_model_data(
        selected_quantum_qubits=selected_quantum_qubits,
        selected_quantum_dataset_source=selected_quantum_dataset_source,
    )
    render_header(model_data)
    selected_model, selected_quantum_qubits, selected_quantum_dataset_source, current_step = render_sidebar_controls(
        model_data,
        selected_quantum_qubits,
        selected_quantum_dataset_source,
    )
    if st.session_state.get("selected_quantum_qubits") != selected_quantum_qubits:
        st.session_state["selected_quantum_qubits"] = selected_quantum_qubits
    if st.session_state.get("selected_quantum_dataset_source") != selected_quantum_dataset_source:
        st.session_state["selected_quantum_dataset_source"] = selected_quantum_dataset_source

    if current_step == "1. Panorama":
        render_overview_tab(model_data, selected_model)
    elif current_step == "2. Laboratorio":
        render_lab_tab(model_data, selected_model, selected_quantum_qubits, selected_quantum_dataset_source)
    elif current_step == "3. Analisis":
        render_analysis_tab(model_data, selected_model, selected_quantum_dataset_source)
    elif current_step == "4. Demo":
        render_demo_tab(model_data, selected_model)
    else:
        render_conclusion_tab(model_data, selected_model)


if __name__ == "__main__":
    main()
