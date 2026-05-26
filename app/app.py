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


PRIMARY_BLUE = "#002B5C"
SECONDARY_BLUE = "#005187"
ACCENT_YELLOW = "#F4C430"
BACKGROUND = "#F7F9FC"
TEXT = "#1F2937"
MUTED_TEXT = "#64748B"
SUCCESS = "#0F766E"
DANGER = "#B91C1C"

RESULTS_DIR = Path("results")
DATASET_PATH = Path("data/dataset.csv")
CLASSICAL_RESULTS_PATH = RESULTS_DIR / "classical_metrics.json"
QUANTUM_SIMULATED_RESULTS_PATH = RESULTS_DIR / "quantum_simulated_metrics.json"
CLASSICAL_MODEL_PATH = RESULTS_DIR / "random_forest_model.joblib"
SCALER_PATH = RESULTS_DIR / "scaler.joblib"
PCA_PATH = RESULTS_DIR / "pca.joblib"
SUPPORTED_QUANTUM_QUBITS = (2, 4, 6, 8)

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
    "Modelo cuantico simulado": {
        "label": "Modelo cuantico simulado",
        "short_label": "QML simulado",
        "description": "Circuito variacional ejecutado en simulador ideal.",
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


def configure_page() -> None:
    st.set_page_config(
        page_title="Quantum IDS Dashboard",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="collapsed",
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
                background: var(--background);
                color: var(--text);
                font-family: "Segoe UI", sans-serif;
            }}

            input[type="radio"] {{
                accent-color: {ACCENT_YELLOW};
            }}

            [data-testid="stWidgetLabel"] p,
            .stRadio label,
            .stCheckbox label {{
                color: var(--text);
            }}

            [data-testid="stHeader"] {{
                background: rgba(247, 249, 252, 0.92);
                backdrop-filter: blur(8px);
            }}

            .block-container {{
                max-width: 1240px;
                padding-top: 1rem;
                padding-bottom: 2rem;
            }}

            h1, h2, h3 {{
                color: var(--primary-blue);
                letter-spacing: 0;
            }}

            .hero {{
                background: #FFFFFF;
                border: 1px solid #E5EAF2;
                border-left: 6px solid var(--accent-yellow);
                border-radius: 8px;
                padding: 1rem 1.2rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
            }}

            .hero h1 {{
                margin: 0 0 0.3rem 0;
                font-size: 1.8rem;
            }}

            .hero p {{
                color: var(--muted-text);
                margin: 0;
                max-width: 980px;
            }}

            .badge-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.75rem;
            }}

            .badge {{
                background: #EEF4FB;
                border: 1px solid #D6E2F0;
                border-radius: 8px;
                color: var(--primary-blue);
                display: inline-flex;
                font-size: 0.78rem;
                font-weight: 800;
                padding: 0.26rem 0.55rem;
            }}

            .badge.accent {{
                background: #FFF6D5;
                border-color: #F5D76E;
            }}

            .section-intro {{
                color: var(--muted-text);
                margin-top: -0.2rem;
                margin-bottom: 0.75rem;
                max-width: 900px;
            }}

            .info-card, .metric-card, .result-card, .compact-card {{
                background: #FFFFFF;
                border: 1px solid #E5EAF2;
                border-radius: 8px;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }}

            .info-card, .compact-card {{
                padding: 0.85rem 0.95rem;
            }}

            .metric-card {{
                border-top: 4px solid var(--accent-yellow);
                padding: 0.8rem 0.9rem;
                min-height: 112px;
            }}

            .card-label {{
                color: var(--muted-text);
                font-size: 0.76rem;
                font-weight: 800;
                text-transform: uppercase;
                margin-bottom: 0.25rem;
            }}

            .card-value {{
                color: var(--primary-blue);
                font-size: 1.12rem;
                font-weight: 850;
                margin-bottom: 0.15rem;
            }}

            .card-help {{
                color: var(--muted-text);
                font-size: 0.86rem;
            }}

            .metric-value {{
                color: var(--primary-blue);
                font-size: 1.65rem;
                font-weight: 850;
                margin-top: 0.05rem;
            }}

            .metric-caption {{
                color: var(--muted-text);
                font-size: 0.8rem;
            }}

            .result-card {{
                padding: 1rem;
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
                background: #EEF4FB;
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

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.5rem;
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 8px 8px 0 0;
                padding-left: 0.25rem;
                padding-right: 0.25rem;
                color: var(--muted-text);
            }}

            .stTabs [data-baseweb="tab-highlight"] {{
                background: var(--accent-yellow) !important;
                height: 3px !important;
            }}

            .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                color: var(--primary-blue);
                font-weight: 800;
            }}

            div[data-testid="stButton"] > button {{
                background: var(--primary-blue);
                color: white;
                border: 1px solid var(--accent-yellow);
                border-radius: 8px;
            }}

            div[data-testid="stButton"] > button p,
            div[data-testid="stButton"] > button span {{
                color: white !important;
            }}

            div[data-testid="stButton"] > button:hover {{
                background: var(--secondary-blue);
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


def get_quantum_results_path(qubits: int) -> Path:
    return RESULTS_DIR / f"quantum_simulated_metrics_{qubits}q.json"


def load_quantum_simulated_results(qubits: int | None = None) -> dict | None:
    results_path = QUANTUM_SIMULATED_RESULTS_PATH if qubits is None else get_quantum_results_path(qubits)
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
        "results_path": str(results_path),
    }


def get_model_data(selected_quantum_qubits: int = 4) -> dict:
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

    quantum_simulated_results = load_quantum_simulated_results(selected_quantum_qubits)
    if quantum_simulated_results is not None:
        model_data["Modelo cuantico simulado"].update(
            {
                "accuracy": quantum_simulated_results["accuracy"],
                "precision": quantum_simulated_results["precision"],
                "recall": quantum_simulated_results["recall"],
                "f1_score": quantum_simulated_results["f1_score"],
                "confusion_matrix": quantum_simulated_results["confusion_matrix"],
                "source": "real",
                "source_label": "Resultado real",
                "description": (
                    f"VQC simulado cargado desde results/quantum_simulated_metrics_{selected_quantum_qubits}q.json. "
                    "Permite comparar QML en simulacion frente al baseline clasico."
                ),
                "trained_model_name": quantum_simulated_results["model_name"],
                "pca_components": quantum_simulated_results["pca_components"],
                "num_qubits": quantum_simulated_results["num_qubits"],
                "sample_size": quantum_simulated_results["sample_size"],
                "selected_qubits": selected_quantum_qubits,
                "execution_time": quantum_simulated_results["execution_time_seconds"]
                if quantum_simulated_results["execution_time_seconds"] is not None
                else model_data["Modelo cuantico simulado"]["execution_time"],
            }
        )
    else:
        model_data["Modelo cuantico simulado"].update(
            {
                "source": "missing",
                "source_label": "Pendiente",
                "selected_qubits": selected_quantum_qubits,
                "description": (
                    f"Todavia no se entreno el VQC con {selected_quantum_qubits} qubits. "
                    f"Ejecutar: python -m src.quantum.train_vqc_simulator --qubits {selected_quantum_qubits}"
                ),
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
    classical_badge = "Clasico real conectado" if model_data["Modelo clasico"]["source"] == "real" else "Clasico en modo mock"
    st.markdown(
        f"""
        <section class="hero">
            <h1>Quantum IDS Dashboard</h1>
            <p>
                Comparacion entre modelo clasico, simulador cuantico y hardware real para deteccion
                de anomalias en trafico de red, con un laboratorio interactivo para ejecutar pruebas desde el front.
            </p>
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


def render_model_switcher(model_data: dict, selected_quantum_qubits: int) -> tuple[str, int]:
    st.markdown("#### Enfoque activo")
    selected_model = st.radio(
        "Seleccion de enfoque",
        options=list(model_data.keys()),
        index=list(model_data.keys()).index(st.session_state.get("selected_model", "Modelo clasico")),
        horizontal=True,
        key="model_switcher_radio",
        label_visibility="collapsed",
    )
    st.session_state["selected_model"] = selected_model
    model = model_data[selected_model]
    source_class = "real" if model["source"] == "real" else "mock"
    st.markdown(
        f"""
        <div class="compact-card">
            <div class="card-label">{model["short_label"]}</div>
            <div class="card-help">
                <span class="status-pill {source_class}">{model["source_label"]}</span>
                {model["description"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if selected_model == "Modelo cuantico simulado":
        st.write("")
        quantum_selection = st.radio(
            "Resultado VQC a visualizar",
            options=[f"{qubits} qubits" for qubits in SUPPORTED_QUANTUM_QUBITS],
            index=list(SUPPORTED_QUANTUM_QUBITS).index(selected_quantum_qubits),
            horizontal=True,
            key="quantum_results_radio",
        )
        chosen_qubits = int(quantum_selection.split()[0])
        if chosen_qubits != selected_quantum_qubits:
            st.session_state["selected_quantum_qubits"] = chosen_qubits
            st.session_state.pop("quantum_lab_results", None)
            st.session_state.pop("quantum_lab_results_qubits", None)
            st.rerun()
        selected_quantum_qubits = chosen_qubits
        st.session_state["selected_quantum_qubits"] = chosen_qubits
    return selected_model, selected_quantum_qubits


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


def build_quantum_runs_dataframe() -> pd.DataFrame:
    rows = []
    for qubits in SUPPORTED_QUANTUM_QUBITS:
        quantum_results = load_quantum_simulated_results(qubits)
        if quantum_results is None:
            rows.append(
                {
                    "Qubits": qubits,
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
            "QML simulado": SECONDARY_BLUE,
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
            "QML simulado": SECONDARY_BLUE,
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
    simulated = model_data["Modelo cuantico simulado"]
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


def classify_mock_connection(packet_rate: int, failed_logins: int, protocol_risk: int, selected_model: str) -> tuple[str, float]:
    model_bias = {
        "Modelo clasico": 0.02,
        "Modelo cuantico simulado": 0.05,
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
        "Resumen compacto del estado actual del experimento, con foco en que el modelo clasico ya usa resultados reales.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card("Dataset", "data/dataset.csv", "Base activa del experimento clasico.")
    with col2:
        render_info_card("Modelo clasico", model_data["Modelo clasico"]["source_label"], model_data["Modelo clasico"]["description"])
    with col3:
        render_info_card(
            "Entornos",
            "3 enfoques",
            (
                f"Clasico: {model_data['Modelo clasico']['source_label']} | "
                f"QML: {model_data['Modelo cuantico simulado']['source_label']} | "
                "Hardware real: Pendiente"
            ),
        )

    if model_data["Modelo cuantico simulado"]["source"] != "real":
        st.warning(
            f"Todavia no se entreno el VQC con {model_data['Modelo cuantico simulado']['selected_qubits']} qubits. "
            f"Ejecutar: python -m src.quantum.train_vqc_simulator --qubits {model_data['Modelo cuantico simulado']['selected_qubits']}"
        )

    st.write("")
    st.plotly_chart(make_global_comparison_chart(model_data), width="stretch")

    model = model_data[selected_model]

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Accuracy", model["accuracy"], "Correctas totales")
    with metric_cols[1]:
        render_metric_card("Precision", model["precision"], "Alertas confiables")
    with metric_cols[2]:
        render_metric_card("Recall", model["recall"], "Intrusiones detectadas")
    with metric_cols[3]:
        render_metric_card("F1-Score", model["f1_score"], "Balance global")

    st.write("")
    chart_col, info_col = st.columns([1.3, 1])
    with chart_col:
        st.plotly_chart(make_confusion_chart(model["confusion_matrix"], height=300), width="stretch")
    with info_col:
        render_info_card("Fuente", model["source_label"], "Indica si las metricas provienen de una ejecucion real o de valores mock.")
        st.write("")
        render_info_card("Tiempo estimado", f"{model['execution_time']:.2f}s", "Costo de ejecucion asociado al enfoque actual.")


def render_lab_tab(model_data: dict, selected_model: str, selected_quantum_qubits: int) -> None:
    section_header(
        "Laboratorio de prueba",
        "Esta vista ejecuta solo el enfoque activo para que no tengas que decidir lo mismo dos veces.",
    )
    if selected_model == "Modelo cuantico simulado":
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("#### VQC simulado")
            st.caption(
                "Entrena y evalua un Variational Quantum Classifier sobre una muestra reducida y balanceada del dataset."
            )
            quantum_button = st.button(
                f"Ejecutar prueba cuantica simulada ({selected_quantum_qubits}q)",
                width="stretch",
                type="primary",
            )
            st.caption(
                f"Muestra por defecto: 200 benignos + 200 ataques, PCA a {selected_quantum_qubits} componentes y {selected_quantum_qubits} qubits."
            )

        with right:
            quantum_results_path = get_quantum_results_path(selected_quantum_qubits)
            status_label = "Resultado real" if quantum_results_path.exists() else "Pendiente"
            render_info_card(
                "Estado VQC",
                status_label,
                f"Se actualiza cuando se genera {quantum_results_path.as_posix()}.",
            )
            st.write("")
            render_info_card(
                "Comando equivalente",
                f"python -m src.quantum.train_vqc_simulator --qubits {selected_quantum_qubits}",
                "La misma prueba que puede ejecutarse desde consola.",
            )

        if quantum_button:
            progress_placeholder = st.empty()
            try:
                from src.quantum.train_vqc_simulator import train_quantum_simulator

                log_messages = []

                def ui_logger(message: str) -> None:
                    log_messages.append(message)
                    progress_placeholder.info(message)

                with st.spinner("Entrenando VQC simulado y evaluando resultados..."):
                    quantum_results = train_quantum_simulator(
                        num_qubits=selected_quantum_qubits,
                        logger=ui_logger,
                    )

                progress_placeholder.empty()
                st.session_state["quantum_lab_results"] = quantum_results
                st.session_state["quantum_lab_results_qubits"] = selected_quantum_qubits
                st.session_state["selected_quantum_qubits"] = selected_quantum_qubits
            except Exception as error:
                progress_placeholder.empty()
                st.error(f"No pude ejecutar la prueba cuantica simulada: {error}")

        quantum_lab_results = st.session_state.get("quantum_lab_results")
        quantum_lab_results_qubits = st.session_state.get("quantum_lab_results_qubits")
        if quantum_lab_results and quantum_lab_results_qubits == selected_quantum_qubits:
            st.success("Prueba cuantica simulada finalizada.")
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
            )
            st.caption(
                f"Este boton entrena el VQC con {selected_quantum_qubits} qubits y actualiza results/quantum_simulated_metrics_{selected_quantum_qubits}q.json y results/quantum_simulated_metrics.json."
            )
        elif quantum_lab_results and quantum_lab_results_qubits is not None:
            st.info(
                f"Los ultimos resultados visibles del laboratorio corresponden a {quantum_lab_results_qubits} qubits. "
                f"Si queres ver {selected_quantum_qubits} qubits, ejecuta esa configuracion."
            )
        return

    if selected_model == "Hardware cuantico real":
        st.info(
            "La ejecucion sobre hardware cuantico real sigue pendiente de implementacion. "
            "Usa por ahora el analisis comparativo y el VQC simulado."
        )
        return

    # Si el enfoque activo es clasico, solo se muestra el flujo clasico.
    if selected_model == "Modelo clasico":
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("#### Baseline clasico")
            st.caption("Evalua el modelo Random Forest ya entrenado sobre el dataset actual o sobre un CSV que subas.")
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
                "Clasico actual",
                model_data["Modelo clasico"]["source_label"],
                "El dashboard toma metricas reales del clasico si encuentra results/classical_metrics.json.",
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
                    "Estas metricas pueden verse casi perfectas porque el modelo clasico obtuvo valores muy altos "
                    "en el holdout. Ahora se muestran con mas precision para evitar que 99.97% se vea como 100.0%."
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
                st.plotly_chart(make_confusion_chart(lab_results["confusion_matrix"]), width="stretch")


def render_analysis_tab(model_data: dict, selected_model: str) -> None:
    section_header(
        "Comparacion y analisis",
        "Lectura mas ordenada de rendimiento, matriz clasica, ruido cuantico y tiempos.",
    )
    comp_tab, quantum_tab, noise_tab, time_tab = st.tabs(["Modelos", "Corridas VQC", "Ruido cuantico", "Tiempos"])

    with comp_tab:
        if model_data["Modelo cuantico simulado"]["source"] != "real":
            st.info(
                f"Todavia no se entreno el VQC con {model_data['Modelo cuantico simulado']['selected_qubits']} qubits. "
                f"Ejecutar: python -m src.quantum.train_vqc_simulator --qubits {model_data['Modelo cuantico simulado']['selected_qubits']}"
            )
        st.plotly_chart(make_global_comparison_chart(model_data, height=380), width="stretch")
        table_df = build_metrics_dataframe(model_data).pivot(index="Modelo", columns="Metrica", values="Valor")
        table_df = table_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
        st.dataframe(table_df.style.format("{:.1%}"), width="stretch")
        st.caption(
            f"Clasico usa resultados reales; el simulador VQC refleja la corrida seleccionada de {model_data['Modelo cuantico simulado']['selected_qubits']} qubits si existe su JSON. Hardware real sigue pendiente."
        )

    with quantum_tab:
        quantum_runs_df = build_quantum_runs_dataframe()
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
                f"Mejor corrida VQC disponible hasta ahora: {int(best_row['Qubits'])} qubits "
                f"con F1-score {best_row['F1-Score']:.2%} y tiempo {best_row['Tiempo (s)']:.2f}s."
            )
        else:
            st.info("Todavia no hay corridas VQC disponibles para comparar.")

    with noise_tab:
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.plotly_chart(make_noise_chart(model_data, height=350), width="stretch")
        with col2:
            simulated = model_data["Modelo cuantico simulado"]
            hardware = model_data["Hardware cuantico real"]
            render_info_card("Caida de Accuracy", f"{(simulated['accuracy'] - hardware['accuracy']):.1%}", "Simulador ideal frente a hardware real.")
            st.write("")
            render_info_card("Caida de F1-Score", f"{(simulated['f1_score'] - hardware['f1_score']):.1%}", "Impacto combinado sobre precision y recall.")
            st.write("")
            render_info_card("Modelo destacado", model_data[selected_model]["short_label"], "Enfoque activo en la lectura actual del dashboard.")

    with time_tab:
        st.plotly_chart(make_time_chart(model_data, height=350), width="stretch")
        st.caption("El clasico sigue siendo el mas eficiente; el hardware real conserva el mayor costo temporal.")


def render_demo_tab(model_data: dict, selected_model: str) -> None:
    section_header(
        "Demo rapida de conexion",
        "Una interaccion simple para tener feedback inmediato sin cargar datasets.",
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
            render_info_card("Estado", "Esperando simulacion", "Ajusta parametros y ejecuta la demo.")


def render_conclusion_tab(model_data: dict, selected_model: str) -> None:
    section_header(
        "Conclusiones visuales",
        "Cierre rapido para usar el dashboard como apoyo de tesis y no como una landing eterna.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        render_info_card("Clasico", "Mejor rendimiento", "Ya conectado a resultados reales del pipeline.")
    with col2:
        render_info_card("QML simulado", "Escenario ideal", "Sirve para medir potencial teorico sin ruido.")
    with col3:
        render_info_card("Hardware real", "Brecha NISQ", "Muestra degradacion por ruido y mayor tiempo.")

    st.write("")
    st.markdown(
        """
        <div class="compact-card">
            <div class="card-label">Lectura preliminar</div>
            <div class="card-help">
                El modelo clasico ofrece hoy la referencia mas solida para deteccion de anomalias en este entorno.
                El valor de QML aparece con mas claridad como linea experimental comparativa: primero en simulacion
                ideal y despues en hardware real, donde el ruido todavia limita la performance practica.
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
    model_data = get_model_data(selected_quantum_qubits=selected_quantum_qubits)
    render_header(model_data)
    selected_model, selected_quantum_qubits = render_model_switcher(model_data, selected_quantum_qubits)
    if st.session_state.get("selected_quantum_qubits") != selected_quantum_qubits:
        st.session_state["selected_quantum_qubits"] = selected_quantum_qubits

    overview_tab, lab_tab, analysis_tab, demo_tab, conclusion_tab = st.tabs(
        ["Vision general", "Probar modelo", "Analisis", "Simulacion", "Conclusiones"]
    )

    with overview_tab:
        render_overview_tab(model_data, selected_model)

    with lab_tab:
        render_lab_tab(model_data, selected_model, selected_quantum_qubits)

    with analysis_tab:
        render_analysis_tab(model_data, selected_model)

    with demo_tab:
        render_demo_tab(model_data, selected_model)

    with conclusion_tab:
        render_conclusion_tab(model_data, selected_model)


if __name__ == "__main__":
    main()
