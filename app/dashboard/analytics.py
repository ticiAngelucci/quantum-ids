from __future__ import annotations

import random
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app.dashboard.constants import (
    ACCENT_YELLOW,
    CLASSICAL_MODEL_PATH,
    DATASET_PATH,
    LIVE_CAPTURE_PATH,
    LIVE_CLASSICAL_INCOMPATIBLE_MESSAGE,
    LIVE_TRAINING_DATASET_PATH,
    MUTED_TEXT,
    PCA_PATH,
    PRIMARY_BLUE,
    SCALER_PATH,
    SECONDARY_BLUE,
    SUPPORTED_QUANTUM_QUBITS,
    TEXT,
)
from app.dashboard.data import load_classical_artifacts, load_quantum_simulated_results
from app.dashboard.types import ModelData
from src.live_detection.compatibility import compare_feature_sets, load_expected_classical_features
from src.classical.train_model import convert_to_binary_label, find_label_column


def build_metrics_dataframe(model_data: ModelData) -> pd.DataFrame:
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


def build_time_dataframe(model_data: ModelData) -> pd.DataFrame:
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


def make_global_comparison_chart(model_data: ModelData, height: int = 320) -> go.Figure:
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


def make_time_chart(model_data: ModelData, height: int = 320) -> go.Figure:
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


def make_noise_chart(model_data: ModelData, height: int = 320) -> go.Figure:
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
        _ = X_train
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


def predict_classical_live_batch(live_df: pd.DataFrame) -> dict:
    expected_features = load_expected_classical_features(DATASET_PATH)
    comparison = compare_feature_sets(live_df.columns.tolist(), expected_features)
    if not comparison["compatible"]:
        return {
            "compatible": False,
            "missing": comparison["missing"],
            "extra": comparison["extra"],
            "message": LIVE_CLASSICAL_INCOMPATIBLE_MESSAGE,
        }

    artifacts = load_classical_artifacts()
    if artifacts is None:
        raise FileNotFoundError("No se encontraron modelo, scaler o PCA clasicos en results/.")

    transformed = artifacts["scaler"].transform(live_df[expected_features])
    projected = artifacts["pca"].transform(transformed)
    predictions = artifacts["model"].predict(projected)
    return {
        "compatible": True,
        "predictions": predictions.tolist(),
        "prediction_counts": {
            "normal": int((predictions == 0).sum()),
            "intrusion": int((predictions == 1).sum()),
        },
    }


def capture_live_monitoring_batch(
    *,
    duration: int,
    windows: int,
    iface: str | None = None,
    count: int = 0,
    output_path: Path = LIVE_CAPTURE_PATH,
    label: str | None = None,
    append_to_training: bool = False,
    logger=lambda _message: None,
) -> pd.DataFrame:
    from src.live_detection.capture import capture_packets, save_features
    from src.live_detection.feature_extractor import extract_live_features

    if duration <= 0 or windows <= 0:
        raise ValueError("La duracion y la cantidad de ventanas deben ser mayores a 0.")

    batch_rows: list[dict[str, float | str]] = []
    if output_path.exists():
        output_path.unlink()

    for window_index in range(windows):
        logger(f"Capturando ventana {window_index + 1} de {windows}...")
        packets, elapsed = capture_packets(duration=duration, iface=iface, count=count)
        features = extract_live_features(packets=packets, duration_seconds=elapsed)
        if label is not None:
            features["Label"] = label
        batch_rows.append(features)
        save_features(features, output_path, append=window_index > 0)
        if append_to_training and label is not None:
            save_features(features, LIVE_TRAINING_DATASET_PATH, append=True)

    return pd.DataFrame(batch_rows)


def predict_quantum_live_batch(
    *,
    live_df: pd.DataFrame,
    num_qubits: int,
    test_size: float,
    logger=lambda _message: None,
) -> dict:
    from src.preprocessing.quantum_preprocessing import prepare_quantum_dataset
    from src.quantum.train_vqc_simulator import build_vqc

    if not LIVE_TRAINING_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"No existe {LIVE_TRAINING_DATASET_PATH.as_posix()}. Genera capturas benign y attack antes de inferir en vivo."
        )

    dataset_bundle = prepare_quantum_dataset(
        dataset_path=LIVE_TRAINING_DATASET_PATH,
        pca_components=num_qubits,
        test_size=test_size,
    )
    training_df = pd.read_csv(LIVE_TRAINING_DATASET_PATH)
    training_df.columns = [str(col).strip() for col in training_df.columns]
    label_column = find_label_column(training_df)
    expected_features = training_df.drop(columns=[label_column]).select_dtypes(include=[np.number]).columns.tolist()
    comparison = compare_feature_sets(live_df.columns.tolist(), expected_features)
    if not comparison["compatible"]:
        raise ValueError(
            "Las features capturadas no coinciden con el dataset live de entrenamiento. "
            f"Faltantes: {comparison['missing']} | Extras: {comparison['extra']}"
        )

    logger("Entrenando VQC live en simulador para inferencia inmediata...")
    vqc, _ = build_vqc(num_qubits=num_qubits, execution_target="simulator", logger=logger)
    vqc.fit(dataset_bundle.X_train, dataset_bundle.y_train)
    batch_scaled = dataset_bundle.scaler.transform(live_df[expected_features])
    batch_pca = dataset_bundle.pca.transform(batch_scaled)
    predictions = np.asarray(vqc.predict(batch_pca)).astype(int)
    return {
        "prediction_counts": {
            "normal": int((predictions == 0).sum()),
            "intrusion": int((predictions == 1).sum()),
        },
        "predictions": predictions.tolist(),
    }


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
