from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.classical.train_model import convert_to_binary_label, find_label_column
from src.live_detection.feature_engineering import curate_live_windows, enrich_live_feature_frame, looks_like_live_feature_frame


@dataclass
class QuantumDatasetBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    quantum_selector: SelectKBest
    quantum_scaler: MinMaxScaler
    qubits: int
    sample_size: int
    feature_count: int
    label_column: str
    live_curation_report: dict | None = None
    live_proxy_baseline_metrics: dict | None = None


def select_balanced_quantum_subset(
    features: np.ndarray,
    labels: np.ndarray,
    samples_per_class: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Selecciona una cohorte binaria, balanceada y reproducible."""
    features = np.asarray(features)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("La evaluación QSVM requiere exactamente dos clases.")

    selected_indices: list[int] = []
    for class_label in classes:
        class_indices = np.flatnonzero(labels == class_label)
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"La clase {class_label} necesita al menos "
                f"{samples_per_class} muestras."
            )
        selected_indices.extend(class_indices[:samples_per_class].tolist())

    return features[selected_indices], labels[selected_indices]


def load_and_clean_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def sample_balanced_binary_dataset(
    df: pd.DataFrame,
    benign_samples: int,
    attack_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    label_column = find_label_column(df)
    working_df = df.copy()
    working_df["_binary_label"] = working_df[label_column].apply(convert_to_binary_label)

    benign_df = working_df[working_df["_binary_label"] == 0]
    attack_df = working_df[working_df["_binary_label"] == 1]

    benign_count = min(benign_samples, len(benign_df))
    attack_count = min(attack_samples, len(attack_df))

    if benign_count == 0 or attack_count == 0:
        raise ValueError("No hay suficientes registros de ambas clases para generar una muestra balanceada.")

    sampled_df = pd.concat(
        [
            benign_df.sample(n=benign_count, random_state=random_state),
            attack_df.sample(n=attack_count, random_state=random_state),
        ],
        axis=0,
    ).sample(frac=1.0, random_state=random_state)

    return sampled_df.drop(columns=["_binary_label"])


def prepare_quantum_dataset(
    dataset_path: Path,
    benign_samples: int = 200,
    attack_samples: int = 200,
    qubits: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_source: str | None = None,
) -> QuantumDatasetBundle:
    df = pd.read_csv(dataset_path)
    # 0. Carga y Muestreo
    df = load_and_clean_dataset(dataset_path)
    df = df.loc[:, df.apply(pd.Series.nunique) != 1]
    label_column = find_label_column(df)
    
    sampled_df = sample_balanced_binary_dataset(
        df, benign_samples=benign_samples, attack_samples=attack_samples, random_state=random_state
    )

    y = sampled_df[label_column].apply(convert_to_binary_label).to_numpy()
    X = sampled_df.drop(columns=[label_column]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    # Limpieza inicial
    is_live_dataset = dataset_source == "live" or looks_like_live_feature_frame(X.columns.tolist())
    if is_live_dataset:
        X = enrich_live_feature_frame(X).fillna(0.0)
    else:
        valid_mask = ~X.isna().any(axis=1)
        X, y = X.loc[valid_mask], y[valid_mask.to_numpy()]

    # 1. ESCALADO ROBUSTO:
    # Mitiga el impacto de valores extremos (outliers) típicos en ataques DDoS.
    robust_scaler = RobustScaler()
    X_robust = robust_scaler.fit_transform(X)
    X = pd.DataFrame(X_robust, columns=X.columns)

    # 2. FILTRO DE CORRELACIÓN:
    # Elimina variables redundantes que "ahogan" la capacidad de aprendizaje de los qubits.
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    X = X.drop(columns=to_drop)
    print(f"DEBUG: Features eliminadas por alta correlación: {len(to_drop)}")
    print(f"DEBUG: Features restantes: {X.shape[1]}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Escalado base para el selector
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. SELECCIÓN CUÁNTICA (KBest)
    quantum_selector = SelectKBest(score_func=f_classif, k=min(qubits, X.shape[1]))
    X_train_selected = quantum_selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = quantum_selector.transform(X_test_scaled)

    # 5. ESCALADO DE FASE (ZZFeatureMap requiere rango -1 a 1)
    quantum_scaler = MinMaxScaler(feature_range=(-1, 1)) 
    X_train_quantum = quantum_scaler.fit_transform(X_train_selected)
    X_test_quantum = quantum_scaler.transform(X_test_selected)

    return QuantumDatasetBundle(
        X_train=X_train_quantum, 
        X_test=X_test_quantum,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler, 
        quantum_selector=quantum_selector,
        quantum_scaler=quantum_scaler,
        qubits=qubits,
        sample_size=len(X),
        feature_count=X.shape[1],
        label_column=label_column
    )
