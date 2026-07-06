from dataclasses import dataclass
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.classical.train_model import convert_to_binary_label, find_label_column
from src.live_detection.feature_engineering import curate_live_windows, enrich_live_feature_frame, looks_like_live_feature_frame


@dataclass
class QuantumDatasetBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    pca: PCA
    pca_components: int
    sample_size: int
    feature_count: int
    label_column: str
    live_curation_report: dict | None = None
    live_proxy_baseline_metrics: dict | None = None


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
    pca_components: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
    dataset_source: str | None = None,
) -> QuantumDatasetBundle:
    if not 0 < test_size < 1:
        raise ValueError(f"test_size invalido: {test_size}. Debe estar entre 0 y 1.")

    df = load_and_clean_dataset(dataset_path)
    label_column = find_label_column(df)
    working_df = df.copy()
    working_df["_binary_label"] = working_df[label_column].apply(convert_to_binary_label)
    class_counts = working_df["_binary_label"].value_counts().to_dict()
    benign_available = int(class_counts.get(0, 0))
    attack_available = int(class_counts.get(1, 0))
    minimum_test_samples = 2
    minimum_total_samples = ceil(minimum_test_samples / test_size)

    if benign_available < 2 or attack_available < 2:
        raise ValueError(
            "No hay suficientes capturas para entrenar el VQC. "
            f"Capturas benign disponibles: {benign_available}. "
            f"Capturas attack disponibles: {attack_available}. "
            f"Necesitas al menos 2 muestras por clase y al menos {minimum_total_samples} filas totales; "
            "en la practica conviene 10 o mas por clase."
        )

    if benign_available + attack_available < minimum_total_samples:
        raise ValueError(
            "No hay suficientes capturas totales para reservar un conjunto de test valido. "
            f"Filas disponibles: {benign_available + attack_available}. "
            f"Con test_size={test_size} necesitas al menos {minimum_total_samples} filas totales "
            "para cubrir ambas clases en test."
        )

    sampled_df = sample_balanced_binary_dataset(
        df,
        benign_samples=benign_samples,
        attack_samples=attack_samples,
        random_state=random_state,
    )

    y = sampled_df[label_column].apply(convert_to_binary_label).to_numpy()
    metadata_columns = [column for column in ("Scenario", "SimulatorVersion") if column in sampled_df.columns]
    metadata_df = sampled_df[metadata_columns].copy() if metadata_columns else None
    X = sampled_df.drop(columns=[label_column]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    live_curation_report = None
    live_proxy_baseline_metrics = None
    is_live_dataset = dataset_source == "live" or looks_like_live_feature_frame(X.columns.tolist())
    if is_live_dataset:
        X = enrich_live_feature_frame(X)
        # El dataset live puede evolucionar de version y agregar columnas nuevas.
        # Para no descartar todas las capturas viejas por schema drift, imputamos
        # faltantes con 0.0 antes de curar el dataset.
        X = X.fillna(0.0)
        valid_mask = np.ones(len(X), dtype=bool)
    else:
        valid_mask = ~X.isna().any(axis=1)
        X = X.loc[valid_mask]
        y = y[valid_mask.to_numpy()]

    if is_live_dataset:
        X, y, curation_report = curate_live_windows(
            X.reset_index(drop=True),
            y,
            metadata=metadata_df,
            random_state=random_state,
        )
        live_curation_report = curation_report.__dict__

    class_counts_after_clean = pd.Series(y).value_counts().to_dict()
    benign_after_clean = int(class_counts_after_clean.get(0, 0))
    attack_after_clean = int(class_counts_after_clean.get(1, 0))
    total_after_clean = int(len(y))

    if benign_after_clean < 2 or attack_after_clean < 2:
        raise ValueError(
            "No quedaron suficientes muestras validas despues de limpiar el dataset live. "
            f"Capturas benign validas: {benign_after_clean}. "
            f"Capturas attack validas: {attack_after_clean}. "
            f"Necesitas al menos 2 muestras por clase y al menos {minimum_total_samples} filas totales; "
            "en la practica conviene 10 o mas por clase."
        )

    if total_after_clean < minimum_total_samples:
        raise ValueError(
            "No quedaron suficientes muestras validas para el split train/test. "
            f"Filas validas: {total_after_clean}. "
            f"Con test_size={test_size} necesitas al menos {minimum_total_samples} filas totales."
        )

    test_rows = ceil(test_size * total_after_clean)
    train_rows = total_after_clean - test_rows
    max_pca_components = min(train_rows, X.shape[1])

    if train_rows < 2:
        raise ValueError(
            "No quedaron suficientes muestras de entrenamiento despues del split. "
            f"Filas de entrenamiento: {train_rows}. Ajusta test_size o agrega mas capturas."
        )

    if X.shape[1] < pca_components:
        raise ValueError(
            f"No hay suficientes features numericas para {pca_components} componentes PCA. "
            f"Features disponibles: {X.shape[1]}"
        )

    if pca_components > max_pca_components:
        raise ValueError(
            f"No se puede entrenar con {pca_components} qubits/componentes PCA. "
            f"Con {total_after_clean} filas validas y test_size={test_size}, el train queda con {train_rows} muestras, "
            f"asi que el maximo soportado es {max_pca_components}. "
            "Reduce qubits o agrega mas capturas."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if is_live_dataset:
        proxy_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        proxy_model.fit(X_train, y_train)
        proxy_predictions = proxy_model.predict(X_test)
        live_proxy_baseline_metrics = {
            "accuracy": round(float(accuracy_score(y_test, proxy_predictions)), 4),
            "precision": round(float(precision_score(y_test, proxy_predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, proxy_predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, proxy_predictions, zero_division=0)), 4),
        }

    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return QuantumDatasetBundle(
        X_train=X_train_pca,
        X_test=X_test_pca,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        pca=pca,
        pca_components=pca_components,
        sample_size=len(X),
        feature_count=X.shape[1],
        label_column=label_column,
        live_curation_report=live_curation_report,
        live_proxy_baseline_metrics=live_proxy_baseline_metrics,
    )
