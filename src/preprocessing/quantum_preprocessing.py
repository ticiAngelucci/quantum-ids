from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.classical.train_model import convert_to_binary_label, find_label_column


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
) -> QuantumDatasetBundle:
    df = load_and_clean_dataset(dataset_path)
    label_column = find_label_column(df)
    sampled_df = sample_balanced_binary_dataset(
        df,
        benign_samples=benign_samples,
        attack_samples=attack_samples,
        random_state=random_state,
    )

    y = sampled_df[label_column].apply(convert_to_binary_label).to_numpy()
    X = sampled_df.drop(columns=[label_column]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_mask = ~X.isna().any(axis=1)
    X = X.loc[valid_mask]
    y = y[valid_mask.to_numpy()]

    if X.shape[1] < pca_components:
        raise ValueError(
            f"No hay suficientes features numericas para {pca_components} componentes PCA. "
            f"Features disponibles: {X.shape[1]}"
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
        sample_size=len(sampled_df),
        feature_count=X.shape[1],
        label_column=label_column,
    )
