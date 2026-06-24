from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np


ModelName = Literal["Modelo clasico", "Modelo cuantico", "Hardware cuantico real"]
QuantumDatasetSource = Literal["cicids", "live"]
QuantumExecutionTarget = Literal["simulator", "ibm_validate", "ibm_quantum"]
SectionName = Literal[
    "1. Resumen",
    "2. Experimentar",
    "3. Live",
    "4. Analisis",
    "5. Conclusiones",
]


@dataclass(frozen=True)
class SidebarSelection:
    selected_model: str
    selected_quantum_qubits: int
    selected_quantum_dataset_source: QuantumDatasetSource
    current_step: SectionName


class DashboardModel(TypedDict, total=False):
    short_label: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    execution_time: float
    description: str
    source: str
    source_label: str
    confusion_matrix: np.ndarray
    trained_model_name: str
    pca_components: int | None
    num_qubits: int | None
    sample_size: int | None
    selected_qubits: int
    selected_dataset_source: QuantumDatasetSource
    dataset_source_label: str
    dataset_path: str | None
    results_path: str
    ibm_backend_name: str | None
    hardware_diagnostics: dict[str, Any]
    hardware_gap_vs_simulator: dict[str, Any]
    hardware_gap_vs_local_subset: dict[str, Any]


ModelData = dict[str, DashboardModel]


class ClassicalArtifacts(TypedDict):
    model: Any
    scaler: Any
    pca: Any
