from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.constants import (
    CLASSICAL_RESULTS_PATH,
    QUANTUM_HARDWARE_RESULTS_PATH,
    QUANTUM_IBM_HARDWARE_RESULTS_PATH,
    QUANTUM_LIVE_HARDWARE_RESULTS_PATH,
    QUANTUM_LIVE_IBM_HARDWARE_RESULTS_PATH,
)
from dashboard.types import ModelData
from dashboard.ui import render_info_card, render_spotlight_panel


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _metric(payload: dict[str, Any] | None, name: str) -> float | None:
    if not payload:
        return None
    value = payload.get("metrics", {}).get(name)
    return float(value) if value is not None else None


def _format_percent(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/d"


def _format_seconds(value: Any) -> str:
    return f"{float(value):.2f} s" if value is not None else "n/d"


def _hardware_evidence_row(
    *,
    label: str,
    dataset: str,
    path: Path,
    expected_target: str,
) -> dict[str, str]:
    payload = _load_json(path)
    if not payload or payload.get("execution_target") != expected_target:
        return {
            "Enfoque": label,
            "Dataset": dataset,
            "Estado": "Pendiente",
            "Backend": "n/d",
            "Cohorte train/test": "4 / 4 planificada",
            "Carga ejecutada": "10 train + 16 prueba = 26 circuitos",
            "Accuracy": "n/d",
            "Precision": "n/d",
            "Recall": "n/d",
            "F1": "n/d",
            "Tiempo / uso": "n/d",
            "Evidencia": path.name,
        }

    train_samples = payload.get("train_sample_size", 4)
    test_samples = payload.get("sample_size", payload.get("rows", 4))
    train_circuits = payload.get("train_circuit_count", 10)
    test_circuits = payload.get("test_circuit_count", 16)
    total_circuits = payload.get(
        "circuit_count",
        int(train_circuits) + int(test_circuits),
    )
    if expected_target == "ibm_quantum":
        backend = str(payload.get("ibm_backend_name") or "IBM Quantum")
        usage = payload.get("ibm_total_usage_seconds")
        timing = f"{_format_seconds(payload.get('execution_time_seconds'))} / "
        timing += f"{_format_seconds(usage)} QPU"
    else:
        backend = "SpinQ Triangulum"
        timing = _format_seconds(payload.get("execution_time_seconds"))

    return {
        "Enfoque": label,
        "Dataset": dataset,
        "Estado": "Ejecutado",
        "Backend": backend,
        "Cohorte train/test": f"{train_samples} / {test_samples}",
        "Carga ejecutada": (
            f"{train_circuits} train + {test_circuits} prueba = "
            f"{total_circuits} circuitos"
        ),
        "Accuracy": _format_percent(_metric(payload, "accuracy")),
        "Precision": _format_percent(_metric(payload, "precision")),
        "Recall": _format_percent(_metric(payload, "recall")),
        "F1": _format_percent(_metric(payload, "f1_score")),
        "Tiempo / uso": timing,
        "Evidencia": path.name,
    }


def _local_simulator_row(dataset_source: str) -> dict[str, str]:
    dataset_label = "Live" if dataset_source == "live" else "CICIDS2017"
    session_payload = st.session_state.get("quantum_lab_results")
    session_source = st.session_state.get("quantum_lab_results_source")
    is_current_qsvm = (
        isinstance(session_payload, dict)
        and session_payload.get("execution_target") == "simulator"
        and session_source == dataset_source
    )

    if dataset_source == "live":
        default_cohort = "400 / 100 máximo"
        default_load = "79.800 train + 40.000 prueba = 119.800 circuitos"
    else:
        default_cohort = "480 / 120 default"
        default_load = "114.960 train + 57.600 prueba = 172.560 circuitos"

    if not is_current_qsvm:
        return {
            "Enfoque": "QSVM local",
            "Dataset": dataset_label,
            "Estado": "Implementado; sin JSON QSVM vigente",
            "Backend": "Qiskit local ideal",
            "Cohorte train/test": default_cohort,
            "Carga ejecutada": default_load,
            "Accuracy": "n/d",
            "Precision": "n/d",
            "Recall": "n/d",
            "F1": "n/d",
            "Tiempo / uso": "n/d",
            "Evidencia": "Resultado de sesión pendiente",
        }

    train_samples = session_payload.get("train_sample_size", "n/d")
    test_samples = session_payload.get("sample_size", session_payload.get("rows", "n/d"))
    train_circuits = session_payload.get("train_circuit_count")
    test_circuits = session_payload.get("test_circuit_count")
    load = (
        f"{train_circuits} train + {test_circuits} prueba = "
        f"{int(train_circuits) + int(test_circuits)} circuitos"
        if train_circuits is not None and test_circuits is not None
        else default_load
    )
    return {
        "Enfoque": "QSVM local",
        "Dataset": dataset_label,
        "Estado": "Ejecutado en sesión",
        "Backend": "Qiskit local ideal",
        "Cohorte train/test": f"{train_samples} / {test_samples}",
        "Carga ejecutada": load,
        "Accuracy": _format_percent(_metric(session_payload, "accuracy")),
        "Precision": _format_percent(_metric(session_payload, "precision")),
        "Recall": _format_percent(_metric(session_payload, "recall")),
        "F1": _format_percent(_metric(session_payload, "f1_score")),
        "Tiempo / uso": _format_seconds(session_payload.get("execution_time_seconds")),
        "Evidencia": "Sesión Streamlit; no persistida",
    }


def _classical_row(model_data: ModelData) -> dict[str, str]:
    classical = model_data["Modelo clasico"]
    is_real = classical.get("source") == "real"
    return {
        "Enfoque": "Random Forest",
        "Dataset": "CICIDS2017",
        "Estado": "Ejecutado" if is_real else "Pendiente",
        "Backend": "CPU",
        "Cohorte train/test": "152.728 / 38.183",
        "Carga ejecutada": "190.911 muestras válidas",
        "Accuracy": _format_percent(float(classical["accuracy"])) if is_real else "n/d",
        "Precision": _format_percent(float(classical["precision"])) if is_real else "n/d",
        "Recall": _format_percent(float(classical["recall"])) if is_real else "n/d",
        "F1": _format_percent(float(classical["f1_score"])) if is_real else "n/d",
        "Tiempo / uso": "No registrado en el JSON",
        "Evidencia": CLASSICAL_RESULTS_PATH.name,
    }


def _selected_hardware_payloads(dataset_source: str) -> tuple[dict | None, dict | None]:
    if dataset_source == "live":
        return (
            _load_json(QUANTUM_LIVE_IBM_HARDWARE_RESULTS_PATH),
            _load_json(QUANTUM_LIVE_HARDWARE_RESULTS_PATH),
        )
    return (
        _load_json(QUANTUM_IBM_HARDWARE_RESULTS_PATH),
        _load_json(QUANTUM_HARDWARE_RESULTS_PATH),
    )


def render_analysis_tab(
    model_data: ModelData,
    selected_model: str,
    selected_quantum_dataset_source: str,
) -> None:
    dataset_label = (
        "Live" if selected_quantum_dataset_source == "live" else "CICIDS2017"
    )
    ibm_payload, spinq_payload = _selected_hardware_payloads(
        selected_quantum_dataset_source
    )

    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 2px solid #FDB913; margin-bottom: 2rem;">
            <h1 style="margin: 0.3rem 0; font-size: 2.6rem; color: #FFFFFF; font-weight: 900;">Análisis y Síntesis</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_spotlight_panel(
        "Conclusión de la auditoría",
        "El modelo cuántico actual es un Quantum Kernel (QSVM)",
        (
            "El circuito calcula fidelidades entre muestras y genera una matriz de kernel. "
            "La clasificación final se realiza con una SVM clásica de kernel precomputado. "
            "No existen ansatz entrenable, optimizador variacional ni fine-tuning en hardware."
        ),
    )

    st.write("")
    st.markdown("### Carga experimental predeterminada")
    st.caption(
        "Desglose de entrenamiento y prueba. En el clásico se cuentan muestras; "
        "en el QSVM se cuentan circuitos lógicos de fidelidad."
    )
    workload_df = pd.DataFrame(
        [
            {
                "Caso": "Random Forest",
                "Cohorte": "152.728 train / 38.183 prueba",
                "Entrenamiento": "152.728 muestras",
                "Prueba": "38.183 muestras",
                "Total": "190.911 muestras",
                "Motivo": "Split clásico 80/20 después de limpiar CICIDS2017.",
            },
            {
                "Caso": "QSVM local CICIDS",
                "Cohorte": "480 train / 120 prueba",
                "Entrenamiento": "114.960 circuitos",
                "Prueba": "57.600 circuitos",
                "Total": "172.560 circuitos",
                "Motivo": "Triángulo sin diagonal n(n−1)/2 y kernel cruzado m·n.",
            },
            {
                "Caso": "QSVM local Live",
                "Cohorte": "400 train / 100 prueba (máximo)",
                "Entrenamiento": "79.800 circuitos",
                "Prueba": "40.000 circuitos",
                "Total": "119.800 circuitos",
                "Motivo": "Máximo default de 250 muestras por clase y split 80/20.",
            },
            {
                "Caso": "SpinQ · CICIDS2017",
                "Cohorte": "4 train / 4 prueba",
                "Entrenamiento": "10 circuitos",
                "Prueba": "16 circuitos",
                "Total": "26 circuitos",
                "Motivo": "Cohorte CICIDS reducida para limitar el uso del hardware NMR.",
            },
            {
                "Caso": "SpinQ · Live",
                "Cohorte": "4 train / 4 prueba",
                "Entrenamiento": "10 circuitos",
                "Prueba": "16 circuitos",
                "Total": "26 circuitos",
                "Motivo": "Cohorte Live reducida para limitar el uso del hardware NMR.",
            },
            {
                "Caso": "IBM Quantum · CICIDS2017",
                "Cohorte": "4 train / 4 prueba",
                "Entrenamiento": "10 circuitos",
                "Prueba": "16 circuitos",
                "Total": "26 circuitos · 2 jobs",
                "Motivo": "Cohorte CICIDS equivalente a SpinQ y consumo QPU acotado.",
            },
            {
                "Caso": "IBM Quantum · Live",
                "Cohorte": "4 train / 4 prueba",
                "Entrenamiento": "10 circuitos",
                "Prueba": "16 circuitos",
                "Total": "26 circuitos · 2 jobs",
                "Motivo": "Cohorte Live equivalente a SpinQ y consumo QPU acotado.",
            },
        ]
    )
    st.dataframe(workload_df, hide_index=True, width="stretch")
    st.caption(
        "Los shots son repeticiones de medición de cada circuito; no aumentan la "
        "cantidad de circuitos lógicos indicada en la tabla."
    )

    st.write("")
    st.markdown("### Evidencia experimental vigente")
    st.caption(
        "Las filas se construyen desde los JSON actuales. Los artefactos antiguos que "
        "declaran Variational Quantum Classifier se excluyen de esta lectura."
    )
    evidence_rows = [
        _classical_row(model_data),
        _local_simulator_row(selected_quantum_dataset_source),
        _hardware_evidence_row(
            label="SpinQ",
            dataset="CICIDS2017",
            path=QUANTUM_HARDWARE_RESULTS_PATH,
            expected_target="spinq",
        ),
        _hardware_evidence_row(
            label="SpinQ",
            dataset="Live",
            path=QUANTUM_LIVE_HARDWARE_RESULTS_PATH,
            expected_target="spinq",
        ),
        _hardware_evidence_row(
            label="IBM Quantum",
            dataset="CICIDS2017",
            path=QUANTUM_IBM_HARDWARE_RESULTS_PATH,
            expected_target="ibm_quantum",
        ),
        _hardware_evidence_row(
            label="IBM Quantum",
            dataset="Live",
            path=QUANTUM_LIVE_IBM_HARDWARE_RESULTS_PATH,
            expected_target="ibm_quantum",
        ),
    ]
    st.dataframe(pd.DataFrame(evidence_rows), hide_index=True, width="stretch")
    st.warning(
        "Las métricas físicas se calcularon sobre sólo 4 muestras de prueba. Son "
        "evidencia de integración y ejecución, no una estimación generalizable del IDS."
    )

    st.write("")
    st.markdown(f"### Lectura física seleccionada · {dataset_label}")
    physical_cols = st.columns(2)
    with physical_cols[0]:
        if ibm_payload and ibm_payload.get("execution_target") == "ibm_quantum":
            render_spotlight_panel(
                "IBM Quantum",
                str(ibm_payload.get("ibm_backend_name") or "Backend IBM"),
                (
                    "Evidencia fuerte de ejecución: backend, timestamps, matrices, "
                    "identificadores de jobs, uso QPU y diagnósticos quedaron persistidos."
                ),
                meta=[
                    ("Accuracy", _format_percent(_metric(ibm_payload, "accuracy"))),
                    ("F1", _format_percent(_metric(ibm_payload, "f1_score"))),
                    (
                        "Uso QPU",
                        _format_seconds(ibm_payload.get("ibm_total_usage_seconds")),
                    ),
                ],
            )
            noise = ibm_payload.get("quantum_noise", {})
            st.caption(
                "Desviación del kernel vs referencia local: "
                f"media {float(noise.get('mean_absolute_deviation', 0)):.4f} · "
                f"máxima {float(noise.get('max_absolute_deviation', 0)):.4f}."
            )
        else:
            render_spotlight_panel(
                "IBM Quantum",
                "Ejecución pendiente",
                f"No existe un resultado IBM vigente para {dataset_label}.",
            )

    with physical_cols[1]:
        if spinq_payload and spinq_payload.get("execution_target") == "spinq":
            render_spotlight_panel(
                "SpinQ",
                "Triangulum · 3 qubits",
                (
                    "El JSON prueba que el pipeline produjo métricas físicas, pero su "
                    "trazabilidad es menor: no registra task ID de SpinQuasar, timestamp "
                    "interno ni duración total."
                ),
                meta=[
                    ("Accuracy", _format_percent(_metric(spinq_payload, "accuracy"))),
                    ("F1", _format_percent(_metric(spinq_payload, "f1_score"))),
                    ("Circuitos", "10 + 16 = 26"),
                ],
            )
        else:
            render_spotlight_panel(
                "SpinQ",
                "Ejecución pendiente",
                f"No existe un resultado SpinQ vigente para {dataset_label}.",
            )

    st.info(
        "SpinQuasar no es el clasificador: es la capa de operación y gestión del "
        "equipo SpinQ. SpinQit construye y compila los circuitos; SpinQuasar recibe "
        "la tarea por red y controla la ejecución sobre Triangulum; la SVM se entrena "
        "después, de forma clásica, con la matriz de fidelidades."
    )

    st.write("")
    st.markdown("### Limitaciones que condicionan la interpretación")
    limitations_left, limitations_right = st.columns(2)
    with limitations_left:
        st.markdown(
            """
- **Muestra física mínima:** 4 train y 4 test; la accuracy cambia en saltos de 25 puntos porcentuales.
- **No hay ventaja cuántica demostrada:** el objetivo logrado es validar el pipeline híbrido sobre hardware real.
- **Comparación no pareada:** los JSON no guardan índices o hashes de las muestras; IBM y SpinQ no deben presentarse como un head-to-head idéntico.
- **Prevalidación débil:** actualmente alcanza con detectar al menos un verdadero positivo; no garantiza separar correctamente ambas clases.
            """
        )
    with limitations_right:
        st.markdown(
            """
- **Fuga de información:** RobustScaler y el filtro de correlación se ajustan antes del split train/test.
- **Historial sobrescribible:** los archivos `latest` y por qubits se reemplazan en corridas posteriores.
- **SpinQ menos auditable:** no persiste task ID, tiempo, shots ni timestamp; además `spinqit` no está declarado en `requirements.txt`.
- **Código histórico:** todavía existen módulos y JSON VQC, pero no describen el modelo activo del dashboard.
            """
        )

    st.write("")
    st.markdown("---")
    st.markdown("### Síntesis de la investigación")
    synthesis_cols = st.columns(3)
    classical = model_data["Modelo clasico"]
    with synthesis_cols[0]:
        render_info_card(
            "Baseline clásico",
            f"Acc {float(classical['accuracy']):.2%} · F1 {float(classical['f1_score']):.2%}",
            (
                "Random Forest continúa siendo la referencia operativa por escala, "
                "estabilidad y desempeño sobre CICIDS2017."
            ),
        )
    with synthesis_cols[1]:
        render_info_card(
            "Contribución cuántica",
            "Pipeline QSVM verificable",
            (
                "Se implementó el mismo principio de fidelidad en simulador local, "
                "SpinQ e IBM, seguido por una SVM clásica."
            ),
        )
    with synthesis_cols[2]:
        render_info_card(
            "Alcance demostrado",
            "Integración, no superioridad",
            (
                "La evidencia demuestra ejecución física y medición de ruido; no "
                "respalda superioridad predictiva ni ventaja cuántica."
            ),
        )

    st.write("")
    st.markdown(
        """
        <div style="background: rgba(10, 30, 64, 0.65); border-left: 4px solid #FDB913; padding: 1rem 1.25rem; border-radius: 0 10px 10px 0;">
            <p style="color: #FFFFFF; font-weight: 800; margin: 0 0 0.4rem 0;">Conclusión integradora</p>
            <p style="color: #C8D6E5; font-size: 0.95rem; line-height: 1.55; margin: 0;">
                La contribución actual de la tesina es una arquitectura híbrida auditable: Random Forest funciona
                como baseline clásico y el QSVM permite estudiar kernels de fidelidad en un simulador ideal y en
                dos tecnologías físicas diferentes, IBM Quantum y SpinQ Triangulum. Los resultados físicos validan
                la integración técnica, pero su cohorte reducida exige presentar las métricas como evidencia
                exploratoria y no como una demostración de ventaja cuántica.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        "Quantum IDS · Tesina de Licenciatura en Sistemas · "
        "Síntesis construida desde código y artefactos vigentes."
    )
