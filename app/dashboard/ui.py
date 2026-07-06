from __future__ import annotations

import streamlit as st

from dashboard.constants import ENABLED_MODEL_OPTIONS, SUPPORTED_QUANTUM_QUBITS
from dashboard.types import ModelData, QuantumDatasetSource, SectionName, SidebarSelection


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


def render_header(model_data: ModelData) -> None:
    _ = model_data
    st.markdown(
        """
        <section class="hero">
            <h1>Quantum IDS Dashboard</h1>
            <div class="badge-row">
                <span class="badge accent">Tesis</span>
                <span class="badge">IDS</span>
                <span class="badge">QML</span>
                <span class="badge">NISQ</span>
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


def _reset_quantum_lab_state() -> None:
    st.session_state.pop("quantum_lab_results", None)
    st.session_state.pop("quantum_lab_results_qubits", None)
    st.session_state.pop("quantum_lab_results_source", None)


def render_sidebar_controls(
    model_data: ModelData,
    selected_quantum_qubits: int,
    selected_quantum_dataset_source: QuantumDatasetSource,
) -> SidebarSelection:
    all_section_options: list[SectionName] = [
        "1. Resumen",
        "2. Experimentar",
        "3. Live",
        "4. Analisis",
        "5. Conclusiones",
    ]
    with st.sidebar:
        current_step = st.session_state.get("journey_radio", st.session_state.get("current_step", "1. Resumen"))
        config_container = st.container()
        section_container = st.container()
        st.markdown("## Quantum IDS")
        st.caption(
            "Esta app compara dos caminos para detectar trafico anomalo en red: uno clasico y otro cuantico. "
            "La idea es mostrar resultados, limites y valor experimental de cada enfoque en un lenguaje claro."
        )

        with config_container:
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

            if selected_model == "Modelo clasico" and current_step == "3. Live":
                current_step = "2. Experimentar"
                st.session_state["journey_radio"] = current_step
                st.session_state["current_step"] = current_step

            if selected_model == "Modelo cuantico":
                if current_step == "2. Experimentar":
                    selected_quantum_dataset_source = "cicids"
                    st.session_state["selected_quantum_dataset_source"] = "cicids"
                elif current_step == "3. Live":
                    selected_quantum_dataset_source = "live"
                    st.session_state["selected_quantum_dataset_source"] = "live"
                else:
                    selected_quantum_dataset_source = st.session_state.get(
                        "selected_quantum_dataset_source",
                        selected_quantum_dataset_source,
                    )
                    st.session_state["selected_quantum_dataset_source"] = selected_quantum_dataset_source

                chosen_qubits = st.selectbox(
                    "Cantidad de qubits",
                    options=list(SUPPORTED_QUANTUM_QUBITS),
                    index=list(SUPPORTED_QUANTUM_QUBITS).index(selected_quantum_qubits),
                    key="quantum_results_selectbox",
                )
                if chosen_qubits != selected_quantum_qubits:
                    st.session_state["selected_quantum_qubits"] = chosen_qubits
                    _reset_quantum_lab_state()
                    st.rerun()
                selected_quantum_qubits = chosen_qubits
                st.session_state["selected_quantum_qubits"] = chosen_qubits

        section_options = (
            all_section_options
            if selected_model == "Modelo cuantico"
            else [option for option in all_section_options if option != "3. Live"]
        )
        if current_step not in section_options:
            current_step = section_options[0]
            st.session_state["current_step"] = current_step

        with section_container:
            st.markdown("---")
            st.markdown("### Seccion")
            current_step = st.radio(
                "Seccion",
                options=section_options,
                key="journey_radio",
                index=section_options.index(st.session_state.get("current_step", current_step))
                if st.session_state.get("current_step", current_step) in section_options
                else 0,
                label_visibility="collapsed",
            )
            st.session_state["current_step"] = current_step

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

    return SidebarSelection(
        selected_model=selected_model,
        selected_quantum_qubits=selected_quantum_qubits,
        selected_quantum_dataset_source=selected_quantum_dataset_source,
        current_step=current_step,
    )
