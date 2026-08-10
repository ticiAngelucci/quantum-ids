from __future__ import annotations

import streamlit as st

from dashboard.constants import ENABLED_MODEL_OPTIONS, SUPPORTED_QUANTUM_QUBITS
from dashboard.types import ModelData, QuantumDatasetSource, SectionName, SidebarSelection


def section_header(title: str, description: str) -> None:
    st.subheader(title)
    st.caption(description)


def render_spotlight_panel(eyebrow: str, title: str, body: str, meta: list[tuple[str, str]] | None = None) -> None:
    with st.container(border=True):
        st.caption(eyebrow)
        st.markdown(f"#### {title}")
        st.write(body)
        if meta:
            meta_cols = st.columns(len(meta))
            for col, (label, value) in zip(meta_cols, meta):
                with col:
                    st.caption(label)
                    st.markdown(f"**{value}**")


def render_story_card(step: str, title: str, body: str) -> None:
    with st.container(border=True):
        st.caption(step)
        st.markdown(f"**{title}**")
        st.write(body)


def render_info_card(label: str, value: str, help_text: str) -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**{value}**")
        st.write(help_text)


def render_metric_card(label: str, value: float, caption: str) -> None:
    st.metric(label, f"{value * 100:.2f}%")
    st.caption(f"{caption} · valor crudo: {value:.4f}")


def render_header(model_data: ModelData) -> None:
    selected_model = st.session_state.get("selected_model", "Modelo clasico")
    selected_qubits = st.session_state.get("selected_quantum_qubits", 4)
    selected_dataset_source = st.session_state.get("selected_quantum_dataset_source", "cicids")
    selected_platform = st.session_state.get("selected_quantum_execution_target", "simulator")
    model = model_data[selected_model]
    dataset_label = "Live simulador" if selected_dataset_source == "live" else "CICIDS2017"
    platform_label = "IBM validate" if selected_platform == "ibm_validate" else "Simulador local"
    status_label = model["source_label"]

    left, right = st.columns([1.6, 1])
    with left:
        st.caption("Quantum Machine Learning aplicado a deteccion de anomalias en red")
        st.title("Quantum IDS Dashboard")
        st.caption("Consola tecnica para seguimiento de baseline clasico, VQC en simulacion y validacion sobre hardware real.")
    with right:
        st.caption(
            f"Dataset · {dataset_label}\n\n"
            f"Modelo · {model['short_label']}\n\n"
            f"Qubits · {selected_qubits}\n\n"
            f"Plataforma · {platform_label}\n\n"
            f"Estado · {status_label}"
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
        st.caption("Consola de experimentacion QML para ciberseguridad.")

        with config_container:
            st.markdown("---")
            st.markdown("### Experimento")
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

            active_dataset = "Live simulador" if selected_quantum_dataset_source == "live" else "CICIDS2017"
            active_platform = st.session_state.get("selected_quantum_execution_target", "simulator")
            active_platform_label = "IBM validate" if active_platform == "ibm_validate" else "Simulador local"
            model = model_data[selected_model]
            st.caption("Estado tecnico")
            st.write(
                f"Dataset: **{active_dataset}**\n\n"
                f"Plataforma: **{active_platform_label}**\n\n"
                f"Qubits: **{selected_quantum_qubits if selected_model == 'Modelo cuantico' else '-'}**\n\n"
                f"Estado: **{model['source_label']}**"
            )

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
            st.markdown("### Navegacion")
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

        st.markdown("---")
        model = model_data[selected_model]
        st.caption("Proyecto")
        st.write(
            "Dataset: **ok**\n\n"
            "Baseline: **ok**\n\n"
            "VQC: **ok**\n\n"
            "Live: **parcial**\n\n"
            "SpinQ: **pendiente**\n\n"
            "Braket: **futuro**"
        )
        st.markdown("---")
        st.caption("Lectura rapida")
        st.write(
            f"Modelo: **{model['short_label']}**\n\n"
            f"Metrica actual: **{model['accuracy']:.1%} acc**\n\n"
            f"Referencia: **{model['description']}**"
        )

    return SidebarSelection(
        selected_model=selected_model,
        selected_quantum_qubits=selected_quantum_qubits,
        selected_quantum_dataset_source=selected_quantum_dataset_source,
        current_step=current_step,
    )
