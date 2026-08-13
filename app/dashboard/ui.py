from __future__ import annotations

import streamlit as st

from dashboard.constants import ENABLED_MODEL_OPTIONS, SUPPORTED_QUANTUM_QUBITS
from dashboard.types import ModelData, QuantumDatasetSource, SectionName, SidebarSelection


def section_header(title: str, description: str) -> None:
    st.markdown(f"### {title}")
    st.caption(description)
    st.markdown("---")


def render_spotlight_panel(eyebrow: str, title: str, body: str, meta: list[tuple[str, str]] | None = None) -> None:
    with st.container():
        st.caption(eyebrow.upper())
        st.markdown(f"#### {title}")
        st.write(body)
        if meta:
            meta_cols = st.columns(len(meta))
            for col, (label, value) in zip(meta_cols, meta):
                with col:
                    st.caption(label)
                    st.markdown(f"**{value}**")


def render_story_card(step: str, title: str, body: str) -> None:
    with st.container():
        st.caption(step.upper())
        st.markdown(f"**{title}**")
        st.write(body)


def render_info_card(label: str, value: str, help_text: str) -> None:
    with st.container():
        st.caption(label.upper())
        st.markdown(f"### {value}")
        st.write(help_text)


def render_metric_card(label: str, value: float, caption: str) -> None:
    st.metric(label, f"{value * 100:.1f}%")
    st.caption(f"{caption} (Efectividad: {value:.4f})")


def render_header(model_data: ModelData) -> None:
    selected_model = st.session_state.get("selected_model", "Modelo clasico")
    selected_qubits = st.session_state.get("selected_quantum_qubits", 4)
    selected_dataset_source = st.session_state.get("selected_quantum_dataset_source", "cicids")
    selected_platform = st.session_state.get("selected_quantum_execution_target", "simulator")
    model = model_data[selected_model]
    
    dataset_label = "Tráfico en vivo (Simulador)" if selected_dataset_source == "live" else "Base histórica (CICIDS)"
    platform_label = "Hardware Real (SpinQ / IBM)" if selected_platform == "ibm_validate" else "Simulador Local"

    st.markdown(
        f"""
        <div style="padding: 1rem 0; border-bottom: 2px solid #FDB913; margin-bottom: 1.5rem;">
            <span style="color: #FDB913; font-weight: 800; font-size: 0.85rem; text-transform: uppercase;">Ciberseguridad & Computación Cuántica</span>
            <h1 style="margin: 0.2rem 0; font-size: 2.2rem; color: #FFFFFF;">Panel de Control · Quantum IDS</h1>
            <p style="color: #A0B3C6; margin: 0; font-size: 1rem;">
                Modelo activo: <b>{model['short_label']}</b> | Entorno: <b>{dataset_label}</b> | Qubits: <b>{selected_qubits}</b>
            </p>
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
        
        # 1. QUANTUM IDS (Primero)
        st.markdown("## 🛡️ Quantum IDS")
        st.caption("Detección de intrusiones con algoritmos cuánticos.")
        st.markdown("---")

        # 2. CONFIGURACIÓN (Segundo)
        st.markdown("### ⚙️ Configuración")
        available_models = [model_name for model_name in ENABLED_MODEL_OPTIONS if model_name in model_data]
        default_model = st.session_state.get("selected_model", "Modelo clasico")
        if default_model not in available_models:
            default_model = "Modelo clasico"
            
        selected_model = st.selectbox(
            "Seleccionar Modelo",
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
                "Número de Qubits",
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

        st.markdown("---")

        # 3. MENÚ (Tercero)
        st.markdown("### 🧭 Navegación")
        current_step = st.radio(
            "Sección",
            options=section_options,
            key="journey_radio",
            index=section_options.index(st.session_state.get("current_step", current_step))
            if st.session_state.get("current_step", current_step) in section_options
            else 0,
            label_visibility="collapsed",
        )
        st.session_state["current_step"] = current_step

        st.markdown("---")

        # 4. RESUMEN (Cuarto - Título cambiado y prolijo)
        model = model_data[selected_model]
        st.markdown("### 📊 Estado del Modelo")
        st.write(
            f"• Modelo: **{model['short_label']}**\n\n"
            f"• Precisión: **{model['accuracy']:.1%}**\n\n"
            f"• Enfoque: **{model['description']}**"
        )

    return SidebarSelection(
        selected_model=selected_model,
        selected_quantum_qubits=selected_quantum_qubits,
        selected_quantum_dataset_source=selected_quantum_dataset_source,
        current_step=current_step,
    )