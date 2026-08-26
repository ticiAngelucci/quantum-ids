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


def render_quantum_noise_card(noise_data: dict | None) -> None:
    """Muestra la desviación del kernel físico respecto del kernel ideal."""
    if noise_data:
        mean_noise = float(noise_data.get("mean_absolute_deviation", 0.0))
        max_noise = float(noise_data.get("max_absolute_deviation", 0.0))
        value_html = f"{mean_noise * 100:.2f}%"
        detail_html = f"Desviación máxima observada: <b>{max_noise * 100:.2f}%</b>."
    else:
        value_html = "Pendiente"
        detail_html = "Se calculará al finalizar la próxima ejecución física completa."

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(10, 30, 64, 0.96), rgba(14, 42, 88, 0.96)); border: 1px solid rgba(253, 185, 19, 0.55); border-left: 5px solid #FDB913; border-radius: 14px; padding: 1.2rem 1.4rem; margin: 1rem 0 1.4rem 0;">
            <div style="color: #FDB913; font-size: 0.76rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Ruido cuántico estimado</div>
            <div style="color: #FFFFFF; font-size: 2rem; font-weight: 900; margin: 0.15rem 0;">{value_html}</div>
            <div style="color: #D9E4F2; font-size: 0.88rem;">{detail_html}</div>
            <div style="color: #A0B3C6; font-size: 0.76rem; line-height: 1.45; margin-top: 0.65rem;">
                El ruido cuántico son pequeñas alteraciones no deseadas que modifican el resultado ideal de un circuito.
                Esta estimación compara el kernel ideal con el medido por el backend físico e incluye efectos del hardware,
                variación estadística de los shots y diferencias de ejecución; no representa una tasa física pura del dispositivo.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(model_data: ModelData) -> None:
    selected_model = st.session_state.get("selected_model", "Modelo clasico")
    selected_qubits = st.session_state.get("selected_quantum_qubits", 4)
    selected_dataset_source = st.session_state.get("selected_quantum_dataset_source", "cicids")
    model = model_data[selected_model]
    
    dataset_label = "Tráfico en vivo (Simulador)" if selected_dataset_source == "live" else "Base histórica (CICIDS)"
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
        "4. Análisis y Síntesis",
    ]
    with st.sidebar:
        current_step = st.session_state.get("journey_radio", st.session_state.get("current_step", "1. Resumen"))
        
        # 1. QUANTUM IDS (Primero)
        st.markdown("## Quantum IDS")
        st.caption("Detección de intrusiones con algoritmos cuánticos.")
        st.markdown("---")

        # 2. CONFIGURACIÓN (Segundo)
        st.markdown("### Configuración")
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

            fixed_three_qubit_hardware_selected = (
                current_step == "2. Experimentar"
                and st.session_state.get("quantum_execution_target_radio")
                in {"spinq", "ibm_quantum"}
            ) or (
                current_step == "3. Live"
                and st.session_state.get("live_quantum_execution_target")
                in {"spinq", "ibm_quantum"}
            )
            if fixed_three_qubit_hardware_selected:
                selected_quantum_qubits = 3
                st.session_state["selected_quantum_qubits"] = 3
                st.session_state["quantum_results_selectbox"] = 3

            qubit_options = list(SUPPORTED_QUANTUM_QUBITS)
            widget_qubits = st.session_state.get(
                "quantum_results_selectbox",
                selected_quantum_qubits,
            )
            if widget_qubits not in qubit_options:
                widget_qubits = selected_quantum_qubits
            st.session_state["quantum_results_selectbox"] = widget_qubits

            chosen_qubits = st.selectbox(
                "Número de Qubits",
                options=qubit_options,
                key="quantum_results_selectbox",
                disabled=fixed_three_qubit_hardware_selected,
                help=(
                    "Las evaluaciones físicas acotadas de SpinQ e IBM Quantum "
                    "utilizan exactamente 3 qubits."
                    if fixed_three_qubit_hardware_selected
                    else None
                ),
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
            st.session_state["journey_radio"] = current_step

        st.markdown("---")

        # 3. MENÚ (Tercero)
        st.markdown("### Navegación")
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
        st.markdown("### Estado del Modelo")
        if selected_model == "Modelo cuantico":
            workload_explanation_label = "¿Por qué esta cantidad de circuitos?"
            active_target = (
                st.session_state.get("live_quantum_execution_target", "simulator")
                if current_step == "3. Live"
                else st.session_state.get("quantum_execution_target_radio", "simulator")
            )
            target_label = {
                "simulator": "Simulador Local Qiskit",
                "ibm_quantum": "IBM Quantum Cloud",
                "spinq": "SpinQ Triangulum",
            }.get(active_target, str(active_target))

            if active_target == "simulator":
                if current_step == "3. Live":
                    circuit_summary = (
                        "79.800 entrenamiento + 40.000 prueba = 119.800 "
                        "(máximo default)"
                    )
                    circuit_explanation = """
**Simulación local Live — configuración predeterminada máxima**

El flujo toma como máximo 500 muestras balanceadas: 250 benignas y 250 ataques. Con el split predeterminado 80/20 quedan 400 muestras de entrenamiento y 100 de prueba.

- Entrenamiento: `400·399/2 = 79.800` circuitos.
- Prueba: `100·400 = 40.000` circuitos.
- Total: `119.800` circuitos lógicos de fidelidad.

Si el dataset Live tiene menos de 250 muestras por clase, la cantidad real será menor. Los *shots* no son circuitos adicionales: son repeticiones de medición.
                    """
                else:
                    circuit_summary = (
                        "114.960 entrenamiento + 57.600 prueba = 172.560 "
                        "(default)"
                    )
                    circuit_explanation = """
**Simulación local CICIDS2017 — configuración predeterminada**

El flujo toma 600 muestras balanceadas: 300 benignas y 300 ataques. Con el split predeterminado 80/20 quedan 480 muestras de entrenamiento y 120 de prueba.

- Entrenamiento: `480·479/2 = 114.960` circuitos.
- Prueba: `120·480 = 57.600` circuitos.
- Total: `172.560` circuitos lógicos de fidelidad.

La diagonal se fija analíticamente en 1 y no se ejecuta. Si la limpieza descarta filas inválidas, la cantidad real puede ser ligeramente menor. Los *shots* no son circuitos adicionales: son repeticiones de medición.
                    """
            elif active_target == "spinq":
                circuit_summary = "10 entrenamiento + 16 prueba = 26"
                circuit_explanation = """
**SpinQ Triangulum**

Se usan 4 muestras de entrenamiento y 4 de prueba. En hardware se mide el triángulo superior incluyendo la diagonal: `4·5/2 = 10` circuitos de entrenamiento. Luego se calculan `4·4 = 16` circuitos de prueba. La reducción a 26 circuitos permite controlar el tiempo de uso del equipo físico. La prueba rápida de conexión ejecuta solamente 1 circuito.

Los *shots* no son circuitos adicionales: son repeticiones de medición de cada circuito.
                """
            else:
                circuit_summary = "10 entrenamiento + 16 prueba = 26"
                circuit_explanation = """
**IBM Quantum**

Se usan 4 muestras de entrenamiento y 4 de prueba. El kernel de entrenamiento mide el triángulo superior incluyendo la diagonal: `4·5/2 = 10` circuitos. El kernel de prueba compara las 4 muestras de prueba contra las 4 de entrenamiento: `4·4 = 16` circuitos. Los 26 circuitos se agrupan en 2 jobs —uno de entrenamiento y otro de prueba— para controlar el uso QPU.

Los *shots* no son circuitos adicionales: son repeticiones de medición de cada circuito.
                """

            st.write(
                f"• Modelo: **QSVM (Quantum Kernel)**\n\n"
                f"• Entorno: **{target_label}**\n\n"
                f"• Qubits: **{selected_quantum_qubits}**\n\n"
                f"• Circuitos: **{circuit_summary}**\n\n"
                f"• Accuracy de referencia: **{model['accuracy']:.1%}**"
            )
        else:
            workload_explanation_label = "¿Por qué esta carga predeterminada?"
            circuit_explanation = """
**Enfoque tradicional — Random Forest**

La configuración predeterminada parte de 190.911 registros válidos de CICIDS2017 y utiliza un split 80/20.

- Entrenamiento: `152.728` muestras.
- Prueba: `38.183` muestras.
- Total procesado: `190.911` muestras.

Todo se ejecuta en CPU mediante Random Forest; los circuitos cuánticos no aplican en este caso.
            """
            st.write(
                "• Modelo: **Random Forest**\n\n"
                "• Enfoque: **Baseline clásico supervisado**\n\n"
                "• Carga default: **152.728 muestras entrenamiento + "
                "38.183 prueba = 190.911**\n\n"
                f"• Accuracy de referencia: **{model['accuracy']:.1%}**"
            )

        with st.expander(workload_explanation_label, expanded=False):
            st.markdown(circuit_explanation)

    return SidebarSelection(
        selected_model=selected_model,
        selected_quantum_qubits=selected_quantum_qubits,
        selected_quantum_dataset_source=selected_quantum_dataset_source,
        current_step=current_step,
    )
