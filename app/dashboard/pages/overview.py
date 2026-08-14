from __future__ import annotations

import numpy as np
import streamlit as st

from dashboard.analytics import build_quantum_runs_dataframe, make_confusion_chart, make_global_comparison_chart
from dashboard.types import ModelData
from dashboard.ui import render_info_card, render_metric_card, render_spotlight_panel, section_header


def render_overview_tab(model_data: ModelData, selected_model: str) -> None:
    # --- HEADER PRINCIPAL ---
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.5rem 0; border-bottom: 2px solid #FDB913; margin-bottom: 2rem;">
            <span style="color: #FDB913; font-weight: 800; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;">Tesina de Licenciatura en Sistemas</span>
            <h1 style="margin: 0.3rem 0; font-size: 2.6rem; color: #FFFFFF; font-weight: 900;">Quantum IDS</h1>
            <p style="color: #A0B3C6; margin: 0; font-size: 1.1rem; line-height: 1.5;">
                Plataforma de investigación que contrasta los sistemas de detección de intrusiones en redes (IDS) 
                basados en <b>computación clásica (Random Forest)</b> frente a arquitecturas de <b>Machine Learning Cuántico (QSVM)</b>. 
                La hipótesis central busca demostrar cómo la proyección de datos a espacios de Hilbert permite resolver problemas 
                complejos de ciberseguridad, explotando la naturaleza probabilística de la computación cuántica.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SECCIÓN: LA HIPÓTESIS Y LA DIFERENCIA CLAVE ---
    st.markdown("### Hipótesis y Diferencia de Paradigmas")
    st.caption("El fundamento técnico que separa ambos enfoques de procesamiento.")
    
    col_h1, col_h2 = st.columns(2)
    
    with col_h1:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.5rem; height: 100%;">
                <span style="color: #FDB913; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Enfoque Tradicional (Clásico)</span>
                <h3 style="color: #FFFFFF; font-size: 1.3rem; margin: 0.4rem 0 0.8rem 0;">Determinista & Tabular</h3>
                <p style="color: #C8D6E5; font-size: 0.95rem; line-height: 1.5; margin: 0;">
                    La computación clásica procesa los flujos de red evaluando variables de manera secuencial o mediante árboles de decisión (como Random Forest). Cada regla o nodo toma decisiones deterministas o basadas en frecuencias fijas sobre atributos estáticos.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col_h2:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.5rem; height: 100%;">
                <span style="color: #FDB913; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Enfoque Cuántico (QSVM / RMN)</span>
                <h3 style="color: #FFFFFF; font-size: 1.3rem; margin: 0.4rem 0 0.8rem 0;">Probabilístico & Amplitudes</h3>
                <p style="color: #C8D6E5; font-size: 0.95rem; line-height: 1.5; margin: 0;">
                    A diferencia de medir una sola vez y descartar, <b>la computación cuántica explota superposiciones y entrelazamiento</b>: el circuito se ejecuta múltiples veces (shots) para extraer una <b>distribución de probabilidades y calcular un valor promedio</b>. Esto captura correlaciones ocultas que el método clásico pasa por alto.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("---")

    # --- SECCIÓN: EJES DE COMPARACIÓN DIRECTA ---
    st.markdown("### Ejes de Comparación")
    st.caption("Qué se está contrastando analíticamente a lo largo de la investigación.")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.4rem; height: 100%;">
                <span style="color: #FDB913; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Contraste 01</span>
                <h4 style="color: #FFFFFF; font-size: 1.15rem; margin: 0.3rem 0 0.6rem 0;">Modelos: Clásico vs. Cuántico</h4>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.45; margin: 0;">
                    <b>Se compara:</b> El rendimiento de un clasificador tabular tradicional (<b>Random Forest</b>) frente al clasificador de Kernel Cuántico (<b>QSVM</b>), evaluando cuál detecta mejor las anomalías de red bajo las mismas métricas.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col_p2:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.4rem; height: 100%;">
                <span style="color: #FDB913; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Contraste 02</span>
                <h4 style="color: #FFFFFF; font-size: 1.15rem; margin: 0.3rem 0 0.6rem 0;">Entornos: Dataset vs. Live</h4>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.45; margin: 0;">
                    <b>Se compara:</b> La estabilidad de los modelos al operar sobre un entorno estático de referencia (<b>CICIDS2017</b>) frente a un escenario dinámico con <b>captura de tráfico en tiempo real por ventanas</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col_p3:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.3); border-radius: 14px; padding: 1.4rem; height: 100%;">
                <span style="color: #FDB913; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Contraste 03</span>
                <h4 style="color: #FFFFFF; font-size: 1.15rem; margin: 0.3rem 0 0.6rem 0;">Infraestructura: Simulación vs. Física</h4>
                <p style="color: #C8D6E5; font-size: 0.9rem; line-height: 1.45; margin: 0;">
                    <b>Se compara:</b> El comportamiento de los circuitos ideales ejecutados en un <b>simulador local</b> frente al impacto del ruido real en el <b>hardware físico de Resonancia Magnética Nuclear (SpinQ)</b>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("---")

    # --- RECUPERACIÓN DE DATOS (CONEXIÓN DINÁMICA CON EL LABORATORIO) ---
    classical = model_data["Modelo clasico"]
    quantum = model_data["Modelo cuantico"]
    hardware = model_data["Hardware cuantico real"]
    
    # Si el usuario corrió una prueba clásica en el laboratorio, pisamos los valores base con los reales
    lab_results = st.session_state.get("lab_results")
    if lab_results and "metrics" in lab_results:
        classical["accuracy"] = lab_results["metrics"]["accuracy"]
        classical["f1_score"] = lab_results["metrics"]["f1_score"]

    # Una corrida de simulador actualiza QSVM; una corrida SpinQ actualiza
    # exclusivamente la referencia de hardware físico.
    quantum_lab_results = st.session_state.get("quantum_lab_results")
    quantum_qubits_used = st.session_state.get("quantum_lab_results_qubits", model_data["Modelo cuantico"]["selected_qubits"])
    if quantum_lab_results and "metrics" in quantum_lab_results:
        runtime_metrics = quantum_lab_results["metrics"]
        runtime_target = quantum_lab_results.get("execution_target", "simulator")
        destination = hardware if runtime_target == "spinq" else quantum
        destination.update(
            {
                "accuracy": runtime_metrics["accuracy"],
                "precision": runtime_metrics["precision"],
                "recall": runtime_metrics["recall"],
                "f1_score": runtime_metrics["f1_score"],
                "confusion_matrix": np.array(
                    quantum_lab_results["confusion_matrix"]
                ),
                "source": "real",
                "source_label": "Resultado real",
            }
        )
        if runtime_target == "spinq":
            destination["quantum_noise"] = quantum_lab_results.get(
                "quantum_noise"
            )

    selected_qubits = quantum_qubits_used
    
    st.markdown("### Rendimiento")
    st.caption("Resultados obtenidos (actualizados con las ejecuciones recientes del laboratorio o valores base de referencia).")

    comp_cols = st.columns(3)
    with comp_cols[0]:
        render_info_card("Enfoque Clásico", f"{classical['accuracy'] * 100:.2f}% Acc", f"Baseline Random Forest\n\nF1-Score: {classical['f1_score'] * 100:.2f}%")
    with comp_cols[1]:
        render_info_card("Enfoque QSVM", f"{quantum['accuracy'] * 100:.2f}% Acc", f"Fidelity Quantum Kernel ({selected_qubits}q)\n\nF1-Score: {quantum['f1_score'] * 100:.2f}%")
    with comp_cols[2]:
        hardware_noise = hardware.get("quantum_noise") or {}
        hardware_noise_text = (
            f" | Ruido est.: {hardware_noise['mean_absolute_deviation'] * 100:.2f}%"
            if "mean_absolute_deviation" in hardware_noise
            else ""
        )
        render_info_card(
            "Hardware Físico RMN",
            f"{hardware['accuracy'] * 100:.2f}% Acc",
            (
                "Validación en SpinQ\n\n"
                f"F1-Score: {hardware['f1_score'] * 100:.2f}% | "
                f"Muestra: {hardware.get('sample_size') or 0} registros"
                f"{hardware_noise_text}"
            ),
        )

    st.write("")
    
    # --- SECCIÓN EXPLICATIVA DE LA GRÁFICA ---
    st.markdown(
        """
        <div style="background: rgba(10, 30, 64, 0.6); border-left: 4px solid #FDB913; padding: 0.9rem 1.2rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
            <p style="color: #E2E8F0; font-size: 0.92rem; margin: 0; line-height: 1.4;">
                <b>Guía de lectura:</b> Cada bloque vertical agrupa las métricas de evaluación (Accuracy, Precision, Recall y F1-Score).
                Permite contrastar visualmente la diferencia de rendimiento entre el modelo tabular tradicional, el simulador cuántico ideal 
                y el desgaste de rendimiento provocado por el ruido térmico en el equipo físico de RMN (SpinQ).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.plotly_chart(make_global_comparison_chart(model_data), width="stretch", key="landing_global_comparison_chart")

    st.write("")
    st.markdown("---")

    # --- SECCIÓN: GLOSARIO TÉCNICO DE MÉTRICAS ---
    st.markdown("### ¿Qué significan las métricas evaluadas?")
    st.caption("Desglose conceptual de los indicadores utilizados en la consola para medir el éxito del IDS.")

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.2); border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <h5 style="color: #FDB913; margin: 0 0 0.3rem 0;">Accuracy (Precisión Global)</h5>
                <p style="color: #C8D6E5; font-size: 0.9rem; margin: 0; line-height: 1.4;">
                    Porcentaje total de decisiones acertadas (tanto tráfico normal como ataques bien clasificados) sobre el total de muestras analizadas.
                </p>
            </div>
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.2); border-radius: 12px; padding: 1.2rem;">
                <h5 style="color: #FDB913; margin: 0 0 0.3rem 0;">Precision (Confiabilidad de Alertas)</h5>
                <p style="color: #C8D6E5; font-size: 0.9rem; margin: 0; line-height: 1.4;">
                    Indica qué tan seguro es que un evento marcado como "ataque" lo sea realmente. Evita los falsos positivos (alertas falsas a los analistas).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            """
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.2); border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;">
                <h5 style="color: #FDB913; margin: 0 0 0.3rem 0;">Recall (Tasa de Detección)</h5>
                <p style="color: #C8D6E5; font-size: 0.9rem; margin: 0; line-height: 1.4;">
                    Mide la capacidad del modelo para encontrar y atrapar todos los ataques reales que ocurrieron, evitando que amenazas pasen desapercibidas (falsos negativos).
                </p>
            </div>
            <div style="background: rgba(10, 30, 64, 0.85); border: 1px solid rgba(253, 185, 19, 0.2); border-radius: 12px; padding: 1.2rem;">
                <h5 style="color: #FDB913; margin: 0 0 0.3rem 0;">F1-Score (Equilibrio General)</h5>
                <p style="color: #C8D6E5; font-size: 0.9rem; margin: 0; line-height: 1.4;">
                    Es la <b>media armónica entre Precision y Recall</b>. Es la métrica clave en ciberseguridad porque resume en un solo valor si el modelo detecta bien los ataques sin generar falsas alarmas.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- FOOTER ACADÉMICO ---
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 1.5rem 0; color: #A0B3C6; font-size: 0.9rem;">
            <p style="margin: 0; color: #FFFFFF; font-weight: 700;">Quantum IDS · Tesina de Licenciatura en Sistemas</p>
            <p style="margin: 0.3rem 0 0 0;">Autor: <b>Ticiana Angelucci</b> | Universidad Champagnat | 2026</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
