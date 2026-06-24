from __future__ import annotations

import streamlit as st

from app.dashboard.analytics import (
    build_metrics_dataframe,
    build_quantum_runs_dataframe,
    make_global_comparison_chart,
    make_noise_chart,
    make_time_chart,
)
from app.dashboard.types import ModelData
from app.dashboard.ui import render_info_card, section_header


def render_analysis_tab(model_data: ModelData, selected_model: str, selected_quantum_dataset_source: str) -> None:
    section_header(
        "Comparacion y analisis",
        "Lectura guiada de rendimiento, ruido, tiempos y diferencias entre los enfoques.",
    )
    st.markdown("#### Resumen general")
    if model_data["Modelo cuantico"]["source"] != "real":
        command = (
            f"python -m src.quantum.train_vqc_simulator --dataset-source live --qubits {model_data['Modelo cuantico']['selected_qubits']}"
            if selected_quantum_dataset_source == "live"
            else f"python -m src.quantum.train_vqc_simulator --qubits {model_data['Modelo cuantico']['selected_qubits']}"
        )
        st.info(
            f"Todavia no se entreno el VQC {'live' if selected_quantum_dataset_source == 'live' else 'CICIDS'} "
            f"con {model_data['Modelo cuantico']['selected_qubits']} qubits. Ejecutar: {command}"
        )
    st.plotly_chart(make_global_comparison_chart(model_data, height=380), width="stretch", key="analysis_global_comparison_chart")
    table_df = build_metrics_dataframe(model_data).pivot(index="Modelo", columns="Metrica", values="Valor")
    table_df = table_df[["Accuracy", "Precision", "Recall", "F1-Score"]]
    st.dataframe(table_df.style.format("{:.1%}"), width="stretch")
    st.caption(
        f"El clasico muestra su referencia real. El bloque cuantico refleja la corrida "
        f"{model_data['Modelo cuantico'].get('dataset_source_label', 'CICIDS2017')} "
        f"de {model_data['Modelo cuantico']['selected_qubits']} qubits si existe un resultado guardado."
    )

    st.write("")
    st.markdown("#### Corridas VQC disponibles")
    quantum_runs_df = build_quantum_runs_dataframe(dataset_source=selected_quantum_dataset_source)
    st.dataframe(
        quantum_runs_df.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-Score": "{:.2%}",
                "Tiempo (s)": "{:.2f}",
                "Sample": "{:.0f}",
            },
            na_rep="Sin correr",
        ),
        width="stretch",
    )
    trained_runs = quantum_runs_df[quantum_runs_df["Estado"] == "Entrenado"]
    if not trained_runs.empty:
        best_row = trained_runs.sort_values(["F1-Score", "Accuracy"], ascending=False).iloc[0]
        st.caption(
            f"Mejor corrida VQC disponible en {best_row['Fuente']}: {int(best_row['Qubits'])} qubits "
            f"con F1-score {best_row['F1-Score']:.2%} y tiempo {best_row['Tiempo (s)']:.2f}s."
        )
    else:
        st.info("Todavia no hay corridas VQC disponibles para comparar.")

    st.write("")
    st.markdown("#### Ruido y limites del hardware")
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.plotly_chart(make_noise_chart(model_data, height=350), width="stretch", key="analysis_noise_chart")
    with col2:
        simulated = model_data["Modelo cuantico"]
        hardware = model_data["Hardware cuantico real"]
        render_info_card("Caida de Accuracy", f"{(simulated['accuracy'] - hardware['accuracy']):.1%}", "Perdida al pasar del ideal al hardware real.")
        st.write("")
        render_info_card("Caida de F1-Score", f"{(simulated['f1_score'] - hardware['f1_score']):.1%}", "Perdida general al salir del simulador ideal.")
        st.write("")
        diagnostics = hardware.get("hardware_diagnostics", {})
        limitation_flags = diagnostics.get("limitation_flags") or []
        render_info_card("Backend IBM", str(hardware.get("ibm_backend_name", "Pendiente")), "Procesador cuantico real usado en la validacion IBM.")
        st.write("")
        render_info_card("Alertas del hardware", ", ".join(limitation_flags) if limitation_flags else "Sin flags", "Senales resumidas de cola, ruido o conectividad limitada detectadas en el backend.")
        st.write("")
        hardware_gap_local = hardware.get("hardware_gap_vs_local_subset", {})
        if hardware_gap_local:
            render_info_card(
                "Caida vs local",
                f"Acc {hardware_gap_local.get('accuracy_drop', 0):.1%} | F1 {hardware_gap_local.get('f1_drop', 0):.1%}",
                "Diferencia entre IBM y la misma muestra evaluada localmente con los mismos pesos.",
            )
            st.write("")
        render_info_card("Modelo destacado", model_data[selected_model]["short_label"], "Enfoque activo en la lectura actual del dashboard.")
        if diagnostics:
            st.caption(
                f"T1 medio: {diagnostics.get('avg_t1_us', 'n/d')} us | "
                f"T2 medio: {diagnostics.get('avg_t2_us', 'n/d')} us | "
                f"Pending jobs: {diagnostics.get('pending_jobs', 'n/d')}"
            )

    st.write("")
    st.markdown("#### Costos de tiempo")
    st.plotly_chart(make_time_chart(model_data, height=350), width="stretch", key="analysis_time_chart")
    st.caption("El clasico sigue siendo el mas eficiente; el hardware real conserva el mayor costo temporal.")
