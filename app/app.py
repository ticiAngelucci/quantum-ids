from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.dashboard.data import get_model_data
from app.dashboard.theme import configure_page, inject_css
from app.dashboard.ui import render_header, render_sidebar_controls
from app.dashboard.views import (
    render_analysis_tab,
    render_conclusion_tab,
    render_demo_tab,
    render_lab_tab,
    render_overview_tab,
)


def main() -> None:
    configure_page()
    inject_css()

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "1. Vision general"

    selected_quantum_qubits = st.session_state.get("selected_quantum_qubits", 4)
    selected_quantum_dataset_source = st.session_state.get("selected_quantum_dataset_source", "cicids")

    model_data = get_model_data(
        selected_quantum_qubits=selected_quantum_qubits,
        selected_quantum_dataset_source=selected_quantum_dataset_source,
    )

    render_header(model_data)
    selected_model, selected_quantum_qubits, selected_quantum_dataset_source, current_step = render_sidebar_controls(
        model_data,
        selected_quantum_qubits,
        selected_quantum_dataset_source,
    )

    st.session_state["selected_quantum_qubits"] = selected_quantum_qubits
    st.session_state["selected_quantum_dataset_source"] = selected_quantum_dataset_source
    st.session_state["current_step"] = current_step

    if current_step == "1. Vision general":
        render_overview_tab(model_data, selected_model)
    elif current_step == "2. Probar modelo":
        render_lab_tab(model_data, selected_model, selected_quantum_qubits, selected_quantum_dataset_source)
    elif current_step == "3. Analisis":
        render_analysis_tab(model_data, selected_model, selected_quantum_dataset_source)
    elif current_step == "4. Simulacion":
        render_demo_tab(model_data, selected_model)
    else:
        render_conclusion_tab(model_data, selected_model)


if __name__ == "__main__":
    main()
