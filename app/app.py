from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from dashboard.data import get_model_data
from dashboard.theme import configure_page, inject_css
from dashboard.types import SidebarSelection
from dashboard.ui import render_header, render_sidebar_controls
from dashboard.views import (
    render_analysis_tab,
    render_conclusion_tab,
    render_lab_tab,
    render_live_tab,
    render_overview_tab,
)


def main() -> None:
    configure_page()
    inject_css()

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = "1. Resumen"

    selected_quantum_qubits = st.session_state.get("selected_quantum_qubits", 4)
    selected_quantum_dataset_source = st.session_state.get("selected_quantum_dataset_source", "cicids")

    model_data = get_model_data(
        selected_quantum_qubits=selected_quantum_qubits,
        selected_quantum_dataset_source=selected_quantum_dataset_source,
    )

    render_header(model_data)
    selection: SidebarSelection = render_sidebar_controls(
        model_data,
        selected_quantum_qubits,
        selected_quantum_dataset_source,
    )

    st.session_state["selected_quantum_qubits"] = selection.selected_quantum_qubits
    st.session_state["selected_quantum_dataset_source"] = selection.selected_quantum_dataset_source
    st.session_state["current_step"] = selection.current_step

    if selection.current_step == "1. Resumen":
        render_overview_tab(model_data, selection.selected_model)
    elif selection.current_step == "2. Experimentar":
        render_lab_tab(
            model_data,
            selection.selected_model,
            selection.selected_quantum_qubits,
            selection.selected_quantum_dataset_source,
        )
    elif selection.current_step == "3. Live":
        render_live_tab(model_data, selection.selected_quantum_qubits)
    elif selection.current_step == "4. Analisis":
        render_analysis_tab(model_data, selection.selected_model, selection.selected_quantum_dataset_source)
    else:
        render_conclusion_tab(model_data, selection.selected_model)


if __name__ == "__main__":
    main()
