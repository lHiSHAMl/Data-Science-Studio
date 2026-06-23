"""
visualizations.py  –  Enhanced version
----------------------------------------
Changes vs. original:
  • Global transformations UI moved into a collapsible sidebar-style expander
    so it doesn't dominate the screen
  • apply_global_transformations is the single source of truth; the page no
    longer re-imports visualization_execution helpers directly
  • Statistical chart types (Correlation Matrix, Distribution Overview, KDE,
    QQ Plot) flow through the same pipeline as regular charts
  • Better session-state guard: never crashes when pipeline is missing
"""

import streamlit as st
import pandas as pd
from constants import DataManager
from viz_lib import visualization_ui, visualization_save, visualization_execution

data_manager = DataManager()


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run():
    # ---- Session state ----
    _init_session_state()

    st.markdown(data_manager.label_style, unsafe_allow_html=True)
    st.markdown("""
    <div class="custom-container">
        <h2 class="custom-header">📊 Interactive Visualizations Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)

    # ---- Save-file banner ----
    if st.session_state.get("save_file_upload"):
        st.info(f"📁 Project save file: {st.session_state.save_file_upload}")
    else:
        st.warning("⚠️ No save file configured. Go to Transform page to set a save file path.")

    # ---- Data guard ----
    if st.session_state.get("load") or st.session_state.get("df_original") is None:
        st.warning("Please upload data first in the Transform page.")
        return

    # ---- Global transformations (collapsible) ----
    with st.expander("🌐 Global Data Transformations  –  applied to ALL visualizations", expanded=False):
        st.info("Transformations selected here stack on top of the original data "
                "and are inherited by every chart on this page.")

        available = ["original"] + [
            s["name"] for s in st.session_state.pipeline.get("transformations", [])
        ]
        global_transformations = st.multiselect(
            "Select global transformations:",
            options=available,
            default=st.session_state.global_transformations,
            key="global_transformations_selector",
        )
        st.session_state.global_transformations = global_transformations

        global_data = apply_global_transformations()
        col_info, col_shape = st.columns(2)
        with col_info:
            st.caption(f"Rows: {global_data.shape[0]}  |  Columns: {global_data.shape[1]}")
        with col_shape:
            st.caption(f"Active steps: {', '.join(global_transformations)}")

        with st.expander("📋 Preview first 10 rows"):
            st.dataframe(global_data.head(10))

    # ---- Main layout ----
    col_creator, col_controls = st.columns([3, 1])

    with col_creator:
        visualization_ui.show_visualization_creator(st.session_state.df_original)
        visualization_ui.show_edit_modal(st.session_state.df_original)

    with col_controls:
        visualization_ui.show_dashboard_controls()

    # ---- Dashboard ----
    st.markdown("---")
    st.subheader("📈 Visualization Dashboard")
    visualization_ui.show_dashboard(st.session_state.df_original)

    # ---- Export ----
    st.markdown("---")
    visualization_save.show_export_options()

    if st.button("📂 Load from Project", type="secondary"):
        visualization_save.load_dashboard_from_project()


# ---------------------------------------------------------------------------
# Global transformation helper  (imported by viz_lib modules)
# ---------------------------------------------------------------------------

def apply_global_transformations() -> pd.DataFrame:
    """
    Return the base DataFrame after all globally-selected pipeline steps
    have been applied.  Safe to call even when session state is incomplete.
    """
    df_original = st.session_state.get("df_original")
    if df_original is None or (hasattr(df_original, "empty") and df_original.empty):
        return pd.DataFrame()

    steps = st.session_state.get("global_transformations", ["original"])

    if not steps or steps == ["original"]:
        return df_original.copy()

    df = df_original.copy()
    pipeline_steps = st.session_state.get("pipeline", {}).get("transformations", [])

    for step_name in steps:
        if step_name == "original":
            continue
        step = next((s for s in pipeline_steps if s["name"] == step_name), None)
        if step is None:
            st.warning(f"Global transformation '{step_name}' not found – skipping.")
            continue
        try:
            from transformations import transformation_execution
            df = transformation_execution.execute_transformation(
                step["type"], "execution", {"df": df, "step": step}
            )
        except Exception as exc:
            st.error(f"Error applying global transformation '{step_name}': {exc}")

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _init_session_state():
    defaults = {
        "visualizations": [],
        "dashboard_layout": "Single Column",
        "current_viz_id": 0,
        "global_transformations": ["original"],
        "pipeline": {"transformations": []},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val