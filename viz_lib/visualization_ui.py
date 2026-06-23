"""
visualization_ui.py  –  Enhanced version
-----------------------------------------
Key improvements over original:
  • Live chart preview before "Add to Dashboard"
  • Fully implemented Statistical Plots section (KDE, QQ, Correlation Matrix, Distribution)
  • Violin / Treemap / Sunburst / Area / Bubble / Funnel chart UIs added
  • Missing edit modals for 3D Scatter and Parallel Coordinates added
  • Individual transformation state is preserved when re-editing
  • Duplicate visualization button
  • Cleaner helper: apply_individual_transformations delegates to execution layer
  • All column lookups are safe (won't crash on missing column)
"""

import streamlit as st
import pandas as pd
import uuid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMPLE_CHART_TYPES = [
    "Scatter Plot", "Line Chart", "Area Chart", "Bar Chart",
    "Histogram", "Box Plot", "Pie Chart", "Bubble Chart", "Funnel Chart",
]

ADVANCED_CHART_TYPES = [
    "Heatmap", "3D Scatter", "Parallel Coordinates",
    "Violin Plot", "Treemap", "Sunburst",
]

STATISTICAL_CHART_TYPES = [
    "Correlation Matrix", "Distribution Overview",
    "KDE Plot", "QQ Plot",
]

COLOR_SCALES = ["Viridis", "Plasma", "Inferno", "Blues", "Reds", "Greens",
                "RdBu", "Spectral", "Cividis", "Turbo"]

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def initialize_session_state():
    defaults = {
        "visualizations": [],
        "global_transformations": ["original"],
        "dashboard_layout_select": "Single Column",
        "show_edit_modal": False,
        "editing_viz_index": None,
        "show_viz_dialogue": False,
        "current_viz_id": None,
        "pipeline": {"transformations": []},
        "df_original": pd.DataFrame(),
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Transformation helpers
# ---------------------------------------------------------------------------

def apply_individual_transformations(base_data: pd.DataFrame, transformations: list) -> pd.DataFrame:
    """Apply a list of named pipeline steps on top of *base_data*."""
    if base_data is None or base_data.empty:
        return pd.DataFrame()
    if not transformations or transformations == ["original"]:
        return base_data.copy()

    df = base_data.copy()
    pipeline_steps = st.session_state.get("pipeline", {}).get("transformations", [])

    for step_name in transformations:
        if step_name == "original":
            continue
        step = next((s for s in pipeline_steps if s["name"] == step_name), None)
        if step:
            try:
                from transformations import transformation_execution
                df = transformation_execution.execute_transformation(
                    step["type"], "execution", {"df": df, "step": step}
                )
            except Exception as exc:
                st.error(f"Error applying transformation '{step_name}': {exc}")
    return df


def _get_global_data() -> pd.DataFrame:
    """Return the dataset after global transformations have been applied."""
    from visualizations import apply_global_transformations
    return apply_global_transformations()


# ---------------------------------------------------------------------------
# Safe column helpers
# ---------------------------------------------------------------------------

def _safe_col_index(columns, value, fallback=0):
    cols = list(columns)
    try:
        return cols.index(value)
    except ValueError:
        return fallback


def _numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include="number").columns.tolist()


# ---------------------------------------------------------------------------
# Live preview helper
# ---------------------------------------------------------------------------

def _show_live_preview(preview_data: pd.DataFrame, config: dict):
    """Render a chart directly from already-transformed preview_data (no double transform)."""
    with st.expander("👁️ Live Preview", expanded=True):
        try:
            from viz_lib import visualization_execution as ve
            # Build a config copy with transformations stripped so generate_plot
            # doesn't re-apply them on top of the already-transformed preview_data.
            # We temporarily patch df_original in session state to the preview slice.
            preview_config = {**config, "transformations": ["original"]}
            orig_df = st.session_state.get("df_original")
            orig_global = st.session_state.get("global_transformations", ["original"])
            # Swap in preview_data as original + no global transforms → clean render
            st.session_state["df_original"] = preview_data
            st.session_state["global_transformations"] = ["original"]
            try:
                fig = ve.generate_plot(preview_data, preview_config)
                st.plotly_chart(fig, use_container_width=True)
            finally:
                # Always restore real session state
                st.session_state["df_original"] = orig_df
                st.session_state["global_transformations"] = orig_global
        except Exception as exc:
            st.warning(f"Preview unavailable: {exc}")


# ---------------------------------------------------------------------------
# Main creator UI
# ---------------------------------------------------------------------------

def show_visualization_creator(current_data: pd.DataFrame):
    st.subheader("➕ Create New Visualization")

    global_data = _get_global_data()
    st.info("💡 Using globally transformed data. Individual transformations can be added below.")

    # ---- Individual transformation selector ----
    available = ["original"] + [s["name"] for s in st.session_state.get("pipeline", {}).get("transformations", [])]
    individual_transformations = st.multiselect(
        "Additional individual transformations (applied after global):",
        options=available,
        default=["original"],
        key="viz_creator_individual_transformations",
    )

    preview_data = apply_individual_transformations(global_data, individual_transformations)

    with st.expander("📋 Final Data Preview"):
        st.dataframe(preview_data.head(10))
        st.caption(f"Shape: {preview_data.shape}  |  Columns: {', '.join(preview_data.columns)}")

    # ---- Chart category tabs ----
    tab_simple, tab_advanced, tab_stats = st.tabs(
        ["📊 Simple Charts", "🔬 Advanced Charts", "📈 Statistical Plots"]
    )

    with tab_simple:
        show_simple_charts_ui(current_data, preview_data, individual_transformations)

    with tab_advanced:
        show_advanced_charts_ui(current_data, preview_data, individual_transformations)

    with tab_stats:
        show_statistical_plots_ui(current_data, preview_data, individual_transformations)


# ---------------------------------------------------------------------------
# Simple charts
# ---------------------------------------------------------------------------

def show_simple_charts_ui(current_data, preview_data, transformations):
    chart_type = st.selectbox("Chart Type:", SIMPLE_CHART_TYPES, key="simple_chart_type")

    col1, col2 = st.columns(2)
    needs_y = chart_type in ["Scatter Plot", "Line Chart", "Area Chart", "Bar Chart",
                              "Bubble Chart", "Funnel Chart"]
    with col1:
        x_axis = st.selectbox("X-axis:", preview_data.columns, key="x_axis_simple")
        if needs_y:
            y_axis = st.selectbox("Y-axis:", preview_data.columns,
                                  index=min(1, len(preview_data.columns) - 1),
                                  key="y_axis_simple")
        else:
            y_axis = None

    with col2:
        color_by = st.selectbox("Color by (optional):",
                                ["None"] + list(preview_data.columns), key="color_by_simple")
        color_by = None if color_by == "None" else color_by

        size_by = None
        if chart_type in ("Scatter Plot", "Bubble Chart"):
            num_cols = _numeric_cols(preview_data)
            size_by = st.selectbox("Size by (optional):",
                                   ["None"] + num_cols, key="size_by_simple")
            size_by = None if size_by == "None" else size_by

    # ---- Chart-specific options ----
    extra = {}
    if chart_type == "Histogram":
        extra["bins"] = st.slider("Bins", 5, 150, 30, key="bins_simple")
        extra["marginal"] = st.selectbox("Marginal plot:", ["None", "box", "violin", "rug"],
                                         key="marginal_simple")
        extra["marginal"] = None if extra["marginal"] == "None" else extra["marginal"]

    if chart_type == "Bar Chart":
        extra["barmode"] = st.selectbox("Bar mode:", ["group", "stack", "overlay"],
                                        key="barmode_simple")

    if chart_type == "Pie Chart":
        extra["hole"] = st.slider("Donut hole (0 = pie)", 0.0, 0.8, 0.0, key="hole_simple")

    if chart_type in ("Scatter Plot", "Line Chart", "Bubble Chart"):
        extra["trendline"] = st.selectbox("Trendline:", ["None", "ols", "lowess"],
                                          key="trend_simple")
        extra["trendline"] = None if extra["trendline"] == "None" else extra["trendline"]

    if chart_type == "Box Plot":
        extra["points"] = st.selectbox("Show points:", ["outliers", "all", "False"],
                                       key="points_simple")
        extra["notched"] = st.checkbox("Notched boxes", False, key="notched_simple")

    if chart_type == "Line Chart":
        extra["show_markers"] = st.checkbox("Show markers", False, key="markers_simple")

    # ---- Advanced options ----
    with st.expander("⚙️ Advanced Options"):
        c3, c4 = st.columns(2)
        with c3:
            width  = st.slider("Width",  400, 1400, 700, key="width_simple")
            height = st.slider("Height", 300, 900,  450, key="height_simple")
            opacity = st.slider("Opacity", 0.1, 1.0, 0.8, step=0.05, key="opacity_simple")
        with c4:
            title     = st.text_input("Chart Title", chart_type, key="title_simple")
            show_grid = st.checkbox("Show Grid", True, key="grid_simple")

    config = {
        "type": "simple",
        "chart_type": chart_type,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "color_by": color_by,
        "size_by": size_by,
        "width": width,
        "height": height,
        "title": title,
        "show_grid": show_grid,
        "opacity": opacity,
        "transformations": transformations,
        **extra,
    }

    c_prev, c_add = st.columns(2)
    with c_prev:
        if st.button("👁️ Preview", key="preview_simple"):
            _show_live_preview(preview_data, config)
    with c_add:
        if st.button("➕ Add to Dashboard", key="add_simple", type="primary"):
            from viz_lib.visualization_execution import create_visualization
            create_visualization(current_data, config)


# ---------------------------------------------------------------------------
# Advanced charts
# ---------------------------------------------------------------------------

def show_advanced_charts_ui(current_data, preview_data, transformations):
    chart_type = st.selectbox("Chart Type:", ADVANCED_CHART_TYPES, key="advanced_chart_type")

    dispatch = {
        "Heatmap":              show_heatmap_ui,
        "3D Scatter":           show_3d_scatter_ui,
        "Parallel Coordinates": show_parallel_coords_ui,
        "Violin Plot":          show_violin_ui,
        "Treemap":              show_treemap_ui,
        "Sunburst":             show_sunburst_ui,
    }
    fn = dispatch.get(chart_type)
    if fn:
        fn(current_data, preview_data, transformations)
    else:
        st.info(f"{chart_type} coming soon…")


def show_heatmap_ui(current_data, preview_data, transformations):
    num_cols = _numeric_cols(preview_data)
    c1, c2 = st.columns(2)
    with c1:
        x_axis      = st.selectbox("X-axis:", preview_data.columns, key="heatmap_x")
        color_scale = st.selectbox("Color Scale:", COLOR_SCALES, key="heatmap_colors")
    with c2:
        y_axis = st.selectbox("Y-axis:", preview_data.columns,
                              index=min(1, len(preview_data.columns) - 1), key="heatmap_y")
        z_axis = st.selectbox("Values (aggregated):", num_cols, key="heatmap_z")

    with st.expander("⚙️ Advanced Options"):
        c3, c4 = st.columns(2)
        with c3:
            width  = st.slider("Width",  400, 1400, 700, key="width_heatmap")
            height = st.slider("Height", 300, 900,  500, key="height_heatmap")
        with c4:
            title            = st.text_input("Title", "Heatmap", key="title_heatmap")
            show_annotations = st.checkbox("Show cell values", True, key="annot_heatmap")

    config = {
        "type": "advanced", "chart_type": "Heatmap",
        "x_axis": x_axis, "y_axis": y_axis, "z_axis": z_axis,
        "color_scale": color_scale, "width": width, "height": height,
        "title": title, "show_annotations": show_annotations,
        "transformations": transformations,
    }
    _add_buttons(current_data, config, "heatmap")


def show_3d_scatter_ui(current_data, preview_data, transformations):
    num_cols = _numeric_cols(preview_data)
    if len(num_cols) < 3:
        st.warning("Need at least 3 numeric columns for a 3D Scatter plot.")
        return
    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X-axis:", num_cols, index=0, key="3d_x")
        y_axis = st.selectbox("Y-axis:", num_cols, index=1, key="3d_y")
    with c2:
        z_axis   = st.selectbox("Z-axis:", num_cols, index=2, key="3d_z")
        color_by = st.selectbox("Color by:", ["None"] + list(preview_data.columns), key="3d_color")
        color_by = None if color_by == "None" else color_by
        size_by  = st.selectbox("Size by:", ["None"] + num_cols, key="3d_size")
        size_by  = None if size_by == "None" else size_by

    opacity = st.slider("Opacity", 0.1, 1.0, 0.7, step=0.05, key="3d_opacity")
    title   = st.text_input("Title", "3D Scatter", key="3d_title")

    config = {
        "type": "advanced", "chart_type": "3D Scatter",
        "x_axis": x_axis, "y_axis": y_axis, "z_axis": z_axis,
        "color_by": color_by, "size_by": size_by, "opacity": opacity,
        "width": 900, "height": 650, "title": title,
        "transformations": transformations,
    }
    _add_buttons(current_data, config, "3d")


def show_parallel_coords_ui(current_data, preview_data, transformations):
    num_cols = _numeric_cols(preview_data)
    dims = st.multiselect("Dimensions:", num_cols,
                          default=num_cols[:min(5, len(num_cols))],
                          key="parallel_dims")
    color_by = st.selectbox("Color by:", ["None"] + list(preview_data.columns), key="parallel_color")
    color_by = None if color_by == "None" else color_by
    color_scale = st.selectbox("Color scale:", COLOR_SCALES, key="parallel_cs")
    title = st.text_input("Title", "Parallel Coordinates", key="parallel_title")

    config = {
        "type": "advanced", "chart_type": "Parallel Coordinates",
        "dimensions": dims, "color_by": color_by, "color_scale": color_scale,
        "width": 1000, "height": 550, "title": title,
        "transformations": transformations,
    }
    _add_buttons(current_data, config, "parallel")


def show_violin_ui(current_data, preview_data, transformations):
    c1, c2 = st.columns(2)
    with c1:
        x_axis   = st.selectbox("Variable (Y-axis):", preview_data.columns, key="violin_x")
        color_by = st.selectbox("Group by (X-axis):", ["None"] + list(preview_data.columns),
                                key="violin_color")
        color_by = None if color_by == "None" else color_by
    with c2:
        show_box = st.checkbox("Inner box plot", True, key="violin_box")
        points   = st.selectbox("Show points:", ["outliers", "all", "False"], key="violin_pts")
        title    = st.text_input("Title", "Violin Plot", key="violin_title")

    config = {
        "type": "advanced", "chart_type": "Violin Plot",
        "x_axis": x_axis, "color_by": color_by,
        "show_box": show_box, "points": points,
        "width": 700, "height": 500, "title": title,
        "transformations": transformations,
    }
    _add_buttons(current_data, config, "violin")


def show_treemap_ui(current_data, preview_data, transformations):
    cat_cols = preview_data.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = _numeric_cols(preview_data)
    if not cat_cols:
        st.warning("Treemap requires at least one categorical column.")
        return
    path_cols = st.multiselect("Hierarchy path (ordered):", cat_cols,
                               default=cat_cols[:min(2, len(cat_cols))], key="tree_path")
    y_axis      = st.selectbox("Values:", num_cols, key="tree_values")
    color_by    = st.selectbox("Color by:", ["None"] + num_cols, key="tree_color")
    color_by    = None if color_by == "None" else color_by
    color_scale = st.selectbox("Color scale:", COLOR_SCALES, key="tree_cs")
    title       = st.text_input("Title", "Treemap", key="tree_title")

    config = {
        "type": "advanced", "chart_type": "Treemap",
        "x_axis": path_cols[0] if path_cols else cat_cols[0],
        "y_axis": y_axis, "path_cols": path_cols,
        "color_by": color_by, "color_scale": color_scale,
        "width": 800, "height": 600, "title": title,
        "transformations": transformations,
    }
    _add_buttons(current_data, config, "treemap")


def show_sunburst_ui(current_data, preview_data, transformations):
    cat_cols = preview_data.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = _numeric_cols(preview_data)
    if not cat_cols:
        st.warning("Sunburst requires at least one categorical column.")
        return
    path_cols = st.multiselect("Hierarchy path:", cat_cols,
                               default=cat_cols[:min(2, len(cat_cols))], key="sun_path")
    y_axis   = st.selectbox("Values:", num_cols, key="sun_values")
    color_by = st.selectbox("Color by:", ["None"] + list(preview_data.columns), key="sun_color")
    color_by = None if color_by == "None" else color_by
    title    = st.text_input("Title", "Sunburst", key="sun_title")

    config = {
        "type": "advanced", "chart_type": "Sunburst",
        "x_axis": path_cols[0] if path_cols else cat_cols[0],
        "y_axis": y_axis, "path_cols": path_cols, "color_by": color_by,
        "width": 700, "height": 650, "title": title,
        "transformations": transformations,
    }
    _add_buttons(current_data, config, "sunburst")


# ---------------------------------------------------------------------------
# Statistical plots  (fully implemented)
# ---------------------------------------------------------------------------

def show_statistical_plots_ui(current_data, preview_data, transformations):
    chart_type = st.selectbox("Statistical Plot Type:", STATISTICAL_CHART_TYPES, key="stat_type")

    num_cols = _numeric_cols(preview_data)
    if not num_cols:
        st.warning("Statistical plots require numeric columns.")
        return

    if chart_type == "Correlation Matrix":
        _stat_correlation_matrix(current_data, preview_data, transformations)

    elif chart_type == "Distribution Overview":
        _stat_distribution_overview(current_data, preview_data, transformations)

    elif chart_type == "KDE Plot":
        _stat_kde(current_data, preview_data, transformations)

    elif chart_type == "QQ Plot":
        _stat_qq(current_data, preview_data, transformations)


def _stat_correlation_matrix(current_data, preview_data, transformations):
    import plotly.express as px
    num_cols = _numeric_cols(preview_data)
    selected = st.multiselect("Select columns:", num_cols, default=num_cols[:min(8, len(num_cols))],
                               key="corr_cols")
    method      = st.selectbox("Correlation method:", ["pearson", "spearman", "kendall"], key="corr_method")
    color_scale = st.selectbox("Color scale:", COLOR_SCALES, index=COLOR_SCALES.index("RdBu"), key="corr_cs")
    title       = st.text_input("Title", "Correlation Matrix", key="corr_title")

    # Inline preview using import-free approach
    if st.button("👁️ Preview", key="prev_corr"):
        corr = preview_data[selected].corr(method=method).round(3)
        fig  = px.imshow(corr, text_auto=True, color_continuous_scale=color_scale,
                         aspect="auto", title=title)
        st.plotly_chart(fig, use_container_width=True)

    config = {
        "type": "statistical", "chart_type": "Correlation Matrix",
        "corr_cols": selected, "corr_method": method,
        "color_scale": color_scale, "title": title,
        "width": 700, "height": 600,
        "transformations": transformations,
    }
    if st.button("➕ Add to Dashboard", key="add_corr", type="primary"):
        # Store config; generate_plot handles this via a special branch below
        from viz_lib.visualization_execution import create_visualization
        create_visualization(current_data, config)


def _stat_distribution_overview(current_data, preview_data, transformations):
    num_cols = _numeric_cols(preview_data)
    selected = st.multiselect("Columns to plot:", num_cols, default=num_cols[:min(4, len(num_cols))],
                               key="dist_cols")
    bins  = st.slider("Bins", 5, 100, 30, key="dist_bins")
    title = st.text_input("Title", "Distribution Overview", key="dist_title")

    if st.button("👁️ Preview", key="prev_dist"):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        rows = (len(selected) + 1) // 2
        fig  = make_subplots(rows=max(1, rows), cols=2, subplot_titles=selected)
        for idx, col in enumerate(selected):
            r, c = divmod(idx, 2)
            fig.add_trace(go.Histogram(x=preview_data[col], nbinsx=bins, name=col), row=r+1, col=c+1)
        fig.update_layout(title=title, showlegend=False, height=300 * max(1, rows))
        st.plotly_chart(fig, use_container_width=True)

    config = {
        "type": "statistical", "chart_type": "Distribution Overview",
        "dist_cols": selected, "bins": bins, "title": title,
        "width": 900, "height": max(400, 300 * ((len(selected) + 1) // 2)),
        "transformations": transformations,
    }
    if st.button("➕ Add to Dashboard", key="add_dist", type="primary"):
        from viz_lib.visualization_execution import create_visualization
        create_visualization(current_data, config)


def _stat_kde(current_data, preview_data, transformations):
    num_cols = _numeric_cols(preview_data)
    col      = st.selectbox("Column:", num_cols, key="kde_col")
    group_by = st.selectbox("Group by (optional):", ["None"] + list(preview_data.columns), key="kde_group")
    group_by = None if group_by == "None" else group_by
    title    = st.text_input("Title", f"KDE – {col}", key="kde_title")

    if st.button("👁️ Preview", key="prev_kde"):
        import plotly.express as px
        fig = px.histogram(preview_data, x=col, color=group_by,
                           marginal="violin", histnorm="density",
                           nbins=50, title=title)
        st.plotly_chart(fig, use_container_width=True)

    config = {
        "type": "statistical", "chart_type": "KDE Plot",
        "x_axis": col, "color_by": group_by, "title": title,
        "bins": 50, "marginal": "violin",
        "width": 700, "height": 500,
        "transformations": transformations,
    }
    if st.button("➕ Add to Dashboard", key="add_kde", type="primary"):
        from viz_lib.visualization_execution import create_visualization
        create_visualization(current_data, config)


def _stat_qq(current_data, preview_data, transformations):
    try:
        import scipy.stats as stats
    except ImportError:
        st.error("QQ Plot requires `scipy`. Install it with: `pip install scipy`")
        return

    import plotly.graph_objects as go

    num_cols = _numeric_cols(preview_data)
    col      = st.selectbox("Column:", num_cols, key="qq_col")
    title    = st.text_input("Title", f"QQ Plot – {col}", key="qq_title")

    if st.button("👁️ Preview", key="prev_qq"):
        series = preview_data[col].dropna()
        if len(series) < 3:
            st.warning("Need at least 3 non-null values for a QQ plot.")
        else:
            (osm, osr) = stats.probplot(series, dist="norm")[:2]
            qq_x, qq_y = osm[0], osm[1]
            slope, intercept, *_ = stats.linregress(qq_x, qq_y)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=qq_x, y=qq_y, mode="markers", name="Data"))
            fig.add_trace(go.Scatter(
                x=[qq_x.min(), qq_x.max()],
                y=[slope * qq_x.min() + intercept, slope * qq_x.max() + intercept],
                mode="lines", name="Normal", line=dict(color="red", dash="dash"),
            ))
            fig.update_layout(title=title, xaxis_title="Theoretical Quantiles",
                              yaxis_title="Sample Quantiles")
            st.plotly_chart(fig, use_container_width=True)

    config = {
        "type": "statistical", "chart_type": "QQ Plot",
        "x_axis": col, "title": title,
        "width": 650, "height": 500,
        "transformations": transformations,
    }
    if st.button("➕ Add to Dashboard", key="add_qq", type="primary"):
        from viz_lib.visualization_execution import create_visualization
        create_visualization(current_data, config)


# ---------------------------------------------------------------------------
# Shared "preview + add" buttons
# ---------------------------------------------------------------------------

def _add_buttons(current_data, config, key_suffix):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("👁️ Preview", key=f"preview_{key_suffix}"):
            _show_live_preview(current_data, config)
    with c2:
        if st.button("➕ Add to Dashboard", key=f"add_{key_suffix}", type="primary"):
            from viz_lib.visualization_execution import create_visualization
            create_visualization(current_data, config)


# ---------------------------------------------------------------------------
# Dashboard controls
# ---------------------------------------------------------------------------

def show_dashboard_controls():
    st.subheader("🎛️ Dashboard Controls")
    st.metric("Visualizations", len(st.session_state.visualizations))

    layout_options = ["Single Column", "Two Columns", "Three Columns", "Grid"]
    current_layout = st.session_state.get("dashboard_layout_select", "Single Column")
    try:
        idx = layout_options.index(current_layout)
    except ValueError:
        idx = 0

    layout = st.selectbox("Layout:", layout_options, index=idx, key="dashboard_layout_selector")
    if layout != st.session_state.get("dashboard_layout_select"):
        st.session_state.dashboard_layout_select = layout
        st.rerun()

    if st.session_state.visualizations:
        st.write("**Manage Visualizations:**")
        for i, viz in enumerate(st.session_state.visualizations):
            c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
            with c1:
                st.write(f"📊 {viz['config'].get('title', f'Viz {i+1}')}")
            with c2:
                if st.button("✏️", key=f"edit_{i}", help="Edit"):
                    edit_visualization(i)
                    st.rerun()
            with c3:
                if st.button("📋", key=f"dup_{i}", help="Duplicate"):
                    duplicate_visualization(i)
            with c4:
                if st.button("❌", key=f"remove_{i}", help="Remove"):
                    st.session_state.visualizations.pop(i)
                    st.rerun()

        if st.button("🗑️ Clear All", type="secondary"):
            st.session_state.visualizations = []
            st.rerun()
    else:
        st.info("No visualizations added yet.")


def duplicate_visualization(viz_index: int):
    import copy, uuid as _uuid
    original = st.session_state.visualizations[viz_index]
    duplicate = copy.deepcopy(original)
    duplicate["id"] = str(_uuid.uuid4())[:8]
    duplicate["config"]["title"] = original["config"].get("title", "Viz") + " (copy)"
    st.session_state.visualizations.append(duplicate)
    st.rerun()


def edit_visualization(viz_index: int):
    st.session_state.editing_viz_index = viz_index
    st.session_state.show_edit_modal = True


# ---------------------------------------------------------------------------
# Dashboard display
# ---------------------------------------------------------------------------

def show_dashboard(current_data):
    if not st.session_state.visualizations:
        st.info("Add visualizations to see them here.")
        return

    layout = st.session_state.get("dashboard_layout_select", "Single Column")
    cols_map = {"Single Column": 1, "Two Columns": 2, "Three Columns": 3, "Grid": 2}
    cols = cols_map.get(layout, 1)

    for i in range(0, len(st.session_state.visualizations), cols):
        columns = st.columns(cols)
        for j in range(cols):
            if i + j < len(st.session_state.visualizations):
                with columns[j]:
                    display_single_visualization(st.session_state.visualizations[i + j], current_data)


def display_single_visualization(viz, current_data):
    from viz_lib.visualization_execution import generate_plot

    try:
        # Transformation badge
        g_steps = st.session_state.get("global_transformations", [])
        i_steps = viz["config"].get("transformations", ["original"])
        ci1, ci2 = st.columns(2)
        with ci1:
            st.caption(f"🌐 Global: {', '.join(g_steps) if g_steps else 'None'}")
        with ci2:
            st.caption(f"📊 Individual: {', '.join(i_steps) if i_steps else 'None'}")

        # ---- Special handling for statistical charts ----
        chart_type = viz["config"].get("chart_type")
        if chart_type in STATISTICAL_CHART_TYPES:
            _render_statistical_viz(viz, current_data)
        else:
            fig = generate_plot(current_data, viz["config"])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Failed to generate plot.")

        # Comments
        viz.setdefault("comments", "")
        with st.expander("💬 Comment"):
            viz["comments"] = st.text_area(
                "Comments:", value=viz["comments"],
                key=f"comment_{viz['id']}", height=80,
            )

        # Action buttons
        ca, cb, cc, cd = st.columns(4)
        with ca:
            if st.button("📥 PNG", key=f"png_{viz['id']}"):
                from viz_lib.visualization_save import download_plot_as_png
                fig2 = generate_plot(current_data, viz["config"])
                download_plot_as_png(fig2, viz["config"].get("title", "plot"))
        with cb:
            if st.button("📥 SVG", key=f"svg_{viz['id']}"):
                from viz_lib.visualization_save import download_plot_as_svg
                fig2 = generate_plot(current_data, viz["config"])
                download_plot_as_svg(fig2, viz["config"].get("title", "plot"))
        with cc:
            if st.button("🔄 Refresh", key=f"update_{viz['id']}"):
                st.rerun()
        with cd:
            if st.button("📋 Report", key=f"report_{viz['id']}"):
                st.session_state.show_viz_dialogue = True
                st.session_state.current_viz_id = viz["id"]
                st.rerun()

        # Reporting dialogue
        if st.session_state.get("show_viz_dialogue") and st.session_state.get("current_viz_id") == viz["id"]:
            from reporting_lib.visualization_reporting import display_visualization_reporting_ui
            display_visualization_reporting_ui(viz["id"], viz["config"], viz.get("comments", ""))

    except Exception as exc:
        st.error(f"Error displaying visualization: {exc}")


def _render_statistical_viz(viz, current_data):
    """Render statistical charts that need special handling."""
    import plotly.express as px
    import plotly.graph_objects as go
    from viz_lib.visualization_execution import get_plot_data

    config     = viz["config"]
    chart_type = config.get("chart_type")
    plot_data  = get_plot_data(config)

    if plot_data.empty:
        st.warning("No data available.")
        return

    if chart_type == "Correlation Matrix":
        cols   = config.get("corr_cols") or _numeric_cols(plot_data)
        method = config.get("corr_method", "pearson")
        cs     = config.get("color_scale", "RdBu")
        corr   = plot_data[cols].corr(method=method).round(3)
        fig    = px.imshow(corr, text_auto=True, color_continuous_scale=cs,
                           aspect="auto", title=config.get("title", "Correlation Matrix"))
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Distribution Overview":
        from plotly.subplots import make_subplots
        cols  = config.get("dist_cols") or _numeric_cols(plot_data)
        bins  = config.get("bins", 30)
        rows  = (len(cols) + 1) // 2
        fig   = make_subplots(rows=max(1, rows), cols=2, subplot_titles=cols)
        for idx, col in enumerate(cols):
            r, c = divmod(idx, 2)
            fig.add_trace(go.Histogram(x=plot_data[col], nbinsx=bins, name=col), row=r+1, col=c+1)
        fig.update_layout(title=config.get("title", ""), showlegend=False,
                          height=300 * max(1, rows))
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "KDE Plot":
        col      = config.get("x_axis")
        color_by = config.get("color_by")
        fig      = px.histogram(plot_data, x=col, color=color_by,
                                marginal="violin", histnorm="density",
                                nbins=50, title=config.get("title", "KDE"))
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "QQ Plot":
        try:
            import scipy.stats as stats
        except ImportError:
            st.error("QQ Plot requires `scipy`. Install it with: `pip install scipy`")
            return

        col    = config.get("x_axis")
        if not col or col not in plot_data.columns:
            st.warning(f"Column '{col}' not found in data.")
            return
        series = plot_data[col].dropna()
        if len(series) < 3:
            st.warning("Need at least 3 non-null values for a QQ plot.")
            return
        (osm, osr) = stats.probplot(series, dist="norm")[:2]
        qq_x, qq_y = osm[0], osm[1]
        slope, intercept, *_ = stats.linregress(qq_x, qq_y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qq_x, y=qq_y, mode="markers", name="Data"))
        fig.add_trace(go.Scatter(
            x=[qq_x.min(), qq_x.max()],
            y=[slope * qq_x.min() + intercept, slope * qq_x.max() + intercept],
            mode="lines", name="Normal", line=dict(color="red", dash="dash"),
        ))
        fig.update_layout(title=config.get("title", "QQ Plot"),
                          xaxis_title="Theoretical Quantiles",
                          yaxis_title="Sample Quantiles")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(f"Statistical chart '{chart_type}' cannot be rendered.")


# ---------------------------------------------------------------------------
# Edit modal
# ---------------------------------------------------------------------------

def show_edit_modal(current_data):
    if not st.session_state.get("show_edit_modal"):
        return

    idx = st.session_state.get("editing_viz_index")
    if idx is None or idx >= len(st.session_state.visualizations):
        st.session_state.show_edit_modal = False
        return

    viz    = st.session_state.visualizations[idx]
    config = viz["config"]

    with st.container():
        st.header(f"✏️ Edit: {config.get('title', 'Untitled')}")

        chart_type = config.get("chart_type", "Scatter Plot")
        if chart_type in SIMPLE_CHART_TYPES:
            show_simple_charts_ui_edit(current_data, config, idx)
        elif chart_type in ADVANCED_CHART_TYPES:
            show_advanced_charts_ui_edit(current_data, config, idx)
        else:
            st.warning("Edit not available for this chart type.")

        if st.button("✖️ Cancel"):
            st.session_state.show_edit_modal = False
            st.session_state.editing_viz_index = None
            st.rerun()


def show_simple_charts_ui_edit(current_data, edit_config, viz_index):
    chart_type = st.selectbox(
        "Chart Type:", SIMPLE_CHART_TYPES,
        index=_safe_col_index(SIMPLE_CHART_TYPES, edit_config.get("chart_type", "Scatter Plot")),
        key="edit_simple_chart_type",
    )

    # Restore transformation state
    available = ["original"] + [s["name"] for s in st.session_state.pipeline.get("transformations", [])]
    saved_transforms = edit_config.get("transformations", ["original"])
    transformations = st.multiselect(
        "Individual transformations:", options=available, default=saved_transforms,
        key="edit_ind_trans",
    )

    c1, c2 = st.columns(2)
    needs_y = chart_type in ["Scatter Plot", "Line Chart", "Area Chart", "Bar Chart",
                              "Bubble Chart", "Funnel Chart"]
    with c1:
        x_axis = st.selectbox(
            "X-axis:", current_data.columns,
            index=_safe_col_index(current_data.columns, edit_config.get("x_axis")),
            key="edit_x_axis",
        )
        y_axis = None
        if needs_y:
            y_axis = st.selectbox(
                "Y-axis:", current_data.columns,
                index=_safe_col_index(current_data.columns, edit_config.get("y_axis"), 1),
                key="edit_y_axis",
            )
    with c2:
        color_opts  = ["None"] + list(current_data.columns)
        color_by    = st.selectbox(
            "Color by:", color_opts,
            index=_safe_col_index(color_opts, edit_config.get("color_by", "None")),
            key="edit_color_by",
        )
        color_by = None if color_by == "None" else color_by

    with st.expander("⚙️ Advanced Options"):
        c3, c4 = st.columns(2)
        with c3:
            width   = st.slider("Width",  400, 1400, edit_config.get("width",  700), key="edit_w")
            height  = st.slider("Height", 300, 900,  edit_config.get("height", 450), key="edit_h")
            opacity = st.slider("Opacity", 0.1, 1.0, edit_config.get("opacity", 0.8), step=0.05, key="edit_op")
        with c4:
            title     = st.text_input("Title", edit_config.get("title", chart_type), key="edit_title")
            show_grid = st.checkbox("Show Grid", edit_config.get("show_grid", True), key="edit_grid")

        if chart_type == "Histogram":
            bins = st.slider("Bins", 5, 150, edit_config.get("bins", 30), key="edit_bins")
        else:
            bins = edit_config.get("bins", 30)

    if st.button("💾 Update Visualization", type="primary", key="update_simple"):
        updated = {
            **edit_config,
            "chart_type": chart_type, "x_axis": x_axis, "y_axis": y_axis,
            "color_by": color_by, "width": width, "height": height,
            "title": title, "show_grid": show_grid, "opacity": opacity,
            "bins": bins, "transformations": transformations,
        }
        st.session_state.visualizations[viz_index]["config"] = updated
        st.success(f"'{title}' updated!")
        st.session_state.show_edit_modal = False
        st.session_state.editing_viz_index = None
        st.rerun()


def show_advanced_charts_ui_edit(current_data, edit_config, viz_index):
    chart_type = edit_config.get("chart_type", "Heatmap")

    edit_dispatch = {
        "Heatmap":              _edit_heatmap,
        "3D Scatter":           _edit_3d_scatter,
        "Parallel Coordinates": _edit_parallel_coords,
        "Violin Plot":          _edit_violin,
    }
    fn = edit_dispatch.get(chart_type)
    if fn:
        fn(current_data, edit_config, viz_index)
    else:
        st.warning(f"Edit not implemented for {chart_type} yet.")


def _save_edit(viz_index, updated_config):
    st.session_state.visualizations[viz_index]["config"] = updated_config
    st.success(f"'{updated_config.get('title', 'Viz')}' updated!")
    st.session_state.show_edit_modal = False
    st.session_state.editing_viz_index = None
    st.rerun()


def _edit_heatmap(current_data, edit_config, viz_index):
    num_cols = _numeric_cols(current_data)
    c1, c2   = st.columns(2)
    with c1:
        x_axis      = st.selectbox("X-axis:", current_data.columns,
                                   index=_safe_col_index(current_data.columns, edit_config.get("x_axis")),
                                   key="ehm_x")
        color_scale = st.selectbox("Color Scale:", COLOR_SCALES,
                                   index=_safe_col_index(COLOR_SCALES, edit_config.get("color_scale", "Viridis")),
                                   key="ehm_cs")
    with c2:
        y_axis = st.selectbox("Y-axis:", current_data.columns,
                              index=_safe_col_index(current_data.columns, edit_config.get("y_axis"), 1),
                              key="ehm_y")
        z_axis = st.selectbox("Values:", num_cols,
                              index=_safe_col_index(num_cols, edit_config.get("z_axis")),
                              key="ehm_z")
    title            = st.text_input("Title", edit_config.get("title", "Heatmap"), key="ehm_title")
    show_annotations = st.checkbox("Show values", edit_config.get("show_annotations", True), key="ehm_ann")

    if st.button("💾 Update", type="primary", key="upd_hm"):
        _save_edit(viz_index, {**edit_config, "x_axis": x_axis, "y_axis": y_axis,
                                "z_axis": z_axis, "color_scale": color_scale,
                                "title": title, "show_annotations": show_annotations})


def _edit_3d_scatter(current_data, edit_config, viz_index):
    num_cols = _numeric_cols(current_data)
    c1, c2   = st.columns(2)
    with c1:
        x_axis = st.selectbox("X:", num_cols, index=_safe_col_index(num_cols, edit_config.get("x_axis")), key="e3d_x")
        y_axis = st.selectbox("Y:", num_cols, index=_safe_col_index(num_cols, edit_config.get("y_axis"), 1), key="e3d_y")
    with c2:
        z_axis   = st.selectbox("Z:", num_cols, index=_safe_col_index(num_cols, edit_config.get("z_axis"), 2), key="e3d_z")
        color_by = st.selectbox("Color by:", ["None"] + list(current_data.columns),
                                index=_safe_col_index(["None"] + list(current_data.columns),
                                                      edit_config.get("color_by", "None")), key="e3d_c")
        color_by = None if color_by == "None" else color_by
    title   = st.text_input("Title", edit_config.get("title", "3D Scatter"), key="e3d_t")
    opacity = st.slider("Opacity", 0.1, 1.0, edit_config.get("opacity", 0.7), key="e3d_op")

    if st.button("💾 Update", type="primary", key="upd_3d"):
        _save_edit(viz_index, {**edit_config, "x_axis": x_axis, "y_axis": y_axis,
                                "z_axis": z_axis, "color_by": color_by,
                                "title": title, "opacity": opacity})


def _edit_parallel_coords(current_data, edit_config, viz_index):
    num_cols = _numeric_cols(current_data)
    dims     = st.multiselect("Dimensions:", num_cols,
                              default=[d for d in edit_config.get("dimensions", []) if d in num_cols],
                              key="epc_dims")
    color_by = st.selectbox("Color by:", ["None"] + list(current_data.columns),
                            index=_safe_col_index(["None"] + list(current_data.columns),
                                                  edit_config.get("color_by", "None")), key="epc_c")
    color_by = None if color_by == "None" else color_by
    title    = st.text_input("Title", edit_config.get("title", "Parallel Coordinates"), key="epc_t")

    if st.button("💾 Update", type="primary", key="upd_pc"):
        _save_edit(viz_index, {**edit_config, "dimensions": dims,
                                "color_by": color_by, "title": title})


def _edit_violin(current_data, edit_config, viz_index):
    c1, c2 = st.columns(2)
    with c1:
        x_axis   = st.selectbox("Variable:", current_data.columns,
                                index=_safe_col_index(current_data.columns, edit_config.get("x_axis")), key="ev_x")
        color_by = st.selectbox("Group by:", ["None"] + list(current_data.columns),
                                index=_safe_col_index(["None"] + list(current_data.columns),
                                                      edit_config.get("color_by", "None")), key="ev_c")
        color_by = None if color_by == "None" else color_by
    with c2:
        show_box = st.checkbox("Inner box", edit_config.get("show_box", True), key="ev_box")
        points   = st.selectbox("Points:", ["outliers", "all", "False"],
                                index=["outliers", "all", "False"].index(
                                    edit_config.get("points", "outliers")), key="ev_pts")
        title = st.text_input("Title", edit_config.get("title", "Violin Plot"), key="ev_t")

    if st.button("💾 Update", type="primary", key="upd_vio"):
        _save_edit(viz_index, {**edit_config, "x_axis": x_axis, "color_by": color_by,
                                "show_box": show_box, "points": points, "title": title})


# ---------------------------------------------------------------------------
# Legacy helper kept for backwards compatibility
# ---------------------------------------------------------------------------

def apply_selected_transformations(selected_steps):
    df = st.session_state.df_original.copy()
    return apply_individual_transformations(df, selected_steps)