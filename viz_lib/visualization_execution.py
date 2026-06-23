import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import uuid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_transformations(base_df: pd.DataFrame, transformation_names: list) -> pd.DataFrame:
    """
    Apply a list of named pipeline transformation steps to *base_df*.
    Returns a copy – never mutates the input.
    """
    if not transformation_names or transformation_names == ["original"]:
        return base_df.copy()

    df = base_df.copy()
    pipeline_steps = st.session_state.get("pipeline", {}).get("transformations", [])

    for step_name in transformation_names:
        if step_name == "original":
            continue
        step = next((s for s in pipeline_steps if s["name"] == step_name), None)
        if step is None:
            st.warning(f"Transformation '{step_name}' not found in pipeline – skipping.")
            continue
        try:
            from transformations import transformation_execution
            df = transformation_execution.execute_transformation(
                step["type"], "execution", {"df": df, "step": step}
            )
        except Exception as exc:
            st.error(f"Error applying transformation '{step_name}': {exc}")
    return df


def get_plot_data(config: dict) -> pd.DataFrame:
    """
    Build the final DataFrame for a single visualization by layering:
      1. Original data
      2. Global transformations  (from session state)
      3. Per-chart individual transformations  (from config)

    This is the single source of truth – both the dashboard renderer and
    the report renderer must call this function instead of duplicating logic.
    """
    df_original = st.session_state.get("df_original")
    if df_original is None or df_original.empty:
        return pd.DataFrame()

    # 1. Global layer
    global_steps = st.session_state.get("global_transformations", ["original"])
    df = _apply_transformations(df_original, global_steps)

    # 2. Individual layer
    individual_steps = config.get("transformations", ["original"])
    df = _apply_transformations(df, individual_steps)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_visualization(current_data: pd.DataFrame, config: dict):
    """Create and append a new visualization to the dashboard."""
    viz_id = str(uuid.uuid4())[:8]
    visualization = {"id": viz_id, "config": config, "comments": ""}

    if "visualizations" not in st.session_state:
        st.session_state.visualizations = []

    st.session_state.visualizations.append(visualization)
    st.success(f"Visualization '{config.get('title', 'Untitled')}' added to dashboard!")


def generate_plot(current_data: pd.DataFrame, config: dict):
    """
    Generate and return a Plotly figure for *config*.

    *current_data* is accepted for backwards-compatibility but is ignored –
    the function always rebuilds the data via ``get_plot_data`` so that
    global + individual transformations are applied consistently.
    """
    plot_data = get_plot_data(config)
    if plot_data.empty:
        return create_error_plot("No data available after applying transformations.")

    chart_type = config.get("chart_type", "")

    builders = {
        "Scatter Plot":         create_scatter_plot,
        "Line Chart":           create_line_chart,
        "Bar Chart":            create_bar_chart,
        "Histogram":            create_histogram,
        "Box Plot":             create_box_plot,
        "Pie Chart":            create_pie_chart,
        "Heatmap":              create_heatmap,
        "3D Scatter":           create_3d_scatter,
        "Parallel Coordinates": create_parallel_coords,
        "Violin Plot":          create_violin_plot,
        "Treemap":              create_treemap,
        "Sunburst":             create_sunburst,
        "Area Chart":           create_area_chart,
        "Bubble Chart":         create_bubble_chart,
        "Funnel Chart":         create_funnel_chart,
        "KDE Plot": create_kde_plot,
        "Correlation Matrix": create_correlation_matrix,
    }

    builder = builders.get(chart_type)
    if builder is None:
        return create_default_plot(plot_data, config)

    try:
        fig = builder(plot_data, config)
        from viz_lib.visualization_save import apply_report_theme
        return apply_report_theme(fig)
    except Exception as exc:
        st.error(f"Error generating '{chart_type}': {exc}")
        return create_error_plot(str(exc))


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
def create_kde_plot(data, config):

    import plotly.figure_factory as ff

    column = config.get("x_axis")

    if column not in data.columns:
        return create_error_plot(
            f"Column '{column}' not found."
        )

    values = (
        data[column]
        .dropna()
        .astype(float)
        .values
    )

    if len(values) < 2:
        return create_error_plot(
            "Not enough data points for KDE."
        )

    fig = ff.create_distplot(
        [values],
        [column],
        show_hist=False,
        show_rug=False
    )

    fig.update_layout(
        title=config.get(
            "title",
            f"KDE - {column}"
        ),
        width=config.get("width", 700),
        height=config.get("height", 500),
        xaxis_title=column,
        yaxis_title="Density"
    )

    return fig
def create_correlation_matrix(data, config):

    import plotly.express as px

    selected_cols = config.get("corr_cols", [])

    if len(selected_cols) < 2:
        return create_error_plot(
            "Correlation Matrix requires at least 2 columns."
        )

    method = config.get("corr_method", "pearson")

    corr = (
        data[selected_cols]
        .corr(method=method)
        .round(3)
    )

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale=config.get(
            "color_scale",
            "RdBu"
        ),
        aspect="auto",
        title=config.get(
            "title",
            "Correlation Matrix"
        )
    )

    fig.update_layout(
        width=config.get("width", 700),
        height=config.get("height", 600)
    )

    return fig
def _base_layout(config: dict) -> dict:
    return dict(
        title=config.get("title", ""),
        width=config.get("width", 600),
        height=config.get("height", 400),
    )


def _apply_grid(fig, config: dict):
    if config.get("show_grid", True):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    return fig


def create_scatter_plot(data, config):
    color_by = config.get("color_by")
    # color_continuous_scale is only valid when color maps to a numeric column;
    # passing it with a categorical column raises a Plotly ValueError.
    use_continuous_scale = (
        color_by is not None
        and color_by in data.columns
        and pd.api.types.is_numeric_dtype(data[color_by])
    )
    fig = px.scatter(
        data,
        x=config["x_axis"],
        y=config["y_axis"],
        color=color_by,
        size=config.get("size_by"),
        trendline=config.get("trendline"),
        color_continuous_scale=config.get("color_scale", "Viridis") if use_continuous_scale else None,
        **_base_layout(config),
    )
    fig.update_traces(marker=dict(opacity=config.get("opacity", 0.7)))
    return _apply_grid(fig, config)


def create_line_chart(data, config):
    fig = px.line(
        data,
        x=config["x_axis"],
        y=config["y_axis"],
        color=config.get("color_by"),
        markers=config.get("show_markers", False),
        **_base_layout(config),
    )
    return _apply_grid(fig, config)


def create_area_chart(data, config):
    fig = px.area(
        data,
        x=config["x_axis"],
        y=config["y_axis"],
        color=config.get("color_by"),
        **_base_layout(config),
    )
    return _apply_grid(fig, config)


def create_bar_chart(data, config):
    fig = px.bar(
        data,
        x=config["x_axis"],
        y=config["y_axis"],
        color=config.get("color_by"),
        barmode=config.get("barmode", "group"),   # group | stack | overlay
        **_base_layout(config),
    )
    return fig


def create_histogram(data, config):
    fig = px.histogram(
        data,
        x=config["x_axis"],
        color=config.get("color_by"),
        nbins=config.get("bins", 30),
        marginal=config.get("marginal"),          # e.g. "box", "violin", "rug"
        **_base_layout(config),
    )
    return fig


def create_box_plot(data, config):
    color_by = config.get("color_by")
    fig = px.box(
        data,
        x=color_by,                               # None → single group; category → grouped
        y=config["x_axis"],
        color=color_by,
        points=config.get("points", "outliers"),
        notched=config.get("notched", False),
        **_base_layout(config),
    )
    return fig


def create_violin_plot(data, config):
    color_by = config.get("color_by")
    fig = px.violin(
        data,
        x=color_by,                               # None → single group; category → grouped
        y=config["x_axis"],
        color=color_by,
        box=config.get("show_box", True),
        points=config.get("points", "outliers"),
        **_base_layout(config),
    )
    return fig


def create_pie_chart(data, config):
    y_axis = config.get("y_axis")
    fig = px.pie(
        data,
        names=config["x_axis"],
        values=y_axis if y_axis and y_axis in data.columns else None,
        hole=config.get("hole", 0.0),
        title=config.get("title", "Pie Chart"),
    )
    # px.pie ignores width/height kwargs – must set via update_layout
    fig.update_layout(width=config.get("width", 600), height=config.get("height", 400))
    return fig


def create_funnel_chart(data, config):
    fig = px.funnel(
        data,
        x=config["y_axis"],
        y=config["x_axis"],
        color=config.get("color_by"),
        **_base_layout(config),
    )
    return fig


def create_treemap(data, config):
    path_cols = config.get("path_cols") or [config["x_axis"]]
    if not path_cols:
        from viz_lib.visualization_execution import create_error_plot
        return create_error_plot("Treemap requires at least one path column.")
    color_col = config.get("color_by") or config.get("y_axis")
    use_continuous = (
        color_col and color_col in data.columns
        and pd.api.types.is_numeric_dtype(data[color_col])
    )
    fig = px.treemap(
        data,
        path=[px.Constant("all")] + path_cols,
        values=config.get("y_axis"),
        color=color_col,
        color_continuous_scale=config.get("color_scale", "RdBu") if use_continuous else None,
        **_base_layout(config),
    )
    return fig


def create_sunburst(data, config):
    path_cols = config.get("path_cols") or [config["x_axis"]]
    if not path_cols:
        return create_error_plot("Sunburst requires at least one path column.")
    fig = px.sunburst(
        data,
        path=path_cols,
        values=config.get("y_axis"),
        color=config.get("color_by"),
        **_base_layout(config),
    )
    return fig


def create_heatmap(data, config):
    try:
        pivot_data = data.pivot_table(
            values=config["z_axis"],
            index=config["y_axis"],
            columns=config["x_axis"],
            aggfunc="mean",
        )
    except Exception as exc:
        return create_error_plot(f"Heatmap pivot failed: {exc}")

    show_ann = config.get("show_annotations", True)
    z_vals   = pivot_data.values
    # Round safely – may contain NaN
    try:
        text_vals = z_vals.round(2).tolist() if show_ann else None
    except Exception:
        text_vals = None
        show_ann  = False

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals.tolist(),
            x=pivot_data.columns.tolist(),
            y=pivot_data.index.tolist(),
            colorscale=config.get("color_scale", "Viridis"),
            hoverongaps=False,
            text=text_vals,
            texttemplate="%{text}" if show_ann else None,
        )
    )
    fig.update_layout(
        xaxis_title=config["x_axis"],
        yaxis_title=config["y_axis"],
        **_base_layout(config),
    )
    return fig


def create_3d_scatter(data, config):
    fig = px.scatter_3d(
        data,
        x=config["x_axis"],
        y=config["y_axis"],
        z=config["z_axis"],
        color=config.get("color_by"),
        size=config.get("size_by"),
        opacity=config.get("opacity", 0.7),
        title=config.get("title", "3D Scatter"),
        width=config.get("width", 800),
        height=config.get("height", 600),
    )
    return fig


def create_parallel_coords(data, config):
    dims     = config.get("dimensions", [])
    color_by = config.get("color_by")
    # parallel_coordinates requires color to be numeric
    if color_by and color_by in data.columns and not pd.api.types.is_numeric_dtype(data[color_by]):
        color_by = None  # silently drop non-numeric color

    fig = px.parallel_coordinates(
        data,
        dimensions=dims if dims else None,
        color=color_by,
        color_continuous_scale=config.get("color_scale", "Viridis"),
        title=config.get("title", "Parallel Coordinates"),
        width=config.get("width", 900),
        height=config.get("height", 500),
    )
    return fig


def create_bubble_chart(data, config):
    """Scatter with mandatory size column, logged if requested."""
    size_by = config.get("size_by")
    # Validate size column exists and is numeric
    if size_by and (size_by not in data.columns or not pd.api.types.is_numeric_dtype(data[size_by])):
        size_by = None
    fig = px.scatter(
        data,
        x=config["x_axis"],
        y=config["y_axis"],
        size=size_by,
        color=config.get("color_by"),
        log_x=config.get("log_x", False),
        log_y=config.get("log_y", False),
        **_base_layout(config),
    )
    return _apply_grid(fig, config)


# ---------------------------------------------------------------------------
# Fallback / error plots
# ---------------------------------------------------------------------------

def create_default_plot(data, config):
    fig = go.Figure()
    fig.add_annotation(
        text=f"Chart type '{config.get('chart_type')}' not implemented",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16),
    )
    fig.update_layout(**_base_layout(config))
    return fig


def create_error_plot(message: str):
    fig = go.Figure()
    fig.add_annotation(
        text=f"⚠️ {message}",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="red"),
    )
    fig.update_layout(width=400, height=300, margin=dict(l=20, r=20, t=20, b=20))
    return fig