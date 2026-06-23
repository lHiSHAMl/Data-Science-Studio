# models/scripts/DBSCAN/model_report.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_array(value):
    """Return a numpy array whether value is already ndarray, list, or repr string."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.array(value)
    import re
    s = str(value)
    s = re.sub(r"array\(", "", s)
    s = re.sub(r",\s*shape=\([^)]*\)\s*\)", "", s)
    s = s.rstrip(")")
    try:
        return np.array(eval(s))
    except Exception:
        return np.array([])


def _safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. Summary text
# ---------------------------------------------------------------------------

def create_ml_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Styled HTML card with high-level DBSCAN clustering metrics."""
    r = metrics_snapshot

    n_clusters  = r.get("N Clusters",     "N/A")
    n_noise     = r.get("N Noise Points", "N/A")
    n_samples   = r.get("N Samples",      "N/A")
    features    = r.get("features",       [])
    best_params = r.get("Best Parameters", {})
    use_gs      = r.get("use_grid_search", False)

    silhouette = _safe_float(r.get("Silhouette Score"))
    calinski   = _safe_float(r.get("Calinski-Harabasz Score"))
    davies     = _safe_float(r.get("Davies-Bouldin Index"))

    sil_str = f"{silhouette:.4f}" if silhouette is not None else "N/A"
    cal_str = f"{calinski:.4f}"   if calinski   is not None else "N/A"
    dav_str = f"{davies:.4f}"     if davies     is not None else "N/A"

    noise_pct = (
        f"{100 * n_noise / n_samples:.1f}%" if isinstance(n_noise, int)
        and isinstance(n_samples, int) and n_samples > 0
        else "N/A"
    )

    params_html = "".join(
        f"<strong>{k}:</strong> {v}<br>" for k, v in best_params.items()
    )

    html = f"""
    <div class="report-container">
        <h3>DBSCAN Clustering Report</h3>
        <div class="metric-card">
            <strong>Features Used:</strong> {", ".join(str(f) for f in features)}<br>
            <strong>Parameter Selection:</strong> {"Grid Search" if use_gs else "Manual"}<br>
            <strong>Total Samples:</strong> {n_samples}
        </div>
        <div class="metric-card">
            <strong>Clusters Found:</strong> {n_clusters}<br>
            <strong>Noise Points:</strong> {n_noise} ({noise_pct})<br>
        </div>
        <div class="metric-card">
            <strong>Silhouette Score:</strong> {sil_str}<br>
            <strong>Calinski-Harabasz Score:</strong> {cal_str}<br>
            <strong>Davies-Bouldin Index:</strong> {dav_str}
        </div>
        <div class="metric-card">
            <strong>Best Parameters:</strong><br>{params_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    return {"type": "text", "content": " "}


# ---------------------------------------------------------------------------
# 2. Clustering metrics table
# ---------------------------------------------------------------------------

def create_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """DataFrame table of clustering quality metrics."""
    import pandas as pd

    r = metrics_snapshot

    silhouette = _safe_float(r.get("Silhouette Score"))
    calinski   = _safe_float(r.get("Calinski-Harabasz Score"))
    davies     = _safe_float(r.get("Davies-Bouldin Index"))

    rows = [
        {
            "Metric":      "Silhouette Score",
            "Value":       f"{silhouette:.4f}" if silhouette is not None else "N/A",
            "Range":       "[-1, 1]",
            "Optimum":     "Higher is better",
            "Interpretation": (
                "Strong"     if silhouette is not None and silhouette > 0.7  else
                "Reasonable" if silhouette is not None and silhouette > 0.5  else
                "Weak"       if silhouette is not None and silhouette > 0.25 else
                "Poor"       if silhouette is not None else "N/A"
            ),
        },
        {
            "Metric":      "Calinski-Harabasz Score",
            "Value":       f"{calinski:.4f}" if calinski is not None else "N/A",
            "Range":       "[0, ∞)",
            "Optimum":     "Higher is better",
            "Interpretation": (
                "Good" if calinski is not None and calinski > 100 else
                "Moderate" if calinski is not None else "N/A"
            ),
        },
        {
            "Metric":      "Davies-Bouldin Index",
            "Value":       f"{davies:.4f}" if davies is not None else "N/A",
            "Range":       "[0, ∞)",
            "Optimum":     "Lower is better",
            "Interpretation": (
                "Good"     if davies is not None and davies < 1.0 else
                "Moderate" if davies is not None and davies < 2.0 else
                "Poor"     if davies is not None else "N/A"
            ),
        },
        {
            "Metric":      "N Clusters",
            "Value":       str(r.get("N Clusters", "N/A")),
            "Range":       "≥ 0",
            "Optimum":     "Domain-dependent",
            "Interpretation": "Clusters found (excl. noise)",
        },
        {
            "Metric":      "N Noise Points",
            "Value":       str(r.get("N Noise Points", "N/A")),
            "Range":       "≥ 0",
            "Optimum":     "Lower is better",
            "Interpretation": "Points labelled as outliers (label = -1)",
        },
    ]

    df = pd.DataFrame(rows)
    return {"type": "dataframe", "content": df}


# ---------------------------------------------------------------------------
# 3. Cluster distribution plot
# ---------------------------------------------------------------------------

def create_cluster_distribution_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """PCA scatter + cluster size bar chart side by side."""
    r              = metrics_snapshot
    X_scaled       = _parse_array(r.get("X_scaled", np.array([])))
    cluster_labels = _parse_array(r.get("cluster_labels", np.array([])))
    cluster_sizes  = r.get("Cluster Sizes", [])
    n_clusters     = r.get("N Clusters", 0)

    if X_scaled.ndim < 2 or X_scaled.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need ≥ 2 features for PCA visualisation",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return {"type": "plot", "content": fig}

    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    unique_labels = np.array(sorted(set(cluster_labels)))
    cmap          = plt.cm.tab10
    colors        = {lbl: cmap(i / max(len(unique_labels), 1))
                     for i, lbl in enumerate(unique_labels)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── PCA scatter ────────────────────────────────────────────────────────
    for lbl in unique_labels:
        mask   = cluster_labels == lbl
        label  = "Noise" if lbl == -1 else f"Cluster {lbl}"
        marker = "x"     if lbl == -1 else "o"
        alpha  = 0.4     if lbl == -1 else 0.7
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=[colors[lbl]], label=label,
                    marker=marker, alpha=alpha, s=30)

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax1.set_title("Cluster Visualisation (PCA)")
    ax1.legend(loc="best", fontsize=8)

    # ── Cluster size bar ───────────────────────────────────────────────────
    if cluster_sizes:
        bar_labels = [f"Cluster {i}" for i in range(len(cluster_sizes))]
        bar_colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes)))
        bars = ax2.bar(bar_labels, cluster_sizes, color=bar_colors)
        for bar, sz in zip(bars, cluster_sizes):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                     f"{sz}", ha="center", va="bottom")
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Number of Points")
        ax2.set_title("Cluster Sizes (excl. noise)")
        ax2.tick_params(axis="x", rotation=45)

        # Annotate noise separately
        n_noise = r.get("N Noise Points", 0)
        if n_noise:
            ax2.text(0.98, 0.97, f"Noise points: {n_noise}",
                     transform=ax2.transAxes, ha="right", va="top",
                     fontsize=9, color="red",
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="red"))
    else:
        ax2.text(0.5, 0.5, "No cluster size data", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.set_axis_off()

    plt.tight_layout()
    return {"type": "plot", "content": fig}


# ---------------------------------------------------------------------------
# 4. Quality metrics plot
# ---------------------------------------------------------------------------

def create_quality_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Bar chart of the three quality scores with interpretation bands."""
    r          = metrics_snapshot
    silhouette = _safe_float(r.get("Silhouette Score"))
    calinski   = _safe_float(r.get("Calinski-Harabasz Score"))
    davies     = _safe_float(r.get("Davies-Bouldin Index"))

    available = {k: v for k, v in {
        "Silhouette\nScore":          silhouette,
        "Calinski-Harabasz\nScore":   calinski,
        "Davies-Bouldin\nIndex":      davies,
    }.items() if v is not None}

    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No quality metrics available\n(need ≥ 2 clusters)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        return {"type": "plot", "content": fig}

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar(list(available.keys()), list(available.values()),
                  color=bar_colors[: len(available)])

    max_val = max(available.values())
    for bar, val in zip(bars, available.values()):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01 * max_val,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Score")
    ax.set_title("DBSCAN Clustering Quality Metrics")
    ax.set_ylim(0, max_val * 1.15)
    plt.tight_layout()
    return {"type": "plot", "content": fig}


# ---------------------------------------------------------------------------
# 5. Parameter configuration plot
# ---------------------------------------------------------------------------

def create_parameter_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Visual summary of DBSCAN parameter configuration and cluster stats."""
    r           = metrics_snapshot
    best_params = r.get("Best Parameters", {})
    n_clusters  = r.get("N Clusters",      0)
    n_noise     = r.get("N Noise Points",  0)
    n_samples   = r.get("N Samples",       1)
    use_gs      = r.get("use_grid_search", False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("DBSCAN Parameter Configuration & Cluster Composition", fontsize=13)

    # ── Left: parameter table ──────────────────────────────────────────────
    ax_table = axes[0]
    ax_table.set_axis_off()

    rows      = [[k, str(v)] for k, v in best_params.items()]
    rows.append(["Selection method", "Grid Search" if use_gs else "Manual"])

    table = ax_table.table(
        cellText=rows,
        colLabels=["Parameter", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style header row
    for col in range(2):
        table[0, col].set_facecolor("#4C72B0")
        table[0, col].set_text_props(color="white", fontweight="bold")

    ax_table.set_title("Best Parameters", pad=12)

    # ── Right: composition pie ─────────────────────────────────────────────
    ax_pie = axes[1]

    cluster_sizes = r.get("Cluster Sizes", [])
    labels, sizes, colors = [], [], []

    for i, sz in enumerate(cluster_sizes):
        labels.append(f"Cluster {i}\n({sz})")
        sizes.append(sz)
        colors.append(plt.cm.tab10(i / max(len(cluster_sizes), 1)))

    if n_noise > 0:
        labels.append(f"Noise\n({n_noise})")
        sizes.append(n_noise)
        colors.append("#cccccc")

    if sizes:
        wedges, texts, autotexts = ax_pie.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax_pie.set_title(f"Sample Composition\n(total = {n_samples})", pad=12)
    else:
        ax_pie.text(0.5, 0.5, "No cluster data", ha="center", va="center",
                    transform=ax_pie.transAxes)
        ax_pie.set_axis_off()

    plt.tight_layout()
    return {"type": "plot", "content": fig}


# ---------------------------------------------------------------------------
# Orchestration layer
# ---------------------------------------------------------------------------

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    """Render every report section for the DBSCAN model."""
    create_ml_summary_text(metrics_snapshot)

    st.subheader("📋 Clustering Metrics Table")
    asset = create_classification_report_table(metrics_snapshot)
    st.dataframe(asset["content"], use_container_width=True)

    st.subheader("🌐 Cluster Distribution")
    asset = create_cluster_distribution_plot(metrics_snapshot)
    st.pyplot(asset["content"])
    plt.close(asset["content"])

    st.subheader("📊 Quality Metrics")
    asset = create_quality_metrics_plot(metrics_snapshot)
    st.pyplot(asset["content"])
    plt.close(asset["content"])

    st.subheader("⚙️ Parameter Configuration")
    asset = create_parameter_importance_plot(metrics_snapshot)
    st.pyplot(asset["content"])
    plt.close(asset["content"])


def model_report():
    """Entry point called by the ML page for DBSCAN."""
    found        = False
    model_results = {}

    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == "DBSCAN":
            if "metrics_snapshot" in model_info:
                found        = True
                model_results = model_info["metrics_snapshot"]
                break

    if not found:
        st.error("No DBSCAN model results found. Please create a model first.")
        return

    st.markdown("---")
    st.markdown("## DBSCAN Clustering Model Analysis (Live View)")
    display_ml_report(model_results)