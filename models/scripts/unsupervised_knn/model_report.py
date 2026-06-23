# models/scripts/unsupervised_knn/model_report.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Helper: safely parse the metrics_snapshot values that may have been
# serialised as strings (numpy arrays / scalars stored as repr strings).
# ---------------------------------------------------------------------------

def _parse_array(value):
    """Return a numpy array whether *value* is already an ndarray or a string repr."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.array(value)
    # String representation produced by str()/repr() of a numpy array
    import re
    # Strip 'array(...)' wrapper and 'shape=...' suffix if present
    s = str(value)
    s = re.sub(r"array\(", "", s)
    s = re.sub(r",\s*shape=\([^)]*\)\s*\)", "", s)
    s = s.rstrip(")")
    try:
        return np.array(eval(s))          # safe enough for internal snapshot data
    except Exception:
        return np.array([])


def _parse_cluster_sizes(value):
    """Return a plain list of ints from cluster_sizes (may contain 'np.int64(n)' strings)."""
    import re
    result = []
    for item in value:
        s = str(item)
        m = re.search(r"(\d+)", s)
        if m:
            result.append(int(m.group(1)))
        else:
            try:
                result.append(int(item))
            except (ValueError, TypeError):
                pass
    return result


# ---------------------------------------------------------------------------
# 1. Summary / common metrics text block
# ---------------------------------------------------------------------------

def create_ml_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Styled HTML card with high-level clustering metrics."""
    r = metrics_snapshot

    silhouette = r.get("Silhouette Score")
    calinski   = r.get("Calinski-Harabasz Score")
    davies     = r.get("Davies-Bouldin Score")
    n_clusters = r.get("Number of Clusters", "N/A")
    features   = r.get("features", [])
    method     = r.get("method_used", r.get("linkage", "KNN-based"))
    params     = r.get("parameters", r.get("compute_full_tree", "N/A"))

    sil_str = f"{silhouette:.4f}" if silhouette is not None else "N/A"
    cal_str = f"{calinski:.4f}"   if calinski   is not None else "N/A"
    dav_str = f"{davies:.4f}"     if davies     is not None else "N/A"

    html = f"""
    <div class="report-container">
        <h3>Unsupervised KNN Clustering Report</h3>
        <div class="metric-card">
            <strong>Features Used:</strong> {", ".join(str(f) for f in features)}<br>
            <strong>Clustering Method:</strong> {method}<br>
            <strong>Number of Clusters Found:</strong> {n_clusters}
        </div>
        <div class="metric-card">
            <strong>Silhouette Score:</strong> {sil_str}<br>
            <strong>Calinski-Harabasz Index:</strong> {cal_str}<br>
            <strong>Davies-Bouldin Index:</strong> {dav_str}
        </div>
        <div class="metric-card">
            <strong>Parameters:</strong> {params}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    return {"type": "text", "content": " "}


# ---------------------------------------------------------------------------
# 2. Cluster distribution – bar + pie
# ---------------------------------------------------------------------------

def create_cluster_distribution_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Bar chart and pie chart of cluster sizes."""
    r = metrics_snapshot
    raw_sizes  = r.get("Cluster Sizes", [])
    sizes      = _parse_cluster_sizes(raw_sizes)
    n_clusters = len(sizes)

    if n_clusters == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No cluster size data available",
                ha="center", va="center", transform=ax.transAxes)
        return {"type": "plot", "content": fig}

    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    labels = [f"Cluster {i}" for i in range(n_clusters)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=90)
    ax1.set_title("Cluster Proportion")

    # Bar
    bars = ax2.bar(range(n_clusters), sizes,
                   color=plt.cm.viridis(np.linspace(0, 1, n_clusters)))
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Number of Points")
    ax2.set_title("Cluster Sizes Distribution")
    ax2.set_xticks(range(n_clusters))
    ax2.set_xticklabels(labels, rotation=45, ha="right")

    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                 f"{size}", ha="center", va="bottom")

    plt.tight_layout()
    return {"type": "plot", "content": fig}


# ---------------------------------------------------------------------------
# 3. PCA 2-D scatter
# ---------------------------------------------------------------------------

def create_PCA_projection_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """2-D PCA scatter coloured by cluster label."""
    r              = metrics_snapshot
    X_scaled       = _parse_array(r.get("X_scaled", np.array([])))
    cluster_labels = _parse_array(r.get("cluster_labels", np.array([])))

    if X_scaled.ndim < 2 or X_scaled.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Need at least 2 features for PCA visualisation",
                ha="center", va="center", transform=ax.transAxes)
        return {"type": "plot", "content": fig}

    pca   = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=cluster_labels, cmap="tab10", alpha=0.7, s=40)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_title("Cluster Visualisation (PCA Projection)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    return {"type": "plot", "content": fig}


# ---------------------------------------------------------------------------
# 4. t-SNE 2-D scatter (skipped for large datasets)
# ---------------------------------------------------------------------------

def create_tsne_projection_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """2-D t-SNE scatter coloured by cluster label (≤ 1 000 samples only)."""
    r              = metrics_snapshot
    X_scaled       = _parse_array(r.get("X_scaled", np.array([])))
    cluster_labels = _parse_array(r.get("cluster_labels", np.array([])))

    if X_scaled.ndim < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No scaled data available for t-SNE",
                ha="center", va="center", transform=ax.transAxes)
        return {"type": "plot", "content": fig}

    n_samples = X_scaled.shape[0]

    if n_samples > 1000:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5,
                f"t-SNE skipped: dataset has {n_samples} samples (limit 1 000).\n"
                "Use PCA projection instead.",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, wrap=True)
        ax.set_axis_off()
        return {"type": "plot", "content": fig}

    perplexity = min(30, n_samples - 1)
    tsne       = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne     = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                         c=cluster_labels, cmap="tab10", alpha=0.7, s=40)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title("Cluster Visualisation (t-SNE)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()
    return {"type": "plot", "content": fig}


# ---------------------------------------------------------------------------
# 5. Silhouette plot
# ---------------------------------------------------------------------------

def create_silhouette_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Per-cluster silhouette bar chart."""
    r              = metrics_snapshot
    X_scaled       = _parse_array(r.get("X_scaled", np.array([])))
    cluster_labels = _parse_array(r.get("cluster_labels", np.array([])))
    silhouette_avg = r.get("Silhouette Score")
    n_clusters     = r.get("Number of Clusters", 0)

    if silhouette_avg is None or X_scaled.ndim < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Silhouette score not available (requires ≥ 2 clusters).",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return {"type": "plot", "content": fig}

    sil_vals = silhouette_samples(X_scaled, cluster_labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 0
    y_ticks = []

    for i in range(n_clusters):
        vals = np.sort(sil_vals[cluster_labels == i])
        y_upper = y_lower + len(vals)
        color = plt.cm.viridis(i / n_clusters)
        ax.barh(range(y_lower, y_upper), vals,
                height=1.0, edgecolor="none", color=color)
        y_ticks.append((y_lower + y_upper) / 2)
        y_lower = y_upper

    ax.axvline(silhouette_avg, color="red", linestyle="--",
               label=f"Average: {silhouette_avg:.3f}")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"Cluster {i}" for i in range(n_clusters)])
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Plot for Clusters")
    ax.legend()
    plt.tight_layout()
    return {"type": "plot", "content": fig}


# ---------------------------------------------------------------------------
# 6. Quality metrics bar chart
# ---------------------------------------------------------------------------

def create_quality_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Bar chart summarising the three clustering quality scores."""
    r          = metrics_snapshot
    silhouette = r.get("Silhouette Score")
    calinski   = r.get("Calinski-Harabasz Score")
    davies     = r.get("Davies-Bouldin Score")

    available = {k: v for k, v in {
        "Silhouette\nScore":          silhouette,
        "Calinski-Harabasz\nIndex":   calinski,
        "Davies-Bouldin\nIndex":      davies,
    }.items() if v is not None}

    if not available:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No quality metrics available.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return {"type": "plot", "content": fig}

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(list(available.keys()), list(available.values()),
                  color=["#4C72B0", "#55A868", "#C44E52"])

    for bar, val in zip(bars, available.values()):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01 * max(available.values()),
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Score")
    ax.set_title("Clustering Quality Metrics")
    ax.set_ylim(0, max(available.values()) * 1.15)
    plt.tight_layout()
    return {"type": "plot", "content": fig}


# ---------------------------------------------------------------------------
# 7. Cluster feature means table
# ---------------------------------------------------------------------------

def create_cluster_feature_means_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """DataFrame of mean feature values per cluster."""
    import pandas as pd

    r              = metrics_snapshot
    X_scaled       = _parse_array(r.get("X_scaled", np.array([])))
    cluster_labels = _parse_array(r.get("cluster_labels", np.array([])))
    features       = r.get("features", [])

    if X_scaled.ndim < 2 or len(features) == 0:
        return {"type": "text", "content": "<p>Feature data not available.</p>"}

    df = pd.DataFrame(X_scaled, columns=[str(f) for f in features])
    df["Cluster"] = cluster_labels.astype(int)
    means = df.groupby("Cluster").mean().round(4)
    return {"type": "dataframe", "content": means}


# ---------------------------------------------------------------------------
# 8. Performance interpretation text
# ---------------------------------------------------------------------------

def create_performance_interpretation(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Human-readable interpretation of the clustering quality scores."""
    r          = metrics_snapshot
    silhouette = r.get("Silhouette Score")
    davies     = r.get("Davies-Bouldin Score")
    n_clusters = r.get("Number of Clusters", 0)
    raw_sizes  = r.get("Cluster Sizes", [])
    sizes      = _parse_cluster_sizes(raw_sizes)

    lines = ["<ul>"]

    # Silhouette
    if silhouette is not None:
        if silhouette > 0.7:
            quality = "strong"
        elif silhouette > 0.5:
            quality = "reasonable"
        elif silhouette > 0.25:
            quality = "weak"
        else:
            quality = "no substantial"
        lines.append(
            f"<li><strong>Silhouette Score ({silhouette:.4f}):</strong> "
            f"Indicates <em>{quality}</em> clustering structure. "
            f"{'Consider refining the number of clusters.' if silhouette < 0.3 else 'Clusters are well-separated.'}</li>"
        )

    # Davies-Bouldin
    if davies is not None:
        lines.append(
            f"<li><strong>Davies-Bouldin Index ({davies:.4f}):</strong> "
            f"{'Lower values are better. ' if davies < 1.0 else 'Values above 1.0 suggest overlapping clusters. '}"
            f"{'Good separation between clusters.' if davies < 1.0 else 'Consider adjusting cluster parameters.'}</li>"
        )

    # Cluster count / outliers
    if -1 in (r.get("cluster_labels_unique") or []):
        outlier_idx = list(r.get("cluster_labels_unique", [])).index(-1)
        lines.append(
            f"<li><strong>Outliers:</strong> {sizes[outlier_idx] if outlier_idx < len(sizes) else '?'} "
            f"noise points detected (label = -1).</li>"
        )

    if n_clusters == 1:
        lines.append("<li>⚠️ Only one cluster found — the data may lack natural groupings.</li>")
    elif n_clusters > 10:
        lines.append(f"<li>ℹ️ {n_clusters} clusters found — verify whether this granularity is meaningful.</li>")
    else:
        lines.append(f"<li>✅ {n_clusters} distinct clusters identified.</li>")

    lines.append("</ul>")

    html = f"""
    <div style="padding:15px;border-radius:5px;border-left:4px solid #17a2b8;background:black;">
        <h4 style="margin-top:0;">📊 Performance Interpretation</h4>
        {"".join(lines)}
    </div>
    """
    return {"type": "text", "content": html}


# ---------------------------------------------------------------------------
# Orchestration layer  (mirrors hierarchical clustering pattern)
# ---------------------------------------------------------------------------

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    """Render every report section for the unsupervised KNN clustering model."""

    create_ml_summary_text(metrics_snapshot)

    st.subheader("📊 Cluster Size Distribution")
    asset = create_cluster_distribution_plot(metrics_snapshot)
    st.pyplot(asset["content"])
    plt.close(asset["content"])

    st.subheader("📐 2-D PCA Projection of Clusters")
    asset = create_PCA_projection_plot(metrics_snapshot)
    st.pyplot(asset["content"])
    plt.close(asset["content"])

    st.subheader("🔍 t-SNE Visualisation")
    asset = create_tsne_projection_plot(metrics_snapshot)
    st.pyplot(asset["content"])
    plt.close(asset["content"])

    st.subheader("🎭 Silhouette Analysis")
    asset = create_silhouette_plot(metrics_snapshot)
    st.pyplot(asset["content"])
    plt.close(asset["content"])

    st.subheader("📈 Clustering Quality Metrics")
    asset = create_quality_metrics_plot(metrics_snapshot)
    st.pyplot(asset["content"])
    plt.close(asset["content"])

    st.subheader("📋 Average Feature Values by Cluster")
    asset = create_cluster_feature_means_table(metrics_snapshot)
    if asset["type"] == "dataframe":
        st.dataframe(asset["content"].style.background_gradient(cmap="Blues"),
                     use_container_width=True)
    else:
        st.markdown(asset["content"], unsafe_allow_html=True)

    st.subheader("💡 Performance Interpretation")
    asset = create_performance_interpretation(metrics_snapshot)
    st.markdown(asset["content"], unsafe_allow_html=True)


def model_report():
    """Entry point called by the ML page for unsupervised KNN clustering."""
    found = False
    model_results = {}

    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == "Unsupervised KNN Clustering":
            if "metrics_snapshot" in model_info:
                found = True
                model_results = model_info["metrics_snapshot"]
                break

    if not found:
        st.error(
            "No Unsupervised KNN Clustering model results found. "
            "Please create a model first."
        )
        return

    st.markdown("---")
    st.markdown("## Unsupervised KNN Clustering Model Analysis (Live View)")
    display_ml_report(model_results)