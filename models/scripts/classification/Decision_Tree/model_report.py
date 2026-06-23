import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import plot_tree
from typing import Dict, Any
import joblib, base64
from io import BytesIO
from html import escape

def _load_model(metrics_snapshot: Dict[str, Any]):
    """Helper: deserialise the base64-encoded model stored in the snapshot."""
    model_b64 = metrics_snapshot.get("model_b64")
    if model_b64 is None:
        return None
    buffer = BytesIO(base64.b64decode(model_b64))
    return joblib.load(buffer)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Summary (Text)
# ─────────────────────────────────────────────────────────────────────────────
def _get_best_params(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Support both saved key styles for best parameters."""
    for key in ("best_params", "Best Parameters", "best_parameters"):
        value = metrics_snapshot.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def create_dt_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Returns an HTML summary card with features, target and key metrics."""
    features   = metrics_snapshot.get("features", [])
    target     = metrics_snapshot.get("target", "N/A")
    metrics    = metrics_snapshot.get("metrics", {})
    gs_used    = metrics_snapshot.get("use_grid_search", False)
    best_params = _get_best_params(metrics_snapshot)
 
    gs_badge = (
        '<span style="color:#28a745;font-weight:bold;">✔ Grid Search Optimised</span>'
        if gs_used else
        '<span style="color:#6c757d;">Default Parameters</span>'
    )
 
    best_params_html = ""
    if gs_used and best_params:
        rows = "".join(
            f"<tr><td style='padding:2px 8px'><b>{escape(str(k))}</b></td>"
            f"<td style='padding:2px 8px'>{escape(str(v))}</td></tr>"
            for k, v in best_params.items()
        )
        best_params_html = f"""
        <p><strong>Best Parameters (Grid Search):</strong></p>
        <table style='border-collapse:collapse;font-size:0.9em'>{rows}</table>
        """
 
    html = f"""
    <div style="background:black;padding:18px;border-radius:10px;
                border-left:5px solid #4e79a7;margin-bottom:12px;">
        <h3 style="margin-top:0;">🌳 Decision Tree Classifier – Model Summary</h3>
        <p><strong>Training Mode:</strong> {gs_badge}</p>
        <p><strong>Features ({len(features)}):</strong> {', '.join(features)}</p>
        <p><strong>Target:</strong> {target}</p>
        <hr style="border:none;border-top:1px solid #ccc;"/>
        <table style="border-collapse:collapse;width:100%;font-size:0.95em;">
            <tr>
                <td style="padding:4px 12px"><b>Accuracy</b></td>
                <td style="padding:4px 12px">{metrics.get('Accuracy', 0):.4f}</td>
                <td style="padding:4px 12px"><b>Precision</b></td>
                <td style="padding:4px 12px">{metrics.get('Precision', 0):.4f}</td>
            </tr>
            <tr>
                <td style="padding:4px 12px"><b>Recall</b></td>
                <td style="padding:4px 12px">{metrics.get('Recall', 0):.4f}</td>
                <td style="padding:4px 12px"><b>F1 Score</b></td>
                <td style="padding:4px 12px">{metrics.get('F1 Score', 0):.4f}</td>
            </tr>
        </table>
        {best_params_html}
    </div>
    """
    return {"type": "text", "content": html}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Performance Metrics Bar Chart (Plot)
# ─────────────────────────────────────────────────────────────────────────────
def create_dt_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Bar chart comparing Accuracy, Precision, Recall and F1 Score."""
    metrics = metrics_snapshot.get("metrics", {})
    labels  = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values  = [metrics.get(k, 0) for k in labels]
    colors  = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
 
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Decision Tree – Performance Metrics")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return {"type": "plot", "content": fig}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix (Plot)
# ─────────────────────────────────────────────────────────────────────────────
def create_dt_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Heatmap of the confusion matrix."""
    conf_matrix = metrics_snapshot.get("metrics", {}).get("Confusion Matrix", [])
    conf_matrix = np.array(conf_matrix)
    classes = [str(i) for i in range(len(conf_matrix))]
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Predicted Labels", fontsize=11)
    ax.set_ylabel("True Labels", fontsize=11)
    ax.set_title("Confusion Matrix", fontsize=13)
    plt.tight_layout()
    return {"type": "plot", "content": fig}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. Classification Report Table
# ─────────────────────────────────────────────────────────────────────────────
def create_dt_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """HTML table with per-class precision, recall and F1 from the stored report."""
    report = metrics_snapshot.get("classification_report", {})
    if not report:
        return {"type": "text", "content": "<p>No classification report available.</p>"}
 
    rows_html = ""
    skip_keys = {"accuracy", "macro avg", "weighted avg"}
    for label, row_data in report.items():
        if label in skip_keys or not isinstance(row_data, dict):
            continue
        rows_html += (
            f"<tr>"
            f"<td style='padding:6px 14px'>{label}</td>"
            f"<td style='padding:6px 14px;text-align:center'>{row_data.get('precision', 0):.3f}</td>"
            f"<td style='padding:6px 14px;text-align:center'>{row_data.get('recall', 0):.3f}</td>"
            f"<td style='padding:6px 14px;text-align:center'>{row_data.get('f1-score', 0):.3f}</td>"
            f"<td style='padding:6px 14px;text-align:center'>{int(row_data.get('support', 0))}</td>"
            f"</tr>"
        )
    # Averages footer
    for avg_key in ("macro avg", "weighted avg"):
        if avg_key in report and isinstance(report[avg_key], dict):
            d = report[avg_key]
            rows_html += (
                f"<tr style='background:#blue;font-weight:bold'>"
                f"<td style='padding:6px 14px'>{avg_key.title()}</td>"
                f"<td style='padding:6px 14px;text-align:center'>{d.get('precision',0):.3f}</td>"
                f"<td style='padding:6px 14px;text-align:center'>{d.get('recall',0):.3f}</td>"
                f"<td style='padding:6px 14px;text-align:center'>{d.get('f1-score',0):.3f}</td>"
                f"<td style='padding:6px 14px;text-align:center'>{int(d.get('support',0))}</td>"
                f"</tr>"
            )
 
    html = f"""
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse;width:100%;font-size:0.92em;
                    border:1px solid #dee2e6;border-radius:6px;">
        <thead style="background:#4e79a7;color:black;">
          <tr>
            <th style="padding:8px 14px;text-align:left">Class</th>
            <th style="padding:8px 14px">Precision</th>
            <th style="padding:8px 14px">Recall</th>
            <th style="padding:8px 14px">F1-Score</th>
            <th style="padding:8px 14px">Support</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """
    return {"type": "text", "content": html}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. Decision Tree Structure Visualisation (Plot)
# ─────────────────────────────────────────────────────────────────────────────
def create_dt_tree_structure_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Renders the full decision-tree diagram using sklearn's plot_tree."""
    model    = _load_model(metrics_snapshot)
    features = metrics_snapshot.get("features", [])
    conf_matrix = metrics_snapshot.get("metrics", {}).get("Confusion Matrix", [])
    n_classes   = len(conf_matrix) if conf_matrix else 2
    classes     = [str(i) for i in range(n_classes)]
 
    if model is None:
        return {"type": "text", "content": "<p>Model not available for tree plot.</p>"}
 
    max_d = model.get_depth() if hasattr(model, "get_depth") else 10
    fig_w = min(30, max(15, max_d * 3))
    fig, ax = plt.subplots(figsize=(fig_w, 10))
    plot_tree(
        model,
        feature_names=features,
        class_names=classes,
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=max(6, 10 - max_d),
        impurity=True,
        proportion=False,
    )
    ax.set_title("Decision Tree Structure", fontsize=14)
    plt.tight_layout()
    return {"type": "plot", "content": fig}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature Importance (Plot)
# ─────────────────────────────────────────────────────────────────────────────
def create_dt_feature_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Horizontal bar chart of Gini-based feature importances."""
    model    = _load_model(metrics_snapshot)
    features = metrics_snapshot.get("features", [])
 
    if model is None or not hasattr(model, "feature_importances_"):
        return {"type": "text", "content": "<p>Feature importances not available.</p>"}
 
    importances = model.feature_importances_
    indices     = np.argsort(importances)          # ascending for horizontal bar
 
    fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.45)))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    ax.barh([features[i] for i in indices], importances[indices], color=colors)
    ax.set_xlabel("Gini Importance")
    ax.set_title("Decision Tree – Feature Importance")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return {"type": "plot", "content": fig}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. Performance Interpretation (Text)
# ─────────────────────────────────────────────────────────────────────────────
def create_dt_performance_interpretation(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a plain-language interpretation of the model's performance."""
    metrics = metrics_snapshot.get("metrics", {})
    acc = metrics.get("Accuracy", 0)
    f1  = metrics.get("F1 Score", 0)
    prec = metrics.get("Precision", 0)
    rec  = metrics.get("Recall", 0)
 
    def _grade(v):
        if v >= 0.90: return "Excellent", "#28a745"
        if v >= 0.75: return "Good",      "#5bc0de"
        if v >= 0.60: return "Moderate",  "#f0ad4e"
        return "Needs Improvement", "#d9534f"
 
    acc_g,  acc_c  = _grade(acc)
    f1_g,   f1_c   = _grade(f1)
    prec_g, prec_c = _grade(prec)
    rec_g,  rec_c  = _grade(rec)
 
    html = f"""
    <div style="background:black;padding:18px;border-radius:10px;
                border-left:5px solid #f0ad4e;margin-bottom:12px;">
        <h4 style="margin-top:0;">📝 Performance Interpretation</h4>
        <ul style="line-height:1.8;font-size:0.95em;">
            <li><b>Accuracy {acc:.3f}</b> –
                <span style="color:{acc_c};font-weight:bold">{acc_g}</span>.
                {acc*100:.1f}% of all samples were classified correctly.</li>
            <li><b>Precision {prec:.3f}</b> –
                <span style="color:{prec_c};font-weight:bold">{prec_g}</span>.
                Of all positive predictions, {prec*100:.1f}% were actually positive.</li>
            <li><b>Recall {rec:.3f}</b> –
                <span style="color:{rec_c};font-weight:bold">{rec_g}</span>.
                The model detected {rec*100:.1f}% of all actual positive cases.</li>
            <li><b>F1 Score {f1:.3f}</b> –
                <span style="color:{f1_c};font-weight:bold">{f1_g}</span>.
                Harmonic mean of Precision and Recall.</li>
        </ul>
        {'<p style="color:#856404;background:#fff3cd;padding:8px;border-radius:4px;">'
         '⚠️ Consider tuning max_depth or min_samples_leaf to reduce overfitting '
         'if train accuracy is significantly higher than test accuracy.</p>'
         if acc < 0.70 else ''}
    </div>
    """
    return {"type": "text", "content": html}


# ─────────────────────────────────────────────────────────────────────────────
# 8. Live Display & Compatibility Entry Points
# ─────────────────────────────────────────────────────────────────────────────
def display_ml_report(metrics_snapshot: Dict[str, Any]):
    """
    Renders all Decision Tree report components inline on the model page
    using the granular helper functions above.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return

    st.markdown("---")
    st.markdown("## 🌳 Decision Tree Model Analysis (Live View)")

    summary_asset      = create_dt_summary_text(metrics_snapshot)
    metrics_plot_asset = create_dt_performance_metrics_plot(metrics_snapshot)
    conf_matrix_asset  = create_dt_confusion_matrix_plot(metrics_snapshot)
    report_table_asset = create_dt_classification_report_table(metrics_snapshot)
    interp_asset       = create_dt_performance_interpretation(metrics_snapshot)
    feat_imp_asset     = create_dt_feature_importance_plot(metrics_snapshot)
    tree_plot_asset    = create_dt_tree_structure_plot(metrics_snapshot)

    # Summary card
    st.markdown(summary_asset["content"], unsafe_allow_html=True)

    # Performance metrics bar chart
    st.subheader("📈 Key Metrics Visualization")
    st.pyplot(metrics_plot_asset["content"])

    # Confusion matrix
    st.subheader("🎯 Confusion Matrix")
    st.pyplot(conf_matrix_asset["content"])

    # Per-class classification report
    st.subheader("📊 Detailed Classification Report")
    st.markdown(report_table_asset["content"], unsafe_allow_html=True)

    # Plain-language interpretation
    st.markdown(interp_asset["content"], unsafe_allow_html=True)

    # Feature importance
    st.subheader("🔍 Feature Importance")
    if feat_imp_asset["type"] == "plot":
        st.pyplot(feat_imp_asset["content"])
    else:
        st.markdown(feat_imp_asset["content"], unsafe_allow_html=True)

    # Full tree diagram — wrapped in expander since it can be very wide
    with st.expander("🌲 Decision Tree Structure", expanded=False):
        if tree_plot_asset["type"] == "plot":
            st.pyplot(tree_plot_asset["content"])
        else:
            st.markdown(tree_plot_asset["content"], unsafe_allow_html=True)


def model_report():
    """
    Compatibility entry point: locates the Decision Tree snapshot stored in
    the pipeline and delegates to display_ml_report.
    """
    found = False
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == "Decision Tree Classifier":
            if "metrics_snapshot" in model_info:
                found = True
                model_results = model_info["metrics_snapshot"]
                break

    if not found:
        st.error("No Decision Tree results found. Please train the model first.")
        return

    display_ml_report(model_results)