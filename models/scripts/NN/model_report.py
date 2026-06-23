import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# --- Helper to safely retrieve metrics ---
def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common metrics from the snapshot."""
    try:
        accuracy = metrics_snapshot.get('Accuracy', 0)
        class_report = metrics_snapshot.get('Classification Report', {})
        
        precision_avg = recall_avg = f1_avg = 0
        if isinstance(class_report, dict):
            if 'macro avg' in class_report and isinstance(class_report['macro avg'], dict):
                macro_avg = class_report['macro avg']
                precision_avg = macro_avg.get('precision', 0)
                recall_avg = macro_avg.get('recall', 0)
                f1_avg = macro_avg.get('f1-score', 0)
            elif 'weighted avg' in class_report and isinstance(class_report['weighted avg'], dict):
                weighted_avg = class_report['weighted avg']
                precision_avg = weighted_avg.get('precision', 0)
                recall_avg = weighted_avg.get('recall', 0)
                f1_avg = weighted_avg.get('f1-score', 0)
        
        conf_matrix_data = metrics_snapshot.get('Confusion Matrix', [])
        if isinstance(conf_matrix_data, list) and len(conf_matrix_data) > 0:
            conf_matrix_array = np.array(conf_matrix_data)
        else:
            conf_matrix_array = np.array([])
        
        training_history = metrics_snapshot.get('training_history', {})
        num_classes = metrics_snapshot.get('num_classes', 2)
        
        class_df = pd.DataFrame()
        if isinstance(class_report, dict) and class_report:
            report_data = {}
            for key, value in class_report.items():
                if isinstance(value, dict):
                    report_data[key] = value
            if report_data:
                class_df = pd.DataFrame(report_data).transpose().round(4)
        
        return accuracy, precision_avg, recall_avg, f1_avg, class_df, conf_matrix_array, training_history, num_classes
    
    except Exception:
        return 0, 0, 0, 0, pd.DataFrame(), np.array([]), {}, 2

# --- Report Asset Generation Functions ---
def create_nn_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block."""
    accuracy, precision_avg, recall_avg, f1_avg, _, _, _, num_classes = _get_common_metrics(metrics_snapshot)
    
    best_params = metrics_snapshot.get('Best Parameters', {})
    if isinstance(best_params, dict):
        params_str = ", ".join([f"{k}: {v}" for k, v in best_params.items()])
    else:
        params_str = str(best_params)
    
    features = metrics_snapshot.get('features', [])
    
    text = f"""
    ### Neural Network Classifier Performance Summary
    - **Features:** {', '.join(features) if features else 'None'}
    - **Target:** {metrics_snapshot.get('target', 'Unknown')}
    - **Number of Classes:** {num_classes}
    - **Accuracy:** **{accuracy:.4f}**
    - **Precision (Macro):** {precision_avg:.4f}
    - **Recall (Macro):** {recall_avg:.4f}
    - **F1-Score (Macro):** {f1_avg:.4f}
    - **Best Parameters:** {params_str}
    - **Training Method:** {"Grid Search" if metrics_snapshot.get('use_grid_search', False) else "Manual"}
    """
    
    return {"type": "text", "content": text}

def create_nn_confusion_matrix_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Confusion Matrix plot."""
    _, _, _, _, _, conf_matrix_array, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if conf_matrix_array.size > 0 and conf_matrix_array.ndim == 2:
        sns.heatmap(conf_matrix_array, 
                    annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix - Neural Network')
    else:
        ax.text(0.5, 0.5, 'No confusion matrix data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Confusion Matrix - Neural Network')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_nn_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the Classification Report table."""
    _, _, _, _, class_df, _, _, _ = _get_common_metrics(metrics_snapshot)
    return {"type": "dataframe", "content": class_df}

def create_nn_performance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main metrics bar chart."""
    accuracy, precision_avg, recall_avg, f1_avg, _, _, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision_avg, recall_avg, f1_avg]
    
    bars = ax.bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Neural Network Performance Metrics')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_nn_training_history_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the training history plots."""
    _, _, _, _, _, _, training_history, _ = _get_common_metrics(metrics_snapshot)
    
    if not training_history:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No training history available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training History')
        plt.tight_layout()
        plt.close(fig)
        return {"type": "plot", "content": fig}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    if 'accuracy' in training_history:
        ax1.plot(training_history['accuracy'], label='Training Accuracy', marker='o', markersize=3)
    if 'val_accuracy' in training_history:
        ax1.plot(training_history['val_accuracy'], label='Validation Accuracy', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    if 'loss' in training_history:
        ax2.plot(training_history['loss'], label='Training Loss', marker='o', markersize=3)
    if 'val_loss' in training_history:
        ax2.plot(training_history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_nn_architecture_summary(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates architecture summary as a table."""
    best_params = metrics_snapshot.get('Best Parameters', {})
    
    architecture_data = {
        'Parameter': ['Hidden Layers', 'Layer Neurons', 'Dropout Rates', 'Activation', 'Output Activation'],
        'Value': [
            str(best_params.get('num_layers', 'N/A')),
            str(best_params.get('layer_neurons', 'N/A')),
            str(best_params.get('layer_dropouts', 'N/A')),
            best_params.get('activation', 'N/A'),
            best_params.get('output_activation', 'N/A')
        ]
    }
    
    training_data = {
        'Parameter': ['Epochs', 'Batch Size', 'Learning Rate', 'Optimizer', 'Validation Split', 'Early Stopping'],
        'Value': [
            str(best_params.get('epochs', 'N/A')),
            str(best_params.get('batch_size', 'N/A')),
            str(best_params.get('learning_rate', 'N/A')),
            best_params.get('optimizer', 'N/A'),
            str(best_params.get('validation_split', 'N/A')),
            str(best_params.get('early_stopping', 'N/A'))
        ]
    }
    
    arch_df = pd.DataFrame(architecture_data)
    train_df = pd.DataFrame(training_data)
    
    return {"type": "dataframe_pair", "content": (arch_df, train_df)}

# --- Main Model Report Function ---
def model_report():
    """Main function to find and display Neural Network model results."""
    
    found = False
    model_results = None
    
    if hasattr(st.session_state, 'pipeline') and "ML" in st.session_state.pipeline:
        for model_info in st.session_state.pipeline["ML"]:
            model_name = (model_info.get('model name') or 
                         model_info.get('name') or 
                         model_info.get('model_type') or 
                         model_info.get('type', ''))
            
            if model_name == 'Neural Network Classifier':
                if 'metrics_snapshot' in model_info:
                    model_results = model_info['metrics_snapshot']
                    found = True
                    break
    
    if not found:
        st.error("No Neural Network model results found. Please train a model first.")
        return
    
    st.markdown("---")
    st.markdown("## 🧠 Neural Network Model Analysis")
    
    try:
        summary_asset = create_nn_summary_text(model_results)
        table_asset = create_nn_classification_report_table(model_results)
        metrics_plot_asset = create_nn_performance_plot(model_results)
        conf_matrix_asset = create_nn_confusion_matrix_plot(model_results)
        history_asset = create_nn_training_history_plot(model_results)
        architecture_asset = create_nn_architecture_summary(model_results)
        
        # Display Summary
        st.markdown(summary_asset['content'])
        
        # Display Architecture Summary
        st.subheader("🏗️ Model Architecture")
        arch_df, train_df = architecture_asset['content']
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(arch_df, use_container_width=True, hide_index=True)
        with col2:
            st.dataframe(train_df, use_container_width=True, hide_index=True)
        
        # Display Training History
        st.subheader("📈 Training History")
        st.pyplot(history_asset['content'])
        
        # Display Metrics Plot
        st.subheader("📊 Key Metrics")
        st.pyplot(metrics_plot_asset['content'])
        
        # Display Confusion Matrix
        st.subheader("🎯 Confusion Matrix")
        st.pyplot(conf_matrix_asset['content'])
        
        # Display Classification Table
        st.subheader("📋 Detailed Classification Report")
        if not table_asset['content'].empty:
            st.dataframe(table_asset['content'], use_container_width=True)
        else:
            st.info("Classification report not available")
        
        # Performance Interpretation
        st.subheader("📊 Performance Interpretation")
        
        accuracy = model_results.get('Accuracy', 0)
        if accuracy >= 0.9:
            st.success("**Excellent Performance** - The neural network shows outstanding classification ability!")
        elif accuracy >= 0.8:
            st.info("**Good Performance** - The neural network performs well on the classification task.")
        elif accuracy >= 0.7:
            st.warning("**Fair Performance** - The neural network has acceptable performance but could be improved with more data or tuning.")
        else:
            st.error("**Poor Performance** - Consider increasing training epochs, adjusting architecture, or collecting more data.")
            
    except Exception as e:
        st.error(f"Error displaying Neural Network model report: {e}")