import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from typing import Dict, Any


def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common metrics from the snapshot."""
    if not metrics_snapshot:
        return None, None, None, None, None, pd.DataFrame(), {}, None
    
    mse = metrics_snapshot.get('MSE', 0)
    rmse = metrics_snapshot.get('RMSE', 0)
    mae = metrics_snapshot.get('MAE', 0)
    r2 = metrics_snapshot.get('R2 Score', 0)
    best_params = metrics_snapshot.get('Best Parameters', {})
    feature_importance = metrics_snapshot.get('feature_importance', [])
    
    # Create a summary DataFrame
    metrics_summary = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score'],
        'Value': [mse, rmse, mae, r2]
    })
    
    return mse, rmse, mae, r2, best_params, metrics_summary, feature_importance

# --- Granular Report Asset Generation Functions ---

def create_dtr_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block (Report Asset Type: text)."""
    mse, rmse, mae, r2, best_params, _, _ = _get_common_metrics(metrics_snapshot)
    
    text = """
    ### Decision Tree Regressor Performance Summary
    - **Features:** {features}
    - **Target:** {target}
    - **Mean Squared Error:** **{mse:.4f}**
    - **Root Mean Squared Error:** {rmse:.4f}
    - **Mean Absolute Error:** {mae:.4f}
    - **R² Score:** {r2:.4f}
    - **Best Parameters:** {best_params}
    - **Training Method:** {grid_search}
    """.format(
        features=", ".join(metrics_snapshot.get('features', [])),
        target=metrics_snapshot.get('target', 'Unknown'),
        mse=mse,
        rmse=rmse,
        mae=mae,
        r2=r2,
        best_params=best_params,
        grid_search="Grid Search" if metrics_snapshot.get('use_grid_search', False) else "Manual"
    )
    return {"type": "text", "content": text}


def create_metrics_summary_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the metrics summary table (Report Asset Type: dataframe)."""
    _, _, _, _, _, metrics_summary, _ = _get_common_metrics(metrics_snapshot)
    return {"type": "dataframe", "content": metrics_summary}


def create_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the performance metrics bar chart (Report Asset Type: plot)."""
    mse, rmse, mae, r2, _, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['MSE', 'RMSE', 'MAE', 'R² Score']
    values = [mse, rmse, mae, r2]
    
    # Use different colors for different metrics
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax.bar(metrics, values, color=colors)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}


def create_feature_importance_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates feature importance visualization (Report Asset Type: plot)."""
    _, _, _, _, _, _, feature_importance = _get_common_metrics(metrics_snapshot)
    features = metrics_snapshot.get('features', [])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if feature_importance and len(feature_importance) == len(features):
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        bars = sns.barplot(data=importance_df, x='Importance', y='Feature', 
                          ax=ax, palette='viridis')
        ax.set_title('Feature Importance')
        
        # Add value labels
        for i, (_, row) in enumerate(importance_df.iterrows()):
            ax.text(row['Importance'] + 0.001, i, 
                   f'{row["Importance"]:.3f}', 
                   va='center', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Feature importance data not available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Feature Importance')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}


def create_feature_importance_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates feature importance table (Report Asset Type: dataframe)."""
    _, _, _, _, _, _, feature_importance = _get_common_metrics(metrics_snapshot)
    features = metrics_snapshot.get('features', [])
    
    if feature_importance and len(feature_importance) == len(features):
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).round(4)
        return {"type": "dataframe", "content": importance_df}
    else:
        return {"type": "dataframe", "content": pd.DataFrame({'Message': ['No feature importance data available']})}


def create_decision_tree_visualization(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates decision tree visualization (Report Asset Type: plot)."""
    for model_info in st.session_state.pipeline.get("ML", []):
        model_type = model_info.get("model type", "")
        # Check for various Decision Tree identifiers
        if any(keyword in str(model_type) for keyword in ["Decision Tree Regressor", "Decision Tree", "DTR"]):
            model_found = True
            metrics_snapshot = model_info.get("metrics_snapshot", {})
            # Try to get the actual model if available
            if 'model' in model_info:
                try:
                    model = model_info['model']

                except Exception as e:
                    st.warning(f"Could not load saved model: {e}")
            break
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = metrics_snapshot.get('features', [])
    
    if model is not None and features:
        try:
            # Limit tree depth for better visualization if it's too large
            max_depth_to_show = min(model.get_depth() if hasattr(model, 'get_depth') else 3, 3)
            
            plot_tree(model, 
                     feature_names=features,
                     filled=True,
                     rounded=True,
                     ax=ax,
                     max_depth=max_depth_to_show)  # Limit depth for readability
            ax.set_title(f'Decision Tree Visualization (Depth: {max_depth_to_show})')
            
        except Exception as e:
            error_msg = f'Tree visualization error:\n{str(e)[:100]}'
            ax.text(0.5, 0.5, error_msg, 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Decision Tree Visualization')
    else:
        if model is None:
            msg = 'Model not available for visualization\n\nPlease train the model first'
        else:
            msg = 'Features not available for visualization'
        
        ax.text(0.5, 0.5, msg, 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.set_title('Decision Tree Visualization')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}


def create_actual_vs_predicted_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:

    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_test = metrics_snapshot.get('y_test')
    y_pred = metrics_snapshot.get('y_pred')
    
    if y_test is not None and y_pred is not None and len(y_test) > 0:
        try:
            # Convert to numpy arrays if they aren't already
            y_test = np.array(y_test)
            y_pred = np.array(y_pred)
            
            # Create scatter plot with some styling
            scatter = ax.scatter(y_test, y_pred, alpha=0.6, color='steelblue', 
                               edgecolors='white', s=50)
            
            # Add perfect prediction line (y = x)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=2, label='Perfect Prediction')
            
            # Calculate and display R² score
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Actual Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title('Actual vs Predicted Values', fontsize=14)
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            error_msg = f'Error creating plot:\n{str(e)[:80]}'
            ax.text(0.5, 0.5, error_msg, 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Actual vs Predicted Values')
    else:
        # Provide helpful guidance
        missing_data = []
        if y_test is None:
            missing_data.append("test data (y_test)")
        if y_pred is None:
            missing_data.append("predictions (y_pred)")
        
        msg = 'Actual vs Predicted plot requires:\n'
        msg += '- Test data stored in model_results\n'
        msg += '- Predictions from the model\n\n'
        
        if missing_data:
            msg += f'Missing: {", ".join(missing_data)}'
        else:
            msg += 'No data available'
        
        ax.text(0.5, 0.5, msg, 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Actual vs Predicted Values')
    
    plt.tight_layout()
    plt.close(fig)
    
    return {"type": "plot", "content": fig}

def create_hyperparameters_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates hyperparameters display (Report Asset Type: text)."""
    _, _, _, _, best_params, _, _ = _get_common_metrics(metrics_snapshot)
    
    if isinstance(best_params, dict) and best_params:
        params_text = "**Hyperparameters:**\n"
        for param, value in best_params.items():
            params_text += f"- **{param}:** {value}\n"
    else:
        params_text = f"**Best Parameters:** {best_params}\n"
    
    params_text += f"\n**Training Method:** {'Grid Search' if metrics_snapshot.get('use_grid_search', False) else 'Manual'}"
    
    return {"type": "text", "content": params_text}


def create_dataset_info_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates dataset information display (Report Asset Type: text)."""
    for model_info in st.session_state.pipeline.get("ML", []):
        model_type = model_info.get("model type", "")
        # Check for various Decision Tree identifiers
        if any(keyword in str(model_type) for keyword in ["Decision Tree Regressor", "Decision Tree", "DTR"]):
            model_found = True
            metrics_snapshot = model_info.get("metrics_snapshot", {})
            # Try to get the actual model if available
            if 'model' in model_info:
                try:
                    model = model_info['model']

                except Exception as e:
                    st.warning(f"Could not load saved model: {e}")
            break
    
    num_features = len(metrics_snapshot.get('features', []))
    target = metrics_snapshot.get('target', 'Unknown')
    
    tree_depth = "N/A"
    num_leaves = "N/A"
    
    if model is not None:
        if hasattr(model, 'get_depth'):
            tree_depth = model.get_depth()
        if hasattr(model, 'get_n_leaves'):
            num_leaves = model.get_n_leaves()
    
    info_text = f"""
    **Dataset Information:**
    - Number of features: **{num_features}**
    - Target variable: **{target}**
    - Tree depth: **{tree_depth}**
    - Number of leaves: **{num_leaves}**
    """
    
    return {"type": "text", "content": info_text}


def create_performance_interpretation(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates performance interpretation text (Report Asset Type: text)."""
    _, rmse, _, r2, _, _, _ = _get_common_metrics(metrics_snapshot)
    
    if r2 >= 0.9:
        interpretation = "**🎉 Excellent Performance** - The model explains almost all the variance in the target variable!"
        color = "green"
    elif r2 >= 0.7:
        interpretation = "**👍 Good Performance** - The model has strong predictive power."
        color = "blue"
    elif r2 >= 0.5:
        interpretation = "**⚠️ Fair Performance** - The model has moderate predictive ability."
        color = "orange"
    else:
        interpretation = "**❌ Poor Performance** - Consider feature engineering, parameter tuning, or trying a different algorithm."
        color = "red"
    
    interpretation += f"\n\n**Model Statistics:**"
    interpretation += f"\n- R² Score: **{r2:.4f}** (closer to 1 is better)"
    interpretation += f"\n- RMSE: **{rmse:.4f}** (lower is better)"
    
    return {"type": "text", "content": f'<span style="color:{color}">{interpretation}</span>'}


# --- Main Display Function ---

def display_dtr_report(metrics_snapshot: Dict[str, Any], model_results: Any):
    """
    Function to be called on the main ML page. It uses the granular functions
    to generate and display all components using Streamlit commands.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return
    
    if model_results is None:
        model_results = {}

    st.markdown("---")
    st.markdown("## Decision Tree Regressor Analysis")
    
    # Get all the report assets with error handling
    try:
        summary_asset = create_dtr_summary_text(metrics_snapshot)
        metrics_table_asset = create_metrics_summary_table(metrics_snapshot)
        metrics_plot_asset = create_performance_metrics_plot(metrics_snapshot)
        feature_importance_plot_asset = create_feature_importance_plot(metrics_snapshot)
        feature_importance_table_asset = create_feature_importance_table(metrics_snapshot)
        tree_viz_asset = create_decision_tree_visualization(metrics_snapshot)
        actual_vs_predicted_asset = create_actual_vs_predicted_plot(metrics_snapshot, model_results)
        params_asset = create_hyperparameters_table(metrics_snapshot)
        dataset_asset = create_dataset_info_table(metrics_snapshot)
        interpretation_asset = create_performance_interpretation(metrics_snapshot)
    except Exception as e:
        st.error(f"Error generating report assets: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return
    
    # Display Summary
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    # Performance Interpretation
    st.markdown("### 📊 Performance Interpretation")
    st.markdown(interpretation_asset['content'], unsafe_allow_html=True)
    
    # Display Metrics Plot
    st.markdown("### 📈 Performance Metrics Visualization")
    st.pyplot(metrics_plot_asset['content'])
    
    # Display Metrics Table
    st.markdown("### 📋 Performance Metrics Summary")
    st.dataframe(metrics_table_asset['content'], use_container_width=True)
    
    # Display Feature Importance
    st.markdown("### 🔑 Feature Importance")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.pyplot(feature_importance_plot_asset['content'])
    
    with col2:
        st.dataframe(feature_importance_table_asset['content'], use_container_width=True)
    
    # Display Decision Tree Visualization
    st.markdown("### 🌳 Decision Tree Visualization")
    st.pyplot(tree_viz_asset['content'])
    
    # Display Actual vs Predicted Plot
    st.markdown("### 📊 Actual vs Predicted Values")
    st.pyplot(actual_vs_predicted_asset['content'])
    
    # Configuration Details
    st.markdown("### ⚙️ Model Configuration")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown(params_asset['content'], unsafe_allow_html=True)
    
    with col4:
        st.markdown(dataset_asset['content'], unsafe_allow_html=True)


# --- Compatibility function for your existing system ---

def model_report():
    """Main function to maintain compatibility with your existing system."""
    # Try multiple ways to find the model data
    metrics_snapshot = None
    model_results = None
    model_found = False
    
    # Method 1: Look for Decision Tree in pipeline ML
    for model_info in st.session_state.pipeline.get("ML", []):
        model_type = model_info.get("model type", "")
        # Check for various Decision Tree identifiers
        if any(keyword in str(model_type) for keyword in ["Decision Tree Regressor", "Decision Tree", "DTR"]):
            model_found = True
            metrics_snapshot = model_info.get("metrics_snapshot", {})
            # Try to get the actual model if available
            if 'model' in model_info:
                try:
                    model_obj = model_info['model']

                except Exception as e:
                    st.warning(f"Could not load saved model: {e}")
            break
    
    # Method 2: Check session state for model_results
    if not model_found and 'model_results' in st.session_state:
        model_results_data = st.session_state.model_results
        if model_results_data and 'metrics' in model_results_data:
            model_found = True
            metrics_snapshot = model_results_data.get('metrics', {})
            model_results = model_results_data
    
    if not model_found:
        st.error("No Decision Tree Regressor model results found. Please create a model first.")
        return
    
    if not metrics_snapshot:
        st.error("Found model but no metrics data available.")
        return
    
    # Ensure model_results has necessary structure
    if model_results is None:
        model_results = {
            'model': None,
            'y_test': None,
            'y_pred': None
        }
    else:
        if 'model' not in model_results:
            model_results['model'] = None
        if 'y_test' not in model_results:
            model_results['y_test'] = None
        if 'y_pred' not in model_results:
            model_results['y_pred'] = None
    
    # Pass the metrics snapshot to the display function
    try:
        display_dtr_report(metrics_snapshot, model_obj)
    except Exception as e:
        st.error(f"Error displaying report: {str(e)}")
        import traceback
        st.code(traceback.format_exc())