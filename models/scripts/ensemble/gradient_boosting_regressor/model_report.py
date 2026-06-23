import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# --- Helper to safely retrieve metrics and convert JSON-safe lists back to arrays ---

def _get_common_metrics(metrics_snapshot: Dict[str, Any]):
    """Extracts common regression metrics."""
    # --- Regression Metrics ---
    r2 = metrics_snapshot.get('R2 Score', 0)
    mae = metrics_snapshot.get('MAE', 0)
    mse = metrics_snapshot.get('MSE', 0)
    rmse = metrics_snapshot.get('RMSE', 0)
    
    # Placeholder/Dummy for compatibility, not used in regressor
    class_df = pd.DataFrame() 
    conf_matrix_array = np.array([]) 
    
    return r2, mae, mse, rmse, class_df, conf_matrix_array

# --- Granular Report Asset Generation Functions (Input: metrics_snapshot) ---

def create_ml_summary_text(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates the main summary text block (Report Asset Type: text)."""
    # Use R2, MAE, MSE, RMSE
    r2, mae, mse, rmse, _, _ = _get_common_metrics(metrics_snapshot)

    text = """
    ### Gradient Boosting Regressor Performance Summary
    - **Features:** {features}
    - **Target:** {target}
    - **R2 Score:** **{r2:.4f}**
    - **Mean Absolute Error (MAE):** {mae:.4f}
    - **Mean Squared Error (MSE):** {mse:.4f}
    - **Root Mean Squared Error (RMSE):** {rmse:.4f}
    - **Best Parameters:** {best_params}
    - **Training Method:** {grid_search}
    """.format(
        features=", ".join(metrics_snapshot['features']),
        target=metrics_snapshot['target'],
        r2=r2,
        mae=mae,
        mse=mse,
        rmse=rmse,
        best_params=metrics_snapshot['Best Parameters'],
        grid_search="Grid Search" if metrics_snapshot['use_grid_search'] else "Manual"
    )
    return {"type": "text", "content": text}

def create_classification_report_table(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a summary table of regression metrics."""
    
    regression_summary = {
        'Metric': ['R2 Score', 'MAE', 'MSE', 'RMSE'],
        'Value': [
            f"{metrics_snapshot.get('R2 Score', 0):.4f}", 
            f"{metrics_snapshot.get('MAE', 0):.4f}", 
            f"{metrics_snapshot.get('MSE', 0):.4f}",
            f"{metrics_snapshot.get('RMSE', 0):.4f}"
        ]
    }
    df = pd.DataFrame(regression_summary)
    return {"type": "dataframe", "content": df}

def create_performance_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a bar chart for main regression metrics (Report Asset Type: plot)."""
    r2, mae, mse, rmse, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We plot R2 score separately as it's typically between 0 and 1.
    metrics = ['$R^2$ Score']
    values = [r2]
    
    bars = ax.bar(metrics, values, color=['#4CAF50'])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance - $R^2$ Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
                
    plt.close(fig)
    return {"type": "plot", "content": fig}

def create_error_metrics_plot(metrics_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a bar chart for error metrics (Report Asset Type: plot)."""
    r2, mae, mse, rmse, _, _ = _get_common_metrics(metrics_snapshot)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    error_metrics = ['RMSE', 'MSE', 'MAE']
    error_values = [rmse, mse, mae]
    error_colors = ['#FF9800', '#F44336', '#2196F3']
    
    bars_err = ax.bar(error_metrics, error_values, color=error_colors)
    ax.set_ylabel('Error Value')
    ax.set_title('Model Performance - Error Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars_err, error_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.close(fig)
    return {"type": "plot", "content": fig}

# --- ML Page Display Function (Original model_report logic, now decoupled) ---

def display_ml_report(metrics_snapshot: Dict[str, Any]):
    """
    Function to be called on the main ML page. It uses the granular functions
    to generate and display all components using Streamlit commands.
    """
    if not metrics_snapshot:
        st.error("No model data provided for display.")
        return

    st.markdown("---")
    st.markdown("## Gradient Boosting Regressor Analysis (Live View)")
    
    # Get all the report assets
    summary_asset = create_ml_summary_text(metrics_snapshot)
    table_asset = create_classification_report_table(metrics_snapshot)
    metrics_plot_asset = create_performance_metrics_plot(metrics_snapshot)
    error_plot_asset = create_error_metrics_plot(metrics_snapshot)
    
    # Display Summary
    st.markdown(summary_asset['content'], unsafe_allow_html=True)
    
    # Display Metrics Plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 $R^2$ Score")
        st.pyplot(metrics_plot_asset['content'])
    with col2:
        st.subheader("📊 Error Metrics")
        st.pyplot(error_plot_asset['content'])
    
    # Display Regression Metrics Table
    st.subheader("📋 Key Regression Metrics")
    st.dataframe(table_asset['content'], use_container_width=True)
    
    # Performance Interpretation based on R2 score
    r2 = metrics_snapshot.get('R2 Score', 0)
    st.subheader("📊 Performance Interpretation")
    
    if r2 >= 0.9:
        st.success("**Excellent Performance** - The model explains a very high variance in the target variable!")
    elif r2 >= 0.7:
        st.info("**Good Performance** - The model provides a strong fit to the data.")
    elif r2 >= 0.5:
        st.warning("**Fair Performance** - The model has an acceptable predictive power but could be improved.")
    else:
        st.error("**Poor Performance** - The model explains less than 50% of the variance. Consider feature engineering, parameter tuning, or trying a different algorithm.")
        
    # Model Configuration Details
    st.subheader("⚙️ Model Configuration")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**Hyperparameters:**")
        best_params = metrics_snapshot['Best Parameters']
        if isinstance(best_params, dict):
            for param, value in best_params.items():
                st.write(f"- **{param}:** {value}")
        else:
            st.write(f"- {best_params}")
        
        st.write("**Training Method:**")
        st.write(f"- Grid Search: {'✅ Yes' if metrics_snapshot['use_grid_search'] else '❌ No'}")
    
    with col4:
        st.write("**Dataset Information:**")
        st.write(f"- Number of features: **{len(metrics_snapshot['features'])}**")
        st.write(f"- Target variable: **{metrics_snapshot['target']}**")
        
        if metrics_snapshot['use_grid_search']:
            cv_folds = metrics_snapshot.get('cv_folds', 'N/A')
            st.write(f"- CV Folds: **{cv_folds}**")
    
    # Additional GBR-specific sections
    st.subheader("🔧 Gradient Boosting Features")
    
    st.info("""
    **Gradient Boosting Specific Advantages:**
    • Handles complex non-linear relationships well
    • Provides feature importance scores
    • Robust to outliers
    • Can capture complex interactions between features
    """)
    
    # Feature Importance and Residuals Analysis (only if we have the actual model)
    if 'model_results' in st.session_state and st.session_state.model_results.get('model'):
        try:
            # Feature Importance
            st.subheader("🎯 Feature Importances")
            model = st.session_state.model_results['model']
            feature_names = metrics_snapshot['features']
            
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                
                # Create feature importances dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=True)  # Sort for horizontal bar chart
                
                # Create feature importance plot
                fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(importance_df))
                
                ax_importance.barh(y_pos, importance_df['Importance'], color='steelblue')
                ax_importance.set_yticks(y_pos)
                ax_importance.set_yticklabels(importance_df['Feature'])
                ax_importance.set_xlabel('Importance')
                ax_importance.set_title('Feature Importances')
                ax_importance.grid(True, alpha=0.3, axis='x')
                
                st.pyplot(fig_importance)
                
                # Display feature importances table
                st.dataframe(importance_df.sort_values('Importance', ascending=False), use_container_width=True)
            else:
                st.info("Feature importances not available for this model.")
                
        except Exception as e:
            st.error(f"Error generating feature importance: {str(e)}")
    
    else:
        st.subheader("🎯 Feature Importances")
        st.info("Feature importance visualization requires access to the trained model. Create a new model to see this visualization.")

# For compatibility with your old system, we keep the original function name
def model_report():
    found = False 
    # Check for the updated model name
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'GBR_Regressor':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    
    # If the regressor wasn't found, try the old name for robustness
    if not found:
        for model_info in st.session_state.pipeline.get("ML", []):
            if model_info.get("model name") == 'GBR': # Fallback check
                if 'metrics_snapshot' in model_info:
                    found = True
                    model_results = model_info.get("metrics_snapshot", {})
                    break

    if not found:
        st.error("No Gradient Boosting Regressor model results found. Please create a model first.")
        return
    
    # Pass the JSON-safe metrics snapshot 
    display_ml_report(model_results)

# Optional: Add a function to display metrics in a more compact way
def display_compact_metrics():
    """Alternative compact metrics display"""
    found = False
    model_results = {}
    
    # Check for the updated model name
    for model_info in st.session_state.pipeline.get("ML", []):
        if model_info.get("model name") == 'GBR_Regressor':
            if 'metrics_snapshot' in model_info:
                found = True
                model_results = model_info.get("metrics_snapshot", {})
                break
    
    # If the regressor wasn't found, try the old name for robustness
    if not found:
        for model_info in st.session_state.pipeline.get("ML", []):
            if model_info.get("model name") == 'GBR': # Fallback check
                if 'metrics_snapshot' in model_info:
                    found = True
                    model_results = model_info.get("metrics_snapshot", {})
                    break

    if not found:
        return
    
    r2 = model_results.get('R2 Score', 0)
    mse = model_results.get('MSE', 0)
    rmse = model_results.get('RMSE', 0)
    mae = model_results.get('MAE', 0)
    
    # Create a compact metrics card
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.4f}")
    with col3:
        st.metric("MSE", f"{mse:.4f}")
    with col4:
        st.metric("MAE", f"{mae:.4f}")