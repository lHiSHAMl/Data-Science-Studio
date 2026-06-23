import streamlit as st
import pandas as pd
import numpy as np

def model_config(model_data, edit):
    # Check if selected_data exists and has columns
    if 'selected_data' not in st.session_state or st.session_state.selected_data.empty:
        st.error("No data available. Please load data first.")
        return {
            "features": [],
            "target": None,
            "df": pd.DataFrame(),
            "edit": edit,
            "use_grid_search": True,
            "param_grid": {},
            "manual_params": {},
            "cv_folds": 5
        }
    
    numeric_cols = [col for col in st.session_state.selected_data.columns 
                   if st.session_state.selected_data[col].dtype in ['int64', 'float64']]
    
    # Feature and target selection
    col1, col2 = st.columns(2)
    with col1:
        # Handle default features for edit mode
        default_features = []
        if edit and model_data.get('model param'):
            for param in model_data['model param']:
                if param['name'] == 'features':
                    default_features = param['value'] if isinstance(param['value'], list) else []
                    break
        
        features = st.multiselect(
            "Select feature columns:",
            options=numeric_cols if numeric_cols else [],
            default=default_features
        )
    
    with col2:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for target selection")
            target = None
        else:
            # For regression, target should be numeric
            numeric_targets = [col for col in numeric_cols if col not in features]
            default_index = 0
            if edit and model_data.get('model param'):
                target_value = None
                for param in model_data['model param']:
                    if param['name'] == 'target':
                        target_value = param['value']
                        break
                
                if target_value and target_value in numeric_targets:
                    default_index = numeric_targets.index(target_value)
            
            target = st.selectbox(
                "Select target column:",
                options=numeric_targets,
                index=default_index,
                key="target_column"
            )
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    # Default values
    default_use_grid_search = True
    default_cv_folds = 5
    
    # Extract parameters from model_data if in edit mode
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']
        
        default_use_grid_search = param_dict.get('use_grid_search', True)
        default_cv_folds = int(param_dict.get('cv_folds', 5))
    
    # Grid Search Configuration
    use_grid_search = st.checkbox(
        "Use Grid Search for hyperparameter tuning",
        value=default_use_grid_search
    )
    
    if use_grid_search:
        st.info("Grid Search will automatically find the best hyperparameters for Gradient Boosting")
        
        col3, col4 = st.columns(2)
        with col3:
            cv_folds = st.number_input(
                "Cross-validation folds:",
                min_value=2,
                max_value=10,
                value=default_cv_folds
            )
        
        with col4:
            n_estimators_range = st.text_input(
                "N Estimators (comma-separated):",
                value="100,200,300,500"
            )
            try:
                n_estimators_values = [int(x.strip()) for x in n_estimators_range.split(',')]
            except (ValueError, TypeError):
                n_estimators_values = [100, 200, 300, 500]
                st.warning("Invalid format. Using default n_estimators values")
        
        col5, col6 = st.columns(2)
        with col5:
            learning_rate_range = st.text_input(
                "Learning Rate (comma-separated):",
                value="0.01,0.05,0.1,0.2"
            )
            try:
                learning_rate_values = [float(x.strip()) for x in learning_rate_range.split(',')]
            except (ValueError, TypeError):
                learning_rate_values = [0.01, 0.05, 0.1, 0.2]
                st.warning("Invalid format. Using default learning_rate values")
        
        with col6:
            max_depth_range = st.text_input(
                "Max Depth (comma-separated):",
                value="3,4,5,6"
            )
            try:
                max_depth_values = [int(x.strip()) for x in max_depth_range.split(',')]
            except (ValueError, TypeError):
                max_depth_values = [3, 4, 5, 6]
                st.warning("Invalid format. Using default max_depth values")
        
        col7, col8 = st.columns(2)
        with col7:
            min_samples_split_range = st.text_input(
                "Min Samples Split (comma-separated):",
                value="2,5,10"
            )
            try:
                min_samples_split_values = [int(x.strip()) for x in min_samples_split_range.split(',')]
            except (ValueError, TypeError):
                min_samples_split_values = [2, 5, 10]
                st.warning("Invalid format. Using default min_samples_split values")
        
        with col8:
            min_samples_leaf_range = st.text_input(
                "Min Samples Leaf (comma-separated):",
                value="1,2,4"
            )
            try:
                min_samples_leaf_values = [int(x.strip()) for x in min_samples_leaf_range.split(',')]
            except (ValueError, TypeError):
                min_samples_leaf_values = [1, 2, 4]
                st.warning("Invalid format. Using default min_samples_leaf values")
        
        col9, col10 = st.columns(2)
        with col9:
            subsample_range = st.text_input(
                "Subsample (comma-separated):",
                value="0.7,0.8,0.9,1.0"
            )
            try:
                subsample_values = [float(x.strip()) for x in subsample_range.split(',')]
            except (ValueError, TypeError):
                subsample_values = [0.7, 0.8, 0.9, 1.0]
                st.warning("Invalid format. Using default subsample values")
        
        with col10:
            max_features_range = st.text_input(
                "Max Features (comma-separated, use 'sqrt', 'log2', or numbers):",
                value="sqrt,log2,0.5,1.0"
            )
            try:
                max_features_values = []
                for x in max_features_range.split(','):
                    x = x.strip()
                    if x in ['sqrt', 'log2']:
                        max_features_values.append(x)
                    else:
                        max_features_values.append(float(x))
            except (ValueError, TypeError):
                max_features_values = ['sqrt', 'log2', 0.5, 1.0]
                st.warning("Invalid format. Using default max_features values")
        
        param_grid = {
            'n_estimators': n_estimators_values,
            'learning_rate': learning_rate_values,
            'max_depth': max_depth_values,
            'min_samples_split': min_samples_split_values,
            'min_samples_leaf': min_samples_leaf_values,
            'subsample': subsample_values,
            'max_features': max_features_values
        }
        
        manual_params = {}
        
    else:
        # Manual parameter configuration
        st.info("Configure hyperparameters manually for Gradient Boosting")
        
        cv_folds = 5
        
        col3, col4, col5 = st.columns(3)
        with col3:
            n_estimators = st.number_input(
                "N Estimators:",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Number of boosting stages to perform"
            )
        with col4:
            learning_rate = st.number_input(
                "Learning Rate:",
                min_value=0.001,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.3f",
                help="Shrinks the contribution of each tree"
            )
        with col5:
            max_depth = st.number_input(
                "Max Depth:",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                help="Maximum depth of the individual trees"
            )
        
        col6, col7, col8 = st.columns(3)
        with col6:
            min_samples_split = st.number_input(
                "Min Samples Split:",
                min_value=2,
                max_value=20,
                value=2,
                step=1,
                help="Minimum samples required to split a node"
            )
        with col7:
            min_samples_leaf = st.number_input(
                "Min Samples Leaf:",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                help="Minimum samples required in a leaf node"
            )
        with col8:
            subsample = st.number_input(
                "Subsample:",
                min_value=0.1,
                max_value=1.0,
                value=1.0,
                step=0.1,
                format="%.1f",
                help="Fraction of samples for fitting base learners"
            )
        
        col9, col10 = st.columns(2)
        with col9:
            max_features_option = st.selectbox(
                "Max Features:",
                options=['sqrt', 'log2', 'custom'],
                index=0,
                help="Number of features to consider for best split"
            )
            
            if max_features_option == 'custom':
                max_features = st.number_input(
                    "Custom Max Features (fraction):",
                    min_value=0.1,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    format="%.1f"
                )
            else:
                max_features = max_features_option
        
        manual_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'max_features': max_features
        }
        param_grid = {}
    return {
        "features": features,
        "target": target,
        "df": st.session_state.selected_data,
        "edit": edit,
        "use_grid_search": use_grid_search,
        "param_grid": param_grid,
        "manual_params": manual_params,
        "cv_folds": cv_folds
    }

GBR_model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Gradient Boosting Regressor</span>
    </div>
    <div>
        Gradient Boosting is an ensemble learning technique that builds models sequentially, 
        where each new model attempts to correct the errors made by the previous models. 
        It combines multiple weak learners (typically decision trees) to create a strong 
        predictive model by minimizing a loss function using gradient descent.
        
        <strong>Key Features:</strong>
        • Sequential ensemble method
        • Uses gradient descent optimization
        • Builds trees to correct residual errors
        • Provides feature importance scores
        • Highly accurate but can be prone to overfitting
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>R² Score:</strong> Proportion of variance explained (closer to 1 is better)
        • <strong>MSE:</strong> Mean Squared Error (lower is better)
        • <strong>MAE:</strong> Mean Absolute Error (lower is better)
        • <strong>RMSE:</strong> Root Mean Squared Error (lower is better)
        
        <strong>Hyperparameters:</strong>
        • <strong>n_estimators:</strong> Number of boosting stages (100-500 typical)
        • <strong>learning_rate:</strong> Shrinks contribution of each tree (0.01-0.2 typical)
        • <strong>max_depth:</strong> Maximum depth of trees (3-6 typical to prevent overfitting)
        • <strong>min_samples_split:</strong> Minimum samples to split node (2-10)
        • <strong>min_samples_leaf:</strong> Minimum samples in leaf node (1-4)
        • <strong>subsample:</strong> Fraction of samples for fitting trees (0.7-1.0)
        • <strong>max_features:</strong> Number of features for best split ('sqrt', 'log2', or fraction)
    </div>
</div>
"""

GBR_model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Gradient Boosting Regressor Implementation</span>
    </div>
    <div>
        <pre style="background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto;">
<code>
# Gradient Boosting Regressor Implementation
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load and prepare data
data = pd.read_csv('your_data.csv')
X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model and Grid Search
model = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', 0.5, 1.0]
}

# Grid Search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print("Best parameters:", grid_search.best_params_)
print("Feature importances:", best_model.feature_importances_)
</code>
        </pre>
    </div>
</div>
"""