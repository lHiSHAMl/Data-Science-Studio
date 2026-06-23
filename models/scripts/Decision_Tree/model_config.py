import streamlit as st
import numpy as np

def model_config(model_data, edit):
    # Check if selected_data exists and has columns
    if 'selected_data' not in st.session_state or st.session_state.selected_data.empty:
        st.error("No data available. Please load data first.")
        return {
            "features": [],
            "target": None,
            "df": None,
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
            default_index = 0
            if edit and model_data.get('model param'):
                target_value = None
                for param in model_data['model param']:
                    if param['name'] == 'target':
                        target_value = param['value']
                        break
                
                if target_value and target_value in numeric_cols:
                    default_index = numeric_cols.index(target_value)
            
            target = st.selectbox(
                "Select target column:",
                options=numeric_cols,
                index=default_index,
                key="target_column_dt"
            )
    
    # Grid Search Configuration
    st.subheader("Hyperparameter Configuration")
    
    # Default values
    default_use_grid_search = True
    default_max_depth = "3,5,7,10"
    default_min_samples_split = "2,5,10"
    default_cv_folds = 5
    
    # Extract parameters from model_data if in edit mode
    if edit and model_data.get('model param'):
        param_dict = {}
        for param in model_data['model param']:
            param_dict[param['name']] = param['value']
        
        default_use_grid_search = param_dict.get('use_grid_search', True)
        default_max_depth = param_dict.get('max_depth', "3,5,7,10")
        default_min_samples_split = param_dict.get('min_samples_split', "2,5,10")
        default_cv_folds = int(param_dict.get('cv_folds', 5))
    
    use_grid_search = st.checkbox(
        "Use Grid Search for hyperparameter tuning",
        value=default_use_grid_search
    )
    
    if use_grid_search:
        st.info("Grid Search will automatically find the best hyperparameters")
        
        # Grid Search parameters
        col3, col4 = st.columns(2)
        with col3:
            max_depth_options = st.text_input(
                "Max depth values (comma-separated):",
                value=default_max_depth,
                help="Maximum depth of the tree"
            )
        
        with col4:
            min_samples_split_options = st.text_input(
                "Min samples split (comma-separated):",
                value=default_min_samples_split,
                help="Minimum number of samples required to split an internal node"
            )
        
        # Parse parameters
        try:
            max_depth_range = [int(x.strip()) for x in max_depth_options.split(',')]
        except (ValueError, TypeError):
            max_depth_range = [3, 5, 7, 10]
            st.warning("Invalid max depth format. Using default: 3,5,7,10")
        
        try:
            min_samples_split_range = [int(x.strip()) for x in min_samples_split_options.split(',')]
        except (ValueError, TypeError):
            min_samples_split_range = [2, 5, 10]
            st.warning("Invalid min samples split format. Using default: 2,5,10")
        
        cv_folds = st.number_input(
            "Cross-validation folds:",
            min_value=2,
            max_value=10,
            value=default_cv_folds
        )
        
        criterion_options = ['squared_error', 'friedman_mse', 'absolute_error']
        
        param_grid = {
            'max_depth': max_depth_range,
            'min_samples_split': min_samples_split_range,
            'criterion': criterion_options
        }
        
        manual_params = {}
        
    else:
        # Manual parameter configuration
        st.info("Configure hyperparameters manually")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            max_depth = st.number_input(
                "Max depth:",
                min_value=1,
                max_value=50,
                value=5,
                help="Maximum depth of the tree"
            )
        
        with col4:
            min_samples_split = st.number_input(
                "Min samples split:",
                min_value=2,
                max_value=20,
                value=2,
                help="Minimum samples required to split a node"
            )
        
        with col5:
            criterion = st.selectbox(
                "Criterion:",
                options=['squared_error', 'friedman_mse', 'absolute_error'],
                index=0
            )
        
        manual_params = {
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'criterion': criterion
        }
        param_grid = {}
        cv_folds = 5
    
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
    
model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Decision Tree Regressor Implementation</span>
    </div>
    <div>
        <pre><code>
# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np

def train_decision_tree_regressor(df, features, target, use_grid_search=True, cv_folds=5):
    # Prepare data
    X = df[features].values
    y = df[target].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if use_grid_search:
        # Grid Search configuration
        param_grid = {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
        }
        
        dtr = DecisionTreeRegressor(random_state=42)
        grid_search = GridSearchCV(dtr, param_grid, cv=cv_folds, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best parameters: {best_params}")
        
    else:
        # Manual parameters
        best_model = DecisionTreeRegressor(
            max_depth=5,
            min_samples_split=2,
            criterion='squared_error',
            random_state=42
        )
        best_model.fit(X_train, y_train)
        best_params = {'max_depth': 5, 'min_samples_split': 2, 'criterion': 'squared_error'}
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = best_model.feature_importances_
    for feature, importance in zip(features, feature_importance):
        print(f"{feature}: {importance:.4f}")
    
    return {
        'model': best_model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'feature_importance': feature_importance,
        'best_params': best_params
    }

# Usage example:
# results = train_decision_tree_regressor(df, ['feature1', 'feature2'], 'target_column')
        </code></pre>
    </div>
</div>
"""

model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Decision Tree Regressor</span>
    </div>
    <div>
        Decision Tree Regressor is a non-parametric supervised learning method used for 
        regression tasks. It creates a model that predicts the value of a target variable 
        by learning simple decision rules inferred from the data features.
        
        <strong>Key Concepts:</strong>
        • <strong>Tree Structure:</strong> Hierarchical structure of nodes and leaves
        • <strong>Splitting:</strong> Recursive partitioning of feature space
        • <strong>Impurity Measures:</strong> Criteria like MSE for determining splits
        
        <strong>Hyperparameters:</strong>
        • <strong>max_depth:</strong> Maximum depth of the tree
        • <strong>min_samples_split:</strong> Minimum samples required to split a node
        • <strong>criterion:</strong> Function to measure split quality (MSE, MAE, etc.)
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Mean Squared Error (MSE):</strong> Average squared difference between predicted and actual values
        • <strong>Root Mean Squared Error (RMSE):</strong> Square root of MSE
        • <strong>Mean Absolute Error (MAE):</strong> Average absolute difference
        • <strong>R² Score:</strong> Proportion of variance explained by the model
        
        <strong>Advantages:</strong>
        • Easy to interpret and visualize
        • Handles non-linear relationships
        • Requires little data preprocessing
        • Feature importance scores
        
        <strong>Limitations:</strong>
        • Can overfit easily without proper regularization
        • Unstable (small changes in data can result in different trees)
        • Biased towards features with more levels
    </div>
</div>
"""