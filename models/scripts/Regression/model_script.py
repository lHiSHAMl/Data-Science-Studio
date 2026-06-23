import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from constants import DataManager

data_manager = DataManager()

def model_script(df, features, target, model_type, alpha=1.0, edit=False):
    """
    Regression model script with default parameters to avoid missing arguments
    """
    try:
        # Input validation
        if not features or not target:
            st.error("Please select features and target")
            return {"status": "error", "message": "Missing features or target"}
        
        if target in features:
            st.error("Target cannot be in features")
            return {"status": "error", "message": "Target in features"}
        
        # Set default alpha if not provided
        if alpha is None:
            alpha = 1.0
            
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Handle categorical features
        categorical_features = df[features].select_dtypes(include=['object']).columns
        feature_encoders = {}
        
        if len(categorical_features) > 0:
            st.info(f"Encoding categorical features: {list(categorical_features)}")
            for col in categorical_features:
                le = LabelEncoder()
                col_idx = features.index(col)
                X[:, col_idx] = le.fit_transform(X[:, col_idx])
                feature_encoders[col] = le
        
        # Scale features for regularization models
        if model_type in ['ridge', 'lasso']:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        else:
            scaler = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model based on type
        if model_type == 'linear':
            model = LinearRegression()
            st.info("Training Linear Regression model...")
        elif model_type == 'ridge':
            model = Ridge(alpha=alpha)
            st.info(f"Training Ridge Regression model (alpha={alpha})...")
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha)
            st.info(f"Training Lasso Regression model (alpha={alpha})...")
        else:
            st.error(f"Unknown model type: {model_type}")
            return {"status": "error", "message": f"Unknown model type: {model_type}"}
        
        best_model = model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics (JSON-compatible with safe defaults)
        r2 = float(r2_score(y_test, y_pred)) if len(y_test) > 0 else 0.0
        mae = float(mean_absolute_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
        mse = float(mean_squared_error(y_test, y_pred)) if len(y_test) > 0 else 0.0
        rmse = float(np.sqrt(mse)) if mse > 0 else 0.0
        
        # Calculate additional metrics (JSON-compatible with safe defaults)
        if len(y_test) > 0 and not np.all(y_test == 0):
            mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        else:
            mape = 0.0
            
        if len(y_test) > len(features) + 1:
            adjusted_r2 = float(1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1))
        else:
            adjusted_r2 = r2
        
        # Get coefficients for interpretation (JSON-compatible)
        coefficients = {}
        if hasattr(model, 'coef_'):
            coefficients = {str(feature): float(coef) for feature, coef in zip(features, model.coef_)}
        
        intercept = float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
        
        # Create JSON-compatible metrics snapshot WITH ALL REQUIRED KEYS
        metrics_snapshot = {
            'R² Score': r2,
            'Adjusted R²': adjusted_r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Coefficients': coefficients,
            'Intercept': intercept,
            'model_type': str(model_type),
            'features': [str(f) for f in features],
            'target': str(target),
            'alpha': float(alpha)
        }
        
        # Create parameter list for pipeline
        param_list = [
            {"name": "features", "value": [str(f) for f in features]},
            {"name": "target", "value": str(target)},
            {"name": "model_type", "value": str(model_type)},
            {"name": "alpha", "value": float(alpha)}
        ]

        # Create model entry for pipeline with metrics_snapshot
        regression_model = DataManager.create_REG_Model(
            "Linear Regression",
            param_list,
            st.session_state.get('selected_trans', []),
            best_model,
            metrics_snapshot=metrics_snapshot  # Add this parameter
        )
        
        # Update pipeline
        if edit:
            # Find and replace existing model
            model_replaced = False
            for i, item in enumerate(st.session_state.pipeline['ML']):
                if item.get('model name') == "Linear Regression":
                    st.session_state.pipeline['ML'][i] = regression_model
                    model_replaced = True
                    break
            if not model_replaced:
                st.session_state.pipeline['ML'].append(regression_model)
        else:
            # Append if no Regression exists
            st.session_state.pipeline['ML'].append(regression_model)
        
        st.success("Regression Model created successfully!")
        return {"status": "success", "metrics_snapshot": metrics_snapshot}
        
    except Exception as e:
        st.error(f"Error training Regression model: {str(e)}")
        return {"status": "error", "message": str(e)}

def validate_model(params):
    """Validate model parameters with safe defaults"""
    if not params.get('features') or len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False
        
    if not params.get('target'):
        st.error("Please select a target column")
        return False
        
    if params['target'] in params['features']:
        st.error("Target column cannot be one of the features")
        return False
    
    # Check if target is numeric for regression
    if params['df'][params['target']].dtype not in ['int64', 'float64']:
        st.error("Target column must be numeric for regression")
        return False
    
    # Check if alpha is valid for Ridge/Lasso
    model_type = params.get('model_type', 'linear')
    if model_type in ['ridge', 'lasso']:
        alpha = params.get('alpha', 1.0)
        try:
            alpha_val = float(alpha)
            if alpha_val <= 0:
                st.error("Alpha value must be positive for Ridge and Lasso regression")
                return False
        except (ValueError, TypeError):
            st.error("Alpha value must be a valid number")
            return False
    
    return True