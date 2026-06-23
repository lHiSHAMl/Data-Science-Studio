import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from constants import DataManager
import pickle
import base64
from io import BytesIO

data_manager = DataManager()

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        
        # Prepare data
        X = df[features].values
        # Target must be numerical for regression
        y = df[target].values.astype(float)
        
        # Target encoder is NOT needed for numerical regression target
        target_encoder = None
        
        # Scale features for GBR
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model with or without grid search
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            
            # Use GradientBoostingRegressor
            gbr = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(
                gbr, 
                param_grid, 
                cv=cv_folds, 
                # Scoring changed to negative mean squared error for optimization
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            st.success(f"Grid Search completed! Best parameters: {best_params}")
            
        else:
            st.info("Training with manual parameters...")
            # Use GradientBoostingRegressor with proper type conversion
            best_model = GradientBoostingRegressor(
                n_estimators=int(manual_params['n_estimators']),
                learning_rate=float(manual_params['learning_rate']),
                max_depth=int(manual_params['max_depth']),
                min_samples_split=int(manual_params['min_samples_split']),
                min_samples_leaf=int(manual_params['min_samples_leaf']),
                subsample=float(manual_params['subsample']),
                max_features=manual_params['max_features'],
                random_state=42
            )
            best_model.fit(X_train, y_train)
            best_params = manual_params
            
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # --- Calculate Regression Metrics ---
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # --- CRITICAL: Create JSON-SAFE metrics snapshot for the pipeline ---
        metrics_snapshot = {
            'R2 Score': r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'Best Parameters': best_params,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }
        
        # Update st.session_state.model_results for the live-view ML page
        model_results = {
            'model': best_model,  # Keep actual model for immediate use
            'scaler': scaler,
            'target_encoder': target_encoder,
            'metrics': metrics_snapshot, # Use the safe snapshot here too
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }
        st.session_state.model_results = model_results
        
        # Create parameter list for pipeline
        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "cv_folds", "value": cv_folds}
        ]
        
        # Add GBR-specific parameters
        if use_grid_search:
            param_list.extend([
                {"name": "n_estimators_range", "value": str(param_grid.get('n_estimators', []))},
                {"name": "learning_rate_range", "value": str(param_grid.get('learning_rate', []))},
                {"name": "max_depth_range", "value": str(param_grid.get('max_depth', []))},
                {"name": "min_samples_split_range", "value": str(param_grid.get('min_samples_split', []))},
                {"name": "min_samples_leaf_range", "value": str(param_grid.get('min_samples_leaf', []))},
                {"name": "subsample_range", "value": str(param_grid.get('subsample', []))},
                {"name": "max_features_range", "value": str(param_grid.get('max_features', []))}
            ])
        else:
            param_list.extend([
                {"name": "n_estimators", "value": manual_params['n_estimators']},
                {"name": "learning_rate", "value": manual_params['learning_rate']},
                {"name": "max_depth", "value": manual_params['max_depth']},
                {"name": "min_samples_split", "value": manual_params['min_samples_split']},
                {"name": "min_samples_leaf", "value": manual_params['min_samples_leaf']},
                {"name": "subsample", "value": manual_params['subsample']},
                {"name": "max_features", "value": manual_params['max_features']}
            ])

        # Create model entry for pipeline - ONLY metrics_snapshot, no model!
        GBR_model_pipeline_entry = DataManager.create_Gradient_Boosting_REG_Model(
            "GBR_Regressor",
            param_list,
            st.session_state.selected_trans,
            metrics_snapshot,  # Only pass JSON-safe metrics, not the model,
            best_model
        )

        # Update pipeline
        if 'ML' not in st.session_state.pipeline:
            st.session_state.pipeline['ML'] = []
             
        model_name_to_check = "GBR_Regressor" 
        
        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != model_name_to_check else GBR_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(GBR_model_pipeline_entry)
        
        st.success("Gradient Boosting Regressor Model created successfully and results saved to pipeline!")
        return model_results
        
    except Exception as e:
        st.error(f"Error training Gradient Boosting Regressor model: {str(e)}")
        return None
    
def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params['target'] in params['features']:
        st.error("Target column cannot be one of the features")
        return False
    
    # Check if target is numeric and has enough variance (can't check type here as it's passed as df)
    # Target value count is less critical for regression, but it must be numerical.
    # We will assume the user selects a numerical target for regression.
    
    return True