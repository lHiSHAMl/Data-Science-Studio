import streamlit as st
import numpy as np
import pandas as pd
# --- Changed to Regressor ---
from sklearn.neighbors import KNeighborsRegressor
# --- Changed to Regression Metrics ---
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
# LabelEncoder removed as target is expected to be numerical
from constants import DataManager

data_manager = DataManager()

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # Prepare data
        X = df[features].values
        # Target must be numerical for regression
        y = df[target].values.astype(float)
        
        # Target encoder is NOT needed for numerical regression target
        target_encoder = None
        
        # Scale features for KNN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model with or without grid search
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            
            # Use KNeighborsRegressor
            knn = KNeighborsRegressor()
            grid_search = GridSearchCV(
                knn, 
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
            # Use KNeighborsRegressor
            best_model = KNeighborsRegressor(
                n_neighbors=manual_params['n_neighbors'],
                weights=manual_params['weights'],
                algorithm=manual_params['algorithm']
            )
            best_model.fit(X_train, y_train)
            best_params = manual_params
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # --- Calculate Regression Metrics ---
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # --- CRITICAL: Create JSON-SAFE metrics snapshot for the pipeline ---
        metrics_snapshot = {
            'R2 Score': r2,
            'MAE': mae,
            'MSE': mse,
            # Confusion Matrix and Classification Report removed
            'Best Parameters': best_params,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }
        
        # Update st.session_state.model_results for the live-view ML page
        model_results = {
            'model': best_model,
            'scaler': scaler,
            'target_encoder': target_encoder, # None for regression
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
            {"name": "n_neighbors_range", "value": str(param_grid.get('n_neighbors', [])) if use_grid_search else manual_params['n_neighbors']},
            {"name": "weights", "value": str(param_grid.get('weights', [])) if use_grid_search else manual_params['weights']},
            {"name": "algorithm", "value": str(param_grid.get('algorithm', [])) if use_grid_search else manual_params['algorithm']},
            {"name": "cv_folds", "value": cv_folds}
        ]

        # Ensure manual_params values are proper types when using them:
        if not use_grid_search:
            manual_params = {
                'n_neighbors': int(manual_params['n_neighbors']),
                'weights': manual_params['weights'],
                'algorithm': manual_params['algorithm']
            }

        # Create model entry for pipeline, including the JSON-safe snapshot
        KNN_model_pipeline_entry = DataManager.create_KNN_Regressor_Model(
            "KNN_Regressor", # Updated name
            param_list,
            st.session_state.selected_trans,
            best_model,
            metrics_snapshot
        )

        # Update pipeline
        if 'ML' not in st.session_state.pipeline:
             st.session_state.pipeline['ML'] = []
             
        # The key for checking existing model should likely be 'KNN_Regressor' now
        model_name_to_check = "KNN_Regressor" 
        
        if edit:
            # Replace existing entry by matching 'model name'
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != model_name_to_check else KNN_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            # Append 
            st.session_state.pipeline['ML'].append(KNN_model_pipeline_entry)
            # st.write(st.session_state.pipeline)
        
        st.success("KNN Regressor Model created successfully and results saved to pipeline!")
        return model_results
        
    except Exception as e:
        st.error(f"Error training KNN Regressor model: {str(e)}")
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