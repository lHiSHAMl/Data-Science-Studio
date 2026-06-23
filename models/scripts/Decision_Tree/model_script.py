import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from constants import DataManager

data_manager = DataManager()

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # Prepare data
        # Drop NA safely to avoid leakage and training errors
        df_clean = df[features + [target]].dropna()

        if df_clean.empty:
            raise ValueError("Dataset is empty after dropping NA values.")

        X = df_clean[features].values
        y = df_clean[target].values

        if len(features) == 0:
            raise ValueError("No features selected.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model with or without grid search
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            
            dtr = DecisionTreeRegressor(random_state=42)
            grid_search = GridSearchCV(
                dtr, 
                param_grid, 
                cv=cv_folds, 
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            st.success(f"Grid Search completed! Best parameters: {best_params}")
            
        else:
            st.info("Training with manual parameters...")
            best_model = DecisionTreeRegressor(
                max_depth=manual_params['max_depth'],
                min_samples_split=manual_params['min_samples_split'],
                criterion=manual_params['criterion'],
                random_state=42
            )
            best_model.fit(X_train, y_train)
            best_params = manual_params
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = best_model.feature_importances_.tolist()
        
        # --- CRITICAL: Create JSON-SAFE metrics snapshot for the pipeline ---
        metrics_snapshot = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2 Score': float(r2),
            'Best Parameters': best_params,
            'feature_importance': feature_importance,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'X_test': X_test,  # Convert numpy array to list for JSON serialization
            'y_test': y_test,
            'y_pred': y_pred
            # DO NOT save: 'model' (non-primitive object)
        }
        
        # Update st.session_state.model_results for the live-view ML page
        model_results = {
            'model': best_model,
            'X_test': X_test,  # Store for actual vs predicted plots
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': metrics_snapshot,  # Use the safe snapshot here too
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
            {"name": "max_depth", "value": str(param_grid.get('max_depth', [])) if use_grid_search else manual_params['max_depth']},
            {"name": "min_samples_split", "value": str(param_grid.get('min_samples_split', [])) if use_grid_search else manual_params['min_samples_split']},
            {"name": "criterion", "value": str(param_grid.get('criterion', [])) if use_grid_search else manual_params['criterion']},
            {"name": "cv_folds", "value": cv_folds}
        ]
        
        # Create model entry for pipeline, including the JSON-safe snapshot
        DTR_model_pipeline_entry = DataManager.create_DecisionTree_Regressor_Model(
            "Decision Tree Regressor",              # Model name
            param_list,
            st.session_state.selected_trans,
            best_model,          # Pass the trained model
            metrics_snapshot     # Pass the metrics snapshot
        )
        # Add metrics snapshot to the pipeline entry
        DTR_model_pipeline_entry["metrics_snapshot"] = metrics_snapshot

        # Update pipeline
        if 'ML' not in st.session_state.pipeline:
            st.session_state.pipeline['ML'] = []
             
        if edit:
            # Replace existing Decision Tree Regressor entry - FIXED LOGIC
            updated = False
            for i, item in enumerate(st.session_state.pipeline['ML']):
                # Check both model name and model type for better matching
                if (item.get('model name') == "DecisionTreeRegressor" or 
                    item.get('model type') == "Decision Tree Regressor" or
                    "Decision Tree" in item.get('model type', '')):
                    st.session_state.pipeline['ML'][i] = DTR_model_pipeline_entry
                    updated = True
                    break
            
            if not updated:
                # If no existing model found, append as new
                st.session_state.pipeline['ML'].append(DTR_model_pipeline_entry)
        else:
            # Append if no Decision Tree Regressor exists
            st.session_state.pipeline['ML'].append(DTR_model_pipeline_entry)
        
        st.success("Decision Tree Regressor Model created successfully and results saved to pipeline!")
        return model_results
        
    except Exception as e:
        st.error(f"Error training Decision Tree Regressor model: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params['target'] in params['features']:
        st.error("Target column cannot be one of the features")
        return False
    
    # Check if target is numeric for regression
    if params['df'][params['target']].dtype not in ['int64', 'float64']:
        st.error("Target column must be numeric for regression")
        return False
    
    return True