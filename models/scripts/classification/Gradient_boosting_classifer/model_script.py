import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from constants import DataManager

data_manager = DataManager()

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Encode target variable if it's categorical
        if df[target].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_encoder = le
        else:
            target_encoder = None
        
        # Split data - No scaling needed for Gradient Boosting
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with or without grid search
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            
            gb = GradientBoostingClassifier(random_state=42)
            grid_search = GridSearchCV(
                gb, 
                param_grid, 
                cv=cv_folds, 
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            st.success(f"Grid Search completed! Best parameters: {best_params}")
            
        else:
            st.info("Training with manual parameters...")
            
            # Ensure all required parameters are present with defaults
            manual_params_with_defaults = {
                'n_estimators': manual_params.get('n_estimators', 100),
                'learning_rate': manual_params.get('learning_rate', 0.1),
                'max_depth': manual_params.get('max_depth', 3),
                'min_samples_split': manual_params.get('min_samples_split', 2),
                'min_samples_leaf': manual_params.get('min_samples_leaf', 1),  # ADD THIS LINE
                'random_state': 42
            }
            
            best_model = GradientBoostingClassifier(**manual_params_with_defaults)
            best_model.fit(X_train, y_train)
            best_params = manual_params_with_defaults
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Get feature importances
        feature_importances = best_model.feature_importances_.tolist()
        
        # --- CRITICAL: Create JSON-SAFE metrics snapshot for the pipeline ---
        metrics_snapshot = {
            'Accuracy': accuracy,
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix.tolist(), # Convert NumPy array to JSON-safe list
            'Best Parameters': best_params,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'feature_importances': feature_importances
        }
        
        # Update st.session_state.model_results for the live-view ML page
        model_results = {
            'model': best_model,
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
            {"name": "n_estimators_range", "value": str(param_grid.get('n_estimators', [])) if use_grid_search else manual_params.get('n_estimators', 100)},
            {"name": "learning_rate_range", "value": str(param_grid.get('learning_rate', [])) if use_grid_search else manual_params.get('learning_rate', 0.1)},
            {"name": "max_depth_range", "value": str(param_grid.get('max_depth', [])) if use_grid_search else manual_params.get('max_depth', 3)},
            {"name": "min_samples_split_range", "value": str(param_grid.get('min_samples_split', [])) if use_grid_search else manual_params.get('min_samples_split', 2)},
            {"name": "min_samples_leaf_range", "value": str(param_grid.get('min_samples_leaf', [])) if use_grid_search else manual_params.get('min_samples_leaf', 1)},  # ADD THIS LINE
            {"name": "cv_folds", "value": cv_folds}
        ]

        # Ensure manual_params values are proper types when using them:
        if not use_grid_search:
            manual_params = {
                'n_estimators': int(manual_params.get('n_estimators', 100)),
                'learning_rate': float(manual_params.get('learning_rate', 0.1)),
                'max_depth': int(manual_params.get('max_depth', 3)) if manual_params.get('max_depth') else None,
                'min_samples_split': int(manual_params.get('min_samples_split', 2)),
                'min_samples_leaf': int(manual_params.get('min_samples_leaf', 1))  # ADD THIS LINE
            }

        # Create model entry for pipeline, including the JSON-safe snapshot
        GB_model_pipeline_entry = DataManager.create_GB_Model(
            "Gradient Boosting Classifier",
            param_list,
            st.session_state.selected_trans,
            best_model,
            metrics_snapshot
        )

        # Update pipeline
        if 'ML' not in st.session_state.pipeline:
             st.session_state.pipeline['ML'] = []
             
        if edit:
            # Replace existing Gradient Boosting entry
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != 'Gradient Boosting Classifier' else GB_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            # Append if no Gradient Boosting exists
            st.session_state.pipeline['ML'].append(GB_model_pipeline_entry)
        
        st.success("Gradient Boosting Model created successfully and results saved to pipeline!")
        return model_results
        
    except Exception as e:
        st.error(f"Error training Gradient Boosting model: {str(e)}")
        return None

def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params['target'] in params['features']:
        st.error("Target column cannot be one of the features")
        return False
    
    # Check if target has at least 2 classes
    target_values = params['df'][params['target']].nunique()
    if target_values < 2:
        st.error("Target column must have at least 2 unique classes for classification")
        return False
    
    return True