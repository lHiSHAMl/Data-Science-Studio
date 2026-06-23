# import streamlit as st
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from constants import DataManager

# data_manager = DataManager()

# def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
#     try:
#         # Prepare data
#         X = df[features].values
#         y = df[target].values
        
#         # Encode target variable if it's categorical
#         if df[target].dtype == 'object':
#             le = LabelEncoder()
#             y = le.fit_transform(y)
#             target_encoder = le
#         else:
#             target_encoder = None
        
#         # Scale features for KNN
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(
#             X_scaled, y, test_size=0.2, random_state=42, stratify=y
#         )
        
#         # Train model with or without grid search
#         if use_grid_search:
#             st.info("Performing Grid Search for optimal parameters...")
            
#             knn = KNeighborsClassifier()
#             grid_search = GridSearchCV(
#                 knn, 
#                 param_grid, 
#                 cv=cv_folds, 
#                 scoring='accuracy',
#                 n_jobs=-1
#             )
#             grid_search.fit(X_train, y_train)
            
#             # Get best model and parameters
#             best_model = grid_search.best_estimator_
#             best_params = grid_search.best_params_

#             st.success(f"Grid Search completed! Best parameters: {best_params}")
            
#         else:
#             st.info("Training with manual parameters...")
#             best_model = KNeighborsClassifier(
#                 n_neighbors=manual_params['n_neighbors'],
#                 weights=manual_params['weights'],
#                 algorithm=manual_params['algorithm']
#             )
#             best_model.fit(X_train, y_train)
#             best_params = manual_params
        
#         # Make predictions
#         y_pred = best_model.predict(X_test)
#         y_pred_proba = best_model.predict_proba(X_test)
        
#         # Calculate metrics
#         accuracy = accuracy_score(y_test, y_pred)
#         class_report = classification_report(y_test, y_pred, output_dict=True)
#         conf_matrix = confusion_matrix(y_test, y_pred)
        
#         # In the model_script.py, update the results dictionary:
#         model_results = {
#             'model': best_model,
#             'scaler': scaler,
#             'target_encoder': target_encoder,
#             'metrics': {
#                 'Accuracy': accuracy,
#                 'Classification Report': class_report,
#                 'Confusion Matrix': conf_matrix,
#                 'Best Parameters': best_params
#             },
#             'features': features,
#             'target': target,
#             'use_grid_search': use_grid_search,
#             'grid_search_params': param_grid if use_grid_search else {},
#             'manual_params': manual_params if not use_grid_search else {},
#             'cv_folds': cv_folds,
#             'df': df  # Add the dataframe to results
#         }
        
#         # Create parameter list for pipeline
#         param_list = [
#             {"name": "features", "value": features},
#             {"name": "target", "value": target},
#             {"name": "use_grid_search", "value": use_grid_search},
#             {"name": "n_neighbors_range", "value": str(param_grid.get('n_neighbors', [])) if use_grid_search else manual_params['n_neighbors']},
#             {"name": "weights", "value": str(param_grid.get('weights', [])) if use_grid_search else manual_params['weights']},
#             {"name": "algorithm", "value": str(param_grid.get('algorithm', [])) if use_grid_search else manual_params['algorithm']},
#             {"name": "cv_folds", "value": cv_folds}
#         ]

#         # Ensure manual_params values are proper types when using them:
#         if not use_grid_search:
#             manual_params = {
#                 'n_neighbors': int(manual_params['n_neighbors']),
#                 'weights': manual_params['weights'],
#                 'algorithm': manual_params['algorithm']
#             }

#         # Create model entry for pipeline
#         KNN_model = DataManager.create_KNN_Model(
#             "KNN",
#             param_list,
#             st.session_state.selected_trans,
#             best_model
#         )
        
#         # Update pipeline
#         if edit:
#             # Replace existing KNN entry
#             st.session_state.pipeline['ML'] = [
#                 item if item.get('name') != 'KNN' else KNN_model
#                 for item in st.session_state.pipeline['ML']
#             ]
#         else:
#             # Append if no KNN exists
#             st.session_state.pipeline['ML'].append(KNN_model)
        
#         st.success("KNN Model created successfully!")
#         return model_results
        
#     except Exception as e:
#         st.error(f"Error training KNN model: {str(e)}")
#         return None

# def validate_model(params):
#     if len(params['features']) == 0:
#         st.error("Please select at least one feature column")
#         return False
#     elif params['target'] in params['features']:
#         st.error("Target column cannot be one of the features")
#         return False
    
#     # Check if target has at least 2 classes
#     target_values = params['df'][params['target']].nunique()
#     if target_values < 2:
#         st.error("Target column must have at least 2 unique classes for classification")
#         return False
    
#     return True
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        
        # Scale features for KNN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with or without grid search
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            
            knn = KNeighborsClassifier()
            grid_search = GridSearchCV(
                knn, 
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
            best_model = KNeighborsClassifier(
                n_neighbors=manual_params['n_neighbors'],
                weights=manual_params['weights'],
                algorithm=manual_params['algorithm']
            )
            best_model.fit(X_train, y_train)
            best_params = manual_params
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
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
            # DO NOT save: 'model', 'scaler', 'target_encoder', 'df' (non-primitive objects)
        }
        
        # Update st.session_state.model_results for the live-view ML page
        model_results = {
            'model': best_model,
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
        KNN_model_pipeline_entry = DataManager.create_KNN_Model(
            "KNN",
            param_list,
            st.session_state.selected_trans,
            best_model,
            metrics_snapshot
        )
        # KNN_model_pipeline_entry = {
        #     "model name": "KNN",
        #     "model type": "KNN",
        #     "model param": param_list,
        #     "transformations": st.session_state.selected_trans,
        #     "metrics_snapshot": metrics_snapshot # <-- The decoupled, JSON-safe data
        # }

        # Update pipeline
        if 'ML' not in st.session_state.pipeline:
             st.session_state.pipeline['ML'] = []
             
        if edit:
            # Replace existing KNN entry
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != 'KNN' else KNN_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            # Append if no KNN exists
            st.session_state.pipeline['ML'].append(KNN_model_pipeline_entry)
            # st.write(st.session_state.pipeline)
        
        st.success("KNN Model created successfully and results saved to pipeline!")
        return model_results
        
    except Exception as e:
        st.error(f"Error training KNN model: {str(e)}")
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