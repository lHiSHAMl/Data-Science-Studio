import streamlit as st
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from constants import DataManager

data_manager = DataManager()

def model_script(df, features, target, edit, model_type, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Encode target variable if categorical
        if not np.issubdtype(df[target].dtype, np.number):
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_encoder = le
        else:
            target_encoder = None
        
        # Scale features for GaussianNB (other NB types may not need scaling)
        if model_type == 'gaussian':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            scaler = None
            X_scaled = X
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model with or without grid search
        if use_grid_search:
            st.info(f"Performing Grid Search for {model_type.upper()} Naive Bayes...")
            
            # Initialize model
            if model_type == 'gaussian':
                model = GaussianNB()
            elif model_type == 'multinomial':
                model = MultinomialNB()
            else:  # bernoulli
                model = BernoulliNB()
            
            grid_search = GridSearchCV(
                model, 
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
            st.info(f"Training {model_type.upper()} Naive Bayes with manual parameters...")
            
            if model_type == 'gaussian':
                best_model = GaussianNB(
                    var_smoothing=manual_params['var_smoothing']
                )
            elif model_type == 'multinomial':
                best_model = MultinomialNB(
                    alpha=manual_params['alpha'],
                    fit_prior=manual_params['fit_prior']
                )
            else:  # bernoulli
                best_model = BernoulliNB(
                    alpha=manual_params['alpha'],
                    fit_prior=manual_params['fit_prior'],
                    binarize=manual_params.get('binarize', 0.0)
                )
            
            best_model.fit(X_train, y_train)
            best_params = manual_params
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # --- CRITICAL: Create JSON-SAFE metrics snapshot for the pipeline ---
        metrics_snapshot = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Classification Report': class_report,
            'Confusion Matrix': cm.tolist(), # Convert NumPy array to JSON-safe list
            'Best Parameters': best_params,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'model_type': model_type
            # DO NOT save: 'model', 'scaler', 'target_encoder', 'df' (non-primitive objects)
        }
        
        # Store results
        model_results = {
            'model': best_model,
            'scaler': scaler,
            'target_encoder': target_encoder,
            'metrics': metrics_snapshot,
            'features': features,
            'target': target,
            'model_type': model_type,
            'use_grid_search': use_grid_search,
            'grid_search_params': param_grid if use_grid_search else {},
            'manual_params': manual_params if not use_grid_search else {},
            'cv_folds': cv_folds,
            'df': df
        }
        
        # Create parameter list for pipeline
        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "model_type", "value": model_type},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "cv_folds", "value": cv_folds}
        ]
        
        # Add model-specific parameters
        if use_grid_search:
            param_list.append({"name": "param_grid", "value": str(param_grid)})
        else:
            for param_name, param_value in manual_params.items():
                param_list.append({"name": param_name, "value": param_value})
        
        # Create model entry for pipeline
        nb_model = DataManager.create_NB_Model(
            # f"{model_type.upper()}_NaiveBayes",
            model_name='Naive Bayes Classifier',
            model_param= param_list,
            trans=st.session_state.selected_trans,
            model=best_model,
            metrics_snapshot=metrics_snapshot
        )
        
        # Update pipeline
        # model_name = f"{model_type.upper()}_NaiveBayes"
        model_name = 'Naive Bayes Classifier'
        if edit:
            # Replace existing Naive Bayes entry
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != model_name else nb_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            # Append if no Naive Bayes model exists
            st.session_state.pipeline['ML'].append(nb_model)
        
        st.success(f"{model_type.upper()} Naive Bayes Model created successfully!")
        return model_results
        
    except Exception as e:
        st.error(f"Error training {model_type} Naive Bayes model: {str(e)}")
        return None

def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params['target'] in params['features']:
        st.error("Target column cannot be one of the features")
        return False
    
    # Check if we have enough data points
    if len(params['df']) < 10:
        st.error("Insufficient data for Naive Bayes (minimum 10 samples required)")
        return False
    
    # Check target type for different Naive Bayes variants
    if params['model_type'] == 'multinomial' or params['model_type'] == 'bernoulli':
        # For Multinomial and Bernoulli NB, target should be categorical/discrete
        target_unique = params['df'][params['target']].nunique()
        if target_unique < 2:
            st.error("Target must have at least 2 classes for classification")
            return False
    
    return True