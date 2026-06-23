import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from constants import DataManager

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
# from transformations.utils.robust_ml import prepare_classification_frame, scale_array

data_manager = DataManager()

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # --- Basic validation ---
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            st.error("Invalid or empty dataframe provided.")
            return None

        if target not in df.columns:
            st.error(f"Target column '{target}' not found in dataframe.")
            return None

        # --- Build X/y from arbitrary pipeline output ---
        # X_df, y_raw, encoded_features = prepare_classification_frame(df, features, target)
        X = df[features]
        y_raw = df[target]
        encoded_features = features# X = scale_array(X_df.values, "robust")

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        target_encoder = le

        unique_classes = np.unique(y)
        if len(X) < 2 or len(unique_classes) < 2:
            st.warning(
                "Dataset is too small or has one target class only. "
                "Using a constant baseline so the automated pipeline can still produce results."
            )
            X_train = X_test = X
            y_train = y_test = y
            force_dummy = True
        else:
            force_dummy = False
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError:
                st.warning(
                    "Stratified split failed (likely a class with too few samples). "
                    "Falling back to random split."
                )
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                except ValueError:
                    st.warning("Random split failed. Evaluating on the available training data.")
                    X_train = X_test = X
                    y_train = y_test = y

        # Ensure safe params
        if manual_params is None:
            manual_params = {}
        if param_grid is None:
            param_grid = {}
        try:
            cv_folds = max(2, int(cv_folds))
        except Exception:
            cv_folds = 5

        # --- Train model ---
        if force_dummy:
            best_model = DummyClassifier(strategy="most_frequent")
            best_model.fit(X_train, y_train)
            best_params = {"fallback": "DummyClassifier", "reason": "insufficient rows or one target class"}
        elif use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            svm = SVC(probability=True, random_state=42)
            try:
                grid_search = GridSearchCV(
                    svm,
                    param_grid,
                    cv=cv_folds,
                    scoring='accuracy',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                st.success(f"Grid Search completed! Best parameters: {best_params}")
            except Exception as gs_err:
                st.warning(f"Grid search failed ({gs_err}). Falling back to manual parameters.")
                best_model = SVC(
                    C=float(manual_params.get('C', 1.0)),
                    kernel=manual_params.get('kernel', 'rbf'),
                    gamma=manual_params.get('gamma', 'scale'),
                    degree=int(manual_params.get('degree', 3)),
                    probability=True,
                    random_state=42
                )
                best_model.fit(X_train, y_train)
                best_params = manual_params
        else:
            st.info("Training with manual parameters...")
            try:
                best_model = SVC(
                    C=float(manual_params.get('C', 1.0)),
                    kernel=manual_params.get('kernel', 'rbf'),
                    gamma=manual_params.get('gamma', 'scale'),
                    degree=int(manual_params.get('degree', 3)),
                    probability=True,
                    random_state=42
                )
                best_model.fit(X_train, y_train)
                best_params = manual_params
            except Exception as fit_err:
                st.warning(f"SVM fit failed ({fit_err}). Falling back to a constant baseline.")
                best_model = DummyClassifier(strategy="most_frequent")
                best_model.fit(X_train, y_train)
                best_params = {"fallback": "DummyClassifier", "reason": str(fit_err)}

        # --- Predictions & metrics ---
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        metrics_snapshot = {
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Recall': float(recall),
            'F1 Score': float(f1),
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix.tolist(),
            'Best Parameters': best_params,
            'features': encoded_features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }

        model_results = {
            'model': best_model,
            'target_encoder': target_encoder,
            'metrics': metrics_snapshot,
            'features': encoded_features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
        }
        st.session_state.model_results = model_results

        param_list = [
            {"name": "features", "value": encoded_features},
            {"name": "target", "value": target},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "C_range", "value": str(param_grid.get('C', [])) if use_grid_search else manual_params.get('C', 1.0)},
            {"name": "kernel_range", "value": str(param_grid.get('kernel', [])) if use_grid_search else manual_params.get('kernel', 'rbf')},
            {"name": "gamma_range", "value": str(param_grid.get('gamma', [])) if use_grid_search else manual_params.get('gamma', 'scale')},
            {"name": "degree", "value": manual_params.get('degree', 3) if not use_grid_search else 'N/A'},
            {"name": "cv_folds", "value": cv_folds}
        ]

        try:
            SVM_model = DataManager.create_SVM_Model(
                "Support Vector Machine Classifier",
                param_list,
                st.session_state.selected_trans,
                metrics_snapshot
            )
        except Exception:
            SVM_model = data_manager.create_SVM_Model(
                "Support Vector Machine Classifier",
                param_list,
                st.session_state.selected_trans,
                metrics_snapshot,
                best_model
            )

        # Robust pipeline/session key guard
        if 'pipeline' not in st.session_state or not isinstance(st.session_state.pipeline, dict):
            st.session_state.pipeline = {}
        if 'ML' not in st.session_state.pipeline or not isinstance(st.session_state.pipeline.get('ML'), list):
            st.session_state.pipeline['ML'] = []

        model_name_to_check = "Support Vector Machine Classifier"
        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != model_name_to_check and item.get('model name') != model_name_to_check
                else SVM_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(SVM_model)

        st.success("Support Vector Machine Model created successfully!")
        return model_results

    except Exception as e:
        st.error(f"Error training Support Vector Machine model: {str(e)}")
        return None


def validate_model(params):
    features = params.get('features', [])
    df = params.get('df')

    if df is None or (hasattr(df, '__len__') and len(df) == 0):
        st.error("No data available.")
        return False

    if len(features) == 0:
        st.error("Please select at least one feature column")
        return False

    if params.get('target') in features:
        st.error("Target column cannot be one of the features")
        return False

    target = params.get('target')
    if target not in df.columns:
        st.error(f"Target column '{target}' not found in dataframe.")
        return False

    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Features not found in dataframe: {missing}")
        return False

    target_values = df[target].nunique()
    if target_values < 2:
        st.error("Target column must have at least 2 unique classes for classification")
        return False

    return True
