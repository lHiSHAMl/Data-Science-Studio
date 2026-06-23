import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import traceback

def build_ref_selection(df, edit_values=None):
    """Build Recursive Feature Elimination UI"""
    st.write("### Recursive Feature Elimination (RFE)")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        st.error("No numeric features found for RFE")
        return {"n_features": 0, "target": None, "features": [], "estimator_type": "linear"}
    
    # Target selection for regression/classification
    target_options = df.columns.tolist()
    default_target = edit_values.get("target", target_options[0]) if edit_values else target_options[0]
    
    target = st.selectbox(
        "Select target variable",
        target_options,
        index=target_options.index(default_target) if default_target in target_options else 0,
        key="ref_target"
    )
    
    # Feature selection
    feature_options = [col for col in numeric_columns if col != target]
    default_features = edit_values.get("features", feature_options) if edit_values else feature_options
    
    selected_features = st.multiselect(
        "Select features for RFE",
        feature_options,
        default=default_features,
        key="ref_features"
    )
    
    # Number of features to select
    max_features = min(len(selected_features), 20) if selected_features else 1
    default_n_features = edit_values.get("n_features", min(5, max_features)) if edit_values else min(5, max_features)
    
    n_features = st.number_input(
        "Number of features to select",
        min_value=1,
        max_value=max_features,
        value=default_n_features,
        key="ref_n_features"
    )
    
    # Estimator type
    estimator_type = st.selectbox(
        "Estimator type",
        ["linear", "tree"],
        index=0 if edit_values and edit_values.get("estimator_type") == "linear" else 0,
        key="ref_estimator"
    )
    
    return {
        "n_features": n_features,
        "target": target,
        "features": selected_features,
        "estimator_type": estimator_type
    }

def apply_ref_selection(df, step):
    """Apply Recursive Feature Elimination"""
    df_copy = df.copy()
    
    try:
        n_features = step["n_features"]
        target = step["target"]
        features = step["features"]
        estimator_type = step["estimator_type"]
        
        if not features or not target:
            st.warning("No features or target selected for RFE")
            return df_copy
        
        if n_features > len(features):
            st.warning(f"n_features ({n_features}) cannot be greater than number of features ({len(features)})")
            n_features = len(features)
        
        # Prepare data
        X = df_copy[features].fillna(df_copy[features].mean())
        y = df_copy[target]
        
        # Handle categorical target
        if y.dtype == 'object':
            y = pd.factorize(y)[0]
        
        # Select estimator
        if estimator_type == "linear":
            estimator = LogisticRegression(max_iter=1000, random_state=42)
        else:  # tree
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Apply RFE
        selector = RFE(estimator, n_features_to_select=n_features)
        X_new = selector.fit_transform(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = [features[i] for i in range(len(features)) if selected_mask[i]]
        
        # Keep only selected features and target
        df_copy = df_copy[selected_features + [target]]
        
        # Display ranking
        rankings = selector.ranking_
        feature_rankings = {features[i]: rankings[i] for i in range(len(features))}
        sorted_rankings = sorted(feature_rankings.items(), key=lambda x: x[1])
        
        st.info("RFE feature rankings (1 = selected):")
        for feature, rank in sorted_rankings:
            st.write(f"  - {feature}: {rank}")
            
    except Exception as e:
        st.error(f"Error in RFE feature selection: {str(e)}")
        raise
    
    return df_copy