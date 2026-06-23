import streamlit as st
import pandas as pd
import numpy as np
# pip install umap-learn
import umap.umap_ as umap
import traceback

def build_umap(df, edit_values=None):
    """Build UMAP dimensionality reduction UI"""
    st.write("### UMAP Dimensionality Reduction")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        st.error("No numeric features found for UMAP")
        return {"n_components": 2, "features": [], "n_neighbors": 15, "min_dist": 0.1}
    
    # Feature selection
    default_features = edit_values.get("features", numeric_columns) if edit_values else numeric_columns
    selected_features = st.multiselect(
        "Select numeric features for UMAP",
        numeric_columns,
        default=default_features
    )
    
    # Number of components
    max_components = min(len(selected_features), 10) if selected_features else 2
    default_components = edit_values.get("n_components", 2) if edit_values else 2
    
    n_components = st.number_input(
        "Number of components",
        min_value=2,
        max_value=max_components,
        value=default_components,
        key="umap_components"
    )
    
    # UMAP specific parameters
    n_neighbors = st.number_input(
        "Number of neighbors",
        min_value=2,
        max_value=100,
        value=edit_values.get("n_neighbors", 15) if edit_values else 15,
        key="umap_neighbors"
    )
    
    min_dist = st.number_input(
        "Minimum distance",
        min_value=0.0,
        max_value=1.0,
        value=edit_values.get("min_dist", 0.1) if edit_values else 0.1,
        key="umap_min_dist"
    )
    
    return {
        "n_components": n_components,
        "features": selected_features,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist
    }

def apply_umap(df, step):
    """Apply UMAP dimensionality reduction"""
    df_copy = df.copy()
    
    try:
        n_components = step["n_components"]
        features = step["features"]
        n_neighbors = step["n_neighbors"]
        min_dist = step["min_dist"]
        
        if not features:
            st.warning("No features selected for UMAP")
            return df_copy
        
        if n_components > len(features):
            st.warning(f"n_components ({n_components}) cannot be greater than number of features ({len(features)})")
            n_components = len(features)
        
        # Prepare data
        X = df_copy[features].fillna(df_copy[features].mean())
        
        # Apply UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
        embedding = reducer.fit_transform(X)
        
        # Create new columns for the components
        component_columns = [f"UMAP_{i+1}" for i in range(n_components)]
        embedding_df = pd.DataFrame(embedding, columns=component_columns, index=df_copy.index)
        
        # Drop original features and add new components
        df_copy = df_copy.drop(columns=features)
        df_copy = pd.concat([df_copy, embedding_df], axis=1)
        
        st.info(f"Applied UMAP reduction: {len(features)} features → {n_components} components")
        
    except Exception as e:
        st.error(f"Error in UMAP dimensionality reduction: {str(e)}")
        raise
    
    return df_copy