import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import traceback

def build_tsne(df, edit_values=None):
    """Build t-SNE dimensionality reduction UI"""
    st.write("### t-SNE Dimensionality Reduction")
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0:
        st.error("No numeric features found for t-SNE")
        return {"n_components": 2, "features": [], "perplexity": 30.0, "learning_rate": 200.0}
    
    # Feature selection
    default_features = edit_values.get("features", numeric_columns) if edit_values else numeric_columns
    selected_features = st.multiselect(
        "Select numeric features for t-SNE",
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
        key="tsne_components"
    )
    
    # t-SNE specific parameters
    perplexity = st.number_input(
        "Perplexity",
        min_value=5.0,
        max_value=50.0,
        value=edit_values.get("perplexity", 30.0) if edit_values else 30.0,
        key="tsne_perplexity"
    )
    
    learning_rate = st.number_input(
        "Learning rate",
        min_value=10.0,
        max_value=1000.0,
        value=edit_values.get("learning_rate", 200.0) if edit_values else 200.0,
        key="tsne_learning_rate"
    )
    
    return {
        "n_components": n_components,
        "features": selected_features,
        "perplexity": perplexity,
        "learning_rate": learning_rate
    }

def apply_tsne(df, step):
    """Apply t-SNE dimensionality reduction"""
    df_copy = df.copy()
    
    try:
        n_components = step["n_components"]
        features = step["features"]
        perplexity = step["perplexity"]
        learning_rate = step["learning_rate"]
        
        if not features:
            st.warning("No features selected for t-SNE")
            return df_copy
        
        if n_components > len(features):
            st.warning(f"n_components ({n_components}) cannot be greater than number of features ({len(features)})")
            n_components = len(features)
        
        # Prepare data
        X = df_copy[features].fillna(df_copy[features].mean())
        
        # Apply t-SNE
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=42
        )
        
        embedding = reducer.fit_transform(X)
        
        # Create new columns for the components
        component_columns = [f"tSNE_{i+1}" for i in range(n_components)]
        embedding_df = pd.DataFrame(embedding, columns=component_columns, index=df_copy.index)
        
        # Drop original features and add new components
        df_copy = df_copy.drop(columns=features)
        df_copy = pd.concat([df_copy, embedding_df], axis=1)
        
        st.info(f"Applied t-SNE reduction: {len(features)} features → {n_components} components")
        
    except Exception as e:
        st.error(f"Error in t-SNE dimensionality reduction: {str(e)}")
        raise
    
    return df_copy