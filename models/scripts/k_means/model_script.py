import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from constants import DataManager
import json

data_manager = DataManager()

def detect_elbow(inertia_values):
    """Automatically detect elbow point for optimal K"""
    x = np.arange(len(inertia_values))
    y = np.array(inertia_values)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    distances = []
    for i in range(len(x)):
        p = np.array([x[i], y[i]])
        distance = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
        distances.append(distance)
    return x[np.argmax(distances)] + 2

def model_script(df, features, k_method, auto_k, manual_k, max_k, edit):
    try:
        # Prepare data
        X = df[features].values
        
        if np.isnan(X).any():
            st.warning("Missing values found. Filling with column means...")
            X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal K
        if k_method == "auto":
            inertia_values = []
            k_range = range(2, min(max_k + 1, len(df) // 2 + 1))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertia_values.append(kmeans.inertia_)
            
            optimal_k = detect_elbow(inertia_values)
            st.info(f"Auto-detected optimal K: {optimal_k} using Elbow Method")
        else:
            optimal_k = manual_k
            st.info(f"Using manual K: {optimal_k}")
        
        # Fit final model
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate evaluation metrics (JSON-compatible)
        inertia_value = float(kmeans.inertia_)
        silhouette_avg = float(silhouette_score(X_scaled, clusters))
        calinski_harabasz = float(calinski_harabasz_score(X_scaled, clusters))
        davies_bouldin = float(davies_bouldin_score(X_scaled, clusters))
        
        # Cluster analysis (JSON-compatible)
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        cluster_distribution = {
            f"Cluster {int(cluster)}": f"{int(count)} samples ({float(count/len(df)*100):.1f}%)" 
            for cluster, count in cluster_counts.items()
        }
        
        # Create JSON-compatible metrics snapshot
        metrics_snapshot = {
            'Optimal K': int(optimal_k),
            'Inertia': inertia_value,
            'Silhouette Score': silhouette_avg,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Davies-Bouldin Index': davies_bouldin,
            'Cluster Distribution': cluster_distribution,
            'k_method': str(k_method),
            'features': [str(f) for f in features]
        }
        
        # Create parameter list for pipeline
        param_list = [
            {"name": "features", "value": [str(f) for f in features]},
            {"name": "k_method", "value": str(k_method)},
            {"name": "auto_k", "value": bool(auto_k)},
            {"name": "manual_k", "value": int(manual_k)},
            {"name": "max_k", "value": int(max_k)}
        ]

        # Create model entry with metrics_snapshot
        kmeans_model = DataManager.create_KMeans_Model(
            "K-Means",
            param_list,
            st.session_state.selected_trans,
            metrics_snapshot=metrics_snapshot ,
            model=kmeans  # Add this parameter
        )
        
        # Update pipeline
        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('name') != 'K-Means' else kmeans_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(kmeans_model)
        
        st.success("K-Means Model created successfully!")
        return {"status": "success", "metrics_snapshot": metrics_snapshot}
        
    except Exception as e:
        st.error(f"Error training K-Means model: {str(e)}")
        return {"status": "error", "message": str(e)}

def validate_model(params):
    if len(params['features']) < 2:
        st.error("Please select at least two feature columns for clustering")
        return False
    
    for feature in params['features']:
        if params['df'][feature].dtype not in ['int64', 'float64']:
            st.error(f"Feature '{feature}' must be numeric for K-Means clustering")
            return False
    
    if params['k_method'] == "manual" and params['manual_k'] < 2:
        st.error("Number of clusters (K) must be at least 2")
        return False
    
    if params['max_k'] < 3:
        st.error("Maximum K for auto-detection must be at least 3")
        return False
    
    return True