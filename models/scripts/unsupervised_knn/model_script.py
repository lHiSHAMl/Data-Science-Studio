import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from constants import DataManager

data_manager = DataManager()

def model_script(df, features, edit, use_grid_search, param_grid, manual_params, n_clusters):
    try:
        # Prepare data
        X = df[features].values
        
        # Scale features for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal parameters
        if use_grid_search:
            st.info("Finding optimal KNN parameters...")
            
            best_score = -1
            best_params = {}
            best_clusters = None
            best_method = ""
            
            for n_neighbors in param_grid.get('n_neighbors', [5]):
                for algorithm in param_grid.get('algorithm', ['auto']):
                    for metric in param_grid.get('metric', ['euclidean']):
                        
                        clusters, method_used = apply_knn_clustering(
                            X_scaled, n_clusters, n_neighbors, metric, algorithm
                        )
                        
                        if len(np.unique(clusters)) > 1:
                            score = silhouette_score(X_scaled, clusters)
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'n_neighbors': n_neighbors,
                                    'algorithm': algorithm,
                                    'metric': metric
                                }
                                best_clusters = clusters
                                best_method = method_used

            st.success(f"Optimal parameters found: {best_params}")
            
        else:
            st.info("Applying KNN clustering with manual parameters...")
            best_params = manual_params
            best_clusters, best_method = apply_knn_clustering(
                X_scaled, n_clusters,
                manual_params['n_neighbors'],
                manual_params['metric'],
                manual_params['algorithm']
            )
        
        # Calculate clustering metrics
        unique_labels = np.unique(best_clusters)
        n_found_clusters = len(unique_labels)

        if n_found_clusters > 1:
            silhouette = silhouette_score(X_scaled, best_clusters)
            calinski   = calinski_harabasz_score(X_scaled, best_clusters)
            davies     = davies_bouldin_score(X_scaled, best_clusters)
        else:
            silhouette = calinski = davies = None
            st.warning("Only one cluster found. Metrics may not be meaningful.")
        
        # Cluster sizes as a plain list ordered by label
        cluster_sizes = [
            int(np.sum(best_clusters == label))
            for label in sorted(unique_labels)
        ]

        # ----------------------------------------------------------------
        # metrics_snapshot — mirrors the hierarchical clustering structure
        # so the report functions can read it with identical key names
        # ----------------------------------------------------------------
        metrics_snapshot = {
            # Quality scores
            "Silhouette Score":        silhouette,
            "Calinski-Harabasz Score": calinski,
            "Davies-Bouldin Score":    davies,

            # Cluster topology
            "Number of Clusters": n_found_clusters,
            "Cluster Sizes":      cluster_sizes,   # plain list of ints
            "cluster_labels":     best_clusters,   # numpy array

            # Scaled data needed for PCA / t-SNE / silhouette plots
            "X_scaled":  X_scaled,                 # numpy array

            # Feature / parameter metadata
            "features":  features,
            "method_used": best_method,
            "parameters":  best_params,

            # Kept for compatibility with shared helper keys used in
            # hierarchical clustering (report parser may reference these)
            "linkage":           best_params.get('algorithm', 'N/A'),
            "metric":            best_params.get('metric', 'euclidean'),
            "compute_full_tree": False,
        }

        # ----------------------------------------------------------------
        # param_list for the pipeline card
        # ----------------------------------------------------------------
        param_list = [
            {"name": "features",       "value": features},
            {"name": "n_clusters",     "value": n_clusters},
            {"name": "use_grid_search","value": use_grid_search},
            {"name": "n_neighbors",    "value": str(param_grid.get('n_neighbors', [])) if use_grid_search else manual_params['n_neighbors']},
            {"name": "algorithm",      "value": str(param_grid.get('algorithm', []))   if use_grid_search else manual_params['algorithm']},
            {"name": "metric",         "value": str(param_grid.get('metric', []))      if use_grid_search else manual_params['metric']},
        ]

        # ----------------------------------------------------------------
        # Build the pipeline model entry with metrics_snapshot as the
        # last argument — same pattern as hierarchical clustering
        # ----------------------------------------------------------------
        UnsupKNN_model = DataManager.create_UnsupKNN_Model(
            "Unsupervised KNN Clustering",   # must match FUNCTION_MAP key
            param_list,
            st.session_state.selected_trans,
            best_clusters,
            metrics_snapshot                 # <— full snapshot, not raw metrics
        )

        # Update pipeline
        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != 'Unsupervised KNN Clustering'
                else UnsupKNN_model
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(UnsupKNN_model)
        
        st.success("Unsupervised KNN Clustering completed successfully!")

        # Also expose via legacy key so any existing page that reads
        # st.session_state.model_results keeps working
        st.session_state.model_results = {
            'clusters':     best_clusters,
            'scaler':       scaler,
            'df':           df,
            'features':     features,
            'n_clusters':   n_found_clusters,
            'metrics': {
                'silhouette_score':       silhouette,
                'calinski_harabasz_score': calinski,
                'davies_bouldin_score':    davies,
            },
            'cluster_sizes': dict(zip(sorted(unique_labels), cluster_sizes)),
            'method_used':  best_method,
            'parameters':   best_params,
        }

        return st.session_state.model_results
        
    except Exception as e:
        st.error(f"Error performing unsupervised KNN clustering: {str(e)}")
        return None


def apply_knn_clustering(X, n_clusters, n_neighbors, metric, algorithm):
    """Apply KNN-based clustering method."""
    
    if n_clusters is not None:
        # Spectral clustering on a KNN connectivity graph
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
        nn.fit(X)
        adjacency_matrix = nn.kneighbors_graph(X, mode='connectivity')
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        clusters    = spectral.fit_predict(adjacency_matrix)
        method_used = "Spectral Clustering with KNN graph"
    else:
        # Density-based fallback using DBSCAN with KNN-derived eps
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        avg_distances = distances.mean(axis=1)
        
        eps     = np.percentile(avg_distances, 50)
        dbscan  = DBSCAN(eps=eps, min_samples=n_neighbors)
        clusters    = dbscan.fit_predict(X)
        method_used = "DBSCAN with KNN density estimation"
    
    return clusters, method_used


def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False
    
    if len(params['features']) < 2:
        st.warning("Clustering with only one feature may not yield meaningful results")
    
    if params['n_clusters'] is None:
        st.error("Please specify the number of clusters")
        return False
    
    n_samples = len(params['df'])
    if params['n_clusters'] >= n_samples:
        st.error("Number of clusters cannot be greater than or equal to number of samples")
        return False
    
    return True