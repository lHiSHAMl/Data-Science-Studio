import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from constants import DataManager

def hierarchy_script(df, features, n_clusters, linkage, metric, compute_full_tree, distance_threshold, edit):
    # Prepare data
    try:
        X = df[features].values
        
        # Standardize the data (important for clustering)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create and fit the model
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            linkage=linkage,
            compute_full_tree=compute_full_tree,
            distance_threshold=distance_threshold
        )
        
        # Make predictions (cluster labels)
        cluster_labels = model.fit_predict(X_scaled)
        
        # Calculate clustering metrics
        try:
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        except:
            silhouette_avg = None
        
        try:
            calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
        except:
            calinski_score = None
            
        try:
            davies_score = davies_bouldin_score(X_scaled, cluster_labels)
        except:
            davies_score = None
            
        metrics_snapshot = {
            'Silhouette Score': silhouette_avg,
            'Calinski-Harabasz Score': calinski_score,
            'Davies-Bouldin Score': davies_score,
            'Number of Clusters': n_clusters,
            'Cluster Sizes': [sum(cluster_labels == i) for i in range(n_clusters)],
            'cluster_labels': cluster_labels,
            'features': features,
            'X_scaled': X_scaled,
            'linkage': linkage,
            'metric': metric,
            'compute_full_tree': compute_full_tree

        }

        # Store results
        model_results = {
            'model': model,
            'scaler': scaler,
            'metrics': {
                'Silhouette Score': silhouette_avg,
                'Calinski-Harabasz Score': calinski_score,
                'Davies-Bouldin Score': davies_score,
                'Number of Clusters': n_clusters,
                'Cluster Sizes': [sum(cluster_labels == i) for i in range(n_clusters)]
            },
            'cluster_labels': cluster_labels,
            'features': features,
            'X_scaled': X_scaled  # Store scaled data for visualization
        }
        # Prepare parameter list for pipeline
        param_list = [
            {"name": "features", "value": features},
            {"name": "n_clusters", "value": n_clusters},
            {"name": "linkage", "value": linkage},
            {"name": "metric", "value": metric},
            {"name": "compute_full_tree", "value": compute_full_tree},
            {"name": "distance_threshold", "value": distance_threshold}
        ]
        
        best_model = model
        
        model_name_to_check = "Hierarchical Clustering"
        
        Hierarchy_model_pipeline_entry = DataManager.create_Hierarchical_Clustering_Model(
            model_name_to_check, 
            param_list,
            st.session_state.selected_trans,
            best_model,
            metrics_snapshot
        )

        # Update pipeline
        if 'ML' not in st.session_state.pipeline:
             st.session_state.pipeline['ML'] = []
             
        if edit:
            # Replace existing entry by matching 'model name'
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != model_name_to_check else Hierarchy_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            # Append 
            st.session_state.pipeline['ML'].append(Hierarchy_model_pipeline_entry)

        st.success("Hierarchical Clustering model created successfully!")
        
        # --- THIS IS THE CRITICAL FIX ---
        # Save the complete results to session state for the report page
        st.session_state.model_results = model_results
        
        return model_results
    
    except Exception as e:
        st.error(f"An error occurred while creating the model: {e}")
        
        # --- THIS IS THE OTHER HALF OF THE CRITICAL FIX ---
        # If the model fails, save None so the report page knows it failed
        st.session_state.model_results = None 
        
        return None

def hierarchy_validate_model(params):
    if len(params['features']) < 2:
        st.error("Please select at least two feature columns for clustering")
        return False
    
    if params['linkage'] == 'ward' and params['metric'] != 'euclidean':
        st.error("Ward linkage can only be used with Euclidean distance metric")
        return False
    
    if params['n_clusters'] < 2:
        st.error("Number of clusters must be at least 2")
        return False
    
    if params['n_clusters'] > len(params['df']):
        st.error("Number of clusters cannot exceed the number of data points")
        return False
    
    return True