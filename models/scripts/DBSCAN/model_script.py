# DBSCAN Model Training Script
import streamlit as st
import numpy as np
import pandas as pd
import itertools
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from constants import DataManager

data_manager = DataManager()


def model_script(df, features, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # ----------------------------------------------------------------
        # Prepare data
        # ----------------------------------------------------------------
        X = df[features].values

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ----------------------------------------------------------------
        # Grid search OR manual fit
        # ----------------------------------------------------------------
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")

            best_score  = -1
            best_params = None
            best_labels = None
            best_model  = None

            param_combinations = list(itertools.product(
                param_grid.get('eps',         [0.5]),
                param_grid.get('min_samples', [5]),
                param_grid.get('metric',      ['euclidean']),
            ))

            progress_bar    = st.progress(0)
            total_combinations = len(param_combinations)

            for idx, (eps, min_samples, metric) in enumerate(param_combinations):
                model  = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = model.fit_predict(X_scaled)

                n_cls   = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                if n_cls >= 2 and n_noise < len(labels):
                    mask = labels != -1
                    if mask.sum() > 1:
                        try:
                            score = silhouette_score(X_scaled[mask], labels[mask])
                            if score > best_score:
                                best_score  = score
                                best_params = {
                                    'eps':         eps,
                                    'min_samples': min_samples,
                                    'metric':      metric,
                                }
                                best_labels = labels
                                best_model  = model
                        except Exception:
                            pass

                progress_bar.progress((idx + 1) / total_combinations)

            progress_bar.empty()

            if best_model is None:
                st.warning("Grid search couldn't find optimal parameters. Using defaults.")
                best_params = {'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean'}
                best_model  = DBSCAN(**best_params)
                best_labels = best_model.fit_predict(X_scaled)

            st.success(f"Grid Search completed! Best parameters: {best_params}")

        else:
            st.info("Training with manual parameters...")

            eps        = manual_params.get('eps',         0.5)
            min_samples= manual_params.get('min_samples', 5)
            metric     = manual_params.get('metric',      'euclidean')
            algorithm  = manual_params.get('algorithm',   'auto')
            leaf_size  = manual_params.get('leaf_size',   30)

            best_model = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                algorithm=algorithm,
                leaf_size=leaf_size,
            )
            best_labels = best_model.fit_predict(X_scaled)
            best_params = {
                'eps':         float(eps),
                'min_samples': int(min_samples),
                'metric':      metric,
                'algorithm':   algorithm,
                'leaf_size':   int(leaf_size),
            }

        # ----------------------------------------------------------------
        # Compute final cluster stats
        # ----------------------------------------------------------------
        unique_labels   = np.array(sorted(set(best_labels)))
        n_clusters      = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise         = int((best_labels == -1).sum())
        n_samples       = len(best_labels)

        # Cluster sizes — list ordered by label (excluding noise label -1)
        cluster_labels_no_noise = [l for l in unique_labels if l != -1]
        cluster_sizes = [int((best_labels == l).sum()) for l in cluster_labels_no_noise]

        # ----------------------------------------------------------------
        # Quality metrics (on non-noise points when possible)
        # ----------------------------------------------------------------
        silhouette = calinski = davies = None

        if n_clusters >= 2:
            mask = best_labels != -1
            X_valid = X_scaled[mask]
            y_valid = best_labels[mask]

            if len(X_valid) > 1:
                try:
                    silhouette = float(silhouette_score(X_valid, y_valid))
                    calinski   = float(calinski_harabasz_score(X_valid, y_valid))
                    davies     = float(davies_bouldin_score(X_valid, y_valid))
                except Exception as e:
                    st.warning(f"Could not calculate some metrics: {e}")

        # ----------------------------------------------------------------
        # metrics_snapshot — every key the DBSCAN report functions read
        # ----------------------------------------------------------------
        metrics_snapshot = {
            # ── core cluster info ──────────────────────────────────────
            "N Clusters":     n_clusters,
            "N Noise Points": n_noise,
            "N Samples":      n_samples,
            "Cluster Sizes":  cluster_sizes,        # list of ints, no noise

            # ── quality scores ─────────────────────────────────────────
            "Silhouette Score":        silhouette,
            "Calinski-Harabasz Score": calinski,
            "Davies-Bouldin Index":    davies,

            # ── data arrays needed by visualisation functions ──────────
            "cluster_labels": best_labels,          # full array incl. -1
            "X_scaled":       X_scaled,             # numpy array

            # ── parameter / meta info ──────────────────────────────────
            "Best Parameters": best_params,
            "features":        features,
            "use_grid_search": use_grid_search,
            "cv_folds":        cv_folds,
        }

        # ----------------------------------------------------------------
        # Legacy model_results (kept for any live-view page)
        # ----------------------------------------------------------------
        model_results = {
            'model':          best_model,
            'scaler':         scaler,
            'labels':         best_labels,
            'metrics':        metrics_snapshot,
            'features':       features,
            'use_grid_search':use_grid_search,
            'cv_folds':       cv_folds,
            'X_scaled':       X_scaled,
            'df':             df,
        }
        st.session_state.model_results = model_results

        # ----------------------------------------------------------------
        # param_list for pipeline card display
        # ----------------------------------------------------------------
        param_list = [
            {"name": "features",       "value": features},
            {"name": "use_grid_search","value": use_grid_search},
            {"name": "cv_folds",       "value": cv_folds},
        ]

        if use_grid_search:
            param_list.extend([
                {"name": "eps_range",         "value": str(param_grid.get('eps',         []))},
                {"name": "min_samples_range", "value": str(param_grid.get('min_samples', []))},
                {"name": "metric_options",    "value": str(param_grid.get('metric',      []))},
            ])
        else:
            param_list.extend([
                {"name": "eps",         "value": best_params['eps']},
                {"name": "min_samples", "value": best_params['min_samples']},
                {"name": "metric",      "value": best_params['metric']},
                {"name": "algorithm",   "value": best_params.get('algorithm',  'auto')},
                {"name": "leaf_size",   "value": best_params.get('leaf_size',  30)},
            ])

        # ----------------------------------------------------------------
        # Pipeline entry
        # The 4th argument becomes metrics_snapshot inside DataManager,
        # which reporting.py reads as:
        #   st.session_state.pipeline['ML'][index]['metrics_snapshot']
        # ----------------------------------------------------------------
        DBSCAN_pipeline_entry = DataManager.create_DBSCAN_Model(
            "DBSCAN",               # must match pipeline_step_name in report items
            param_list,
            st.session_state.selected_trans,
            metrics_snapshot,       # ← full snapshot, not the raw model object
            best_model,
        )

        # ----------------------------------------------------------------
        # Update pipeline
        # ----------------------------------------------------------------
        if 'ML' not in st.session_state.pipeline:
            st.session_state.pipeline['ML'] = []

        if edit:
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != 'DBSCAN' else DBSCAN_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            st.session_state.pipeline['ML'].append(DBSCAN_pipeline_entry)

        st.success("DBSCAN Model created successfully and results saved to pipeline!")
        return model_results

    except Exception as e:
        st.error(f"Error training DBSCAN model: {str(e)}")
        return None


def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False

    if len(params['df']) < 10:
        st.error("Insufficient data for clustering (minimum 10 samples required)")
        return False

    return True