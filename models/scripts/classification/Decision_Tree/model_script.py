import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from constants import DataManager
data_manager = DataManager()
from sklearn.preprocessing import LabelEncoder # ADDED for target encoding
import pandas as pd # ADDED for feature encoding
import numpy as np # ADDED for metrics correction
import joblib, base64
from io import BytesIO
from sklearn.metrics import classification_report

DT_PARAM_GRID = {
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

def _model_to_b64(model) -> str:
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
 
 
def _safe_classification_report(y_test, y_pred) -> dict:
    """Return classification_report as a plain dict (JSON-safe)."""
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # ensure all nested values are plain Python floats / ints
    safe = {}
    for key, val in report.items():
        if isinstance(val, dict):
            safe[key] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v)
                         for k, v in val.items()}
        else:
            safe[key] = float(val) if isinstance(val, (np.floating, float)) else val
    return safe
 
 
# ─────────────────────────────────────────────────────────────────────────────
# main script
# ─────────────────────────────────────────────────────────────────────────────
def model_script(df, features, target, edit,
                 use_grid_search, max_depth, min_samples_leaf, random_state):
 
    # ── 1. Prepare data ──────────────────────────────────────────────────────
    df_copy = df.copy()
    X = df_copy[features]
    y = df_copy[target].values
 
    X_encoded       = pd.get_dummies(X, drop_first=True)
    encoded_features = X_encoded.columns.tolist()
 
    le = LabelEncoder()
    if y.dtype == "object" or y.dtype.name == "category":
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y.astype(int)
 
    X_final = X_encoded.values
    y_final = y_encoded
 
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y_final, test_size=0.2, random_state=42
    )
 
    # ── 2. Train ──────────────────────────────────────────────────────────────
    best_params = {}
    if use_grid_search:
        st.subheader("Grid Search Optimisation")
        st.write("Running Grid Search for best Decision Tree hyperparameters…")
        dt_base   = DecisionTreeClassifier(random_state=42)
        gs        = GridSearchCV(dt_base, DT_PARAM_GRID, cv=5,
                                 scoring="accuracy", n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        model       = gs.best_estimator_
        best_params = {k: (v if v is not None else "None")
                       for k, v in gs.best_params_.items()}
        st.success(f"Grid Search complete. Best params: {gs.best_params_}")
    else:
        st.subheader("Default Model Training")
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        st.success("Default Decision Tree Classifier trained.")
 
    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    y_pred        = model.predict(X_test)
    unique_classes = np.unique(y_test)
    avg_type       = "binary" if len(unique_classes) <= 2 else "weighted"
 
    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average=avg_type, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, average=avg_type, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, average=avg_type, zero_division=0))
    conf_mat  = confusion_matrix(y_test, y_pred).tolist()   # list-of-lists → JSON safe
    cls_report = _safe_classification_report(y_test, y_pred)
 
    # ── 4. Build metrics_snapshot (100 % JSON-safe) ───────────────────────────
    metrics_snapshot = {
        # raw metrics
        "metrics": {
            "Accuracy":         accuracy,
            "Precision":        precision,
            "Recall":           recall,
            "F1 Score":         f1,
            "Confusion Matrix": conf_mat,
        },
        "classification_report": cls_report,
        # model metadata
        "features":       encoded_features,          # list[str]
        "target":         str(target),
        "use_grid_search": bool(use_grid_search),
        "best_params":    best_params,               # dict[str, str|int|float]
        # serialised model (base64 string — JSON safe)
        "model_b64":      _model_to_b64(model),
    }
 
    # ── 5. Build session-state model_results (still carries live model) ───────
    model_results = {
        "model":          model,
        "metrics":        {
            "Accuracy":         accuracy,
            "Precision":        precision,
            "Recall":           recall,
            "F1 Score":         f1,
            "Confusion Matrix": conf_mat,
        },
        "features":       encoded_features,
        "target":         target,
        "use_grid_search": use_grid_search,
    }
 
    # ── 6. param_list for pipeline ────────────────────────────────────────────
    param_list = [
        {"name": "features",        "value": features},
        {"name": "target",          "value": target},
        {"name": "use_grid_search", "value": bool(use_grid_search)},
    ]
 
    # ── 7. Persist to pipeline ────────────────────────────────────────────────
    DT_entry = DataManager.create_DecisionTree_Model(
        "Decision Tree Classifier",
        param_list,
        st.session_state.selected_trans,
        model,
        metrics_snapshot,           # ← new argument
    )
 
    if edit:
        st.session_state.pipeline["ML"] = [
            item if item.get("name") != "Decision Tree Classifier" else DT_entry
            for item in st.session_state.pipeline["ML"]
        ]
    else:
        st.session_state.pipeline["ML"].append(DT_entry)
 
    st.success("Decision Tree model created successfully!")
    return model_results
 
 
# ─────────────────────────────────────────────────────────────────────────────
# validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_model(params):
    if len(params["features"]) == 0:
        st.error("Please select at least one feature column.")
        return False
    if params["target"] in params["features"]:
        st.error("Target column cannot be one of the features.")
        return False
    unique_classes = params["df"][params["target"]].nunique()
    if unique_classes < 2:
        st.error("Target column must have at least 2 unique classes.")
        return False
    return True