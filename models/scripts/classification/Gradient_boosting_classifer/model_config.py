import streamlit as st
import pandas as pd

def model_config(model_data, edit):        
    numeric_cols = [col for col in st.session_state.selected_data.columns if st.session_state.selected_data[col].dtype in ['int64', 'float64']] 
    
    # Feature and target selection
    col1, col2 = st.columns(2)
    with col1:
        # Handle default features for edit mode
        default_features = []
        if edit and model_data.get('model param'):
            for param in model_data['model param']:
                if param['name'] == 'features':
                    default_features = param['value'] if isinstance(param['value'], list) else []
                    break
        
        features = st.multiselect(
            "Select feature columns:",
            options=[col for col in st.session_state.selected_data.columns if st.session_state.selected_data[col].dtype in ['int64', 'float64']] if len(numeric_cols) != 0 else [],
            default=default_features
        )
    
    with col2:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for target selection")
            target = None
        else:
            # Safe index calculation for selectbox
            default_index = 0
            if edit and model_data.get('model param'):
                target_value = None
                for param in model_data['model param']:
                    if param['name'] == 'target':
                        target_value = param['value']
                        break
                
                if target_value and target_value in numeric_cols:
                    default_index = numeric_cols.index(target_value)
            
            target = st.selectbox(
                "Select target column:",
                options=numeric_cols,
                index=default_index,
                key="target_column_gb"
            )    
    
    # Gradient Boosting Hyperparameter Configuration
    st.subheader("🎯 Gradient Boosting Hyperparameters")
    
    col3, col4 = st.columns(2)
    
    with col3:
        n_estimators = st.slider(
            "Number of Estimators:",
            min_value=10,
            max_value=500,
            value=100,
            help="Number of boosting stages to perform"
        )
        
        learning_rate = st.slider(
            "Learning Rate:",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Shrinks the contribution of each tree"
        )
    
    with col4:
        max_depth = st.slider(
            "Max Depth:",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum depth of individual trees"
        )
        
        min_samples_split = st.slider(
            "Min Samples Split:",
            min_value=2,
            max_value=20,
            value=2,
            help="Minimum number of samples required to split a node"
        )
    
    subsample = st.slider(
        "Subsample Ratio:",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Fraction of samples to be used for fitting individual trees"
    )
    
    # Prepare parameters for model script
    manual_params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'subsample': subsample
    }
    
    return {
        "features": features, 
        "target": target, 
        "df": st.session_state.selected_data, 
        "edit": edit,
        "use_grid_search": False,
        "param_grid": {},
        "manual_params": manual_params,
        "cv_folds": 5
    }

model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Gradient Boosting Classifier</span>
    </div>
    <div>
        # Gradient Boosting Classifier Implementation
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Encode target if categorical
        if df[target].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train Gradient Boosting model
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            subsample=1.0,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        st.session_state.model_results = {
            'model': model,
            'metrics': {
                'Accuracy': accuracy,
                'Classification Report': class_report,
                'Confusion Matrix': conf_matrix,
                'Feature Importances': model.feature_importances_
            },
            'features': features,
            'target': target
        }
    </div>
</div>
"""

model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Gradient Boosting Classifier</span>
    </div>
    <div>
        Gradient Boosting is a powerful ensemble method that builds trees sequentially, 
        where each new tree corrects the errors of the previous ones. It combines weak 
        learners (typically decision trees) into a strong learner through gradient 
        descent optimization.
        
        <strong>Key Advantages:</strong>
        • High predictive accuracy
        • Handles mixed data types well
        • Provides feature importance
        • Robust to outliers
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Accuracy:</strong> Overall classification correctness
        • <strong>Precision:</strong> Ratio of true positives to predicted positives
        • <strong>Recall:</strong> Ratio of true positives to actual positives  
        • <strong>F1-Score:</strong> Harmonic mean of precision and recall
        
        <strong>Key Parameters:</strong>
        • <strong>n_estimators:</strong> Number of boosting stages
        • <strong>learning_rate:</strong> Shrinks the contribution of each tree
        • <strong>max_depth:</strong> Maximum depth of individual trees
        • <strong>min_samples_split:</strong> Minimum samples required to split a node
        • <strong>subsample:</strong> Fraction of samples used for fitting trees
    </div>
</div>
"""