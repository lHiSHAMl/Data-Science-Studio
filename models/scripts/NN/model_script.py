import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from constants import DataManager
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_nn_model(input_dim, num_classes, manual_params):
    """Create a neural network model based on parameters"""
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers with individual activations and dropout
    layer_neurons = manual_params['layer_neurons']
    layer_dropouts = manual_params['layer_dropouts']
    layer_activations = manual_params['layer_activations']
    
    for neurons, activation, dropout_rate in zip(layer_neurons, layer_activations, layer_dropouts):
        # Handle special activation functions
        if activation == 'leaky_relu':
            model.add(layers.Dense(neurons))
            model.add(layers.LeakyReLU(alpha=0.1))
        elif activation == 'swish':
            model.add(layers.Dense(neurons))
            model.add(layers.Activation('swish'))
        else:
            model.add(layers.Dense(neurons, activation=activation))
        
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    output_activation = manual_params.get('output_activation', 'softmax')
    
    if num_classes == 2:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation=output_activation))
        loss = 'sparse_categorical_crossentropy'
    
    # Configure optimizer
    optimizer_name = manual_params['optimizer']
    learning_rate = manual_params['learning_rate']
    
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'adagrad':
        optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def grid_search_nn(X_train, y_train, X_val, y_val, param_grid, input_dim, num_classes, manual_params_base):
    """Perform grid search for neural network hyperparameters"""
    best_score = 0
    best_params = {}
    best_model = None
    results = []
    
    total_combinations = len(param_grid.get('epochs', [1])) * len(param_grid.get('batch_size', [1])) * len(param_grid.get('learning_rate', [1]))
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    combination_count = 0
    
    for epochs in param_grid.get('epochs', [manual_params_base['epochs']]):
        for batch_size in param_grid.get('batch_size', [manual_params_base['batch_size']]):
            for learning_rate in param_grid.get('learning_rate', [manual_params_base['learning_rate']]):
                
                combination_count += 1
                status_text.text(f"Testing combination {combination_count}/{total_combinations}: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
                
                # Update parameters
                manual_params = manual_params_base.copy()
                manual_params['epochs'] = epochs
                manual_params['batch_size'] = batch_size
                manual_params['learning_rate'] = learning_rate
                
                # Create and train model
                model = create_nn_model(input_dim, num_classes, manual_params)
                
                # Callbacks
                callbacks = []
                if manual_params.get('early_stopping', True):
                    callbacks.append(
                        keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True
                        )
                    )
                
                # Train
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                val_accuracy = max(history.history['val_accuracy']) if history.history['val_accuracy'] else 0
                
                results.append({
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'val_accuracy': val_accuracy
                })
                
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate
                    }
                    best_model = model
                
                progress_bar.progress(combination_count / total_combinations)
    
    status_text.empty()
    progress_bar.empty()
    
    return best_model, best_params, results

def model_script(df, features, target, edit, use_grid_search, param_grid, manual_params, cv_folds):
    try:
        # Safety check - ensure layer_activations exists
        if 'layer_activations' not in manual_params:
            st.warning("layer_activations not found in parameters. Creating default values.")
            num_layers = manual_params.get('num_layers', 2)
            manual_params['layer_activations'] = ['relu'] * num_layers
        
        # Prepare data
        X = df[features].values
        y = df[target].values
        
        # Encode target variable if it's categorical
        if df[target].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            target_encoder = le
            num_classes = len(le.classes_)
        else:
            target_encoder = None
            num_classes = len(np.unique(y))
        
        # Scale features (important for neural networks)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        input_dim = X_scaled.shape[1]
        
        # Train model with or without grid search
        if use_grid_search:
            st.info("Performing Grid Search for optimal parameters...")
            
            # Split training data for validation during grid search
            X_train_grid, X_val, y_train_grid, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train if len(np.unique(y_train)) > 1 else None
            )
            
            best_model, best_params, grid_results = grid_search_nn(
                X_train_grid, y_train_grid, X_val, y_val, 
                param_grid, input_dim, num_classes, manual_params
            )
            
            # Update manual_params with best params
            for key, value in best_params.items():
                manual_params[key] = value
            
            st.success(f"Grid Search completed! Best parameters: {best_params}")
            
            # Train final model on full training data
            st.info("Training final model with best parameters...")
            final_model = create_nn_model(input_dim, num_classes, manual_params)
            
            callbacks = []
            if manual_params.get('early_stopping', True):
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                )
            
            history = final_model.fit(
                X_train, y_train,
                epochs=manual_params['epochs'],
                batch_size=manual_params['batch_size'],
                validation_split=manual_params.get('validation_split', 0.2),
                callbacks=callbacks,
                verbose=0
            )
            
            best_model = final_model
            training_history = history.history
            
        else:
            st.info("Training Neural Network with manual parameters...")
            
            best_model = create_nn_model(input_dim, num_classes, manual_params)
            
            callbacks = []
            if manual_params.get('early_stopping', True):
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                )
            
            history = best_model.fit(
                X_train, y_train,
                epochs=manual_params['epochs'],
                batch_size=manual_params['batch_size'],
                validation_split=manual_params.get('validation_split', 0.2),
                callbacks=callbacks,
                verbose=0
            )
            
            training_history = history.history
        
        # Make predictions
        y_pred_proba = best_model.predict(X_test, verbose=0)
        if num_classes == 2:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Prepare training history for JSON serialization
        serializable_history = {
            'loss': training_history.get('loss', []),
            'accuracy': training_history.get('accuracy', []),
            'val_loss': training_history.get('val_loss', []),
            'val_accuracy': training_history.get('val_accuracy', [])
        }
        
        # Create JSON-safe metrics snapshot
        metrics_snapshot = {
            'Accuracy': accuracy,
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix.tolist(),
            'Best Parameters': manual_params if not use_grid_search else best_params,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'num_classes': num_classes,
            'input_dim': input_dim,
            'training_history': serializable_history
        }
        
        # Update st.session_state.model_results for live view
        model_results = {
            'model': best_model,
            'scaler': scaler,
            'target_encoder': target_encoder,
            'metrics': metrics_snapshot,
            'features': features,
            'target': target,
            'use_grid_search': use_grid_search,
            'cv_folds': cv_folds,
            'training_history': training_history
        }
        st.session_state.model_results = model_results
        
        # Create parameter list for pipeline
        param_list = [
            {"name": "features", "value": features},
            {"name": "target", "value": target},
            {"name": "use_grid_search", "value": use_grid_search},
            {"name": "num_layers", "value": manual_params['num_layers']},
            {"name": "layer_neurons", "value": manual_params['layer_neurons']},
            {"name": "layer_dropouts", "value": manual_params['layer_dropouts']},
            {"name": "layer_activations", "value": manual_params['layer_activations']},
            {"name": "output_activation", "value": manual_params['output_activation']},
            {"name": "epochs", "value": manual_params['epochs']},
            {"name": "batch_size", "value": manual_params['batch_size']},
            {"name": "learning_rate", "value": manual_params['learning_rate']},
            {"name": "optimizer", "value": manual_params['optimizer']},
            {"name": "validation_split", "value": manual_params.get('validation_split', 0.2)},
            {"name": "early_stopping", "value": manual_params.get('early_stopping', True)},
            {"name": "cv_folds", "value": cv_folds},
            {"name": "use_global_activation", "value": manual_params.get('use_global_activation', True)}
        ]
        
        # Create model entry for pipeline using DataManager
        NN_model_pipeline_entry = DataManager.create_NN_Model(
            model_name="Neural Network Classifier",
            model_param=param_list,
            trans=st.session_state.selected_trans if hasattr(st.session_state, 'selected_trans') else [],
            model=best_model,
            metrics_snapshot=metrics_snapshot,
            scaler=scaler
        )
        
        # Update pipeline
        if 'ML' not in st.session_state.pipeline:
            st.session_state.pipeline['ML'] = []
        
        if edit:
            # Replace existing Neural Network entry
            st.session_state.pipeline['ML'] = [
                item if item.get('model name') != 'Neural Network Classifier' else NN_model_pipeline_entry
                for item in st.session_state.pipeline['ML']
            ]
        else:
            # Append new model
            st.session_state.pipeline['ML'].append(NN_model_pipeline_entry)
        
        # Display training progress summary
        st.success("Neural Network Model created successfully and results saved to pipeline!")
        
        # Show training curves
        st.subheader("📈 Training Progress")
        col1, col2 = st.columns(2)
        
        with col1:
            if training_history.get('accuracy'):
                st.line_chart(pd.DataFrame({
                    'Training Accuracy': training_history.get('accuracy', []),
                    'Validation Accuracy': training_history.get('val_accuracy', [])
                }))
                st.caption("Accuracy over epochs")
        
        with col2:
            if training_history.get('loss'):
                st.line_chart(pd.DataFrame({
                    'Training Loss': training_history.get('loss', []),
                    'Validation Loss': training_history.get('val_loss', [])
                }))
                st.caption("Loss over epochs")
        
        return model_results
        
    except Exception as e:
        st.error(f"Error training Neural Network model: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

def validate_model(params):
    if len(params['features']) == 0:
        st.error("Please select at least one feature column")
        return False
    elif params['target'] in params['features']:
        st.error("Target column cannot be one of the features")
        return False
    
    # Check if target has at least 2 classes
    target_values = params['df'][params['target']].nunique()
    if target_values < 2:
        st.error("Target column must have at least 2 unique classes for classification")
        return False
    
    # Check if dataset is large enough for neural network
    if len(params['df']) < 50:
        st.warning("Dataset is small (<50 samples). Neural networks typically perform better with larger datasets.")
    
    return True