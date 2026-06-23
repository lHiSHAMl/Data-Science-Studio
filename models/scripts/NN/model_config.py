import streamlit as st
import pandas as pd

def model_config(model_data, edit):        
    numeric_cols = [col for col in st.session_state.selected_data.columns if st.session_state.selected_data[col].dtype in ['int64', 'float64']] 
    all_cols = st.session_state.selected_data.columns.tolist()

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
            options=all_cols,
            default=default_features
        )
    
    with col2:
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for target selection")
            target = None
        else:
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
                options=all_cols,
                index=default_index,
                key="target_column_nn"
            )    
    
    # Neural Network Hyperparameter Configuration
    st.subheader("🧠 Neural Network Architecture")
    
    col3, col4 = st.columns(2)
    
    with col3:
        num_layers = st.number_input(
            "Number of Hidden Layers:",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of hidden layers in the neural network"
        )
    
    with col4:
        # Global activation function option
        use_global_activation = st.checkbox(
            "Use same activation for all layers",
            value=True,
            help="If checked, all hidden layers use the same activation function"
        )
    
    # Layer configuration with individual activation functions
    st.subheader("📊 Layer Configuration")
    
    layer_neurons = []
    layer_dropouts = []
    layer_activations = []
    
    # Available activation functions
    activation_options = ['relu', 'tanh', 'sigmoid', 'selu', 'elu', 'leaky_relu', 'swish']
    
    if use_global_activation:
        # Global activation selection
        global_activation = st.selectbox(
            "Activation Function (All Layers):",
            options=activation_options,
            index=0,
            help="Activation function for all hidden layers"
        )
        
        # Configure each layer
        for i in range(num_layers):
            st.markdown(f"**Layer {i+1}**")
            col_a, col_b = st.columns(2)
            with col_a:
                neurons = st.number_input(
                    f"Neurons:",
                    min_value=4,
                    max_value=512,
                    value=64 if i == 0 else 32,
                    step=8,
                    key=f"nn_neurons_{i}"
                )
            with col_b:
                dropout = st.slider(
                    f"Dropout Rate:",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    key=f"nn_dropout_{i}"
                )
            layer_neurons.append(neurons)
            layer_dropouts.append(dropout)
            layer_activations.append(global_activation)  # ← This is set correctly
            st.markdown("---")
    else:
        # Individual activation for each layer
        for i in range(num_layers):
            st.markdown(f"**Layer {i+1} Configuration**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                neurons = st.number_input(
                    f"Neurons:",
                    min_value=4,
                    max_value=512,
                    value=64 if i == 0 else 32,
                    step=8,
                    key=f"nn_neurons_{i}"
                )
            with col_b:
                dropout = st.slider(
                    f"Dropout Rate:",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    key=f"nn_dropout_{i}"
                )
            with col_c:
                activation = st.selectbox(
                    f"Activation:",
                    options=activation_options,
                    index=0,
                    key=f"nn_activation_{i}"
                )
            layer_neurons.append(neurons)
            layer_dropouts.append(dropout)
            layer_activations.append(activation)  # ← This is set correctly
            st.markdown("---")
    
    # Training parameters
    st.subheader("⚙️ Training Parameters")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        epochs = st.number_input(
            "Number of Epochs:",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of training epochs"
        )
    
    with col6:
        batch_size = st.selectbox(
            "Batch Size:",
            options=[16, 32, 64, 128, 256],
            index=1,
            help="Batch size for training"
        )
    
    with col7:
        learning_rate = st.selectbox(
            "Learning Rate:",
            options=[0.001, 0.0001, 0.01, 0.0005],
            index=0,
            format_func=lambda x: f"{x:.4f}",
            help="Learning rate for optimizer"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col8, col9 = st.columns(2)
        
        with col8:
            optimizer = st.selectbox(
                "Optimizer:",
                options=['adam', 'rmsprop', 'sgd', 'adagrad', 'nadam'],
                index=0,
                help="Optimizer algorithm"
            )
            
            output_activation = st.selectbox(
                "Output Layer Activation:",
                options=['softmax', 'sigmoid'],
                index=0,
                help="softmax for multi-class, sigmoid for binary classification"
            )
        
        with col9:
            validation_split = st.slider(
                "Validation Split:",
                min_value=0.0,
                max_value=0.3,
                value=0.2,
                step=0.05,
                help="Portion of training data used for validation"
            )
            
            early_stopping = st.checkbox(
                "Early Stopping",
                value=True,
                help="Stop training when validation loss stops improving"
            )
    
    # Grid search option
    use_grid_search = st.checkbox(
        "Use Grid Search for Hyperparameter Tuning",
        value=False,
        help="Perform grid search to find optimal hyperparameters"
    )
    
    param_grid = {}
    cv_folds = 5
    
    if use_grid_search:
        st.subheader("🎯 Grid Search Configuration")
        
        col10, col11 = st.columns(2)
        
        with col10:
            epochs_grid = st.multiselect(
                "Epochs to try:",
                options=[50, 100, 150, 200],
                default=[100],
                help="Select epoch values for grid search"
            )
        
        with col11:
            batch_size_grid = st.multiselect(
                "Batch Sizes to try:",
                options=[16, 32, 64],
                default=[32],
                help="Select batch sizes for grid search"
            )
        
        col12, col13 = st.columns(2)
        
        with col12:
            learning_rate_grid = st.multiselect(
                "Learning Rates to try:",
                options=[0.001, 0.0001, 0.01],
                default=[0.001],
                format_func=lambda x: f"{x:.4f}",
                help="Select learning rates for grid search"
            )
        
        with col13:
            cv_folds = st.number_input(
                "Cross-validation folds:",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of cross-validation folds for grid search"
            )
        
        param_grid = {
            'epochs': epochs_grid if epochs_grid else [100],
            'batch_size': batch_size_grid if batch_size_grid else [32],
            'learning_rate': learning_rate_grid if learning_rate_grid else [0.001]
        }
    
    # Prepare manual parameters - ALWAYS include layer_activations
    manual_params = {
        'num_layers': num_layers,
        'layer_neurons': layer_neurons,
        'layer_dropouts': layer_dropouts,
        'layer_activations': layer_activations,  # ← This is always set now
        'output_activation': output_activation,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'optimizer': optimizer,
        'validation_split': validation_split,
        'early_stopping': early_stopping,
        'use_global_activation': use_global_activation
    }
    
    return {
        "features": features, 
        "target": target, 
        "df": st.session_state.selected_data, 
        "edit": edit,
        "use_grid_search": use_grid_search,
        "param_grid": param_grid,
        "manual_params": manual_params,
        "cv_folds": cv_folds
    }


model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Neural Network Classifier</span>
    </div>
    <div>
        Neural Networks are computing systems inspired by biological neural networks. They consist of interconnected 
        layers of neurons that learn to perform tasks by considering examples, generally without task-specific programming.
        
        <strong>Key Advantages:</strong>
        • Can model complex non-linear relationships
        • Handles high-dimensional data effectively
        • Automatic feature engineering through hidden layers
        • Scalable to large datasets
        
        <strong>Activation Functions:</strong>
        • <strong>ReLU:</strong> Fast, avoids vanishing gradient
        • <strong>Tanh:</strong> Zero-centered, good for hidden layers
        • <strong>Sigmoid:</strong> Good for output layer
        • <strong>Leaky ReLU:</strong> Allows small negative values
        • <strong>Swish:</strong> Self-gated activation
    </div>
</div>
"""

model_reference_code = """
<div class="code-container">
    <div class="code-header">
        <span>REFERENCE CODE</span>
        <span>Neural Network Classifier</span>
    </div>
    <div>
        # Neural Network Implementation
        from tensorflow.keras import layers
        
        # Build model with per-layer activations
        model = keras.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        for neurons, activation, dropout_rate in zip(layer_neurons, layer_activations, layer_dropouts):
            model.add(layers.Dense(neurons, activation=activation))
            if dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(num_classes, activation='softmax'))
    </div>
</div>
"""