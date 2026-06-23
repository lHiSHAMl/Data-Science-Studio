model_description = """
<div class="code-container">
    <div class="code-header">
        <span>MODEL DESCRIPTION</span>
        <span>Decision Tree Regressor</span>
    </div>
    <div>
        Decision Tree Regressor is a non-parametric supervised learning method used for 
        regression tasks. It creates a model that predicts the value of a target variable 
        by learning simple decision rules inferred from the data features.
        
        <strong>Key Concepts:</strong>
        • <strong>Tree Structure:</strong> Hierarchical structure of nodes and leaves
        • <strong>Splitting:</strong> Recursive partitioning of feature space
        • <strong>Impurity Measures:</strong> Criteria like MSE for determining splits
        
        <strong>Hyperparameters:</strong>
        • <strong>max_depth:</strong> Maximum depth of the tree
        • <strong>min_samples_split:</strong> Minimum samples required to split a node
        • <strong>criterion:</strong> Function to measure split quality (MSE, MAE, etc.)
        
        <strong>Key Evaluation Metrics:</strong>
        • <strong>Mean Squared Error (MSE):</strong> Average squared difference between predicted and actual values
        • <strong>Root Mean Squared Error (RMSE):</strong> Square root of MSE
        • <strong>Mean Absolute Error (MAE):</strong> Average absolute difference
        • <strong>R² Score:</strong> Proportion of variance explained by the model
        
        <strong>Advantages:</strong>
        • Easy to interpret and visualize
        • Handles non-linear relationships
        • Requires little data preprocessing
        • Feature importance scores
        
        <strong>Limitations:</strong>
        • Can overfit easily without proper regularization
        • Unstable (small changes in data can result in different trees)
        • Biased towards features with more levels
    </div>
</div>
"""