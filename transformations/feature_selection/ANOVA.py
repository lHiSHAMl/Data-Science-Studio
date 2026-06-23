import streamlit as st
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def apply_ANOVA(df, step):
        """Apply ANOVA dimensionality reduction"""
        target_col = step["target_column"]
        k = step["k"]
        
        X = df.select_dtypes(include=['number'])
        y = df[target_col]
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        
        selected_columns = X.columns[selector.get_support(indices=True)]
        df_reduced = df[selected_columns.tolist() + [target_col]]
        
        return df_reduced

def build_anova_transf(df, edit_values):
    st.subheader("ANOVA Dimensionality Reduction Parameters")
    columns = df.select_dtypes(include=['number']).columns.tolist()
    target_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    default_target_column = edit_values.get("target_column", target_columns[0] if target_columns else "")
    target_column = st.selectbox("Select Target Column", target_columns, index=target_columns.index(default_target_column) if default_target_column in target_columns else 0)
    
    max_k = len(columns)
    default_k = edit_values.get("k", min(5, max_k))
    k = st.number_input("Number of Features to Select (k)", min_value=1, max_value=max_k, value=default_k)
    
    return {
        "category": "ANOVA",
        "target_column": target_column,
        "k": k
    }
