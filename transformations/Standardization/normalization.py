import streamlit as st
def apply_normalization(df, step):
        """mean normalization standarization"""
        col = step["column"]
        df[col] = (df[col] - df[col].mean()) / (df[col].max() - df[col].min())
        return df



def build_normalization_transf(df, edit_values):
    st.subheader("Normalization Standardization Parameters")
    columns = df.select_dtypes(include=['number']).columns.tolist()
    default_column = edit_values.get("column", columns[0] if columns else "")
    column = st.selectbox("Select Column to Normalize", columns, index=columns.index(default_column) if default_column in columns else 0)
    
    return {
        "category": "Mean Normalization",
         "column": column}

