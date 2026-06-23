import streamlit as st

def apply_target_encoding(df,step):
    df_encoded = df.copy()
    for col in step['columns']:
        mean_map = df_encoded.groupby(col)[step['target_column']].mean()
        df_encoded[col] = df_encoded[col].map(mean_map)
    return df_encoded

def build_target_encoding_transf(df, edit_values=None) -> dict:
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    default_cols = edit_values.get("columns", []) if edit_values else []

    selected_columns = st.multiselect(
        "Select columns for Target Encoding",
        options=categorical_cols,
        default=default_cols
    )
    target_column = st.selectbox(
        "Select target column",
        options=df.columns.tolist(),
        index=categorical_cols.index(edit_values.get("target_column")) if edit_values and edit_values.get("target_column") in categorical_cols else 0
    )

    return {
        "columns": selected_columns,
        "target_column": target_column
    }
