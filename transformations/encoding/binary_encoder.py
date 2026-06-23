import streamlit as st
import pandas as pd
import category_encoders as ce
def apply_binary_encoding(df, step):
        """Apply binary Encoding"""
        col = step["column"]
        encoder = ce.BinaryEncoder(cols=[col])
        df_encoded = encoder.fit_transform(df[col])
        df = pd.concat([df.drop(columns=[col]), df_encoded], axis=1)
        return df


def build_binary_encoding_transf(df, edit_values):
    st.subheader("Binary Encoding Parameters")
    columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    default_column = edit_values.get("column", columns[0] if columns else "")
    column = st.selectbox("Select Column for Binary Encoding", columns, index=columns.index(default_column) if default_column in columns else 0)
    
    return {
         "column": column}