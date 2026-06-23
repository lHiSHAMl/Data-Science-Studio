import streamlit as st
import pandas as pd




# -------------------------------------------------------------
# OPTION 2 — VARIANCE THRESHOLD LOGIC
# -------------------------------------------------------------
def reduce_by_variance(df, threshold):
    numeric_df = df.select_dtypes(include='number')

    variances = numeric_df.var()

    low_var_cols = variances[variances < threshold].index.tolist()

    df = df.drop(columns=low_var_cols)

    return df


def build_variance_reduction_transf(df, edit_values=None) -> dict:
    default_threshold = float(edit_values.get("threshold", 0.01)) if edit_values else 0.01

    threshold = st.number_input(
        "Variance Threshold",
        min_value=0.0,
        value=default_threshold,
        step=0.001
    )

    return {
        "category": "variance_threshold",
        "threshold": threshold
    }
