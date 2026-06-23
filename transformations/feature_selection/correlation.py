import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np

def reduce_by_correlation(df, threshold):

    numeric_df = df.select_dtypes(include="number")

    corr_matrix = numeric_df.corr().abs()

    upper_tri = corr_matrix.where(
        np.triu(
            np.ones(corr_matrix.shape),
            k=1
        ).astype(bool)
    )
    # st.write("### Highly Correlated Features")

    results = []

    for col in upper_tri.columns:

        high_corr = upper_tri[col][upper_tri[col] > threshold]

        for feature, corr_value in high_corr.items():

            results.append({
                "Feature 1": feature,
                "Feature 2": col,
                "Correlation": round(corr_value, 4)
            })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values(
            "Correlation",
            ascending=False
        ).reset_index(drop=True)

    return result_df
def build_corr_reduction_transf(df, edit_values=None) -> dict:
    default_threshold = float(edit_values.get("threshold", 0.9)) if edit_values else 0.9

    threshold = st.slider(
        "Correlation Threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        value=default_threshold
    )

    return {
        "category": "correlation_threshold",
        "threshold": threshold
    }

