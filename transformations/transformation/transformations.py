import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def transformations_config(choice,edit_values):
            # Display appropriate fields based on transformation type
            transformation_params = {}
            # transformations = ["delete", "computation", "filter", "group"]
            # choice =st.selectbox("Select Transformation", transformations, index=transformations.index(edit_values.get("type", "delete")))
            if st.session_state.df_original is not None:
                if choice == "delete":
                    transformation_params = build_delete_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "computation":
                    transformation_params = build_computation_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "filter":
                    transformation_params = build_filter_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "group":
                    transformation_params = build_group_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "split":
                    transformation_params= build_split_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values )
                elif choice == "replace":
                    transformation_params = build_replace_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )    
                elif choice == "distinct":
                    transformation_params = build_distinct_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "datatype":
                    transformation_params = build_datatype_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "value_counts":
                    transformation_params = build_value_counts_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
                elif choice == "trim_whitespace":
                    transformation_params = build_trim_whitespace_transf(
                        df=st.session_state.df_to_show if not st.session_state.df_to_show.empty else st.session_state.df_original,
                        edit_values=edit_values
                    )
            return transformation_params
def apply_selected_transformations(df,step):
    """Apply the selected transformations to the original dataframe"""
    
    if step["category"] == "delete":
        df.drop(columns=[step["column"]], inplace=True)
    elif step["category"] == "computation":
        df[step["new_column"]] = calculate_expression(df, step['expr'])
    elif step["category"] == "filter":
        df = df[df[step["column"]] == step["value"]]
    elif step["category"] == "group":
        df = group_and_aggregate(df,step["group_col"], step["target_col"], step["agg_choice"])
    elif step["category"] == "split_train":
        df= aplly_train(step["X_train"], step["y_train"])
    elif step["category"] == "split_test":
        df= aplly_test(step["X_test"], step["y_test"])
    elif step["category"] == "describe":
        df = df.describe()
    elif step["category"] == "summary":
        df_summ = dataframe_summary(df)
        st.write("Rows:", df_summ["total_rows"])
        st.write("Columns:", df_summ["total_columns"])
        st.write("Duplicate Rows:", df_summ["duplicate_rows"])
        df=df_summ['column_summary']
    elif step["category"] == "combined":
        df=step['df']
    elif step["category"] == "replace":
        df = apply_replace_transformation(df, step)
    elif step["category"] == "distinct":
        df = get_distinct_values(df, step["columns"])
    elif step["category"] == "datatype":
        df = convert_datatypes(
            df,
            step["columns"],
            step["datatype"]
        )
    elif step["category"] == "value_counts":
                    df = get_value_counts(
                        df,
                        step["column"]
                    )
    elif step["category"] == "trim_whitespace":
        df = normalize_text_columns(
            df,
            step["columns"]
        )
    return df

def build_trim_whitespace_transf(df, edit_values=None):

    default_cols = (
        edit_values.get("columns", [])
        if edit_values else []
    )

    selected_cols = st.multiselect(
        "Columns",
        df.columns.tolist(),
        default=default_cols
    )

    return {
        "columns": selected_cols
    }
def trim_whitespace(df, columns):

    df = df.copy()

    for col in columns:

        if col not in df.columns:
            continue

        # Only process text/object columns
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    return df
def normalize_text_columns(df, columns):

    df = df.copy()

    for col in columns:

        if df[col].dtype == "object":

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    return df
def build_distinct_transf(df, edit_values=None) -> dict:
    """Build distinct values transformation UI"""

    default_cols = edit_values.get("columns", []) if edit_values else []

    selected_cols = st.multiselect(
        "Select columns",
        options=df.columns.tolist(),
        default=default_cols
    )

    return {
        "columns": selected_cols
    }
def get_distinct_values(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Returns a dataframe containing distinct values
    for each selected column.
    """

    result = {}

    max_len = 0

    for col in columns:
        values = (
            df[col]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        result[col] = values
        max_len = max(max_len, len(values))

    # pad shorter columns
    for col in result:
        result[col] += [None] * (max_len - len(result[col]))

    return pd.DataFrame(result)
def calculate_expression(df, expr: str) -> pd.Series:
    """Evaluates a mathematical expression string with column names in brackets"""
    columns = re.findall(r'\[([^\[\]]+)\]', expr)
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        expr = expr.replace(f"[{col}]", f"df['{col}']")
    
    try:
        return eval(expr)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}")

def group_and_aggregate(df: pd.DataFrame, group_col: str, target_col: str, agg_choice: str) -> pd.DataFrame:

    # Perform groupby
    grouped_df = (
        df.groupby(group_col)[target_col]
        .agg(agg_choice)
        .reset_index()
        .rename(columns={target_col: f"{agg_choice}_{target_col}"})
    )

    st.write("### 📊 Aggregated Data")
    # st.dataframe(grouped_df)

    return grouped_df

def build_group_transf(df, edit_values=None) -> dict:
    """Build the group transformation UI"""
    default_group_col = edit_values.get("group_col", df.columns[0]) if edit_values else df.columns[0]
    group_col = st.selectbox(
        "Column to group by",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_group_col) if edit_values else 0
    )
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    default_target_col = edit_values.get("target_col", numeric_cols[0]) if edit_values else numeric_cols[0]
    target_col = st.selectbox(
        "Numeric column to aggregate",
        numeric_cols,
        index=numeric_cols.index(default_target_col) if edit_values else 0
    )
    
    agg_functions = ['sum', 'mean', 'max', 'min', 'count']
    default_agg = edit_values.get("agg_choice", agg_functions[0]) if edit_values else agg_functions[0]
    agg_choice = st.selectbox(
        "Aggregation function",
        agg_functions,
        index=agg_functions.index(default_agg) if edit_values else 0
    )
    
    return {"group_col": group_col, "target_col": target_col, "agg_choice": agg_choice}
    
def build_delete_transf(df, edit_values=None) -> dict:
    """Build the delete transformation UI"""
    default_col = edit_values.get("column", df.columns[0]) if edit_values else df.columns[0]
    col_to_delete = st.selectbox(
        "Column to delete",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_col) if edit_values else 0
    )
    return {"column": col_to_delete}
def build_computation_transf(df, edit_values=None) -> dict:
    """Build the computation transformation UI"""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    st.markdown(f"**Numeric Columns:** {' | '.join(numeric_cols)}")
    
    default_expr = edit_values.get("expr", "") if edit_values else ""
    expr_cols = st.text_input("Expression", value=default_expr)
    
    default_new_col = edit_values.get("new_column", "") if edit_values else ""
    new_col_name = st.text_input("New column name", value=default_new_col)
    
    return {"expr": expr_cols, "new_column": new_col_name}

def build_filter_transf(df, edit_values=None) -> dict:
    """Build the filter transformation UI"""
    default_col = edit_values.get("column", df.columns[0]) if edit_values else df.columns[0]
    col_to_filter = st.selectbox(
        "Column to filter",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_col) if edit_values else 0
    )
    default_value = edit_values.get("value", "") if edit_values else ""
    value_to_filter = st.text_input("Value to filter by", value=default_value)
    
    return {"column": col_to_filter, "value": value_to_filter}


def split_train_test(df: pd.DataFrame, target_col: str, test_size: float, random_state: int = 42):
    """
    Splits dataframe into X_train, X_test, y_train, y_test
    """
    
    if target_col not in df.columns:
        raise ValueError("Target column not found in dataframe")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
def build_split_transf(df, edit_values=None):

    default_target = edit_values.get("target_col", df.columns[-1]) if edit_values else df.columns[-1]

    target_col = st.selectbox(
        "Target Column",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_target)
    )

    default_test_size = edit_values.get("test_size", 0.2) if edit_values else 0.2

    test_size = st.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.5,
        value=default_test_size,
        step=0.05
    )
    X_train, X_test, y_train, y_test = split_train_test(df, target_col, test_size)


    return {
        "target_col": target_col,
        "test_size": test_size,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def aplly_train(x_train, y_train):
    # Placeholder for training logic
    return  pd.concat([x_train, y_train], axis=1)
def aplly_test(x_test, y_test):
    # Placeholder for training logic
    return  pd.concat([x_test, y_test], axis=1)
def dataframe_summary(df: pd.DataFrame) -> dict:
    """
    Returns full structured summary of dataframe
    """

    summary = {}

    # General info
    summary["total_rows"] = df.shape[0]
    summary["total_columns"] = df.shape[1]
    summary["duplicate_rows"] = df.duplicated().sum()

    # Column-level summary
    column_summary = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null_count": df.count(),
        "null_count": df.isna().sum(),
        "unique_values": df.nunique()
    })

    # Percentage missing
    column_summary["missing_%"] = (
        column_summary["null_count"] / len(df) * 100
    ).round(2)

    summary["column_summary"] = column_summary.reset_index().rename(columns={"index": "column"})

    return summary

def build_replace_transf(df, edit_values=None):

    default_col = edit_values.get("column", df.columns[0]) if edit_values else df.columns[0]

    col_to_replace = st.selectbox(
        "Column",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_col)
    )

    replace_target = st.radio(
        "Replace",
        ["Selected Values", "Null Values"],
        index=0 if edit_values.get("replace_target", "Selected Values") == "Selected Values" else 1
    )

    strategy_options = [
        "Static Value",
        "Mode",
        "Mean",
        "Median",
        "Min",
        "Max"
    ]

    strategy = st.selectbox(
        "Replacement Strategy",
        strategy_options,
        index=strategy_options.index(
            edit_values.get("strategy", "Static Value")
        ) if edit_values else 0
    )

    old_values = []

    if replace_target == "Selected Values":

        unique_vals = sorted(
            [str(v) for v in df[col_to_replace].dropna().unique()]
        )

        old_values = st.multiselect(
            "Values to replace",
            unique_vals,
            default=edit_values.get("old_values", [])
        )

    static_value = ""

    if strategy == "Static Value":
        static_value = st.text_input(
            "Replacement Value",
            value=edit_values.get("static_value", "")
        )

    return {
        "column": col_to_replace,
        "replace_target": replace_target,
        "strategy": strategy,
        "old_values": old_values,
        "static_value": static_value
    }

def build_datatype_transf(df, edit_values=None) -> dict:
    """Build datatype conversion transformation UI"""

    default_cols = edit_values.get("columns", []) if edit_values else []

    selected_cols = st.multiselect(
        "Select columns",
        options=df.columns.tolist(),
        default=default_cols
    )

    datatype_options = [
        "string",
        "integer",
        "float",
        "datetime",
        "boolean"
    ]

    default_dtype = (
        edit_values.get("datatype", "string")
        if edit_values else "string"
    )

    datatype = st.selectbox(
        "Convert To",
        datatype_options,
        index=datatype_options.index(default_dtype)
    )

    return {
        "columns": selected_cols,
        "datatype": datatype
    }

def convert_datatypes(
    df: pd.DataFrame,
    columns: list,
    datatype: str
) -> pd.DataFrame:

    df = df.copy()

    for col in columns:

        # try:
            if datatype == "string":
                df[col] = df[col].astype(str)

            elif datatype == "integer":
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).astype("int64")

            elif datatype == "float":
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            elif datatype == "datetime":
                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

            elif datatype == "boolean":

                # Handle common text representations
                mapping = {
                    "true": True,
                    "false": False,
                    "yes": True,
                    "no": False,
                    "1": True,
                    "0": False
                }

                if df[col].dtype == object:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.lower()
                        .map(mapping)
                    )
                else:
                    df[col] = df[col].astype(bool)

        # except Exception as e:
        #     st.warning(f"Failed converting {col}: {e}")

    return df
def build_value_counts_transf(df, edit_values=None) -> dict:

    default_col = (
        edit_values.get("column", df.columns[0])
        if edit_values else df.columns[0]
    )

    column = st.selectbox(
        "Column",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_col)
    )

    return {
        "column": column
    }
def get_value_counts(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:

    result = (
        df[column]
        .value_counts(dropna=False)
        .reset_index()
    )

    result.columns = [column, "Count"]

    result["Percentage"] = (
        result["Count"] / len(df) * 100
    ).round(2)

    return result

def apply_replace_transformation(df, step):

    df = df.copy()

    column = step["column"]
    strategy = step["strategy"]

    # Determine replacement value
    if strategy == "Static Value":
        replacement = step["static_value"]

    elif strategy == "Mode":
        replacement = df[column].mode().iloc[0]

    elif strategy == "Mean":
        replacement = df[column].mean()

    elif strategy == "Median":
        replacement = df[column].median()

    elif strategy == "Min":
        replacement = df[column].min()

    elif strategy == "Max":
        replacement = df[column].max()

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    # Apply replacement
    if step["replace_target"] == "Null Values":

        df[column] = df[column].fillna(replacement)

    else:

        df[column] = df[column].replace(
            step["old_values"],
            replacement
        )

    return df