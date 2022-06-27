import pandas as pd
import streamlit as st


def get_dataframe():
    """If the input dataset is not none, returns the equivalent Pandas Dataframe."""
    with st.sidebar:
        st.markdown("## **1.First Step** ##")
        data = st.sidebar.file_uploader("Please upload your dataset (CSV format):", type=['csv','xlsx'])
        is_loaded_dataset = st.sidebar.warning("Dataset not uploaded")
        if data is not None:
            is_loaded_dataset.success("Dataset uploaded successfully!")
            data = pd.read_csv(data)
            data.to_csv("df")
    return data 
    # if dataset_file is not None \
    #     else None


def get_missing_values(dataframe):
    """Returns the missing values and the missing percentages for each column."""

    missing_values = dataframe.isnull().sum().sort_values(ascending=False)
    missing_percentage = (dataframe.isnull().sum() / dataframe.isnull().count()).sort_values(ascending=False)
    return missing_values, missing_percentage


@st.cache
def get_linear_correlation(df, label_name, positive):
    """Returns the correlation (positive or negative, based on the input) between the features and the label"""

    corr_matrix = df.corr()
    corr = get_signed_correlations(corr_matrix, label_name, positive=positive)
    corr_df = pd.DataFrame(corr).rename(columns={label_name: 'Correlation'})
    return corr_df


def get_signed_correlations(corr_matrix, label_name, positive=True):
    """Get positive or negative correlations, based on the value of the input."""

    correlation = corr_matrix[label_name][corr_matrix[label_name] >= 0] \
        if positive else corr_matrix[label_name][corr_matrix[label_name] < 0]

    return correlation.iloc[:-1].sort_values(ascending=not positive)


@st.cache
def get_columns_and_label(df):
    """Returns the columns and the label of the input dataframe."""

    column_names = list(df.columns.values)
    return column_names, column_names[len(column_names) - 1]


@st.cache
def get_categorical_columns(df):
    """Returns the list of categorical columns of the input dataframe."""

    return list(df.select_dtypes(exclude=['number']).columns.values)


@st.cache
def get_numeric_columns(df):
    """Returns the list of numerical columns of the input dataframe."""

    return list(df.select_dtypes(['number']).columns.values)


def is_categorical(column):
    return column.dtype.name == 'object'


def color_null_red(val):
    """Coloring in red the NaN values."""

    return 'color: red' if pd.isnull(val) else 'color: black'

