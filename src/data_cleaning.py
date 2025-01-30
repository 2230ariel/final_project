"""This module provides functions for cleaning and preprocessing data.

Functions included:
- remove_missing_values: Removes rows with missing values.
- normalize_columns: Normalizes specified numeric columns.
- remove_outliers: Identifies and removes outliers based on z-score.
- encode_categorical_columns: Encodes categorical columns into numeric values.
"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def remove_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with missing values from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing rows and columns of data.
            - Rows with missing (NaN) values will be removed.

    Returns:
        pd.DataFrame: A new DataFrame with all rows containing missing values removed.
    """
    return dataframe.dropna()


def normalize_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Normalizes the specified numeric columns using Min-Max Scaling.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing numeric and non-numeric columns.
            - Only the specified columns will be normalized.
        columns (List[str]): A list of column names to be normalized.
            - The columns should contain numeric data.

    Returns:
        pd.DataFrame: A new DataFrame where the specified columns are normalized to a range of 0 to 1.
    """
    scaler = MinMaxScaler()
    dataframe[columns] = scaler.fit_transform(dataframe[columns])
    return dataframe


def remove_outliers(dataframe: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.DataFrame:
    """Identifies and removes outliers from the specified column based on the z-score method.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing numeric and non-numeric columns.
        column (str): The name of the column from which outliers are to be removed.
        z_threshold (float, optional): The z-score threshold to identify outliers. Default is 3.0.

    Returns:
        pd.DataFrame: A new DataFrame with rows containing outliers removed.
    """
    # Calculate mean and standard deviation
    mean_value = dataframe[column].mean()
    std_dev = dataframe[column].std()

    # Calculate z-score
    z_scores = (dataframe[column] - mean_value) / std_dev

    # Identify rows to keep
    rows_to_keep = abs(z_scores) <= z_threshold

    # Return original DataFrame if no rows are removed
    if rows_to_keep.all():
        return dataframe.copy()

    # Otherwise, return filtered DataFrame
    return dataframe[rows_to_keep].copy()




def encode_categorical_columns(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Encodes categorical columns into numeric values.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing categorical and non-categorical columns.
            - Only the specified columns will be encoded.
        columns (List[str]): A list of column names to encode into numeric values.
            - The columns should be of type 'category' or 'object'.

    Returns:
        pd.DataFrame: A new DataFrame where the specified categorical columns are encoded into numeric values.
            - Each unique category is mapped to a unique integer starting from 0.
    """
    for column in columns:
        if dataframe[column].dtype.name == "category" or dataframe[column].dtype == "object":
            dataframe[column] = dataframe[column].astype("category").cat.codes
    return dataframe


def save_cleaned_data(dataframe: pd.DataFrame, file_path: str = "cleaned_data.csv") -> None:
    """Saves the cleaned DataFrame to a CSV file.

    Args:
        dataframe (pd.DataFrame): The cleaned DataFrame to save.
        file_path (str, optional): The path where the file will be saved. Default is 'cleaned_data.csv'.

    Returns:
        None
    """
    dataframe.to_csv(file_path, index=False)
