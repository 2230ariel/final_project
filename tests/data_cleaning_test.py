"""Unit tests for the data cleaning functions in the 'data_cleaning' module.

Run these tests with pytest:
    pytest data_cleaning_test.py
"""

import os
import sys

import pandas as pd
import pytest

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120

from src.data_cleaning import (
    encode_categorical_columns,
    normalize_columns,
    remove_missing_values,
    remove_outliers,
)


def test_remove_missing_values() -> None:
    """Tests the 'remove_missing_values' function by checking that rows with missing values are removed from the DataFrame.

    Args:
        None

    Returns:
        None: Asserts that rows with missing values are properly removed.
    """
    data = {"A": [1, 2, None], "B": [4, None, 6]}
    df = pd.DataFrame(data)
    cleaned_df = remove_missing_values(df)

    assert cleaned_df.isnull().sum().sum() == 0, "There are still missing values in the DataFrame."
    assert len(cleaned_df) == 1, "Incorrect number of rows after removing missing values."


def test_remove_missing_values_no_missing() -> None:
    """Tests 'remove_missing_values' when the DataFrame has no missing values.

    Args:
        None

    Returns:
        None: Asserts that the DataFrame remains unchanged.
    """
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    df = pd.DataFrame(data)
    cleaned_df = remove_missing_values(df)

    assert cleaned_df.equals(df), "DataFrame was modified even though there were no missing values."


def test_normalize_columns() -> None:
    """Tests the 'normalize_columns' function by checking that numeric columns are normalized to the range [0, 1].

    Args:
        None

    Returns:
        None: Asserts that all specified columns are normalized correctly.
    """
    data = {"A": [10, 20, 30], "B": [5, 15, 25]}
    df = pd.DataFrame(data)
    normalized_df = normalize_columns(df, ["A", "B"])

    assert normalized_df["A"].min() == 0, "Normalization failed for column 'A'."
    assert normalized_df["A"].max() == 1, "Normalization failed for column 'A'."
    assert normalized_df["B"].min() == 0, "Normalization failed for column 'B'."
    assert normalized_df["B"].max() == 1, "Normalization failed for column 'B'."


def test_remove_outliers() -> None:
    """Tests the 'remove_outliers' function by ensuring that rows with z-scores exceeding the threshold are removed.

    Args:
        None

    Returns:
        None: Asserts that rows with outliers are removed correctly.
    """
    data = {"A": [10, 20, 30, 1000], "B": [1, 2, 3, 4]}  # Outlier is 1000
    df = pd.DataFrame(data)
    cleaned_df = remove_outliers(df, column="A", z_threshold=1.3)

    assert len(cleaned_df) == 3, "Incorrect number of rows after removing outliers."
    assert 1000 not in cleaned_df["A"].values, "Outlier was not removed."


def test_remove_outliers_no_outliers() -> None:
    """Tests 'remove_outliers' when there are no outliers in the DataFrame.

    Args:
        None

    Returns:
        None: Asserts that the DataFrame remains unchanged in structure and values.
    """
    data = {"A": [10, 20, 30], "B": [1, 2, 3]}
    df = pd.DataFrame(data)

    # Apply function
    cleaned_df = remove_outliers(df, column="A")

    # Assertions
    pd.testing.assert_frame_equal(cleaned_df, df, check_dtype=True, check_like=True)


def test_encode_categorical_columns() -> None:
    """Tests the 'encode_categorical_columns' function by checking that categorical columns are encoded into numeric values.

    Args:
        None

    Returns:
        None: Asserts that all categorical columns are encoded correctly.
    """
    data = {"A": ["cat", "dog", "fish"], "B": ["red", "blue", "green"]}
    df = pd.DataFrame(data)
    encoded_df = encode_categorical_columns(df, ["A", "B"])

    assert encoded_df["A"].tolist() == [0, 1, 2], "Encoding failed for column 'A'."
    assert encoded_df["B"].tolist() == [2, 0, 1], "Encoding failed for column 'B'."


def test_encode_categorical_columns_no_categorical() -> None:
    """Tests 'encode_categorical_columns' when there are no categorical columns in the DataFrame.

    Args:
        None

    Returns:
        None: Asserts that the DataFrame remains unchanged.
    """
    data = {"A": [1, 2, 3], "B": [4, 5, 6]}
    df = pd.DataFrame(data)
    encoded_df = encode_categorical_columns(df, ["A", "B"])

    assert encoded_df.equals(df), "DataFrame was modified even though there were no categorical columns."


if __name__ == "__main__":
    """
    Main entry point for running the tests.

    Args:
        None

    Returns:
        None: Executes all tests using pytest and prints the validation results.
    """
    pytest.main()
