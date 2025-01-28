"""Unit tests for the data analysis functions in the 'data_analysis' module.

Run these tests with pytest:
    pytest test_data_analysis.py
"""

import os
import sys

import pandas as pd
import pytest

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120

from src.data_analysis import (
    check_normality,
    compare_groups,
    correlation_matrix,
    descriptive_statistics,
    logistic_regression_analysis,
    perform_analysis,
)


def test_descriptive_statistics() -> None:
    """Tests the 'descriptive_statistics' function by verifying that descriptive statistics are calculated correctly for numeric columns.

    Args:
        None

    Returns:
        None: Asserts that the output contains valid statistical metrics.
    """
    data = {
        "MDVP:Fo(Hz)": [119.992, 122.400, 116.682],
        "MDVP:Fhi(Hz)": [157.302, 148.650, 131.111],
        "MDVP:Flo(Hz)": [74.997, 113.819, 111.555],
    }
    df = pd.DataFrame(data)
    stats = descriptive_statistics(df, ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"])
    assert "mean" in stats.index
    assert stats.loc["mean", "MDVP:Fo(Hz)"] > 0


def test_check_normality() -> None:
    """Tests the 'check_normality' function by verifying the output format of the Shapiro-Wilk test.

    Args:
        None

    Returns:
        None: Asserts that the function correctly identifies normality.
    """
    data = {"MDVP:Fo(Hz)": [119.992, 122.400, 116.682, 116.676]}
    df = pd.DataFrame(data)
    result = check_normality(df, "MDVP:Fo(Hz)", alpha=0.05)
    assert "Shapiro-Wilk Test" in result
    assert "p-value" in result


def test_compare_groups() -> None:
    """Tests the 'compare_groups' function by verifying the output format of the T-Test.

    Args:
        None

    Returns:
        None: Asserts that the function returns valid test results.
    """
    data = {
        "MDVP:Fo(Hz)": [119.992, 122.400, 116.682, 116.676],
        "status": [0, 1, 0, 1],  # 0 = healthy, 1 = patient
    }
    df = pd.DataFrame(data)
    result = compare_groups(df, "MDVP:Fo(Hz)", alpha=0.05)
    assert "T-Test for" in result
    assert "p-value" in result


def test_correlation_matrix() -> None:
    """Tests the 'correlation_matrix' function by verifying that a correlation matrix is generated for numeric columns.

    Args:
        None

    Returns:
        None: Asserts that the matrix contains correlation values.
    """
    data = {
        "MDVP:Fo(Hz)": [119.992, 122.400, 116.682],
        "MDVP:Fhi(Hz)": [157.302, 148.650, 131.111],
        "MDVP:Flo(Hz)": [74.997, 113.819, 111.555],
    }
    df = pd.DataFrame(data)
    result = correlation_matrix(df, ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"])
    assert result.shape == (3, 3)
    assert -1 <= result.values.min() <= 1
    assert -1 <= result.values.max() <= 1


def test_logistic_regression_analysis() -> None:
    """Tests the 'logistic_regression_analysis' function by verifying the model's ability to fit and predict.

    Args:
        None

    Returns:
        None: Asserts that the model is trained and predictions are made.
    """
    data = {
        "MDVP:Fo(Hz)": [119.992, 122.400, 116.682, 116.676],
        "MDVP:Fhi(Hz)": [157.302, 148.650, 131.111, 149.303],
        "status": [0, 1, 0, 1],  # 0 = healthy, 1 = patient
    }
    df = pd.DataFrame(data)
    result = logistic_regression_analysis(df, "status", ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"])
    assert "model" in result
    assert "X_test" in result
    assert "y_test" in result
    assert "y_pred_probs" in result


def test_perform_analysis() -> None:
    """Tests the 'perform_analysis' function by verifying that all analysis steps execute without errors and return results.

    Args:
        None

    Returns:
        None: Asserts that the output contains results for all steps.
    """
    data = {
        "MDVP:Fo(Hz)": [119.992, 122.400, 116.682, 116.676],
        "MDVP:Fhi(Hz)": [157.302, 148.650, 131.111, 149.303],
        "status": [0, 1, 0, 1],  # 0 = healthy, 1 = patient
    }
    df = pd.DataFrame(data)
    result = perform_analysis(df, ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)"], alpha=0.05)
    assert "Descriptive Statistics" in result
    assert "Normality check" in result
    assert "Comparing groups" in result
    assert "Correlation Matrix" in result


if __name__ == "__main__":
    """
    Main entry point for running the tests.

    Args:
        None

    Returns:
        None: Executes all tests using pytest and prints the validation results.
    """
    pytest.main()
