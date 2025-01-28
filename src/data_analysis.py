"""This module performs data analysis for Parkinson's disease research.

It includes functions for:
- Descriptive statistics on stability and loudness measurements.
- Checking normality of data using the Shapiro-Wilk test.
- Comparing means between healthy individuals and Parkinson's patients.
- Generating a correlation matrix for stability and loudness metrics.
- Performing logistic regression for predicting Parkinson's status.

The results of these analyses are saved to an output file for further visualization and reporting.
"""

import pandas as pd
from scipy.stats import shapiro, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Constants
ALPHA = 0.05  # Significance level for statistical tests
STABILITY_COLUMNS = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]


def descriptive_statistics(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Returns descriptive statistics for the specified columns in the dataset.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        columns (list[str]): List of column names to calculate descriptive statistics for.

    Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics.
    """
    return dataframe[columns].describe()


def check_normality(dataframe: pd.DataFrame, column: str, alpha: float) -> str:
    """Checks normality for a given column using the Shapiro-Wilk test.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column (str): The column to check for normality.
        alpha (float): The significance level for the test.

    Returns:
        str: A formatted string with the results of the normality test.
    """
    stat, p_value = shapiro(dataframe[column])
    result = f"Shapiro-Wilk Test for {column}: Stat={stat}, p-value={p_value}\n"
    if p_value > alpha:
        result += f"{column} follows a normal distribution.\n"
    else:
        result += f"{column} does not follow a normal distribution.\n"
    return result


def compare_groups(dataframe: pd.DataFrame, column: str, alpha: float) -> str:
    """Compares the means between healthy individuals and Parkinson's patients using a T-test.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        column (str): The column to compare between groups.
        alpha (float): The significance level for the test.

    Returns:
        str: A formatted string with the results of the T-test.
    """
    healthy = dataframe[dataframe["status"] == 0][column]
    parkinson = dataframe[dataframe["status"] == 1][column]

    stat, p_value = ttest_ind(healthy, parkinson)
    result = f"T-Test for {column}: Stat={stat}, p-value={p_value}\n"
    if p_value < alpha:
        result += f"There is a significant difference in {column} between healthy individuals and Parkinson's patients.\n"
    else:
        result += f"No significant difference in {column} between healthy individuals and Parkinson's patients.\n"
    return result


def correlation_matrix(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Calculates and returns a correlation matrix for specified columns.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        columns (list[str]): List of column names to calculate correlations for.

    Returns:
        pd.DataFrame: A DataFrame containing the correlation matrix.
    """
    return dataframe[columns].corr()


def logistic_regression_analysis(dataframe: pd.DataFrame, target: str, features: list[str]) -> dict:
    """Performs logistic regression to predict a binary target variable.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        target (str): The name of the target column (dependent variable).
        features (list[str]): The list of feature columns (independent variables).

    Returns:
        dict: A dictionary containing the trained model and prediction-related data.
    """
    X = dataframe[features]
    y = dataframe[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred_probs = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_probs": y_pred_probs,
    }


def perform_analysis(dataframe: pd.DataFrame, columns: list[str], alpha: float) -> str:
    """Runs all statistical analysis functions on the provided DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        columns (list[str]): List of columns to analyze.
        alpha (float): The significance level for statistical tests.

    Returns:
        str: A formatted string with all analysis results.
    """
    results = ""

    # Descriptive statistics
    results += "Descriptive Statistics for Stability and Loudness:\n"
    results += str(descriptive_statistics(dataframe, columns)) + "\n"

    # Normality tests
    for column in columns:
        results += f"\nNormality check for {column}:\n"
        results += check_normality(dataframe, column, alpha)

    # Group comparisons
    for column in columns:
        results += f"\nComparing groups for {column}:\n"
        results += compare_groups(dataframe, column, alpha)

    # Correlation matrix
    corr_matrix = correlation_matrix(dataframe, columns)
    results += "\nCorrelation Matrix:\n"
    results += str(corr_matrix) + "\n"

    return results


if __name__ == "__main__":
    # Example usage
    file_path = r"C:\Users\4arie\OneDrive\מסמכים\Project python\project_python_01\cleaned_data.csv"
    df = pd.read_csv(file_path)

    # Define columns
    analysis_results = perform_analysis(df, STABILITY_COLUMNS, ALPHA)

    # Save results to a file
    with open("analysis_results.txt", "w") as file:
        file.write(analysis_results)
