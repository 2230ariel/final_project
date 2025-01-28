"""This module generates visualizations for the analysis results related to Parkinson's disease research.

Functions included:
- Plotting histograms and boxplots for stability and loudness measurements.
- Generating a correlation heatmap for stability and loudness columns.
- Creating comparative visualizations to analyze differences between healthy individuals and Parkinson's patients.
- Creating logistic regression visualization for predicted probabilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_matrix(dataframe: pd.DataFrame) -> plt.Figure:
    """Creates a correlation matrix heatmap for specified columns.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        plt.Figure: A matplotlib figure object containing the heatmap.
    """
    corr_matrix = dataframe[["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]].corr()

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix for Stability and Loudness")

    return fig


def plot_group_comparison(data: dict) -> plt.Figure:
    """Creates a bar chart comparing the means of two groups.

    Args:
        data (dict): A dictionary containing column name, and mean values for two groups.

    Returns:
        plt.Figure: A matplotlib figure object containing the bar chart.
    """
    categories = ["Healthy", "Parkinson's"]
    means = [data["healthy_mean"], data["parkinson_mean"]]

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, means, color=["blue", "orange"])
    ax.set_title(f"Comparison of {data['column']} between Groups")
    ax.set_ylabel(data["column"])
    ax.set_xlabel("Group")

    # Adding values above the bars
    ax.text(0, means[0], f"{means[0]:.2f}", ha="center", va="bottom")
    ax.text(1, means[1], f"{means[1]:.2f}", ha="center", va="bottom")

    return fig


def create_logistic_regression_plot(
    X_test: pd.DataFrame, y_test: pd.Series, y_pred_probs: np.ndarray, feature_name: str
) -> plt.Figure:
    """Creates a logistic regression plot showing the predicted probabilities.

    Args:
        X_test (pd.DataFrame): The test set features (only one feature supported for plotting).
        y_test (pd.Series): The true labels for the test set.
        y_pred_probs (np.ndarray): The predicted probabilities for the test set.
        feature_name (str): The name of the feature used for the plot.

    Returns:
        plt.Figure: A matplotlib figure object containing the logistic regression plot.
    """
    if X_test.shape[1] != 1:
        msg = "Only one feature is supported for logistic regression plotting."
        raise ValueError(msg)

    X = X_test.iloc[:, 0].values  # Extract the first (and only) feature
    y = y_test.values

    # Sort data by feature values for smooth plotting
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    y_pred_sorted = y_pred_probs[sorted_indices]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color="blue", label="True values")
    ax.plot(X_sorted, y_pred_sorted, color="red", label="Regression line")
    ax.set_title(f"Logistic Regression: {feature_name}")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Predicted Probability")
    ax.legend()

    return fig

