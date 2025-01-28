"""Main entry point for Parkinson's data analysis.

This script loads the cleaned dataset from the root directory and runs analysis and visualizations.

Functions:
- load_cleaned_data: Loads the cleaned dataset from a CSV file.
- save_plot: Saves a matplotlib figure to the outputs directory.
- main: Runs analysis and visualization on the cleaned data.
"""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from src.data_analysis import logistic_regression_analysis
from src.data_visualization import create_logistic_regression_plot, plot_correlation_matrix, plot_group_comparison

# ---- CONFIGURATION ----
CLEANED_DATA_PATH = "cleaned_data.csv"  # קריאה מהתיקייה הראשית
OUTPUT_DIR = "outputs"
NUMERIC_COLUMNS = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]
TARGET_COLUMN = "status"
REGRESSION_FEATURE = "MDVP:Fo(Hz)"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)  # noqa: PTH103

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Loads the cleaned dataset from a CSV file.

    Args:
        file_path (str): The path to the cleaned data file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        logging.exception(f"File '{file_path}' not found. Please run the data cleaning script first.")  # noqa: G004
        exit(1)


def save_plot(fig: plt.Figure, filename: str) -> None:
    """Saves a matplotlib figure to the outputs directory.

    Args:
        fig (plt.Figure): The matplotlib figure object to save.
        filename (str): The name of the output file.

    Returns:
        None
    """
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(output_path)
    logging.info(f"Saved plot: {output_path}")


def main() -> None:
    """Runs analysis and visualization on the cleaned data.

    Steps:
    1. Load cleaned dataset.
    2. Perform logistic regression.
    3. Generate and save visualizations.

    Returns:
        None
    """
    logging.info("Loading cleaned data...")
    data = load_cleaned_data(CLEANED_DATA_PATH)

    logging.info("Performing logistic regression...")
    regression_results = logistic_regression_analysis(data, TARGET_COLUMN, [REGRESSION_FEATURE])

    logging.info("Generating visualizations...")

    # Save and show correlation matrix
    fig_corr = plot_correlation_matrix(data)
    save_plot(fig_corr, "correlation_matrix.png")
    plt.show()  # ✅ מציג את הגרף על המסך

    # Save and show group comparison plots
    for column in NUMERIC_COLUMNS:
        healthy_mean = data[data[TARGET_COLUMN] == 0][column].mean()
        parkinson_mean = data[data[TARGET_COLUMN] == 1][column].mean()
        fig_group = plot_group_comparison({
            "column": column,
            "healthy_mean": healthy_mean,
            "parkinson_mean": parkinson_mean
        })
        save_plot(fig_group, f"group_comparison_{column}.png")
        plt.show()

    # Save and show logistic regression plot
    fig_reg = create_logistic_regression_plot(
        X_test=regression_results["X_test"],
        y_test=regression_results["y_test"],
        y_pred_probs=regression_results["y_pred_probs"],
        feature_name=REGRESSION_FEATURE
    )
    save_plot(fig_reg, "logistic_regression.png")
    plt.show()

    logging.info("Analysis and visualization completed!")


if __name__ == "__main__":
    main()
