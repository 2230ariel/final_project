"""Main entry point for Parkinson's data analysis, including data cleaning."""

import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
from src.data_analysis import logistic_regression_analysis
from src.data_cleaning import normalize_columns, remove_missing_values, remove_outliers, save_cleaned_data
from src.data_visualization import create_logistic_regression_plot, plot_correlation_matrix, plot_group_comparison

# ---- CONFIGURATION ----
RAW_DATA_PATH = "parkinsons.data"  # Path to the raw dataset
CLEANED_DATA_PATH = "cleaned_data.csv"  # Path to save the cleaned dataset
OUTPUT_DIR = "outputs"
NUMERIC_COLUMNS = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)"]
TARGET_COLUMN = "status"
REGRESSION_FEATURE = "MDVP:Fo(Hz)"

# Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def clean_data() -> pd.DataFrame:
    """Loads, cleans, and saves the dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    logging.info("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)  # Load raw dataset

    logging.info("Cleaning data...")
    df = remove_missing_values(df)
    df = normalize_columns(df, NUMERIC_COLUMNS)

    for column in NUMERIC_COLUMNS:
        df = remove_outliers(df, column)

    logging.info("Saving cleaned data...")
    save_cleaned_data(df, CLEANED_DATA_PATH)  # Save cleaned dataset to a CSV file

    logging.info("Data cleaning completed. Cleaned dataset saved to cleaned_data.csv")
    return df  # Return the cleaned dataset for further analysis


def save_plot(fig: plt.Figure, filename: str) -> None:
    """Saves a matplotlib figure to the outputs directory.

    Args:
        fig (plt.Figure): The figure to save.
        filename (str): The filename for saving the plot.
    """
    if not filename.endswith(".png"):
        filename += ".png"
    filename = filename.replace(":", "_")  # Replace invalid characters
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(output_path, format="png", bbox_inches="tight")
    logging.info(f"Saved plot: {output_path}")


def main() -> None:
    """Runs data cleaning, analysis, and visualization."""
    logging.info("Starting analysis pipeline...")

    # Step 1: Clean the dataset and save it
    data = clean_data()  # Load, clean, and save the dataset

    # Step 2: Perform logistic regression analysis
    logging.info("Performing logistic regression...")
    regression_results = logistic_regression_analysis(data, TARGET_COLUMN, [REGRESSION_FEATURE])

    # Step 3: Generate visualizations
    logging.info("Generating visualizations...")

    # Save and display the correlation matrix
    fig_corr = plot_correlation_matrix(data)
    save_plot(fig_corr, "correlation_matrix")
    plt.show()

    # Save and display group comparison plots
    for column in NUMERIC_COLUMNS:
        healthy_mean = data[data[TARGET_COLUMN] == 0][column].mean()
        parkinson_mean = data[data[TARGET_COLUMN] == 1][column].mean()
        fig_group = plot_group_comparison({
            "column": column,
            "healthy_mean": healthy_mean,
            "parkinson_mean": parkinson_mean
        })
        save_plot(fig_group, f"group_comparison_{column.replace(':', '_')}")
        plt.show()

    # Save and display the logistic regression plot
    fig_reg = create_logistic_regression_plot(
        X_test=regression_results["X_test"],
        y_test=regression_results["y_test"],
        y_pred_probs=regression_results["y_pred_probs"],
        feature_name=REGRESSION_FEATURE
    )
    save_plot(fig_reg, "logistic_regression")
    plt.show()

    logging.info("Analysis and visualization completed!")


if __name__ == "__main__":
    main()
