# Parkinson's Voice Analysis Project

This project analyzes **voice stability and loudness differences** between **Parkinson's patients and healthy individuals** using Python.  
The analysis includes **data cleaning, statistical analysis, and result visualization**.

## 📌 Project Summary

(🔗 Add a link to the project summary here)

---

## 📂 Project Structure

```plaintext
project_python_01/
├── src/              # Source code for data processing and analysis
│   ├── data_cleaning.py        # Data preprocessing and cleaning functions
│   ├── data_analysis.py        # Statistical analysis functions
│   ├── data_visualization.py   # Visualization functions
│   ├── __init__.py             # Package initializer
│   ├── analysis_results.py     # Script to generate analysis summaries
├── tests/            # Unit tests for validation
│   ├── data_cleaning_test.py   # Tests for data cleaning functions
│   ├── test_data_analysis.py   # Tests for analysis functions
│   ├── __init__.py             # Package initializer
├── outputs/          # Automatically generated results (cleaned data, plots, etc.)
│   ├── cleaned_data.csv       # The cleaned dataset
│   ├── correlation_matrix.png # Heatmap of variable correlations
│   ├── logistic_regression.png # Logistic regression visualization
│   ├── group_comparison_*.png # Comparison charts for different features
├── main.py           # Main script to run the entire analysis
├── parkinsons.data   # Original dataset file
├── pyproject.toml    # Project dependencies and settings
├── README.md         # Project documentation

# **Python version**
**- Python 3.10+

```python
### **Required libraries**

```To run this project, make sure you have the following libraries installed:
- pandas
- scipy
- seaborn
- matplotlib
- scikit-learn
- numpy

```bash
pip install pandas scipy seaborn matplotlib scikit-learn numpy
```

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone להוסיף קישור לגיט האב שלנו
   ```
2. Navigate to the project directory:
   ```bash
   cd project_python_01
   ```
3. Run the main script:
   ```bash
   python main/main.py
   ```

## key functions

### Data Cleaning Functions (src/data_cleaning.py):
1. remove_missing_values: Removes rows with missing values from the dataset.
2. normalize_columns: Normalizes specified numeric columns using Min-Max Scaling.
3. remove_outliers: Identifies and removes outliers from a specific column based on the Z-score method.
4. encode_categorical_columns: Encodes categorical columns into numeric values for further analysis.

### Data Analysis Functions (src/data_analysis.py): 
1. descriptive_statistics: Returns descriptive statistics (mean, median, standard deviation, etc.) for specified columns.
2. check_normality: Tests whether a specific column follows a normal distribution using the Shapiro-Wilk test.
3. compare_groups: Performs a T-test to compare the means of a specific feature between healthy individuals and Parkinson's patients.
4. correlation_matrix: Computes and returns a correlation matrix for specified columns.
5. logistic_regression_analysis: Builds and trains a logistic regression model to predict Parkinson's status based on specified features.
Returns the trained model, test set predictions, and probabilities.
6. perform_analysis: Runs all analysis steps, including descriptive statistics, normality tests, group comparisons, and correlation matrix generation.

### Data Visualization Functions (src/data_visualization.py):
1. plot_correlation_matrix: Generates a heatmap to visualize correlations between features related to stability and loudness.
2. plot_group_comparison: Creates a bar chart comparing means of a specific feature between healthy individuals and Parkinson's patients.
3. create_logistic_regression_plot: Visualizes logistic regression predictions, showing how probabilities change with the selected feature.

### Functions in main.py
1. load_cleaned_data: Loads the cleaned dataset from a specified CSV file.
2. save_plot: Saves a matplotlib figure to the outputs directory.
3. main: Executes the full analysis and visualization pipeline: Loads the cleaned dataset, Performs logistic regression analysis, Generates and saves visualizations for correlation, group comparisons, and logistic regression.

## tests functions

### data_cleaning_test.py
1. test_remove_missing_values: Verifies that rows with missing values are properly removed from the DataFrame.
2. test_remove_missing_values_no_missing: Ensures that a DataFrame with no missing values remains unchanged.
3. test_normalize_columns: Confirms that specified numeric columns are normalized to the range [0, 1].
4. test_remove_outliers: Tests that rows with outliers (based on z-scores) are identified and removed.
5. test_remove_outliers_no_outliers: Ensures that a DataFrame without outliers remains unchanged.
6. test_encode_categorical_columns: Verifies that categorical columns are correctly encoded into numeric values.
7. test_encode_categorical_columns_no_categorical: Ensures that a DataFrame with no categorical columns remains unchanged.

### test_data_analysis.py
1. test_descriptive_statistics: Verifies that descriptive statistics (mean, median, etc.) are calculated correctly for specified columns.
2. test_check_normality: Ensures that the Shapiro-Wilk test for normality is performed correctly and returns the expected output format.
3. test_compare_groups: Confirms that the T-Test between groups (e.g., healthy vs. Parkinson's patients) is calculated correctly.
4. test_correlation_matrix: Verifies that a correlation matrix is generated correctly for numeric columns.
5. test_logistic_regression_analysis: Ensures that the logistic regression model is trained correctly and returns expected results (e.g., predictions, probabilities).
6. test_perform_analysis: Confirms that the overall analysis pipeline executes without errors and returns all expected results.

Run all tests with:
```bash
pytest tests/
```


## outputs

outputs/
├── cleaned_data.csv            # The cleaned dataset after preprocessing
├── correlation_matrix.png      # Heatmap showing correlations between features
├── logistic_regression.png     # Visualization of logistic regression predictions
├── group_comparison_Fo(Hz).png # Comparison of "Fo(Hz)" between groups
├── group_comparison_Fhi(Hz).png # Comparison of "Fhi(Hz)" between groups
├── group_comparison_Flo(Hz).png # Comparison of "Flo(Hz)" between groups


## Credits

- Developed by [Ariel Tzooman, May Mualem, Linoy Elbaz](https://github.com/yourusername)
- Dataset from (https://www.kaggle.com/datasets/gargmanas/parkinsonsdataset)

