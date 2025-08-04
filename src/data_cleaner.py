import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_missing_values_heatmap(df, output_dir='plots'):
    """
    Generates and saves a heatmap visualizing the pattern of missing values in the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to visualize.
        output_dir (str): Directory to save the plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n--- Generating Missing Values Heatmap in '{output_dir}' directory ---")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap (Before Cleaning)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_values_heatmap_before_cleaning.png'))
    plt.close()
    print("Missing values heatmap saved.")


def clean_data(df):
    """
    Performs core data cleaning steps on the DataFrame.
    This function handles columns with a high percentage of missing values,
    imputes remaining numerical missing values with the median, and
    categorical missing values with the mode. It also performs specific
    cleaning for known columns in the Lending Club dataset and removes duplicates.
    Args:
        df (pd.DataFrame): The raw DataFrame.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("\n--- Data Cleaning Process ---")

    # Remove duplicate rows
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        print(f"Removed {initial_rows - df.shape[0]} duplicate rows.")
    else:
        print("No duplicate rows found.")

    # Drop columns with a high percentage of missing values.
    # A threshold of 50% is chosen to balance data loss and feature quality.
    initial_cols = df.shape[1]
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)
    print(f"Dropped {initial_cols - df.shape[1]} columns with more than 50% missing values.")

    # Impute remaining missing numerical values with the median.
    # Median is used instead of mean to be robust against outliers and skewed distributions.
    numerical_cols_with_nulls = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()].tolist()
    for col in numerical_cols_with_nulls:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed missing values in numerical column '{col}' with median: {median_val}")

    # For remaining missing categorical values, impute with the mode.
    # Mode is appropriate for categorical data as it represents the most frequent category.
    categorical_cols_with_nulls = df.select_dtypes(include='object').columns[df.select_dtypes(include='object').isnull().any()].tolist()
    for col in categorical_cols_with_nulls:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Imputed missing values in categorical column '{col}' with mode: '{mode_val}'")
    
    # Drop columns that are irrelevant for modeling or contain unique identifiers
    # 'id', 'url', 'desc', 'title', 'zip_code', 'emp_title', 'member_id' were
    # identified as irrelevant in the previous dataset.
    # For 'loans_full_schema.csv', 'Unnamed: 0' is an index, and 'emp_title' is also dropped.
    cols_to_drop = ['Unnamed: 0', 'emp_title', 'url', 'desc', 'title', 'zip_code', 'member_id']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    print(f"Dropped irrelevant columns: {', '.join([col for col in cols_to_drop if col in df.columns])}")


    print("Finished initial data cleaning and missing value imputation.")
    return df

