import pandas as pd
import numpy as np # Imported for initial_data_overview's dtypes check

def load_data(file_path):
    """
    Loads data from a CSV file into a Pandas DataFrame.
    This function provides basic error handling for file not found issues.
    Args:
        file_path (str): The full path to the CSV file.
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found. Please ensure it exists.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def initial_data_overview(df):
    """
    Provides an initial overview of the DataFrame, including shape, missing values,
    and data types.
    Args:
        df (pd.DataFrame): The DataFrame to inspect.
    """
    print("--- Initial Data Overview ---")
    print(f"Original data shape: {df.shape}")
    print("\nMissing values before cleaning:")
    # Display only columns with missing values for brevity
    print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))
    print("\nData types overview:")
    print(df.info())
    print("\nFirst 5 rows of the original data:")
    print(df.head())

