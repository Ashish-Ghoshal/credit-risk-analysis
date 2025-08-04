import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

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

def clean_data(df):
    """
    Performs core data cleaning steps on the DataFrame.
    This function handles columns with a high percentage of missing values,
    imputes remaining numerical missing values with the median, and
    categorical missing values with the mode. It also performs specific
    cleaning for known columns in the Lending Club dataset.
    Args:
        df (pd.DataFrame): The raw DataFrame.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print("\n--- Data Cleaning Process ---")

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
        # Changed to direct assignment to avoid FutureWarning with inplace=True
        df[col] = df[col].fillna(median_val)
        print(f"Imputed missing values in numerical column '{col}' with median: {median_val}")

    # For remaining missing categorical values, impute with the mode.
    # Mode is appropriate for categorical data as it represents the most frequent category.
    categorical_cols_with_nulls = df.select_dtypes(include='object').columns[df.select_dtypes(include='object').isnull().any()].tolist()
    for col in categorical_cols_with_nulls:
        mode_val = df[col].mode()[0]
        # Changed to direct assignment to avoid FutureWarning with inplace=True
        df[col] = df[col].fillna(mode_val)
        print(f"Imputed missing values in categorical column '{col}' with mode: '{mode_val}'")
    
    # --- Specific cleaning for Lending Club dataset (loans_full_schema.csv) ---
    # The 'term' and 'interest_rate' columns are already numerical (int64/float64)
    # in this specific dataset, so string extraction/replacement is not needed.
    # The previous code assumed they were strings like " 36 months" or "10.65%".
    # These lines are removed as they cause AttributeError.

    # Drop columns that are irrelevant for modeling or contain unique identifiers
    # 'id', 'url', 'desc', 'title', 'zip_code', 'emp_title', 'member_id' were
    # identified as irrelevant in the previous dataset.
    # For 'loans_full_schema.csv', 'Unnamed: 0' is an index, and 'emp_title' is also dropped.
    cols_to_drop = ['Unnamed: 0', 'emp_title', 'url', 'desc', 'title', 'zip_code', 'member_id']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    print(f"Dropped irrelevant columns: {', '.join([col for col in cols_to_drop if col in df.columns])}")

    print("Finished initial data cleaning and missing value imputation.")
    return df

def feature_engineering(df):
    """
    Creates new, more informative features from existing ones.
    This function demonstrates feature engineering by creating a new
    'loan_to_annual_inc_ratio' and 'credit_history_length_months'.
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with newly engineered features.
    """
    print("\n--- Feature Engineering ---")

    # The 'fico_range_low' and 'fico_range_high' columns are NOT present
    # in the 'loans_full_schema.csv' dataset. Removing this feature creation.
    # if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
    #     df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    #     df = df.drop(columns=['fico_range_low', 'fico_range_high'])
    #     print("Created 'fico_score' from FICO range columns and dropped originals.")
    
    # Calculate a loan-to-annual-income ratio.
    # This ratio can be a strong indicator of a borrower's ability to repay.
    # Corrected column names from 'loan_amnt' to 'loan_amount' and 'annual_inc' to 'annual_income'
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        # Add a small epsilon to annual_income to prevent division by zero for any 0 values
        df['loan_to_annual_inc_ratio'] = df['loan_amount'] / (df['annual_income'] + 1e-6)
        print("Created 'loan_to_annual_inc_ratio'.")

    # Example: Creating 'credit_history_length' from 'earliest_credit_line'
    # This feature indicates the longevity of the borrower's credit history.
    # Corrected column name from 'earliest_cr_line' to 'earliest_credit_line'
    if 'earliest_credit_line' in df.columns:
        # Convert to datetime, handling potential parsing errors by coercing to NaT
        df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line'], errors='coerce')
        # Calculate length of credit history in months from the earliest credit line to now
        # Using pd.to_datetime('now') for the current date
        df['credit_history_length_months'] = (pd.to_datetime('now') - df['earliest_credit_line']).dt.days / 30.44
        # Impute any NaT results (e.g., from unparseable dates) with the median length
        # Changed to direct assignment to avoid FutureWarning with inplace=True
        df['credit_history_length_months'] = df['credit_history_length_months'].fillna(df['credit_history_length_months'].median())
        # Drop the original 'earliest_credit_line' column as it's been transformed
        df.drop(columns=['earliest_credit_line'], inplace=True)
        print("Created 'credit_history_length_months' and dropped 'earliest_credit_line'.")

    return df

def handle_outliers(df):
    """
    Identifies and handles outliers in numerical features using the Interquartile Range (IQR) method.
    Outliers are capped at the upper and lower bounds defined by 1.5 * IQR from Q3 and Q1,
    respectively. This approach mitigates the impact of extreme values without removing data.
    Args:
        df (pd.DataFrame): The DataFrame with engineered features.
    Returns:
        pd.DataFrame: The DataFrame with outliers capped.
    """
    print("\n--- Outlier Handling (Capping using IQR) ---")
    numerical_cols = df.select_dtypes(include=np.number).columns
    
    for col in numerical_cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1 # Calculate the Interquartile Range
        
        # Define the lower and upper bounds for outlier detection
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers before capping
        outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outliers_count > 0:
            # Cap values that are below the lower bound to the lower bound
            # Cap values that are above the upper bound to the upper bound
            # Using .loc for explicit assignment to avoid SettingWithCopyWarning and chained assignment issues
            df.loc[:, col] = np.where(df[col] > upper_bound, upper_bound, 
                               np.where(df[col] < lower_bound, lower_bound, df[col]))
            print(f"Capped {outliers_count} outliers in column '{col}'.")
        else:
            print(f"No significant outliers found in column '{col}' to cap.")
    
    print("Outlier capping process completed for all numerical features.")
    return df

def preprocess_data(df):
    """
    Performs final preprocessing steps including categorical encoding and numerical scaling.
    Binary categorical features are encoded using Label Encoding, while multi-class
    categorical features are One-Hot Encoded. All numerical features are then
    Standard Scaled to standardize their range and distribution.
    Args:
        df (pd.DataFrame): The DataFrame with outliers handled.
    Returns:
        pd.DataFrame: The final preprocessed DataFrame.
    """
    print("\n--- Final Preprocessing: Encoding and Scaling ---")
    
    # Identify categorical columns for encoding
    categorical_cols = df.select_dtypes(include='object').columns
    
    for col in categorical_cols:
        unique_values = df[col].nunique()
        if unique_values <= 2: # Binary features (e.g., 'loan_status': 'Fully Paid', 'Charged Off')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            print(f"Label encoded binary feature: '{col}'")
        else: # Multi-class features (e.g., 'purpose', 'home_ownership')
            # One-Hot Encoding: Creates new binary columns for each category.
            # drop_first=True prevents multicollinearity by dropping the first category.
            df = pd.get_dummies(df, columns=[col], prefix=f'{col}', drop_first=True)
            print(f"One-Hot encoded multi-class feature: '{col}' (created {unique_values - 1} new columns).")
            
    # Standard Scaling for all numerical features.
    # This transforms data to have a mean of 0 and standard deviation of 1,
    # which is beneficial for many ML algorithms that are sensitive to feature scales.
    numerical_cols = df.select_dtypes(include=np.number).columns
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        print("Standard scaled all numerical features.")
    else:
        print("No numerical features to scale after encoding.")
            
    return df

def visualize_data(df, output_dir='plots'):
    """
    Generates and saves exploratory data visualizations to understand feature distributions
    and correlations after preprocessing.
    Args:
        df (pd.DataFrame): The final preprocessed DataFrame to visualize.
        output_dir (str): The directory where the generated plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")
        
    print(f"\n--- Generating visualizations in '{output_dir}' directory ---")
    
    # Select a subset of numerical columns for histogram visualization
    # Avoid plotting too many columns at once for clarity and to prevent memory issues with very wide dataframes.
    numerical_cols_for_plot = df.select_dtypes(include=np.number).columns
    if len(numerical_cols_for_plot) > 20: # Limit for readability in plots
        numerical_cols_for_plot = numerical_cols_for_plot[:20] 

    # Visualize feature distributions using histograms
    # This helps to see the effect of scaling and transformations.
    df[numerical_cols_for_plot].hist(bins=30, figsize=(20, 15), edgecolor='black')
    plt.suptitle("Feature Distributions (After Preprocessing)", y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(output_dir, 'feature_distributions_post_preprocessing.png'))
    plt.close()
    print(f"Saved 'feature_distributions_post_preprocessing.png'")
    
    # Visualize correlation matrix of numerical features
    # A heatmap provides a quick overview of feature relationships.
    plt.figure(figsize=(18, 15))
    correlation_matrix = df[numerical_cols_for_plot].corr() # Use subset for clarity
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix of Numerical Features (After Preprocessing)", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_post_preprocessing.png'))
    plt.close()
    print(f"Saved 'correlation_matrix_post_preprocessing.png'")
    
    print("All specified visualizations saved successfully.")

def main():
    """
    Main function to orchestrate the entire data preprocessing pipeline.
    Loads data, cleans it, performs feature engineering, handles outliers,
    applies final encoding and scaling, saves the processed data, and
    generates visualizations.
    """
    file_path = 'data/loans_full_schema.csv' # Updated to the specific Kaggle file name
    output_processed_data_path = 'data/preprocessed_loan_data.csv'
    
    df = load_data(file_path)
    if df is None:
        return # Exit if data loading failed

    initial_data_overview(df.copy()) # Show initial state before modification
        
    # Execute the preprocessing pipeline steps sequentially
    df_cleaned = clean_data(df.copy())
    df_engineered = feature_engineering(df_cleaned.copy())
    df_outliers_handled = handle_outliers(df_engineered.copy())
    df_preprocessed = preprocess_data(df_outliers_handled.copy())
    
    print(f"\nFinal preprocessed data shape: {df_preprocessed.shape}")
    print("\nFirst 5 rows of the final preprocessed data:")
    print(df_preprocessed.head())
    
    # Save the final preprocessed dataset
    try:
        df_preprocessed.to_csv(output_processed_data_path, index=False)
        print(f"\nSuccessfully saved preprocessed data to '{output_processed_data_path}'.")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")
    
    # Generate and save visualizations of the preprocessed data
    visualize_data(df_preprocessed)

if __name__ == "__main__":
    main()
