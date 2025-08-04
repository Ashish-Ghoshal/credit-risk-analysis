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

def plot_categorical_distributions(df, output_dir='plots'):
    """
    Generates and saves count plots for key categorical features to visualize their distributions.
    Args:
        df (pd.DataFrame): The DataFrame containing categorical features.
        output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n--- Generating Categorical Feature Distribution Plots in '{output_dir}' directory ---")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    # Select a few representative categorical columns for plotting
    # Avoid plotting too many if there are many unique values or too many columns
    cols_to_plot = [col for col in categorical_cols if df[col].nunique() < 20 and col != 'loan_status'] # Exclude target for now, handle separately
    if len(cols_to_plot) > 5: # Limit to avoid too many plots
        cols_to_plot = cols_to_plot[:5]

    for col in cols_to_plot:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=df[col], order=df[col].value_counts().index, palette='viridis')
        plt.title(f'Distribution of {col} (Before Encoding)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'categorical_distribution_{col}_before_encoding.png'))
        plt.close()
        print(f"Saved 'categorical_distribution_{col}_before_encoding.png'")
    print("Categorical distribution plots saved.")


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

def plot_outliers_before_handling(df, output_dir='plots'):
    """
    Generates and saves boxplots for key numerical features to visualize outliers BEFORE capping.
    Args:
        df (pd.DataFrame): The DataFrame with engineered features (before outlier handling).
        output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n--- Generating Boxplots for Outlier Visualization (Before Capping) in '{output_dir}' directory ---")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Select a few key numerical columns for plotting outliers
    # Avoid columns with very few unique values or those that are essentially categorical
    cols_to_plot = [
        'annual_income', 'loan_amount', 'installment', 'debt_to_income',
        'total_credit_limit', 'total_credit_utilized', 'interest_rate'
    ]
    # Filter to ensure columns exist in the current DataFrame
    cols_to_plot = [col for col in cols_to_plot if col in numerical_cols]

    for col in cols_to_plot:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[col], palette='pastel')
        plt.title(f'Boxplot of {col} (Before Outlier Capping)')
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'boxplot_before_capping_{col}.png'))
        plt.close()
        print(f"Saved 'boxplot_before_capping_{col}.png'")
    print("Boxplots for outlier visualization saved.")


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

def plot_scaling_impact(df_before_scaling, df_after_scaling, output_dir='plots'):
    """
    Generates and saves plots comparing the distribution of key numerical features
    before and after StandardScaler.
    Args:
        df_before_scaling (pd.DataFrame): DataFrame before scaling.
        df_after_scaling (pd.DataFrame): DataFrame after scaling.
        output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n--- Generating Scaling Impact Comparison Plots in '{output_dir}' directory ---")
    
    # Select a few key numerical columns to show scaling impact
    cols_to_compare = [
        'annual_income', 'loan_amount', 'installment', 'debt_to_income',
        'total_credit_limit', 'total_credit_utilized', 'interest_rate'
    ]
    # Filter to ensure columns exist in both DataFrames
    cols_to_compare = [col for col in cols_to_compare if col in df_before_scaling.columns and col in df_after_scaling.columns]

    for col in cols_to_compare:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df_before_scaling[col], kde=True, ax=axes[0], color='skyblue', edgecolor='black')
        axes[0].set_title(f'Original Distribution of {col}')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')

        sns.histplot(df_after_scaling[col], kde=True, ax=axes[1], color='lightcoral', edgecolor='black')
        axes[1].set_title(f'Scaled Distribution of {col}')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Frequency')
        
        plt.suptitle(f'Distribution Comparison: {col} (Before vs. After Scaling)', y=1.05, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig(os.path.join(output_dir, f'scaling_impact_{col}.png'))
        plt.close()
        print(f"Saved 'scaling_impact_{col}.png'")
    print("Scaling impact plots saved.")

def plot_target_distribution(df, target_col='loan_status', output_dir='plots'):
    """
    Generates and saves a count plot for the target variable distribution.
    Args:
        df (pd.DataFrame): The DataFrame containing the target variable.
        target_col (str): The name of the target column.
        output_dir (str): Directory to save the plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n--- Generating Target Variable Distribution Plot in '{output_dir}' directory ---")
    if target_col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=df[target_col], palette='coolwarm')
        plt.title(f'Distribution of {target_col}')
        plt.xlabel(target_col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'target_distribution_{target_col}.png'))
        plt.close()
        print(f"Saved 'target_distribution_{target_col}.png'")
    else:
        print(f"Warning: Target column '{target_col}' not found in DataFrame. Skipping plot.")
    print("Target variable distribution plot saved.")


def plot_final_distributions_and_correlation(df, output_dir='plots'):
    """
    Generates and saves exploratory data visualizations (histograms and correlation matrix)
    for the final preprocessed data.
    Args:
        df (pd.DataFrame): The final preprocessed DataFrame to visualize.
        output_dir (str): The directory where the generated plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")
        
    print(f"\n--- Generating Final Preprocessed Data Visualizations in '{output_dir}' directory ---")
    
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
    
    print("Final preprocessed data visualizations saved successfully.")

def main():
    """
    Main function to orchestrate the entire data preprocessing pipeline.
    Loads data, cleans it, performs feature engineering, handles outliers,
    applies final encoding and scaling, saves the processed data, and
    generates visualizations.
    """
    file_path = 'data/loans_full_schema.csv' # Updated to the specific Kaggle file name
    output_processed_data_path = 'data/preprocessed_loan_data.csv'
    output_plots_dir = 'plots' # Define output directory for plots

    # Ensure the plots directory exists at the start
    if not os.path.exists(output_plots_dir):
        os.makedirs(output_plots_dir)
        print(f"Created output directory: '{output_plots_dir}'")
    
    df = load_data(file_path)
    if df is None:
        return # Exit if data loading failed

    # --- Initial Data Overview and Pre-cleaning Visualizations ---
    initial_data_overview(df.copy())
    plot_missing_values_heatmap(df.copy(), output_plots_dir) # Plot missing values before cleaning

    df_cleaned = clean_data(df.copy())
    plot_categorical_distributions(df_cleaned.copy(), output_plots_dir) # Plot categorical distributions before encoding

    df_engineered = feature_engineering(df_cleaned.copy())
    plot_outliers_before_handling(df_engineered.copy(), output_plots_dir) # Plot outliers before capping

    df_outliers_handled = handle_outliers(df_engineered.copy())
    
    # --- Post-processing and Final Visualizations ---
    df_preprocessed = preprocess_data(df_outliers_handled.copy())
    
    plot_scaling_impact(df_outliers_handled.copy(), df_preprocessed.copy(), output_plots_dir) # Plot scaling impact
    plot_target_distribution(df_preprocessed.copy(), 'loan_status', output_plots_dir) # Plot target distribution
    plot_final_distributions_and_correlation(df_preprocessed.copy(), output_plots_dir) # Existing final plots

    print(f"\nFinal preprocessed data shape: {df_preprocessed.shape}")
    print("\nFirst 5 rows of the final preprocessed data:")
    print(df_preprocessed.head())
    
    # Save the final preprocessed dataset
    try:
        df_preprocessed.to_csv(output_processed_data_path, index=False)
        print(f"\nSuccessfully saved preprocessed data to '{output_processed_data_path}'.")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")
    
    print("\nData preprocessing and visualization pipeline completed.")

if __name__ == "__main__":
    main()
