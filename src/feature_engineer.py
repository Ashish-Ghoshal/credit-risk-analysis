import pandas as pd
import numpy as np

def feature_engineering(df):
    """
    Creates new, more informative features from existing ones.
    This function demonstrates feature engineering by creating a new
    'loan_to_annual_inc_ratio', 'credit_history_length_months',
    and an interaction feature.
    Args:
        df (pd.DataFrame): The cleaned DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with newly engineered features.
    """
    print("\n--- Feature Engineering ---")

    # Calculate a loan-to-annual-income ratio.
    # This ratio can be a strong indicator of a borrower's ability to repay.
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        # Add a small epsilon to annual_income to prevent division by zero for any 0 values
        df['loan_to_annual_inc_ratio'] = df['loan_amount'] / (df['annual_income'] + 1e-6)
        print("Created 'loan_to_annual_inc_ratio'.")

    # Example: Creating 'credit_history_length' from 'earliest_credit_line'
    # This feature indicates the longevity of the borrower's credit history.
    if 'earliest_credit_line' in df.columns:
        # Convert to datetime, handling potential parsing errors by coercing to NaT
        df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line'], errors='coerce')
        # Calculate length of credit history in months from the earliest credit line to now
        # Using pd.to_datetime('now') for the current date
        df['credit_history_length_months'] = (pd.to_datetime('now') - df['earliest_credit_line']).dt.days / 30.44
        # Impute any NaT results (e.g., from unparseable dates) with the median length
        df['credit_history_length_months'] = df['credit_history_length_months'].fillna(df['credit_history_length_months'].median())
        # Drop the original 'earliest_credit_line' column as it's been transformed
        df.drop(columns=['earliest_credit_line'], inplace=True)
        print("Created 'credit_history_length_months' and dropped 'earliest_credit_line'.")


    # Captures if high loans with high interest rates have a different risk profile.
    if 'loan_amount' in df.columns and 'interest_rate' in df.columns:
        df['loan_x_interest_rate'] = df['loan_amount'] * df['interest_rate']
        print("Created interaction feature 'loan_x_interest_rate'.")


    return df

