import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

