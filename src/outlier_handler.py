import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_outliers_before_handling(df, output_dir='plots'):
    """
    Generates and saves a single image with multiple boxplots for key numerical features
    to visualize outliers BEFORE capping.
    Args:
        df (pd.DataFrame): The DataFrame with engineered features (before outlier handling).
        output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n--- Generating Consolidated Boxplots for Outlier Visualization (Before Capping) in '{output_dir}' directory ---")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Select a few key numerical columns for plotting outliers
    # Aim for a manageable number to fit in one figure (e.g., 6-9 plots)
    cols_to_plot = [
        'annual_income', 'loan_amount', 'installment', 'debt_to_income',
        'total_credit_limit', 'total_credit_utilized', 'interest_rate',
        'total_collection_amount_ever', 'total_debit_limit', 'loan_to_annual_inc_ratio' # Added new engineered feature
    ]
    # Filter to ensure columns exist in the current DataFrame
    cols_to_plot = [col for col in cols_to_plot if col in numerical_cols]

    # Determine grid size for subplots
    n_cols = 3
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols # Calculate rows needed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 6))
    axes = axes.flatten() # Flatten the array of axes for easy iteration

    for i, col in enumerate(cols_to_plot):
        sns.boxplot(y=df[col], ax=axes[i], palette='pastel')
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel(col)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Outlier Visualization: Key Numerical Features (Before Capping)', y=1.02, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(output_dir, 'consolidated_boxplots_before_capping.png'))
    plt.close()
    print("Consolidated boxplots for outlier visualization saved.")


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

