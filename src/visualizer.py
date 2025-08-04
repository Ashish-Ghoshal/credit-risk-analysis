import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    for the final preprocessed data. This includes all numerical features,
    including one-hot encoded ones.
    Args:
        df (pd.DataFrame): The final preprocessed DataFrame to visualize.
        output_dir (str): The directory where the generated plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: '{output_dir}'")
        
    print(f"\n--- Generating Final Preprocessed Data Visualizations in '{output_dir}' directory ---")
    
    # Select ALL numerical columns for plotting, including one-hot encoded ones.
    # Pandas' select_dtypes(include=np.number) will correctly pick up int, float, uint8 (for OHE).
    all_numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    
    # --- Histograms ---
    # Limit for readability if there are too many columns
    hist_cols = all_numerical_cols
    if len(hist_cols) > 30: # Increase limit slightly for more features
        hist_cols = hist_cols[:30] 

    df[hist_cols].hist(bins=30, figsize=(25, 20), edgecolor='black') # Increased figure size
    plt.suptitle("Feature Distributions (After Preprocessing)", y=1.02, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.savefig(os.path.join(output_dir, 'feature_distributions_post_preprocessing.png'))
    plt.close()
    print(f"Saved 'feature_distributions_post_preprocessing.png'")
    
    # --- Correlation Matrix ---
    # Use ALL numerical columns for the correlation matrix calculation
    # Adjust figure size dynamically based on number of columns for better readability
    num_features = len(all_numerical_cols)
    fig_size = max(18, num_features * 0.5) # Dynamic sizing, minimum 18
    plt.figure(figsize=(fig_size, fig_size))

    correlation_matrix = df[all_numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix of All Numerical Features (After Preprocessing)", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_post_preprocessing.png'))
    plt.close()
    print(f"Saved 'correlation_matrix_post_preprocessing.png'")
    
    print("Final preprocessed data visualizations saved successfully.")

