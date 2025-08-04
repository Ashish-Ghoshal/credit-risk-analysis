import os
from .data_loader import load_data, initial_data_overview
from .data_cleaner import clean_data, plot_missing_values_heatmap
from .feature_engineer import feature_engineering
from .outlier_handler import handle_outliers, plot_outliers_before_handling
from .preprocessor import preprocess_data, plot_scaling_impact
from .visualizer import plot_categorical_distributions, plot_target_distribution, plot_final_distributions_and_correlation

def main():
    """
    Main function to orchestrate the entire data preprocessing pipeline.
    Loads data, cleans it, performs feature engineering, handles outliers,
    applies final encoding and scaling, saves the processed data, and
    generates comprehensive visualizations.
    """
    file_path = 'data/loans_full_schema.csv'
    output_processed_data_path = 'data/preprocessed_loan_data.csv'
    output_plots_dir = 'plots'

    # Ensure the plots directory exists at the start
    if not os.path.exists(output_plots_dir):
        os.makedirs(output_plots_dir)
        print(f"Created output directory: '{output_plots_dir}'")
    
    # --- Stage 1: Data Loading and Initial Overview ---
    df = load_data(file_path)
    if df is None:
        return # Exit if data loading failed

    initial_data_overview(df.copy())
    plot_missing_values_heatmap(df.copy(), output_plots_dir) # Plot missing values before cleaning

    # --- Stage 2: Data Cleaning ---
    df_cleaned = clean_data(df.copy())
    plot_categorical_distributions(df_cleaned.copy(), output_plots_dir) # Plot categorical distributions before encoding

    # --- Stage 3: Feature Engineering ---
    df_engineered = feature_engineering(df_cleaned.copy())

    # --- Stage 4: Outlier Handling ---
    plot_outliers_before_handling(df_engineered.copy(), output_plots_dir) # Plot outliers before capping
    df_outliers_handled = handle_outliers(df_engineered.copy())
    
    # --- Stage 5: Final Preprocessing (Encoding and Scaling) ---
    # Plot target distribution BEFORE final preprocessing (encoding)

    plot_target_distribution(df_outliers_handled.copy(), 'loan_status', output_plots_dir) 

    # Keep a copy before scaling to show scaling impact
    df_before_scaling = df_outliers_handled.copy() 
    df_preprocessed = preprocess_data(df_outliers_handled.copy())
    
    plot_scaling_impact(df_before_scaling, df_preprocessed.copy(), output_plots_dir) # Plot scaling impact
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
