# **Project Name: Credit Risk Data Preprocessing Pipeline**

A professional-grade data preprocessing pipeline for credit risk analysis, developed as a demonstration of robust data science principles and practices. This version introduces a modular code structure and enhanced feature engineering.

## **Table of Contents**

- [Introduction](https://www.google.com/search?q=%23introduction)
- [Problem Statement](https://www.google.com/search?q=%23problem-statement)
- [Key Features](https://www.google.com/search?q=%23key-features)
- [Dataset](https://www.google.com/search?q=%23dataset)
- [Technologies Used](https://www.google.com/search?q=%23technologies-used)
- [Project Structure](https://www.google.com/search?q=%23project-structure)
- [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
- [Usage](https://www.google.com/search?q=%23usage)
- [Interpreting the Results](https://www.google.com/search?q=%23interpreting-the-results)
- [Future Enhancements](https://www.google.com/search?q=%23future-enhancements)
- [Contributing](https://www.google.com/search?q=%23contributing)
- [License](https://www.google.com/search?q=%23license)

## **Introduction**

This repository contains a comprehensive solution for cleaning, preprocessing, and performing exploratory data analysis (EDA) on a real-world financial dataset. The primary goal is to transform raw, messy loan application data into a clean, structured format suitable for training machine learning models for credit risk prediction. This project goes beyond basic data handling by incorporating advanced techniques for feature engineering and robust data visualization, now with a more organized, modular code structure.

## **Problem Statement**

In the financial industry, accurately predicting credit risk is paramount for lenders to make informed decisions and minimize losses. However, raw data from lending platforms is often incomplete, inconsistent, and unstructured, posing significant challenges for direct use in machine learning models. This project addresses the critical first step of any robust machine learning workflow: preparing such a dataset for model training. We tackle common data quality challenges including:

- **Missing Data:** Identifying and handling missing values through strategic imputation or removal.
- **Feature Engineering:** Creating new, more informative features from existing raw data points, including interaction terms.
- **Categorical Data:** Converting non-numerical, categorical information into a numerical format that machine learning algorithms can process effectively.
- **Outlier Detection and Treatment:** Systematically identifying and mitigating the impact of extreme values that could skew model performance and lead to inaccurate predictions.
- **Data Skewness:** Addressing skewed distributions in numerical features to improve model stability and meet assumptions of certain algorithms.
- **Duplicate Data:** Identifying and removing redundant entries to ensure data integrity.

## **Key Features**

- **Modular Data Preprocessing Pipeline:** The monolithic script has been refactored into distinct, logical Python modules (data_loader.py, data_cleaner.py, feature_engineer.py, outlier_handler.py, preprocessor.py, visualizer.py, main.py) for improved readability, maintainability, and reusability.
- **Automated Data Cleaning:** Robust handling of missing values (dropping high-null columns, median/mode imputation) and detection/removal of duplicate records.
- **Comprehensive Exploratory Data Analysis (EDA):** Integration of various visualizations at different stages:
  - Missing values heatmap.
  - Categorical feature distributions.
  - Consolidated boxplots for outlier visualization (before capping).
  - Distribution comparisons (before vs. after scaling).
  - Target variable distribution.
  - Histograms and correlation matrix of final preprocessed features.
- **Intelligent Feature Engineering:** Creation of domain-relevant features, including derived ratios and interaction terms, to enhance predictive power.
- **Robust Outlier Handling:** Implementation of the Interquartile Range (IQR) method for capping outliers, preserving data integrity.
- **Professional Documentation:** Extensive docstrings for functions and classes, along with clear inline comments, explaining the 'why' and 'how' of the code's logic.

## **Dataset**

This project utilizes the **Lending Club Loan Dataset** available on Kaggle. Specifically, we will use the loans_full_schema.csv file from the following Kaggle dataset:

**Kaggle Dataset Link:** [Lending Club Loan Dataset](https://www.kaggle.com/datasets/utkarshx27/lending-club-loan-dataset)

This dataset contains 10,000 observations across 55 variables, providing a realistic yet manageable size for demonstrating comprehensive data cleaning and preprocessing techniques without being overly resource-intensive.

## **Technologies Used**

- **Python 3.8+**
- **Pandas:** Essential for efficient data manipulation and analysis.
- **NumPy:** Provides powerful numerical computing capabilities.
- **Matplotlib:** Used for creating static, high-quality data visualizations.
- **Seaborn:** A high-level data visualization library built on Matplotlib, used for generating attractive and informative statistical graphics.
- **Scikit-learn:** Utilized for various preprocessing utilities such as LabelEncoder and StandardScaler, laying the groundwork for future machine learning model training.

## **Project Structure**

credit-risk-analysis/  
├── README.md  
├── requirements.txt  
├── .gitignore  
├── data/  
│ └── loans_full_schema.csv # Raw input dataset (download from Kaggle link above)  
│ └── preprocessed_loan_data.csv # Output: Cleaned and preprocessed data  
├── src/  
│ ├── \__init_\_.py # Makes 'src' a Python package  
│ ├── main.py # Main script to run the entire pipeline  
│ ├── data_loader.py # Functions for loading data and initial overview  
│ ├── data_cleaner.py # Functions for cleaning, imputation, duplicate handling, missing value heatmap  
│ ├── feature_engineer.py # Functions for creating new features  
│ ├── outlier_handler.py # Functions for outlier detection and capping, boxplots  
│ ├── preprocessor.py # Functions for encoding and scaling, scaling impact plots  
│ └── visualizer.py # Functions for final distributions, correlation, and target plots  
└── plots/ # Directory for generated visualizations (created on run)  

## **Setup and Installation**

To set up and run this project locally, follow these steps:

1. **Clone the repository:**  
    git clone <https://github.com/Ashish-Ghoshal/credit-risk-analysis.git>  
    cd credit-risk-analysis  

2. Create a virtual environment (highly recommended):  
    This isolates project dependencies and avoids conflicts with other Python projects.  
    python -m venv cr_venv  
    \# On macOS/Linux:  
    source cr_venv/bin/activate  
    \# On Windows (PowerShell):  
    .\\cr_venv\\Scripts\\Activate.ps1  
    \# On Windows (Command Prompt):  
    .\\cr_venv\\Scripts\\activate.bat  

3. Install the required dependencies:  
    The necessary libraries are listed in requirements.txt.  
    pip install -r requirements.txt  

4. Download and place the dataset:  
    Go to the Kaggle dataset link provided above: Lending Club Loan Dataset.  
    Download the loans_full_schema.csv file and place it into the data/ directory within your cloned repository. Create the data/ directory if it doesn't exist.

## **Usage**


**IMPORTANT: How to Run the Pipeline**

To run the entire data preprocessing pipeline, you must execute the `main.py` script **as a Python module**. This is the standard and most robust way to run Python code organized into packages.

1.  **Navigate to the project's root directory:** Open your terminal (PowerShell or Git Bash) and ensure your current directory is `credit-risk-analysis` (the folder containing `src/`, `data/`, `README.md`, etc.).
    
2.  **Activate your virtual environment:**
    
        # On macOS/Linux:
        source cr_venv/bin/activate
        # On Windows (PowerShell):
        .\cr_venv\Scripts\Activate.ps1
        # On Windows (Command Prompt):
        .\cr_venv\Scripts\activate.bat
        
    
3.  **Run the main script as a module:**
    
        python -m src.main
Upon successful execution, the script will:

1. Load the raw data from data/loans_full_schema.csv.
2. Apply a series of data cleaning, feature engineering, outlier handling, and scaling steps, with detailed console output for each stage.
3. Generate various insightful exploratory plots at different stages of preprocessing and save them as PNG images in the plots/ directory.
4. Save the final cleaned and preprocessed dataset as data/preprocessed_loan_data.csv.

## **Interpreting the Results**

The script's console output and the generated plots provide a clear narrative of the data transformation process and insights into the dataset:

- **Initial Data Overview & Missing Values Heatmap (plots/missing_values_heatmap_before_cleaning.png):** Provides a baseline understanding of the raw data's structure and highlights patterns of missingness.
- **Categorical Distributions (plots/categorical_distribution_\*.png):** Shows the frequency of different categories in key nominal features before encoding.
- **Consolidated Boxplots (plots/consolidated_boxplots_before_capping.png):** Visually identifies outliers in numerical features before they are capped, demonstrating the need for outlier handling.
- **Scaling Impact Plots (plots/scaling_impact_\*.png):** Illustrates the effect of standardization on feature distributions, showing how data is transformed to a consistent scale.
- **Target Distribution (plots/target_distribution_loan_status.png):** Reveals the class balance of the target variable (loan_status), indicating potential data imbalance issues that would need to be addressed in the modeling phase.
- **Final Feature Distributions (plots/feature_distributions_post_preprocessing.png):** Shows the distributions of numerical features after all preprocessing, confirming their readiness for modeling.
- **Correlation Matrix (plots/correlation_matrix_post_preprocessing.png):** Provides a visual summary of relationships between all numerical features in the final dataset.

## **Future Enhancements**

To evolve this project into an even more robust and production-ready solution, consider the following enhancements:

- **Advanced Missing Value Imputation:** Explore and implement more sophisticated missing value imputation methods beyond simple median/mode imputation. This could include K-Nearest Neighbors (KNN) Imputation, MICE (Multiple Imputation by Chained Equations), or even training a separate machine learning model to predict missing values based on other features.
- **Handling High Cardinality Categorical Features:** For categorical columns with many unique values (high cardinality), investigate and implement techniques like Target Encoding, Frequency Encoding, or grouping rare categories to reduce dimensionality and improve model performance.
- **Automated Feature Selection:** Implement advanced feature selection techniques (e.g., Recursive Feature Elimination, SelectKBest, or tree-based feature importance methods) to automatically identify and retain only the most impactful features for the predictive model, reducing dimensionality and improving model performance.
- **Polynomial Feature Generation:** Systematically generate polynomial features for numerical columns to capture non-linear relationships that might exist between features and the target variable.
- **Data Validation and Schema Enforcement:** Integrate a data validation library (e.g., Great Expectations, Pandera) to define and enforce a strict data schema. This ensures data quality at ingestion and throughout the pipeline, preventing unexpected data types, ranges, or missingness from breaking downstream processes.
- **Full Machine Learning Pipeline Integration:** Extend the project to include the complete machine learning workflow. This would involve:
  - Splitting the preprocessed data into training and testing sets.
  - Training and evaluating various classification models (e.g., Logistic Regression, Random Forest, Gradient Boosting, XGBoost) for credit risk prediction.
  - Performing hyperparameter tuning and cross-validation to optimize model performance.
  - Generating comprehensive model evaluation metrics (e.g., accuracy, precision, recall, F1-score, ROC-AUC).
- **Model Deployment:** Containerize the entire solution using Docker, making it portable and easily deployable. Develop a simple API (e.g., using Flask or FastAPI) to serve the trained model's predictions, allowing real-time credit risk assessment.
- **Pipeline Orchestration:** For larger, more complex projects, consider using tools like Apache Airflow or Prefect to orchestrate and schedule the data preprocessing and model training pipeline, ensuring automated and reliable execution.

## **Contributing**

Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/YourFeature).
3. Make your changes.
4. Commit your changes (git commit -m 'Add some feature').
5. Push to the branch (git push origin feature/YourFeature).
6. Open a Pull Request.

## **License**

This project is licensed under the MIT License. See the LICENSE file for details.