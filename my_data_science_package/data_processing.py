
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder


def summarize_dataframe(sdf):
    """
    Summarizes the structure and content of a DataFrame.
    """
    return pd.DataFrame({
        'Column': sdf.columns,
        'Non-Null Count': sdf.notnull().sum(),
        'Data Type': sdf.dtypes,
        'Unique Values': sdf.nunique(),
        'Missing Values': sdf.isnull().sum(),
        'Missing Values %': ((sdf.isnull().sum()*100)/len(sdf)).round(2)
    }).reset_index(drop=True)


def impute_missing_values(df):
    """
    Imputes missing values in a dataset using KNN for low/moderate missingness 
    and Iterative Imputer for high missingness. Categorical columns are encoded 
    using LabelEncoder.

    Parameters:
    df (pd.DataFrame): The input dataset with missing values.

    Returns:
    pd.DataFrame: The dataset with missing values imputed.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_imputed = df.copy()

    # Encode categorical columns
    categorical_columns = ['loan_request_reason', 'applicant_job_type']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_imputed[col] = df_imputed[col].astype(str)  # Convert to string to handle NaN as a category
        df_imputed[col] = le.fit_transform(df_imputed[col])
        label_encoders[col] = le

    # Define column groups for different imputation methods
    columns_knn = [
        'mortgage_amount_due', 'property_current_value', 
        'loan_request_reason', 'applicant_job_type', 
        'age_of_oldest_credit_line', 'existing_credit_lines'
    ]
    columns_iterative = [
        'debt_to_income_ratio', 'major_derogatory_reports', 
        'delinquent_credit_lines', 'years_at_present_job', 
        'recent_credit_inquiries'
    ]

    # KNN Imputation for low/moderate missingness
    knn_imputer = KNNImputer(n_neighbors=5)
    df_imputed[columns_knn] = knn_imputer.fit_transform(df_imputed[columns_knn])

    # Iterative Imputer for high missingness
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    df_imputed[columns_iterative] = iterative_imputer.fit_transform(df_imputed[columns_iterative])

    # Return the fully imputed dataset
    return df_imputed
