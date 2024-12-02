import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


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


from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Needed for IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder


def impute_missing_values(df):
    """
    Imputes missing values in a dataset using KNN for low/moderate missingness 
    and Iterative Imputer for high missingness. Categorical columns are encoded 
    using LabelEncoder and can be converted back to their original values.

    Parameters:
    df (pd.DataFrame): The input dataset with missing values.

    Returns:
    pd.DataFrame: The dataset with missing values imputed.
    dict: A dictionary of LabelEncoder objects for categorical columns.
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
    df_imputed['loan_request_reason'] = df_imputed['loan_request_reason'].round().astype(int)
    df_imputed['applicant_job_type'] = df_imputed['applicant_job_type'].round().astype(int)


    # Iterative Imputer for high missingness
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    df_imputed[columns_iterative] = iterative_imputer.fit_transform(df_imputed[columns_iterative])

    # Return the fully imputed dataset and label encoders
    return df_imputed, label_encoders


def decode_categorical_columns(df, label_encoders):
    """
    Decodes categorical columns back to their original values using LabelEncoder objects.

    Parameters:
    df (pd.DataFrame): The DataFrame with encoded categorical columns.
    label_encoders (dict): A dictionary of LabelEncoder objects for each categorical column.

    Returns:
    pd.DataFrame: The DataFrame with categorical columns decoded to their original values.
    """
    df_decoded = df.copy()
    for col, le in label_encoders.items():
        df_decoded[col] = le.inverse_transform(df_decoded[col].astype(int))
    return df_decoded


def calculate_corr_and_vif(dataframe):
    """
    Calculate the correlation matrix and Variance Inflation Factor (VIF)
    for numerical predictor columns in a given DataFrame, excluding the target variable.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame.
    target_variable (str): The name of the target variable to exclude.

    Returns:
    tuple: A tuple containing the correlation matrix (pd.DataFrame)
           and VIF values (pd.DataFrame).
    """
    
    # Select only numerical columns
    numeric_df = dataframe.select_dtypes(include=[np.number])
    
    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Calculate VIF for each numerical column
    vif_data = pd.DataFrame()
    vif_data["Feature"] = numeric_df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(numeric_df.values, i)
        for i in range(numeric_df.shape[1])
    ]
    
    return corr_matrix, vif_data
