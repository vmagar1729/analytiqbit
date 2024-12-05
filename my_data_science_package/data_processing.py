import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from fancyimpute import KNN


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
    using LabelEncoder and can be converted back to their original values.

    Parameters:
    df (pd.DataFrame): The input dataset with missing values.

    Returns:
    pd.DataFrame: The dataset with missing values imputed.
    dict: A dictionary of LabelEncoder objects for categorical columns.
    """

# Column segregation
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
    
    # Encode categorical columns for KNN
    categorical_columns = ['loan_request_reason', 'applicant_job_type']
    # Convert object columns to category dtype before encoding
    original_categories = {}
    for col in categorical_columns:
        df[col] = df[col].astype('category')  # Convert to category
        original_categories[col] = df[col].cat.categories  # Save the original categories

    for col in categorical_columns:
        df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)  # Encode categories and handle NaN
    
    # Impute with KNN
    knn_imputer = KNN(k=5, verbose=False)
    df[columns_knn] = knn_imputer.fit_transform(df[columns_knn])
    
    # Impute with IterativeImputer
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42, verbose=0)
    df[columns_iterative] = iterative_imputer.fit_transform(df[columns_iterative])

    # Return the fully imputed dataset and label encoders
    return df, original_categories


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
