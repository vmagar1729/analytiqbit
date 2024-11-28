
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
    Imputes missing values using KNN and Iterative Imputer.
    """
    df_imputed = df.copy()
    categorical_columns = ['loan_request_reason', 'applicant_job_type']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    for col, le in label_encoders.items():
        df_imputed[col] = le.fit_transform(df_imputed[col].astype(str))
    knn_imputer = KNNImputer(n_neighbors=5)
    iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
    return df_imputed
