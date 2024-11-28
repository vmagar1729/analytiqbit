
import pandas as pd

def segregate_columns_by_dtype(df):
    """
    Segregates DataFrame columns by their data types.
    """
    return {dtype.name: df.select_dtypes(include=[dtype]).columns.tolist() for dtype in df.dtypes.unique()}
