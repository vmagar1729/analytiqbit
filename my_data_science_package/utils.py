
import pandas as pd
from IPython.display import Markdown, display

def segregate_columns_by_dtype(df):
    """
    Segregates DataFrame columns by their data types.
    """
    return {dtype.name: df.select_dtypes(include=[dtype]).columns.tolist() for dtype in df.dtypes.unique()}

def add_markdown_to_live_notebook(markdown_content):
    """
    Adds a markdown cell to the currently running Jupyter Notebook.

    Parameters:
    markdown_content (str): Content of the markdown cell.

    Returns:
    None
    """
    display(Markdown(markdown_content))
