# Ignore warning messages only
import warnings
warnings.filterwarnings("ignore")

# Importing the Pandas library for data manipulation and analysis
import pandas as pd

# Importing the NumPy library for numerical operations
import numpy as np

# Importing Matplotlib for data visualization
import matplotlib.pyplot as plt

# Importing Seaborn for enhanced data visualization
import seaborn as sns

from scipy.stats import chi2_contingency

# Importing accuracy_score, confusion_matrix, and classification_report for model evaluation and plot_tree to visualize the tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.tree import plot_tree
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Required to use IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

def summarize_dataframe(sdf):
    """
    Summarizes the structure and content of a DataFrame, including column names,
    non-null counts, data types, unique values, missing values, and percentage of missing data.

    Parameters:
    df (pd.DataFrame): The DataFrame to summarize.

    Returns:
    pd.DataFrame: A summary DataFrame containing details for each column.
    """
    summary_df = pd.DataFrame({
        'Column': sdf.columns,
        'Non-Null Count': sdf.notnull().sum(),
        'Data Type': sdf.dtypes,
        'Unique Values': sdf.nunique(),
        'Missing Values': sdf.isnull().sum(),
        'Missing Values %': ((sdf.isnull().sum()*100)/len(sdf)).round(2)
    }).reset_index(drop=True)
    return summary_df


def histogram_boxplot(data, feature, figsize=(12, 7), kde=True, bins=None):
    """
    Boxplot and histogram combined with automatic bins and x-axis ticks.
    Ensures the x-axis starts at zero.

    data: DataFrame
    feature: Column in the DataFrame
    figsize: Size of the figure (default (12, 7))
    kde: Whether to show density curve (default True)
    bins: Number of bins or sequence of bin edges for histogram (default None)
    """
    # Create the subplots for boxplot and histogram
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,      # Number of rows of the subplot grid = 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )

    # Boxplot with the mean indicated
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")

    # Calculate automatic bins if not provided
    if bins is None:
        q75, q25 = data[feature].quantile([0.75, 0.25])  # Interquartile range
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data[feature]) ** (1 / 3))  # Freedman-Diaconis rule
        bins = max(1, int((data[feature].max() - data[feature].min()) / bin_width))  # Ensure at least 1 bin

    # Create the histogram
    sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter")

    # Add mean and median to the histogram
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--", label="Mean")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-", label="Median")
    ax_hist2.legend()

    # Ensure x-axis starts at 0
    ax_hist2.set_xlim(left=0)  # Set the minimum x-axis value to 0

    # Automatically adjust x-axis ticks
    x_min, x_max = ax_hist2.get_xlim()  # Get the updated limits of the x-axis
    ticks = np.linspace(x_min, x_max, num=10)  # Generate evenly spaced ticks
    ax_hist2.set_xticks(ticks)  # Set the new ticks

    plt.show()


def plot_barplots(data, variables, target_variable, nrows=3, ncols=2, figsize=(18, 10)):
    """
    Create bar plots showing the sum of a target variable grouped by each variable in the list.

    Parameters:
    - data (DataFrame): The DataFrame containing the data.
    - variables (list): List of variables to plot.
    - target_variable (str): The target variable to aggregate.
    - nrows (int): Number of rows in the subplot grid.
    - ncols (int): Number of columns in the subplot grid.
    - figsize (tuple): Figure size for the plot.
    """
    # Create subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()  # Flatten the array for easier indexing

    for i, var in enumerate(variables):
        if i >= len(axes):
            break  # Avoid indexing errors if there are more variables than subplots

        # Calculate the sum of the target variable for each category of the variable
        var_sums = data.groupby(var)[target_variable].sum().reset_index()

        # Create the barplot
        sns.barplot(x=var, y=target_variable, data=var_sums, ax=axes[i])
        axes[i].set_title(f'Sum of {target_variable} by {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(f'Sum of {target_variable}')
        axes[i].tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability

    # Remove any extra subplots if the number of variables is less than the grid size
    for j in range(len(variables), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


def plot_binned_features(data, features, status, bins=5, title=None):
    """
    Plots subplots for binned bar charts of multiple features against a status variable.

    Parameters:
        data (pd.DataFrame): The dataset containing the features and status.
        features (list): A list of continuous variables to be binned and plotted.
        status (str): The variable to sum for each bin, plotted on the y-axis.
        bins (int): Number of bins to create for the features.
        title (str, optional): Title of the overall plot. Default is None.

    Returns:
        None: Displays the subplots.
    """
    # Create a figure with subplots
    num_features = len(features)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, 5 * num_features))
    fig.suptitle(title if title else f"Features vs {status}", fontsize=16)

    # Ensure axes is iterable for a single feature
    if num_features == 1:
        axes = [axes]

    # Iterate over features and create plots
    for ax, feature in zip(axes, features):
        # Group by feature and calculate the sum of the status variable
        grouped_data = data.groupby(feature)[status].sum().reset_index()

        # Create bins for the feature
        grouped_data['feature_bin'] = pd.cut(grouped_data[feature], bins=bins)

        # Group by bins and calculate the sum of the status variable
        binned_data = grouped_data.groupby('feature_bin')[status].sum().reset_index()

        # Generate bin labels
        bin_edges = grouped_data['feature_bin'].cat.categories
        bin_labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in bin_edges]

        # Bar plot
        sns.barplot(data=binned_data, x='feature_bin', y=status, ax=ax)

        # Add title and labels
        ax.set_title(f"Leads converted by {feature}")
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel(f"Converted")

        # Set x-axis tick labels
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=90)

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def chi_squared_test(data, cat_cols, target_col):
    # List to store results for each feature
    results = []

    # Loop through each categorical column in the dataset
    for feature in cat_cols:
        # Create a contingency table for the current feature and the target column
        contingency_table = pd.crosstab(data[feature], data[target_col])

        # Perform the chi-squared test for independence
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Append the feature name and p-value to the results list
        results.append({'Feature': feature, 'P-value': p})

    # Convert results to a DataFrame for better visualization
    results_df = pd.DataFrame(results)

    # Sort the results by P-value in ascending order
    sorted_results = results_df.sort_values(by='P-value', ascending=True)

    return sorted_results


def metrics_score(actual, predicted):
    """
    Calculates and displays various classification metrics and a confusion matrix.

    Args:
      actual: The actual target values.
      predicted: The predicted target values.
    """
    
    # Print the classification report
    print(classification_report(actual, predicted))

    # Generate the confusion matrix
    cm = confusion_matrix(actual, predicted)

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', 
                xticklabels=['Not Converted', 'Converted'], 
                yticklabels=['Not Converted', 'Converted'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def model_performance_classification(model, predictors, target):
    """
    Computes and returns classification performance metrics as a DataFrame.

    Args:
      model: The trained classification model.
      predictors: The independent variables used for prediction.
      target: The dependent variable (actual values).

    Returns:
      A pandas DataFrame containing precision, recall, and accuracy.
    """

    # Predict using the model
    pred = model.predict(predictors)

    # Calculate the metrics
    recall = recall_score(target, pred, average='macro')
    precision = precision_score(target, pred, average='macro')
    acc = accuracy_score(target, pred)

    # Create a DataFrame of the metrics
    df_perf = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "Accuracy": acc
    }, index=[0])

    return df_perf


def plot_feature_importance(importances, columns):
    """
    Plots feature importances as a bar chart.

    Args:
      importances: A list or array of feature importances.
      columns: A list of feature names corresponding to the importances.
    """

    # Create a DataFrame for the feature importances
    importance_df = pd.DataFrame(importances, index=columns, columns=['Importance']).sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(13, 13))
    sns.barplot(x=importance_df.Importance, y=importance_df.index, palette="magma") 
    plt.title("Feature Importances", fontsize=16)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.show()



def visualize_decision_tree(dt, X, max_depth=4):
    """
    Visualizes a decision tree using matplotlib.

    Args:
      dt: The trained decision tree model.
      X: The DataFrame or array containing the features used for training.
      max_depth (int): The maximum depth of the tree to display (default: 4).
    """

    plt.figure(figsize=(30, 20))
    plot_tree(dt, 
              max_depth=max_depth, 
              feature_names=list(X.columns), 
              filled=True, 
              fontsize=12, 
              node_ids=True, 
              class_names=None)
    plt.show()


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


# 1. Before vs. After Distribution
def plot_distributions(column, original_data, imputed_data):
    """
    Plots the distributions of a feature before and after imputation.

    Parameters:
    column (str): The column name to plot.
    original_data (DataFrame): The original dataset with missing values.
    imputed_data (DataFrame): The dataset after imputation.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(original_data[column], label="Before Imputation", fill=True, alpha=0.5, color="blue")
    sns.kdeplot(imputed_data[column], label="After Imputation", fill=True, alpha=0.5, color="green")
    plt.title(f'Distribution of {column}: Before vs. After Imputation', fontsize=14)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# # Columns to visualize
# columns_to_plot = ['mortgage_amount_due', 'debt_to_income_ratio', 'major_derogatory_reports', 
#                    'delinquent_credit_lines', 'years_at_present_job', 'property_current_value']

# for column in columns_to_plot:
#     if column in df_original.columns:
#         plot_distributions(column, df_original, df_imputed)

# 2. Correlation Heatmaps
def plot_correlation_heatmap(data, title):
    """
    Plots the correlation heatmap of the dataset.

    Parameters:
    data (DataFrame): The dataset to plot.
    title (str): The title of the heatmap.
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title, fontsize=16)
    plt.show()

# # Correlation heatmaps before and after imputation
# plot_correlation_heatmap(df_original, "Correlation Matrix: Before Imputation")
# plot_correlation_heatmap(df_imputed, "Correlation Matrix: After Imputation")

# 3. Missingness Heatmap
def plot_missingness_heatmap(data, title):
    """
    Plots a heatmap of missing values in the dataset.

    Parameters:
    data (DataFrame): The dataset to plot.
    title (str): The title of the heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title(title, fontsize=16)
    plt.show()

# # Missingness before and after
# plot_missingness_heatmap(df_original, "Missingness Heatmap: Before Imputation")
# plot_missingness_heatmap(df_imputed, "Missingness Heatmap: After Imputation")


def treat_skewness(df, columns, method='log'):
    """
    Reduces skewness in specified columns using transformations.
    
    Parameters:
    df (pd.DataFrame): The dataset.
    columns (list): Columns to treat for skewness.
    method (str): Transformation method ('log', 'sqrt', or 'boxcox').
    
    Returns:
    pd.DataFrame: Dataset with treated skewness.
    """
    df = df.copy()
    
    for col in columns:
        if method == 'log':
            # Apply log transformation (add 1 to avoid log(0))
            df[col] = np.log1p(df[col])
        elif method == 'sqrt':
            # Apply square root transformation
            df[col] = np.sqrt(df[col])
        elif method == 'boxcox':
            from scipy.stats import boxcox
            # Apply Box-Cox transformation (add 1 if necessary)
            df[col], _ = boxcox(df[col] + 1 if (df[col] <= 0).any() else df[col])
        else:
            raise ValueError("Unsupported method. Choose from 'log', 'sqrt', or 'boxcox'.")
    
    return df



# # Treat skewness with log transformation
# skewed_columns = ['mortgage_amount_due', 'property_current_value', 'debt_to_income_ratio']
# df_skew_treated = treat_skewness(df_new, skewed_columns, method='log')



def treat_outliers(df, columns, lower_percentile=0.01, upper_percentile=0.99):
    """
    Caps outliers in specified columns to the given percentiles.
    
    Parameters:
    df (pd.DataFrame): The dataset.
    columns (list): Columns to treat for outliers.
    lower_percentile (float): Lower capping threshold (e.g., 0.01 for 1st percentile).
    upper_percentile (float): Upper capping threshold (e.g., 0.99 for 99th percentile).
    
    Returns:
    pd.DataFrame: Dataset with treated outliers.
    """
    df = df.copy()
    
    for col in columns:
        lower_bound = df[col].quantile(lower_percentile)
        upper_bound = df[col].quantile(upper_percentile)
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    return df



# # Treat outliers by capping at 1st and 99th percentiles
# outlier_columns = ['mortgage_amount_due', 'property_current_value', 'age_of_oldest_credit_line', 'debt_to_income_ratio']
# df_outliers_treated = treat_outliers(df_skew_treated, outlier_columns, lower_percentile=0.01, upper_percentile=0.99)


def segregate_columns_by_dtype(df):
    """
    Segregates DataFrame columns by their data types into separate lists.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    dict: A dictionary with data type names as keys and lists of column names as values.
    """
    column_types = {}
    for dtype in df.dtypes.unique():
        dtype_name = dtype.name
        column_types[dtype_name] = df.select_dtypes(include=[dtype]).columns.tolist()
    return column_types
