# Ignore warning messages
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