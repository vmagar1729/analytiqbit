import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def histogram_boxplot(data, feature, figsize=(12, 7), kde=True, bins=None):
    """
    Boxplot and histogram combined with automatic bins and x-axis ticks.
    Ensures the x-axis starts at zero.

    Parameters:
    - data: DataFrame
    - feature: Column in the DataFrame
    - figsize: Size of the figure (default (12, 7))
    - kde: Whether to show density curve (default True)
    - bins: Number of bins or sequence of bin edges for histogram (default None)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

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
        if len(data[feature]) == 0:  # Handle empty feature
            raise ValueError(f"Feature '{feature}' contains no data.")
        if data[feature].max() - data[feature].min() == 0:  # Handle zero range
            bins = 1
        else:
            bin_width = 2 * iqr / (len(data[feature]) ** (1 / 3))
            if bin_width <= 0:  # Handle zero or negative bin width
                bin_width = 1  # Default fallback
            bins = int((data[feature].max() - data[feature].min()) / bin_width)
            bins = max(1, bins)  # Ensure at least 1 bin

    # Create the histogram
    sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, color="blue")

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
    # Calculate the grid dimensions
    n_rows, n_cols = 6, 2  # Fixed 4 rows and 3 columns
    total_subplots = n_rows * n_cols
    num_features = len(features)

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 15))
    fig.suptitle(title if title else f"Features vs {status}", fontsize=16)

    # Flatten axes for easy indexing
    axes = axes.flatten()

    # Iterate over features and create plots
    for i, feature in enumerate(features):
        ax = axes[i]

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
        ax.set_title(f"{feature} vs {status}")
        ax.set_xlabel(f"{feature}")
        ax.set_ylabel(f"{status}")

        # Set x-axis tick labels
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45)
        total = grouped_data[status].sum()
        for bar in ax.patches:
            height = bar.get_height()
            percentage = (height / total) * 100  # Calculate percentage
            if percentage > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
        sns.despine()
        
    # Turn off extra subplots
    for i in range(num_features, total_subplots):
        axes[i].axis('off')

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Example usage
# df = pd.DataFrame({
#     "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
#     "feature2": [10, 20, 30, 40, 50, 60, 70, 80],
#     "feature3": [5, 4, 3, 2, 1, 0, 1, 2],
#     "status": [0, 1, 0, 1, 0, 1, 1, 0]
# })
# plot_binned_features(data=df, features=["feature1", "feature2", "feature3"], status="status", bins=3)


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
