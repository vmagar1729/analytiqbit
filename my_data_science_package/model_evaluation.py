
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from scipy.stats import chi2_contingency

def metrics_score(actual, predicted):
    """
    Calculates and displays classification metrics and confusion matrix.
    """
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Converted', 'Converted'],
                yticklabels=['Not Converted', 'Converted'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def chi_squared_test(data, target, bins=5):
    """
    Perform the Chi-Squared test for independence between all features
    and the target variable, handling both categorical and continuous features.

    Parameters:
        data (pd.DataFrame): The input dataset.
        target (str): The name of the target variable.
        bins (int): Number of bins to discretize continuous features. Default is 5.

    Returns:
        pd.DataFrame: A DataFrame containing features, their P-values, 
                      and whether they are significant.
    """
    # Separate target column
    target_data = data[target]
    features = data.drop(columns=[target])

    # List to store results for each feature
    results = []

    for feature in features.columns:
        if features[feature].dtype in ['object', 'category']:
            # Categorical feature: Use directly
            feature_data = features[feature]
        else:
            # Continuous feature: Discretize into bins
            feature_data = pd.cut(features[feature], bins=bins, labels=False)

        # Create a contingency table for the current feature and target
        contingency_table = pd.crosstab(feature_data, target_data)

        # Perform the Chi-Squared test for independence
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Append the feature and p-value to the results list
        results.append({'Feature': feature, 'P-value': p, 'Significant': p < 0.05})

    # Convert results to a DataFrame for better visualization
    results_df = pd.DataFrame(results)

    # Sort the results by P-value in ascending order
    sorted_results = results_df.sort_values(by='P-value', ascending=True)

    return sorted_results