
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


def chi_squared_test(data, cat_cols):
    # List to store results for each pair of features
    results = []

    # Loop through each pair of categorical columns
    for feature1, feature2 in combinations(cat_cols, 2):
        # Create a contingency table for the current pair of features
        contingency_table = pd.crosstab(data[feature1], data[feature2])

        # Perform the Chi-squared test for independence
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Append the feature pair and p-value to the results list
        results.append({'Feature 1': feature1, 'Feature 2': feature2, 'P-value': p})

    # Convert results to a DataFrame for better visualization
    results_df = pd.DataFrame(results)

    # Sort the results by P-value in ascending order
    sorted_results = results_df.sort_values(by='P-value', ascending=True)

    return sorted_results