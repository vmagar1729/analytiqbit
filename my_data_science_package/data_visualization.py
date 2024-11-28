
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def histogram_boxplot(data, feature, figsize=(12, 7), kde=True, bins=None):
    """
    Boxplot and histogram combined for a feature.
    """
    f, (ax_box, ax_hist) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)}, figsize=figsize
    )
    sns.boxplot(data=data, x=feature, ax=ax_box, showmeans=True, color="violet")
    sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist, bins=bins, palette="winter")
    plt.show()

def plot_correlation_heatmap(data, title):
    """
    Plots a correlation heatmap of the dataset.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(title, fontsize=16)
    plt.show()
