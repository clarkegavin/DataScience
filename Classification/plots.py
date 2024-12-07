import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
import matplotlib.pyplot as plt


def histogram(dfs, title="Histogram"):
    """ Custom function to display a histogram using plotly"""
    print("Histogram Plot")
    dfs.hist(figsize=(10, 7))
    plt.title(title)
    plt.show()


def custom_plot_tree(dt_clf, df):
    plt.figure(figsize=(25, 16))
    print(f"Decision Tree Classes: \n {dt_clf.classes_} ")
    plot_tree(dt_clf, filled=True, feature_names=list(df.columns), class_names=dt_clf.classes_)
    plt.show()


def custom_scatter_plot(df, x, y, class_label):
    sns.scatterplot(data=df, x=x, y=y, hue=class_label)
    plt.title('Relationship between Age, Hours Per Week, and Label')
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def scatter_plot_matrix(df, hue=None, figsize=(15, 15), marker='o', alpha=0.7):
    """
    Creates a matrix of 2D scatter plots for all combinations of numerical attributes in the DataFrame.

    :param df: Pandas DataFrame containing both numerical and categorical data.
    :param hue: Column name for coloring points (optional, can be categorical or numerical).
    :param figsize: Tuple indicating the size of the overall plot.
    :param marker: Marker style for scatter plots.
    :param alpha: Transparency level for scatter points.
    :return: None (plots the scatter matrix).
    """
    # Select numerical columns only
    numerical_columns = df.select_dtypes(include=['number']).columns

    if len(numerical_columns) < 2:
        print("Not enough numerical columns to create scatter plot matrix.")
        return

    # Encode categorical hue if necessary
    if hue is not None and df[hue].dtype == 'object':
        df[hue] = df[hue].astype('category').cat.codes

    # Set up the figure
    num_cols = len(numerical_columns)
    fig, axes = plt.subplots(num_cols, num_cols, figsize=figsize, sharex=False, sharey=False)

    for i, col1 in enumerate(numerical_columns):
        for j, col2 in enumerate(numerical_columns):
            ax = axes[i, j]

            if i == j:
                # Diagonal plots show histograms of individual variables
                sns.histplot(df[col1], ax=ax, kde=True, color="skyblue")
                ax.set_ylabel('Frequency')
            else:
                # Scatter plot for combinations
                sns.scatterplot(data=df, x=col2, y=col1, hue=hue, marker=marker, alpha=alpha, ax=ax,
                                legend=(i == 0 and j == num_cols - 1))

            # Set axis labels only on edges
            if j == 0:
                ax.set_ylabel(col1)
            else:
                ax.set_ylabel('')

            if i == num_cols - 1:
                ax.set_xlabel(col2)
            else:
                ax.set_xlabel('')

    # Adjust layout and show
    plt.tight_layout()
    plt.show()


# Example usage:
# scatter_plot_matrix(df, hue='label', figsize=(10, 10))


def plot_correlation_matrix(df):
    plt.matshow(df.corr())
    plt.show()


def plot_importance(cols, scores):
    plt.figure(figsize=(10, 6))
    plt.barh(cols, scores)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Permutation Feature Importance')
    plt.show()
