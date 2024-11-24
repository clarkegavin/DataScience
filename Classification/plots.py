import matplotlib.pyplot as plt


def histogram(dfs, title="Histogram"):
    """ Custom function to display a histogram using plotly"""
    print("Histogram Plot")
    dfs.hist(figsize=(10, 7))
    plt.title(title)
    plt.show()
