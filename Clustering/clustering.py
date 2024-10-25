import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline
import plotly.express as px  # for 3D scatter plots
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# TODO: 1) Loops to check some agglomerative/dbscan values

# read dataset into a DataFrame
df = pd.read_csv("datasets/assessment_cluster_dataset.csv")


def describe_data_out(dfs, path, separator):
    dfs.describe().to_csv(path_or_buf=path, sep=separator)


def describe_data(dfs):
    print("Data Info:")
    dfs.info()
    print(f"Data Description: \n {dfs.describe()}")


def calc_variance(dfs):
    print(f"Data Variance: {dfs.var()}")


def null_check(dfs):
    print(f"Null Data Check: \n {dfs.isnull().sum()}")


def duplicate_check(dfs):
    print(f"Duplicate Data Check: \n {dfs[dfs.duplicated()]}")


def histogram(dfs):
    print("Histogram Plot")
    df.hist(figsize=(10, 7))
    plt.show()


def scatter_3d(dfs, x, y, z, title='3D Scatter Plot'):
    print(title)
    fig = px.scatter_3d(dfs, x=x, y=y, z=z)
    fig.update_layout(title=dict(text=title))
    plotly.offline.plot(fig)


def plot_elbow_method(dfs, range_from, range_to, **kmeans_kwargs):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(**kmeans_kwargs)
        kmeans.fit(dfs)
        sse.append(kmeans.inertia_)
    print(f"Elbow Method: {sse}")  # print inertia calculations
    # plot inertia
    plt.plot(range(range_from, range_to + 1), sse)
    plt.xticks(range(range_from, range_to + 1))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title("Elbow Method")
    plt.show()


def plot_kmeans_3d_scatter(dfs, x, y, z, title="3D Scatter Plot", display=True, **kmeans_kwargs):
    kmeans = KMeans(**kmeans_kwargs)
    results = kmeans.fit_predict(dfs[[x, y, z]])
    dfs['Cluster'] = results
    fig = px.scatter_3d(dfs, x=x, y=y, z=z, color='Cluster', title=title)
    if display:
        plotly.offline.plot(fig)
    return fig


def plot_agglomerative_3d_scatter(dfs, n_clusters, linkage, x, y, z, title="Agglomerative Clustering", display=True):
    if linkage not in ('complete', 'single', 'average', 'ward'):
        print(f"Invalid linkage: {linkage}")
    else:
        agg_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        agg_model.fit_predict(dfs)
        dfs['cluster'] = agg_model.labels_
        print(f"Number of datapoints per cluster: {linkage} \n {dfs.groupby('cluster')['cluster'].value_counts()}")
        fig = px.scatter_3d(dfs, x=x, y=y, z=z, color='cluster', title=title)
        if display:
            plotly.offline.plot(fig)
        return fig


def plot_dbscan_3d_scatter(dfs, x, y, z, eps=0.5, min_samples=5, title="DBSCAN 3D Scatter Plot", display=True):
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model.fit_predict(dfs)
    dfs['cluster'] = dbscan_model.labels_
    print(dfs.head())
    print(
        f"Number of datapoints per cluster: eps={eps} | min_samples={min_samples} \n {dfs.groupby('cluster')['cluster'].value_counts()}")
    fig = px.scatter_3d(dfs, x=x, y=y, z=z, color='cluster', title=title)
    if display:
        plotly.offline.plot(fig)
    return fig


def plot_optimal_eps(dfs, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(dfs)
    distances, indices = nbrs.kneighbors(dfs)
    print(distances)
    distances = np.sort(distances[:, n_neighbors - 1])  # sort  to the k-th nearest neighbors across the whole dataset
    plt.figure(figsize=(10, 6))
    plt.plot(distances, marker='o')
    plt.ylabel('k-distance')
    plt.xlabel('Points sorted by distance to k-th nearest neighbor')
    plt.title(f'k-Distance Plot for k={n_neighbors}')
    plt.grid = True
    plt.show()


def dbscan_gridsearch(dfs, eps_from, eps_to, eps_increment, min_samples_from, min_samples_to):
    warnings.filterwarnings("ignore")  # filter warnings from sklearn for when it can't assign a cluster
    param_grid = {
        'eps': np.arange(eps_from, eps_to, eps_increment),
        'min_samples': range(min_samples_from, min_samples_to)
    }
    dbscan = DBSCAN()
    grid_search = GridSearchCV(dbscan, param_grid, scoring=silhouette_score)
    grid_search.fit(dfs)
    print(f"DBSCAN Best Parameters: {grid_search.best_params_}")


def kmeans_gridsearch(dfs):
    param_grid = {
        'n_clustsers': range(2, 10),
        'init': ['k-means++', 'random'],
        'n_init': [5, 10, 15],
        'max_iter': [100, 200, 300, 400, 500],
        'random_state': [1, 16, 34, 57]
    }

    kmeans = KMeans()
    grid_search = GridSearchCV(kmeans, param_grid)
    grid_search.fit(dfs)
    print(f"KMeans Best Parameters: {grid_search.best_params_}")


def grid_search(dfs, param_grid, algo):
    if algo == 'kmeans':
        kmeans = KMeans()
        grid_search = GridSearchCV(kmeans, param_grid)
        grid_search.fit(dfs)
        print(f"KMeans Best Parameters: {grid_search.best_params_}")
    elif algo == 'agglomerative':
        aggl = AgglomerativeClustering()
        grid_search = GridSearchCV(dfs, param_grid)
        grid_search.fit(dfs)
        print(f"DBSCAN Best Parameters: {grid_search.best_params_}")
    elif algo == 'dbscan':
        dbscan = DBSCAN()
        grid_search = GridSearchCV(dbscan, param_grid)
        grid_search.fit(dfs)
        print(f"DBSCAN Best Parameters: {grid_search.best_params_}")
    else:
        print(f"ERROR: Invalid algorithm: {algo}")

# Display a grid of 3d scatter plots, 2 columns wide for an unknown number of plots
def multi_3d_scatter_plot(n_cluster_range, x, y, z, **kmeans_kwargs):
    num_plots = n_cluster_range[-1] - n_cluster_range[0] +1  # upper bound
    num_cols = 2  # always use 2 columns
    num_rows = (num_plots + num_cols - 1) // num_cols  # calculate number of rows rather than relying on parameter
    print(f"number rows: {num_rows}")
    num_cols=2
    num_rows=2
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=[f"Number of clusters: {i}" for i in
                        range(n_cluster_range[0], n_cluster_range[-1]+1)],
                        vertical_spacing=0.1,
                        specs=[[{'type': 'scatter3d'} for _ in range(num_cols)] for _ in range(num_rows)])


    fig.print_grid()
    subplot_index=1
    for i, n_clusters in enumerate(n_cluster_range, start=1):
        # kmeans_kwargs = {"n_clusters": n_clusters, "init": "random", "random_state": 1}
        kmeans_kwargs['n_clusters'] = n_clusters
        kmeans_fig = plot_kmeans_3d_scatter(scaled_df, x, y, z, f"3D Scatter Plot: init: {kmeans_kwargs['init']}<br>" 
                                             f"n_clusters: {kmeans_kwargs['n_clusters']}<br>"
                                                                    f"random_state: {kmeans_kwargs['random_state']}",

                                            **kmeans_kwargs)

        # Determine row & column based on the iteration index
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1
        fig.add_trace(kmeans_fig.data[0], row=row, col=col)

        scene_id = f'scene{i}'
        fig.update_layout({
            scene_id: dict(
                xaxis_title=f"{x}",
                yaxis_title=f"{y}",
                zaxis_title=f"{z}"
            )
        })



    fig.update_layout(height=600 * num_rows, width=600 * num_cols,
                      title_text="K-Means Clustering with Varying n_clusters")
    plotly.offline.plot(fig)

# unscaled data
# describe_data_out(df, "datasets/dataset_description.csv", ',')
# describe_data(df)
# with calc_variance(df)
# null_check(df)
# duplicate_check(df)
# histogram(df)
# scatter_3d(df, 'att1', 'att2', 'att3', '3D distribution of unscaled data')

# scaled data
scaled_df = StandardScaler().fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
# describe_data_out(scaled_df, "datasets/dataset_description.csv", ',')
# describe_data(scaled_df)
# calc_variance(scaled_df)
# null_check(scaled_df)
# duplicate_check(scaled_df)
# histogram(scaled_df)
# scatter_3d(scaled_df, 'att1', 'att2', 'att3', '3D distribution of scaled data')

# K-means Algorithm
#kmeans_kwargs = {"n_clusters": 4, "init": "k-means++", "n_init": 10, "random_state": 1}
# plot_elbow_method(scaled_df, 1, 10, **kmeans_kwargs)
# fig1 = plot_kmeans_3d_scatter(scaled_df, 'att1', 'att2', 'att3', "Scaled 3d k-means (k-means++) scatter plot",
#                       **kmeans_kwargs)

# kmeans_kwargs = {"init": "k-means++", "random_state": 57, "max_iter":100, "n_clusters": 9, "n_init": 5}

# fig2 = plot_kmeans_3d_scatter(scaled_df, 'att1', 'att2', 'att3', "Scaled 3d k-means (random) scatter plot", **kmeans_kwargs)


# k-means multi plot with different states
my_kmeans_kwargs = {"init": "random", "random_state": 1, "max_iter": 100, "n_init": 1}
cluster_range = np.arange(3, 7)
multi_3d_scatter_plot(cluster_range, 'att1', 'att2', 'att3', display=True, **my_kmeans_kwargs)



# Agglomerative/Divisive Clustering
# single linkage - not biased for globular shapes
# complete linkage - biased for globular but sensitive to noise
# Average/Centroid/Ward - biased for globular clusters but not as senstive to noise - potential option

# plot_agglomerative_3d_scatter(scaled_df, 3, 'complete', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Complete] -'
#                                                                                ' scaled dataset')

# plot_agglomerative_3d_scatter(scaled_df, 3, 'single', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Single] -'
#                                                                                ' scaled dataset')

# plot_agglomerative_3d_scatter(scaled_df, 3, 'average', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Average] -'
#                                                                                ' scaled dataset')

# plot_agglomerative_3d_scatter(scaled_df, 3, 'ward', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Ward] -'
#                                                                                ' scaled dataset')


# DBScan
# Note: Below is from a subjective perspective a decent clustering of the data, eps=0.3, min_samples=5
# plot_dbscan_3d_scatter(dfs=scaled_df, x='att1', y='att2', z='att3', eps=0.3, min_samples=5)

# plot_optimal_eps(scaled_df, 5)

# dbscan_gridsearch(scaled_df, eps_from=0.5, eps_to=2, eps_increment=0.25, min_samples_from=5, min_samples_to=10)

param_grid = {
    'n_clusters': range(2, 10),
    'init': ['k-means++', 'random'],
    'n_init': [5, 10, 15],
    'max_iter': [100, 200, 300, 400, 500],
    'random_state': [1, 16, 34, 57]
}
# grid_search(scaled_df, param_grid, 'kmeans')

param_grid = {
    'eps': np.arange(0.5, 2, 0.25),
    'min_samples': range(2, 5),
    'scoring': ['silhouette_score']
}

"""



#create a dataframe with the centroids
centroid_table = pd.DataFrame(krandom_model.cluster_centers_, columns=scaled_df.columns)
centroid_table['cluster'] = ['centroid 0', 'centroid 1', 'centroid 2']
print(centroid_table.head())
#plot the centroids
fig = px.parallel_coordinates(centroid_table, 'cluster')
plt.show()


#Plot Centroids
centroid_table = pd.DataFrame(results.cluster_centers_, columns=scaled_df.columns)
print(centroid_table)
"""

# 1. Understand the Business Requirement
# 2. Understand the data
# 3. Prepare the data
# 5. Prepare the model (k-means, hierarchical clustering or DBScan - suggested, pick 2)
# 5. Evaluate
