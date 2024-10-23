import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline
import plotly.express as px  # for 3D scatter plots
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# TODO: 1) Loops to check some k-means/agglomerative/dbscan values and 2) grid search for k-means and agglomerative

# read dataset into a DataFrame
df = pd.read_csv("datasets/assessment_cluster_dataset.csv")


def describe_data_out(dfs, path, separator):
    dfs.describe().to_csv(path_or_buf=path, sep=separator)


def describe_data(dfs):
    print("Data Info:")
    dfs.info()
    print(f"Data Description: \n {dfs.describe()}")


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
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
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


def plot_kmeans_3d_scatter(k, dfs, x, y, z, title="3D Scatter Plot", **kmeans_kwargs):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    results = kmeans.fit_predict(dfs[[x, y, z]])
    dfs['Cluster'] = results
    fig = px.scatter_3d(dfs, x=x, y=y, z=z, color='Cluster', title=title)
    plotly.offline.plot(fig)


def plot_agglomerative_3d_scatter(dfs, n_clusters, linkage, x, y, z, title="Agglomerative Clustering"):
    if linkage not in ('complete', 'single', 'average', 'ward'):
        print(f"Invalid linkage: {linkage}")
    else:
        agg_model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        agg_model.fit_predict(dfs)
        dfs['cluster'] = agg_model.labels_
        print(f"Number of datapoints per cluster: {linkage} \n {dfs.groupby('cluster')['cluster'].value_counts()}")
        fig = px.scatter_3d(dfs, x=x, y=y, z=z, color='cluster', title=title)
        plotly.offline.plot(fig)


def plot_dbscan_3d_scatter(dfs, x, y, z, eps=0.5, min_samples=5, title="DBSCAN 3D Scatter Plot"):
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model.fit_predict(dfs)
    dfs['cluster'] = dbscan_model.labels_
    print(dfs.head())
    print(f"Number of datapoints per cluster: eps={eps} | min_samples={min_samples} \n {dfs.groupby('cluster')['cluster'].value_counts()}")
    fig = px.scatter_3d(dfs, x=x, y=y, z=z, color='cluster', title=title)
    plotly.offline.plot(fig)


def plot_optimal_eps(dfs, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(dfs)
    distances, indices = nbrs.kneighbors(dfs)
    print(distances)
    distances = np.sort(distances[:, n_neighbors-1])  # sort  to the k-th nearest neighbors across the whole dataset
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



# unscaled data
# describe_data_out(df, "datasets/dataset_description.csv", ',')
# describe_data(df)
# null_check(df)
# duplicate_check(df)
# histogram(df)
# scatter_3d(df, 'att1', 'att2', 'att3', '3D distribution of unscaled data')

# scaled data
scaled_df = StandardScaler().fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
# describe_data_out(scaled_df, "datasets/dataset_description.csv", ',')
# describe_data(scaled_df)
# null_check(scaled_df)
# duplicate_check(scaled_df)
# histogram(scaled_df)
# scatter_3d(scaled_df, 'att1', 'att2', 'att3', '3D distribution of scaled data')

# K-means Algorithm
# kmeans_kwargs = {"init": "k-means++", "n_init": 10, "random_state": 1}
# plot_elbow_method(scaled_df, 1, 10, **kmeans_kwargs)
# plot_kmeans_3d_scatter(4, scaled_df, 'att1', 'att2', 'att3', "Scaled 3d k-means (k-means++) scatter plot",
#                       **kmeans_kwargs)

#kmeans_kwargs = {"init": "random", "random_state": 1}
#plot_kmeans_3d_scatter(4, scaled_df, 'att1', 'att2', 'att3', "Scaled 3d k-means (random) scatter plot", **kmeans_kwargs)

# Agglomerative/Divisive Clustering
# single linkage - not biased for globular shapes
# complete linkage - biased for globular but sensitive to noise
# Average/Centroid/Ward - biased for globular clusters but not as senstive to noise - potential option

#plot_agglomerative_3d_scatter(scaled_df, 3, 'complete', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Complete] -'
#                                                                                ' scaled dataset')

#plot_agglomerative_3d_scatter(scaled_df, 3, 'single', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Single] -'
#                                                                                ' scaled dataset')

#plot_agglomerative_3d_scatter(scaled_df, 3, 'average', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Average] -'
#                                                                                ' scaled dataset')

#plot_agglomerative_3d_scatter(scaled_df, 3, 'ward', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Ward] -'
#                                                                                ' scaled dataset')


# DBScan
# Note: Below is from a subjective perspective a decent clustering of the data, eps=0.3, min_samples=5
# plot_dbscan_3d_scatter(dfs=scaled_df, x='att1', y='att2', z='att3', eps=0.3, min_samples=5)

# plot_optimal_eps(scaled_df, 5)

dbscan_gridsearch(scaled_df, eps_from=0.5, eps_to=2, eps_increment=0.25, min_samples_from=5, min_samples_to=10)

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
