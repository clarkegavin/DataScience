
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline
import plotly.express as px  # for 3D scatter plots
from plotly.subplots import make_subplots
from scipy.spatial.distance import cityblock
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
from itertools import product


# read dataset into a DataFrame
df = pd.read_csv("datasets/assessment_cluster_dataset.csv")


def describe_data_out(dfs, path, separator):
    """ Custom function to print describe information to csv file"""
    dfs.describe().to_csv(path_or_buf=path, sep=separator)


def data_out(dfs, path, separator):
    """ Custom function to print dataframe to csv file"""
    dfs.to_csv(path_or_buf=path, sep=separator)


def describe_data(dfs):
    """ Custom function to print dataframe description to console"""
    print("Data Info:")
    dfs.info()
    print(f"Data Description: \n {dfs.describe()}")


def calc_variance(dfs):
    """ Custom function to print dataframe variance to console"""
    print(f"Data Variance: {dfs.var()}")


def null_check(dfs):
    """ Custom function to print dataframe null value check to console"""
    print(f"Null Data Check: \n {dfs.isnull().sum()}")


def duplicate_check(dfs):
    """ Custom function to print dataframe data duplication to console"""
    print(f"Duplicate Data Check: \n {dfs[dfs.duplicated()]}")


def histogram(dfs):
    """ Custom function to display a histogram using plotly"""
    print("Histogram Plot")
    dfs.hist(figsize=(10, 7))
    plt.show()


def scatter_3d(dfs, x, y, z, title='3D Scatter Plot'):
    """ Custom function to display 3D scatter graphy using plotly"""
    print(title)
    fig = px.scatter_3d(dfs, x=x, y=y, z=z)
    fig.update_layout(title=dict(text=title))
    plotly.offline.plot(fig, "3D Scatter Plot.html")


def plot_elbow_method(dfs, range_from, range_to, **kmeans_kwargs):
    """ Custom function to plot elbow method for k-means using plotly """
    sse = []
    for k in range(range_from, range_to):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(dfs)
        sse.append(kmeans.inertia_)
    print(f"Elbow Method: {sse}")  # print inertia calculations
    # plot inertia
    plt.plot(range(range_from, range_to), sse)
    plt.xticks(range(range_from, range_to))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title("Elbow Method")
    plt.show()


def plot_kmeans_3d_scatter(dfs, x, y, z, title="3D Scatter Plot", display_individual_plot=True, plot_centroids=True,
                           plot_iter=0, display_davies_bouldin_index=True, display_dunn_index=False, **kmeans_kwargs):
    """ Custom function to
        1. Display a 3D scatter plot using plotly for k-means
        2. Calculate centroids and plot centroids using plotly
        3. Display the dunn index and print to console
        4. Display the Davies Bouldin index and print to console
    """
    kmeans = KMeans(**kmeans_kwargs)
    results = kmeans.fit_predict(dfs[[x, y, z]])
    dfs['Cluster'] = results
    fig = px.scatter_3d(dfs, x=x, y=y, z=z, color='Cluster', title=title)
    if display_individual_plot:
        plotly.offline.plot(fig, filename=f"Kmeans 3D Scatter Plot {plot_iter}.html")
    if plot_centroids:
        # results_df = pd.DataFrame(results, columns=['Cluster'])
        centroid_table = parallel_centroid_plot(dfs, kmeans, plot_iter, **kmeans_kwargs)[[x, y, z]]
        if display_dunn_index:
            labels = kmeans.labels_
            print(f"Dunn Index: {dunn_index(dfs, centroid_table, labels)}")
    if display_davies_bouldin_index:
        calc_davies_bouldin_index(dfs, kmeans)
    return fig


def plot_agglomerative_3d_scatter(dfs, n_clusters, linkage, x, y, z, title="Agglomerative Clustering", display=True):
    """ Custom function to plot 3D scatter plot of agglomerative clustering """
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


def plot_dbscan_3d_scatter(dfs, x, y, z, eps=0.5, min_samples=5, plot_iter=0, title="DBSCAN 3D Scatter Plot",
                           display=True, show_centroids=False):
    """ Custom function to
        1. plot 3D scatter plot of DBSCAN algorithm
        2. Print Dunn Index for DBSCAN to console
        3. Print Davies-Bouldin Index for DBSCAN to console
    """

    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model.fit_predict(dfs)
    dfs['cluster'] = dbscan_model.labels_
    # print(dfs.head())
    data_out(dfs, 'datasets/dbscan.csv', ',')
    print(
        f"Number of datapoints per cluster: eps={eps} | min_samples={min_samples} "
        f"\n {dfs.groupby('cluster')['cluster'].value_counts()}")
    fig = px.scatter_3d(dfs, x=x, y=y, z=z, color='cluster', title=f"{title} <br> <span style = 'font-size: 10px'>"
                                                                   f"eps={eps} | min_samples={min_samples}</span>")

    if show_centroids:
        # print(get_centroid_table(dfs, dbscan_model))
        print(f"Dunn Index: {dunn_index(dfs, get_centroid_table(dfs, dbscan_model), dbscan_model.labels_)}")
        print(f"Davies-Bouldin Score: {davies_bouldin_score(dfs, dbscan_model.labels_)}")

    if display:
        plotly.offline.plot(fig, filename=f'DBSCAN 3D Scatter Plot {plot_iter}.html')
    return fig


def plot_optimal_eps(dfs, n_neighbors):
    """ Custom function to calculate optimal eps for DBSCAN """
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


def silhouette_scorer(estimator, X):
    """ Custom scoring function for silhouette score. """
    labels = estimator.fit_predict(X)  # Fit and predict labels
    if len(set(labels)) < 2:  # Check if there's at least 2 clusters
        return -1  # Return a negative score if not enough clusters
    return silhouette_score(X, labels)  # Calculate silhouette score


def dbscan_gridsearch(dfs, eps_from, eps_to, eps_increment, min_samples_from, min_samples_to):
    """ Custom function to perform grid search and print best parameters for DBSCAN """
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
    """ Custom function to perfrorm gird search and print best parameters for K-Means"""
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
    """ Custom function to perform grid search for K-Means, DBSCAN and Agglomerative Clustering Algorithms
        and print the best parameters to the console
    """
    if algo == 'kmeans':
        kmeans = KMeans()
        grid_search = GridSearchCV(kmeans, param_grid, cv=5)  # added cv for cross validation
        grid_search.fit(dfs)
        print(f"KMeans Best Parameters: {grid_search.best_params_}")
    elif algo == 'agglomerative':
        aggl = AgglomerativeClustering()
        grid_search = GridSearchCV(dfs, param_grid, cv=5)
        grid_search.fit(dfs)
        print(f"Agglomerative Best Parameters: {grid_search.best_params_}")
    elif algo == 'dbscan':
        warnings.filterwarnings("ignore")  # filter warnings from sklearn for when it can't assign a cluster
        dbscan = DBSCAN()
        custom_scorer = make_scorer(silhouette_scorer)
        # grid_search = GridSearchCV(dbscan, param_grid, scoring=custom_scorer, cv=5)
        grid_search = GridSearchCV(dbscan, param_grid, scoring=silhouette_score)
        grid_search.fit(dfs)
        print(f"DBSCAN Best Parameters: {grid_search.best_params_}")
    else:
        print(f"ERROR: Invalid algorithm: {algo}")

    return grid_search.best_params_


# Display a grid of 3d scatter plots, 2 columns wide for an unknown number of plots
def multi_3d_scatter_plot(dfs, n_cluster_range, x, y, z, display_individual_plot=True, plot_centroids=0,
                          **kwargs):
    """ Custom function to display multi 3D scatter plots for K-means """
    num_plots = n_cluster_range[-1] - n_cluster_range[0] + 1  # upper bound
    num_cols = 2  # always use 2 columns
    num_rows = int((num_plots + num_cols - 1) // num_cols)  # calculate number of rows required

    fig = make_subplots(rows=num_rows, cols=num_cols,
                        subplot_titles=[f"<span style = 'font-size: 10px'> n_clusters: {i} | "
                                        f"init: {kwargs['init']} | "
                                        f"n_init: {kwargs['n_init']} | "
                                        f"random_state: {kwargs['random_state']} </span>"
                                        for i in range(n_cluster_range[0], n_cluster_range[-1] + 1)],

                        vertical_spacing=0.1,
                        specs=[[{'type': 'scatter3d'} for _ in range(num_cols)] for _ in range(num_rows)])
    fig.print_grid()

    for i, n_clusters in enumerate(n_cluster_range, start=1):
        # kmeans_kwargs = {"n_clusters": n_clusters, "init": "random", "random_state": 1}
        kwargs['n_clusters'] = n_clusters
        kmeans_fig = plot_kmeans_3d_scatter(dfs, x, y, z, f"3D Scatter Plot: init: {kwargs['init']} <br> "
                                                          f"<span style = 'font-size: 10px'>n_clusters: {kwargs['n_clusters']} |"
                                                          f"n_init: {kwargs['n_init']} | "
                                                          f"random_state: {kwargs['random_state']}</span>",
                                            display_individual_plot=display_individual_plot,
                                            plot_centroids=plot_centroids, plot_iter=i,
                                            **kwargs)

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
                      title_text="K-Means Clustering with Varying n_clusters <br>")
    plotly.offline.plot(fig, filename='Multi 3D Scatter Plot.html')


def multi_dbscan_3d_scatter_plot(dfs, x, y, z, display_individual_plot=True,
                                 show_centroids=False, **kwargs):
    """ Custom function to display multi 3D scatter plots for DBSCAN """
    num_plots = (len(kwargs['eps'])) * len(kwargs['min_samples'])  # plot for each eps and min_samples
    print(num_plots)
    num_cols = 2  # always use 2 columns
    # num_rows = int((num_plots + num_cols - 1) // num_cols)  # calculate number of rows required
    num_rows = (num_plots + num_cols - 1) // num_cols

    titles = [
                 f"<span style='font-size: 10px'>eps: {eps} | min_samples: {min_samples} | </span>"
                 for eps, min_samples in product(kwargs['eps'], kwargs['min_samples'])
             ][:num_plots]

    specs = [[{'type': 'scatter3d'} for _ in range(num_cols)] for _ in range(num_rows)]

    fig = make_subplots(rows=num_rows, cols=num_cols,
                        subplot_titles=titles,
                        vertical_spacing=0.05,
                        specs=specs)

    # fig.print_grid()

    for i, min_sample_values in enumerate(kwargs['min_samples']):
        for j, eps_value in enumerate(kwargs['eps']):
            # Calculate the plot index
            plot_index = i * len(kwargs['eps']) + j

            # Calculate row and column for the subplot grid
            row = plot_index // num_cols + 1  # Convert to 1-based index
            col = plot_index % num_cols + 1  # Convert to 1-based index

            # Debug: print current plot index, row, col
            # print(
            #    f"Plot Index: {plot_index}, Row: {row}, Col: {col}, eps: {eps_value}, min_samples: {min_sample_values}")

            dbscan_fig = plot_dbscan_3d_scatter(dfs, x, y, z, eps=eps_value, min_samples=min_sample_values,
                                                plot_iter=plot_index,
                                                title=f"DBSCAN 3D Scatter Plot: <br> "
                                                      f"<span style = 'font-size: 10px'>eps: {eps_value} |"
                                                      f"min_samples: {min_sample_values} | </span>",
                                                display=display_individual_plot, show_centroids=show_centroids)

            # Debug: check if data is present
            # if dbscan_fig.data:
            #    print(f"Adding trace for plot index {plot_index}")
            fig.add_trace(dbscan_fig.data[0], row=row, col=col)
            # else:
            #    print(f"No data found for plot index {plot_index}")

            scene_id = f'scene{plot_index + 1}'
            fig.update_layout({
                scene_id: dict(
                    xaxis_title=f"{x}",
                    yaxis_title=f"{y}",
                    zaxis_title=f"{z}",
                    # aspectratio=dict(x=1, y=1, z=1)
                )
            })

    fig.update_layout(height=600 * num_rows, width=600 * num_cols,
                      # margin=dict(l=10, r=10, t=10, b=10),
                      title_text="DBSCAN Clustering with Varying eps and min_samples <br>",
                      # scene=dict(
                      #    aspectmode='auto',  # Ensures the aspect ratio fits the space
                      # )
                      )

    plotly.offline.plot(fig, filename='Multi DBSCAN 3D Scatter Plot.html')


def parallel_centroid_plot(dfs, kmeans_model, centroid_table, plot_iter=0, **kmeans_kwargs):
    """ Custom function to plot centroids for k-means """
    kmeans_model.fit(dfs)

    centroid_table = pd.DataFrame(kmeans_model.cluster_centers_, columns=dfs.columns)
    centroid_table['cluster'] = range(len(centroid_table))

    # plot the centroids
    fig = px.parallel_coordinates(centroid_table, dimensions=dfs.columns.tolist(), color='cluster',
                                  title=f"Parallel Coordinates Plot of {kmeans_kwargs['n_clusters']} Cluster Centroids")
    plotly.offline.plot(fig, filename=f'Parallel Centroid Plot {plot_iter}.html')
    print(f"Centroid Table for n_clusters={kmeans_kwargs['n_clusters']}\n {centroid_table}")
    data_out(centroid_table, "datasets/centroid_table.csv", ',')
    return centroid_table


def get_centroid_table(dfs, model):
    """ Custom function to get centroids """
    # Fit the model to the data
    model.fit(dfs)

    # Retrieve labels for each point to identify clusters
    labels = model.labels_

    # Filter out noise points, which are labeled as -1
    unique_clusters = [label for label in set(labels) if label != -1]

    # Calculate the mean (centroid) of each cluster and store in a dictionary
    centroids = {}
    for cluster in unique_clusters:
        cluster_points = dfs[labels == cluster]
        centroids[cluster] = cluster_points.mean()

    # Convert the centroid dictionary to a DataFrame
    centroid_table = pd.DataFrame(centroids).T
    centroid_table.columns = dfs.columns
    centroid_table['cluster'] = unique_clusters  # Assign cluster labels

    return centroid_table.reset_index(drop=True)


def dunn_index(dfs, centroid_table, labels):
    """ customer function to calculate the Dunn Index for clustering. """
    # Ensure labels is a NumPy array for indexing
    labels = np.array(labels)

    # Calculate the minimum inter-cluster distance (numerator)
    numerator = float('inf')
    for i, c1 in enumerate(centroid_table.values):
        for j, c2 in enumerate(centroid_table.values):
            if i >= j:
                continue  # Avoid duplicate comparisons
            distance = cityblock(c1, c2)
            numerator = min(numerator, distance)

    # Calculate the maximum intra-cluster distance (denominator)
    denominator = 0
    for cluster_label in np.unique(labels):
        # Filter points that belong to the current cluster
        cluster_points = dfs[labels == cluster_label].values

        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                distance = cityblock(cluster_points[i], cluster_points[j])
                denominator = max(denominator, distance)

    # Return the Dunn Index
    return numerator / denominator if denominator != 0 else float('inf')


def calc_davies_bouldin_index(dfs, kmeans):
    """ Custom function to dcalculate the Davies Bouldin Indix and print to console """
    labels = kmeans.labels_
    print(f"Davies-Bouldin Index: {davies_bouldin_score(dfs, labels)}")


# unscaled data - initial observations
# describe_data_out(df, "datasets/dataset_description_unscaled.csv", ',')
# describe_data(df)
# calc_variance(df)
# null_check(df)
# duplicate_check(df)
# histogram(df)
# scatter_3d(df, 'att1', 'att2', 'att3', '3D distribution of unscaled data')

# scaled data - initial observations
scaled_df = StandardScaler().fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
# describe_data_out(scaled_df, "datasets/dataset_description.csv", ',')
# describe_data(scaled_df)
# calc_variance(scaled_df)
# null_check(scaled_df)
# duplicate_check(scaled_df)
# histogram(scaled_df)
# scatter_3d(scaled_df, 'att1', 'att2', 'att3', '3D distribution of scaled data')

# # K-Means algorithm - specifying number of clusters
# kmeans_kwargs = {"n_clusters": 4, "init": "k-means++", "n_init": 10, "random_state": 1}
# fig1 = plot_kmeans_3d_scatter(scaled_df, 'att1', 'att2', 'att3', "Scaled 3d k-means (k-means++) scatter plot",
#                               **kmeans_kwargs)
# # Plot elbow method
# kmeans_kwargs = {"init": "k-means++", "n_init": 10, "random_state": 1}
# plot_elbow_method(scaled_df, 1, 10, **kmeans_kwargs)
#
# # K-Means algorithm n_clusters = 7
# kmeans_kwargs = {"n_clusters": 7, "init": "k-means++", "random_state": 1, "max_iter": 300,  "n_init": 5}
# fig2 = plot_kmeans_3d_scatter(scaled_df, 'att1', 'att2', 'att3', "Scaled 3d k-means (random) scatter plot",
#                               **kmeans_kwargs)
#
# # K-Means multi plot with different states
# kmeans_kwargs = {"init": "k-means++", "random_state": 1, "max_iter": 100, "n_init": 5}
# cluster_range = np.arange(3, 8)
# multi_3d_scatter_plot(scaled_df, cluster_range, 'att1', 'att2', 'att3', display_individual_plot=True,
#                       plot_centroids=True, display_dunn_index=True, **kmeans_kwargs)


# my_kmeans_kwargs = {'n_clusters': 3, "init": "random", "random_state": 1, "max_iter": 100, "n_init": 1}
# deprecated!
# parallel_centroid_plot(scaled_df, **my_kmeans_kwargs)

# Agglomerative/Divisive Clustering
# single linkage - not biased for globular shapes
# complete linkage - biased for globular but sensitive to noise
# Average/Centroid/Ward - biased for globular clusters but not as senstive to noise - potential option

# plot_agglomerative_3d_scatter(scaled_df, 3, 'complete', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Complete] -'
#                                                                                ' scaled dataset')
#
# plot_agglomerative_3d_scatter(scaled_df, 3, 'single', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Single] -'
#                                                                                ' scaled dataset')
#
# plot_agglomerative_3d_scatter(scaled_df, 3, 'average', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Average] -'
#                                                                                ' scaled dataset')
#
# plot_agglomerative_3d_scatter(scaled_df, 3, 'ward', 'att1', 'att2', 'att3', 'Agglomerative Clustering [Ward] -'
#                                                                                ' scaled dataset')


# DBScan
# Note: Below is from a subjective perspective a decent clustering of the data, eps=0.3, min_samples=5
dbscan_kwargs = {'eps': np.arange(0.25, 2, 0.25),
                'min_samples': range(4, 7),
                'scoring': ['silhouette_score']
                }
#
# plot_dbscan_3d_scatter(dfs=scaled_df, x='att1', y='att2', z='att3', eps=1.25, min_samples=4, plot_iter=0,
#                        show_centroids=False)

plot_dbscan_3d_scatter(dfs=scaled_df, x='att1', y='att2', z='att3', eps=0.25, min_samples=5, plot_iter=0,
                       show_centroids=False)

# # Unused in report
# multi_dbscan_3d_scatter_plot(dfs=scaled_df, x='att1', y='att2', z='att3', display_individual_plot=False,
#                             show_centroids=True, **dbscan_kwargs)

# plot_optimal_eps(scaled_df, 5)
param_grid = {
   'eps': np.arange(0.25, 2, 0.25),
   'min_samples': range(3, 8)
}


# best_params = grid_search(scaled_df, param_grid, 'dbscan')

# dbscan_gridsearch(scaled_df, eps_from=0.25, eps_to=2, eps_increment=0.25, min_samples_from=2, min_samples_to=8)
# param_grid = {
#     'n_clusters': range(2, 10),
#     'init': ['k-means++', 'random'],
#     'n_init': [5, 10, 15],
#     'max_iter': [100, 200, 300, 400, 500],
#     'random_state': [1, 16, 34, 57]
# }
# grid_search(scaled_df, param_grid, 'kmeans')
