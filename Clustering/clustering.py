import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline
import plotly.express as px  # for 3D scatter plots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# TODO: create a loop to go through different iterations of cluster size, init types (k-means++, random), etc.
# use a gird search - param_grid which will look at all types of possible values
# pipeline (sklearn)

# read dataset into a DataFrame
df = pd.read_csv("datasets/assessment_cluster_dataset.csv")


def describe_data_out(dfs, path, separator):
    dfs.describe().to_csv(path_or_buf=path, sep=separator)


print("Dataset Information: ")
df.info()

describe_data_out(df, "datasets/dataset_description.csv", ',')
print("Dataset Description: ")
print(df.describe())

# check if any attribute values are null
print("NULL Check")
print(df.isnull().sum())

# Duplicate check
print("Duplicate Check")
print(df[df.duplicated()])

# Histogram for frequency distribution
df.hist(figsize=(10, 7))
#plt.show()

print("3D Scatter Plot")
fig = px.scatter_3d(df, x='att1', y='att2', z='att3')
fig.update_layout(title=dict(text="3D distribution of unscaled data"))
#plotly.offline.plot(fig)


# scale data into a new dataframe retaining original column names
scaled_df = StandardScaler().fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
print("Scaled data--------------------")
print("Scaled Data Histogram")
scaled_df.hist(figsize=(10, 7))
#plt.show()

print("3D Scatter Plot")
fig = px.scatter_3d(scaled_df, x='att1', y='att2', z='att3')
fig.update_layout(title=dict(text="3D distribution of scaled data"))
#plotly.offline.plot(fig)


print("Elbow Method")
kmeans_kwargs = {"init": "k-means++", "n_init": 10, "random_state": 1}
# Elbow-method
# list to hold SSE (sum of squared errors) values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_df)
    sse.append(kmeans.inertia_)

# print inertia calculations from Elbow-method
print(sse)

# visualise Elbow-method results
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
#plt.show()


krandom_model = KMeans(n_clusters=3, init='random', random_state=1).set_output(transform='pandas')
results = krandom_model.fit_transform(scaled_df)
print(results.head)
print(krandom_model.labels_)
krandom_data = scaled_df.copy()
krandom_data['cluster'] = krandom_model.labels_
print(krandom_data.head())

#create a dataframe with the centroids
centroid_table = pd.DataFrame(krandom_model.cluster_centers_, columns=scaled_df.columns)
centroid_table['cluster'] = ['centroid 0', 'centroid 1', 'centroid 2']
print(centroid_table.head())
#plot the centroids
fig = px.parallel_coordinates(centroid_table, 'cluster')
plt.show()

"""
# Create k-means instance on scaled dataset
kmeans = KMeans(n_clusters=4)
results = kmeans.fit_predict(scaled_df[['att1', 'att2', 'att3']])
scaled_df['Cluster'] = results
fig = px.scatter_3d(scaled_df, x='att1', y='att2', z='att3', color='Cluster')
#plotly.offline.plot(fig)


#Plot Centroids
centroid_table = pd.DataFrame(results.cluster_centers_, columns=scaled_df.columns)
print(centroid_table)
"""

# 1. Understand the Business Requirement
# 2. Understand the data
# 3. Prepare the data
# 5. Prepare the model (k-means, hierarchical clustering or DBScan - suggested, pick 2)
# 5. Evaluate
