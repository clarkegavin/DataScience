import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px #  for 3D scatter plots
import plotly.io as pio
import numpy as np
# import warnings

# warnings.filterwarnings('ignore')

# loading 'iris' data into a DataFrame
iris = pd.read_csv(
    "C:/Users/Gavin/Documents/Gavin/Data Science/M.Sc. Data Science and Analytics/Data Science Algorithms/iris.csv",
    sep=';')

# prints first 5 rows of data
print(iris.head())

print("--iris.info--")
# prints how many rows, columns there are in the dataset, along with column names, datatypes and how many
# non-null values are in each column
iris.info()

print("--missing values--")
# quick way to see if any of the columns have null values
print(iris.isnull().sum())

print("--summary statistics--")
# descriptive statistics that summarize the central tendency, dispersion and shape of a datasets distribution,
# excluding NaN values
print(iris.describe(include='all'))
print(iris.describe(include=object))
print(iris.describe(include=float))

print("--regular attributes--")
# this excludes the target column and assigns the returned object to a new DataFrame 'regular_attributes'
# the purpose was to remove the class column so that only the numeric columns would exist in the new DataFrame
regular_attributes = iris[iris.columns.difference(['iris'])]
print(regular_attributes.sample(7)) # displays a 'random' sample of the data with 7 rows

print("--Slicing")
# Slices up the DataFrame into Series objects (i.e. Columns specified by the index 'iris' in DataFrame iris
label_attribute = iris['iris']  # creates a Series object 'label_attribute' with the class column 'iris'
print(label_attribute.tail(3))  # prints last 3 records in the column


print("--Data Manipulation--")
print(iris[iris.notnull()])  # select all rows where there are no missing values.

# select duplicated rows.  iris.duplicated() returns a Boolean Series indicating whether each row is a duplicate or not
# It returns True for all rows that are duplicates of a previous row & False for the first occurance
# iris[...] is using the Boolean Series returned by duplicated() to filter the rows in the iris DataFrame.  Rows
# where the value is True will be included in the result, meaning it only includes duplicated rows
print(iris[iris.duplicated()])

# select all rows where labels are not missing values
print(iris[iris.iris.notnull()])  # iris.iris is accessing the column called iris in the iris DataFrame

# select all rows labelled as Iris-setosa
print(iris[iris.iris == 'Iris-setosa'])

print("--Inspecting labels distribution--")
# count the number of unique records in the iris column(i.e. Series) in the iris DataFrame
print(iris.iris.value_counts())

print("--displaying data in a bar chart--")
label_attribute.value_counts().plot(kind='bar')  # plot the class, i.e. iris column in a bar chart
plt.xticks(rotation=25)  # provides better readability of the labels on the x axis
# plt.show()  # required to display the bar chat from Pycharm IDE

print(" -- displaying histogram of the numeric values, i.e. the DataFrame 'regular_attributes' ")
# creates individual histograms for each numeric column in regular_attributes DataFrame.
# figsize sets the size of the entire plot to 10" x 7"
regular_attributes.hist(figsize=(10, 7))  # pandas generates histograms using Matplotlib internally
# matplotlib maintains a  statement machine to keep track of the current figure.  Each plot (i.e. our bar chart and
# histogram) are added to the current figure and we can use plt.show() to display them
# if we only included this last plt.show() the histogram would open first (figure 2) and then the bar chart (figure 1)
plt.show()


regular_attributes.plot(kind='hist', subplots=True, layout=(2, 2), figsize=(10, 7))
plt.show()

# Boxplots
print("--Boxplots--")
regular_attributes.boxplot(figsize=(10, 7))
plt.show()

regular_attributes.plot(kind='box', subplots=True, layout=(2, 2), figsize=(10, 7))
# plt.show()

# Scatter plots using seaborn
# Note:seaborn is a powerful statistical data visualisation library built on top of Matplotlib
print("Scatter plots via seaborn")
sns.scatterplot(x='petallength', y='petalwidth', data=iris)
# coloured by class
sns.scatterplot(x='petallength', y='petalwidth', data=iris, hue='iris')
plt.show()

#3D scatter plots
fig = px.scatter_3d(iris, x='sepallength', y='petallength', z='petalwidth', color='iris')
fig.show()
pio.renderers.default = 'png'

print(pio.renderers.default)