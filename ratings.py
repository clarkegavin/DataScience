# Display 3d scatter plot of 3 ratings per person
# Expand this in the future to be filterable by year?

import pandas as pd
import plotly.express as px  # for 3D scatter plots


ratings = pd.read_csv(
    "C:/Users/Gavin/Documents/Gavin/Data Science/M.Sc. Data Science and Analytics/Data Science Algorithms/ratings.csv")

# 3D scatter plots
fig = px.scatter_3d(ratings, x='Client', y='Finance', z='Risk', color='Employee')
fig.show()


