import pandas as pd
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv(
    "C:/Users/Gavin/Documents/Gavin/Data Science/M.Sc. Data Science and Analytics/Data Science Algorithms/iris.csv")

print(iris.head())
