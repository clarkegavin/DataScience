from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import data as data
import adsa_utils as ad
import pandas as pd


df_adult = data.read_data(file_location="datasets/assesment_adult_dataset.csv", sep=';')
df_student = data.read_data(file_location="datasets/assessment_student_dataset.csv", remove_quotes=True, sep=';\s*')

# clean student data - remove extraneous quotes
df_student = data.clean_data(df_student)

# print data statistics
data.print_data_statistics(df_student)
data.print_data_statistics(df_adult)
data.data_out(df_adult.describe(), "dataout/adult_describe.csv", ",")
data.data_out(df_student.describe(), "dataout/student_describe.csv", ",")
data.data_out(data.get_data_info(df_adult), "dataout/adult_info.csv")
data.data_out(data.get_data_info(df_student), "dataout/student_info.csv")

# scale continuous data
data.scale_continuous_data(df_student)
data.scale_continuous_data(df_adult)
data.print_data_statistics(df_student)
data.print_data_statistics(df_adult)
data.data_out(df_adult.describe(), "dataout/adult_describe_scaled.csv", ",")
data.data_out(df_student.describe(), "dataout/student_describe_scaled.csv", ",")


# validate

#
# # create decision tree from adult dataset
# print(df_adult.head())
# # separate the class label
# y = df_adult['label']
# X = df_adult[df_adult.columns.difference(['labels'])]
#
# # split the dataset into training & test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=43
# )
#
# # encode categorical values for use with sklearn
# X_train, encoders = data.encode_categorical_data_le(X_train)
# # check: are there any id columns in the dataset that we need to get rid of?  e.g. education number?
#
# print(X_train.head())
# features = list(X_train.columns)
# print(X_train.head())
# print(y_train.head())
#
# # baseline (default hyperparameters with exception of random_state for deterministic output
# dtree = DecisionTreeClassifier(random_state=43)
# dtree = dtree.fit(X_train, y_train)
#
# # cross validation - see adsa utils for custom_crossvalidation
# ad.custom_crossvalidation(X_train, y_train, dtree)
#
#
# plot_tree(dtree, filled=True, feature_names=features, class_names=dtree.classes_)
# plt.figure(figsize=(25, 16))
# plt.show()

# XGBoost for categorical data