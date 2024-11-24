from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import data as data # custom utility
import adsa_utils as ad
import plots as plts # custom utility
import pandas as pd


df_adult = data.read_data(file_location="datasets/assesment_adult_dataset.csv", sep=';')
df_student = data.read_data(file_location="datasets/assessment_student_dataset.csv", remove_quotes=True, sep=';\s*')


# clean student data - remove extraneous quotes
df_student = data.clean_data(df_student)

# print data statistics
# data.print_data_statistics(df_student)
# data.print_data_statistics(df_adult)
data.data_out(df_adult.describe(), "dataout/adult_describe_numerical.csv", ",")
data.data_out(df_student.describe(), "dataout/student_describe_numerical.csv", ",")
data.data_out(df_adult.describe(include='object'), "dataout/adult_describe_categorical.csv", ",")
data.data_out(df_student.describe(include='object'), "dataout/student_describe_categorical.csv", ",")
data.data_out(data.get_data_info(df_adult), "dataout/adult_info.csv")
data.data_out(data.get_data_info(df_student), "dataout/student_info.csv")



# scale continuous data
df_student_scaled = data.scale_continuous_data(df_student)
df_adult_scaled = data.scale_continuous_data(df_adult)


# data.print_data_statistics(df_student)
# data.print_data_statistics(df_adult)
data.data_out(df_adult_scaled.describe(), "dataout/adult_describe_scaled.csv", ",")
data.data_out(df_student_scaled.describe(), "dataout/student_describe_scaled.csv", ",")

# Separate the descriptive attributes from the class label column
df_adult_descriptive, df_adult_predictive = data.remove_class_label(df_adult, 'label') # unscaled
df_adult_descriptive_scaled, df_adult_predictive_scaled = data.remove_class_label(df_adult_scaled, 'label') # scaled


print(f"Adult Descriptive Features Data Frame: {data.print_data_statistics(df_adult_descriptive)}")
print(f"Adult Predictive Feature Data Frame: {data.print_data_statistics(df_adult_predictive)}")


# print(f"Distribution of adult dataset class label: {df_adult['label'].value_counts()}")

# Encode adult categorical data using one hot encoder
df_adult_encoded, adult_encoder = data.encode_categorical_data_ohe(df_adult_descriptive) #unscaled
df_adult_encoded_scaled, adult_encoder = data.encode_categorical_data_ohe(df_adult_descriptive_scaled) #scaled
data.data_out(df_adult_encoded, "dataout/adult_ohe.csv")

# Alternative - XGBoost?

data.data_out(data.get_data_info(df_adult_encoded), "dataout/adult_ohe_info.csv")

# plts.histogram(df_adult, "Adult data")
# plts.histogram(df_adult_scaled, "Adult data scaled")


# Split the data into training and test sets
X = df_adult_encoded.copy()
y = df_adult_predictive.copy()


# KNN
# Baseline - no scaling, no hyperparameter adjustments
knn_clf = KNeighborsClassifier()
print(f"X baseline {X.head()}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

ad.custom_crossvalidation(X_train, y_train, knn_clf)

# Scaling (using one hot encoder)
X = df_adult_encoded_scaled.copy()
print(f"X after scaling {X.head()}")
y = df_adult_predictive.copy()  # Question: Should I use the scaled data for this?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

ad.custom_crossvalidation(X_train, y_train, knn_clf)




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