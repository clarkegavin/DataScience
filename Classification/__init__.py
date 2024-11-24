from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import data as data  # custom utility
import adsa_utils as ad
import plots as plts  # custom utility
import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np

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
df_student_scaled, student_scaler = data.scale_continuous_data(df_student)
df_adult_scaled, adult_scaler = data.scale_continuous_data(df_adult)

# data.print_data_statistics(df_student)
# data.print_data_statistics(df_adult)
data.data_out(df_adult_scaled.describe(), "dataout/adult_describe_scaled.csv", ",")
data.data_out(df_student_scaled.describe(), "dataout/student_describe_scaled.csv", ",")

# Separate the descriptive attributes from the class label column
df_adult_descriptive, df_adult_predictive = data.remove_class_label(df_adult, 'label')  # unscaled
df_adult_descriptive_scaled, df_adult_predictive_scaled = data.remove_class_label(df_adult_scaled, 'label')  # scaled

# print(f"Distribution of adult dataset class label: {df_adult['label'].value_counts()}")

# Encode adult categorical data using one hot encoder
df_adult_encoded, adult_encoder = data.encode_categorical_data_ohe(df_adult_descriptive)  # unscaled/encoded
df_adult_encoded_scaled, adult_encoder = data.encode_categorical_data_ohe(df_adult_descriptive_scaled)  # scaled/encoded
data.data_out(df_adult_encoded, "dataout/adult_ohe.csv")
data.data_out(data.get_data_info(df_adult_encoded), "dataout/adult_ohe_info.csv")

# plts.histogram(df_adult, "Adult data")
# plts.histogram(df_adult_scaled, "Adult data scaled")

# Check correlation

df_adult_class_encoded = df_adult['label'].apply(lambda x: 1 if x == '>50K' else 0)
# Note: Calculating and displaying scatter plots using onehotencoder is unrealistic given the 'pivot' that OHE creates
# df_adult_correlation = pd.concat([df_adult_encoded, df_adult_class_encoded], axis='columns')
# print(f"Fully encoded: {df_adult_correlation.head()}")
# print(f"Encoded description:\n {df_adult_correlation.info()}")
plts.scatter_plot_matrix(df_adult, hue='label')

# Split the data into training and test sets
X = df_adult_encoded.copy()
y = df_adult_predictive.copy()

# KNN
print("------------KNN--------------------")
# Baseline - no scaling, no hyperparameter adjustments
knn_clf = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

ad.custom_crossvalidation(X_train, y_train, knn_clf)

# Scaled adult dataset (using one hot encoder)
X = df_adult_encoded_scaled.copy()
y = df_adult_predictive.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

ad.custom_crossvalidation(X_train, y_train, knn_clf)

# grid search for best parameters
params = {'n_neighbors': range(3, 11),
          'weights': ['uniform', 'distance'],
          'p': [1, 2, 3]}

# grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=params)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# KNN with best parameters

knn_clf = KNeighborsClassifier(n_neighbors=10, p=1, weights='distance')
print("Grid Search Best Parameters")
ad.custom_crossvalidation(X_train, y_train, knn_clf)

# predict
num_columns = X_train.select_dtypes(include=['number']).columns.tolist()
cat_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', adult_scaler, num_columns),  # scale numerical features
    ('cat', adult_encoder, cat_columns)  # encode categorical columns
])
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', knn_clf)
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"Prediction Report\n: {classification_report(y_test, y_pred)}")
ad.cross_validation_avg_scores(knn_clf, X_test, y_test, cv_=5)
ad.plot_confusion_matrix(y_test, y_pred)

# Decision Trees
print("------------DECISION TREES--------------------")
X = df_adult_encoded.copy()
y = df_adult_predictive.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

dt_clf = DecisionTreeClassifier(random_state=43)
ad.custom_crossvalidation(X_train, y_train, dt_clf)
dt_clf = dt_clf.fit(X_train, y_train)
plts.custom_plot_tree(dt_clf, X_train)
print(f"Tree Depth: {dt_clf.get_depth()} \nNumber of Leaves: {dt_clf.get_n_leaves()}")
print(f"Classes: {dt_clf.classes_}")
# print(f"Decision Path: {dt_clf.decision_path(X_train, check_input=True)}")

# Takes approx. 1 hour to run!
# params = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': range(3, 11),
#     'min_impurity_decrease': np.arange(0.01, 0.3, 0.01),
#     'min_samples_leaf': range(2, 20, 4),
#     'min_samples_split': range(2, 10, 2),
#     'ccp_alpha': [0.003, 0.005]
# }
# grid_search = GridSearchCV(dt_clf, param_grid=params, scoring='f1_macro')
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# Test model on testing portion
dt_clf = DecisionTreeClassifier(random_state=43, ccp_alpha=0.003, criterion='entropy', max_depth=5,
                                min_impurity_decrease=0.01, min_samples_leaf=2, min_samples_split=2)
ad.custom_crossvalidation(X_train, y_train, dt_clf)
dt_clf = dt_clf.fit(X_train, y_train)


# No scaling required for Decision Trees
cat_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
preprocessor = ColumnTransformer([
    ('cat', adult_encoder, cat_columns)  # encode categorical columns
])


# pipeline = Pipeline([
#     ('preprocessing', preprocessor),
#     ('sklearn_dt', grid_search.best_estimator_)
# ])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', dt_clf)
])

# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# print(f"Classification Report\n: {classification_report(y_test, y_pred)}")
# ad.plot_confusion_matrix(y_test, y_pred)
# plts.custom_plot_tree(dt_clf, X_test)
# print(f"Tree Depth: {dt_clf.get_depth()} \nNumber of Leaves: {dt_clf.get_n_leaves()}")
# print(f"Classes: {dt_clf.classes_}")

# Scaled - decision tress do not rely on numerical scale of the features
# they split data based on feature thresholds, these threshold dpeend only on the relative order of values, not
# their magnitude -  so in summary, Scaling for Decision Trees is not relevant
# X = df_adult_encoded_scaled.copy()
# y = df_adult_predictive.copy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
# ad.custom_crossvalidation(X_train, y_train, dt_clf)
# dt_clf = dt_clf.fit(X_train, y_train)

# Naive Bayes


# Ensembles


# Logistic Regression