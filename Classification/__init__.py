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

from Classification.knn import KNN
from processdata import ProcessData


def process_adult_dataset():
    """
    assessment - adult dataset
    """
    return ProcessData(
        data.read_data(file_location="datasets/assesment_adult_dataset.csv", sep=';'), data_out="dataout/adult/")


def process_student_dataset():
    """
    assessment - student dataset
    """
    return ProcessData(
        data.read_data(file_location="datasets/assesment_student_dataset.csv", sep=';'), data_out="dataout/student/")


processed_data = process_adult_dataset()
# processed_data = process_student_dataset()

df = processed_data.process_data() # Output data statistics
processed_data.plot_correlation_matrix() # correlation matrix
df_scaled, scaler = processed_data.scale_data(df) # scale the data
df_encoded_unscaled, df_predictive_unscaled, encoder_unscaled = processed_data.encode_data(df)  # Unscaled Encoded
df_encoded_scaled, df_predictive, encoder = processed_data.encode_data(df_scaled)  # Scaled Encoded

# Copy unscaled data for usage in model
X = df_encoded_unscaled.copy()
y = df_predictive_unscaled.copy()

# KNN
print("------------KNN--------------------")
# Baseline - no scaling, no hyperparameter adjustments

knn_clf = KNN(X=X, y=y, scaler=scaler, encoder=encoder,
              knn_clf=KNeighborsClassifier())
knn_clf.predict()


# knn_clf = KNeighborsClassifier()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
#
# ad.custom_crossvalidation(X_train, y_train, knn_clf)


# Scaled adult dataset (using one hot encoder)
X = df_encoded_scaled.copy()
y = df_predictive.copy()

# Creates KNN classifer, trains and validates
knn_clf = KNN(X=X, y=y, scaler=scaler, encoder=encoder,
              knn_clf=KNeighborsClassifier())

knn_clf.predict()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
#
# ad.custom_crossvalidation(X_train, y_train, knn_clf)


# grid search for best parameters
params = {'n_neighbors': range(3, 11),
          'weights': ['uniform', 'distance'],
          'p': [1, 2, 3]}

# grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=params)
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)
# print(grid_search.best_score_)

# KNN with best parameters

knn_clf = KNN(X=X, y=y, scaler=scaler, encoder=encoder,
              knn_clf=KNeighborsClassifier(n_neighbors=10, p=1, weights='distance'))
knn_clf.predict()


# knn_clf = KNeighborsClassifier(n_neighbors=10, p=1, weights='distance')
# print("Grid Search Best Parameters")
# ad.custom_crossvalidation(X_train, y_train, knn_clf)

# # predict
# num_columns = X_train.select_dtypes(include=['number']).columns.tolist()
# cat_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
#
# preprocessor = ColumnTransformer([
#     ('num', scaler, num_columns),  # scale numerical features
#     ('cat', encoder, cat_columns)  # encode categorical columns
# ])
# pipeline = Pipeline([
#     ('preprocessing', preprocessor),
#     ('classifier', knn_clf)
# ])
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# print(f"Prediction Report\n:")
# ad.custom_crossvalidation(X_test, y_test, knn_clf)
# #ad.cross_validation_avg_scores(knn_clf, X_test, y_test, cv_=5)
# #classification_report(y_test, y_pred)
# #ad.plot_confusion_matrix(y_test, y_pred)


# Naive Bayes


# Ensembles


# Logistic Regression
