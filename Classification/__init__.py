import sys

from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

import Classification.data as data  # custom utility
import Classification.adsa_utils as ad
import Classification.plots as plts  # custom utility
from Classification.knn import KNN
from Classification.processdata import ProcessData
import pandas as pd
import csv
from Classification.decision_tree import DTree
from Classification.naive_bayes import NBayes
from Classification.logistic_regression import LRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
import numpy as np

# Configuration parameters for selective classifiers, set to True if you want to display a train a particular model
PROCESS = 'ADULT'
GRID_SEARCH = False
KNN_CLASSIFIER = True
DT_CLASSIFIER = False
NB_CLASSIFIER = False
LR_CLASSIFIER = False
CORRELATION_MATRIX = False


def process_adult_dataset():
    """
    assessment - adult dataset
    """
    return ProcessData(
        data.read_data(file_location="datasets/assesment_adult_dataset.csv", sep=';'),
        data_out="dataout/adult/"), "label"


def process_student_dataset():
    """
    assessment - student dataset
    """
    processed = ProcessData(
        data.read_data(
            file_location="datasets/assessment_student_dataset.csv",
            sep=';',
            remove_quotes=True), data_out="dataout/student/")
    processed.clean_data()
    return processed, "Pass"


if __name__ == "__main__":

    if PROCESS == 'ADULT':
        processed_data, class_label = process_adult_dataset()
        print("Adult Dataset")
    elif PROCESS == 'STUDENT':
        processed_data, class_label = process_student_dataset()
        print("Student Dataset")
    else:
        # Error
        sys.exit()

    df = processed_data.process_data()  # Output data statistics

    if CORRELATION_MATRIX:
        processed_data.plot_correlation_matrix(class_label)  # correlation matrix

    df_scaled, scaler = processed_data.scale_data(df)  # scale the data
    df_encoded_unscaled, df_predictive_unscaled, encoder_unscaled = processed_data.encode_data(df,
                                                                                               class_label)  # Unscaled Encoded
    df_encoded_scaled, df_predictive, encoder = processed_data.encode_data(df_scaled, class_label)  # Scaled Encoded

    # Copy unscaled data for usage in model
    X = df_encoded_unscaled.copy() # class label has been droped
    y = df_predictive_unscaled.copy() # class label only

    # KNN
    if KNN_CLASSIFIER:
        print("------------KNN--------------------")
        # Baseline - no scaling, no hyperparameter adjustments

        knn_clf = KNN(X=X, y=y, scaler=scaler, encoder=encoder,
                      knn_clf=KNeighborsClassifier())
        knn_clf.predict()

        # Scaled adult dataset (using one hot encoder)
        X = df_encoded_scaled.copy()
        y = df_predictive.copy()

        # Creates KNN classifer, trains and validates
        knn_clf = KNN(X=X, y=y, scaler=scaler, encoder=encoder,
                      knn_clf=KNeighborsClassifier())
        knn_clf.predict()


        # if GRID_SEARCH == True:
        #     # grid search for best parameters
        #     params = {'n_neighbors': range(3, 11),
        #               'weights': ['uniform', 'distance'],
        #               'p': [1, 2, 3]}
        #
        #     grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=params)
        #     grid_search.fit(X_train, y_train)
        #     print(grid_search.best_params_)
        #     print(grid_search.best_score_)

        # KNN with best parameters

        knn_clf = KNN(X=X, y=y, scaler=scaler, encoder=encoder,
                      knn_clf=KNeighborsClassifier(n_neighbors=10, p=1, weights='distance'))
        knn_clf.predict()
        print("Feature Importance")
        knn_clf.feature_importance()

    if DT_CLASSIFIER:
        print("-------------Decision Trees----------------------")
        X = df_encoded_unscaled
        y = df_predictive
        dt_clf = DTree(X=X, y=y, encoder=encoder, dt_clf=DecisionTreeClassifier(random_state=43))
        dt_clf.predict()

        # change entropy
        print("----Shannon's Entropy--------")
        dt_clf = DecisionTreeClassifier(criterion='entropy')

        # pre pruning
        print("----Pre Pruning--------")
        dt_clf = DecisionTreeClassifier(max_depth=10,
                                        min_samples_split=5,
                                        min_samples_leaf=3,
                                        random_state=43)
        dt_clf = DTree(X=X, y=y, encoder=encoder, dt_clf=dt_clf)
        dt_clf.predict()
        # post pruning
        print("----Post Pruning--------")
        dt_clf = DecisionTreeClassifier(ccp_alpha=0.007)
        dt_clf = DTree(X=X, y=y, encoder=encoder, dt_clf=dt_clf)
        dt_clf.predict()
        # combined pre/post pruning
        print("----Combined Pre and Post--------")
        dt_clf = DecisionTreeClassifier(max_depth=10,
                                        min_samples_split=5,
                                        min_samples_leaf=3,
                                        ccp_alpha=0.007,
                                        random_state=43)
        dt_clf = DTree(X=X, y=y, encoder=encoder, dt_clf=dt_clf)
        dt_clf.predict()

    # Naive Bayes
    if NB_CLASSIFIER:
        print("-------------Naive Bayes--------------")
        print("Baseline")

        X = df_encoded_unscaled
        y = df_predictive
        data.data_out(y, "dataout/adult/test_label_split.csv")
        nb_clf = NBayes(X=X, y=y, scaler=None, encoder=encoder,
                        nb_clf=ComplementNB(alpha=0, force_alpha=True, norm=False))
        nb_clf.predict()



        # print("Baseline Scaled")
        # # scale the data in range [0,1] as Naive Bayes doesn't handle negative numbers
        df_scaled, scaler = processed_data.scale_data(df, type='MinMaxScaler')
        df_encoded_scaled, df_predictive, encoder = processed_data.encode_data(df_scaled, class_label, encoder='le')  # Scaled Encoded

        X = df_encoded_scaled
        y = df_predictive
        # nb_clf = NBayes(X=X, y=y, scaler=scaler, encoder=encoder,
        #                 nb_clf=ComplementNB(alpha=0, force_alpha=True, norm=False))
        # nb_clf.predict()

        # change alpha hyper parameter - small
        print("Alpha = 0.00001 | Norm =True")
        nb_clf = NBayes(X=X, y=y, scaler=scaler, encoder=encoder,
                        nb_clf=ComplementNB(alpha=0.00001, norm=True))
        nb_clf.predict()


        print("Alpha = 0.00001 | Norm =False")
        nb_clf = NBayes(X=X, y=y, scaler=scaler, encoder=encoder,
                        nb_clf=ComplementNB(alpha=0.00001, norm=False))
        nb_clf.predict()

        # change alpha hyper parameter - large
        print("Alpha = 10 | Norm =True")
        nb_clf = NBayes(X=X, y=y, scaler=scaler, encoder=encoder,
                        nb_clf=ComplementNB(alpha=10, norm=True))
        nb_clf.predict()

        # change alpha hyper parameter - large
        print("Alpha = 10 | Norm =False")
        nb_clf = NBayes(X=X, y=y, scaler=scaler, encoder=encoder,
                        nb_clf=ComplementNB(alpha=10, norm=True))
        nb_clf.predict()


        if GRID_SEARCH:
            nb_clf.custom_grid_search()

        # for alpha in np.arange(0.1, 3, 0.1):
        #     print(f"alpha={alpha}")
        #     nb_clf = NBayes(X=X, y=y, scaler=scaler, encoder=encoder,
        #                     nb_clf=ComplementNB(alpha=alpha, norm=True), show_confusion=True)

        # Best Params when using OneHotEncoder
        # print("Grid Search Best Params: Alpha = 0.5501 | Norm =True")
        # nb_clf = NBayes(X=X, y=y, scaler=scaler, encoder=encoder,
        #                 nb_clf=ComplementNB(alpha=0.5501, norm=True))
        # nb_clf.predict()

        print("Grid Search Best Params: Alpha = 0.0001 | Norm =False")
        nb_clf = NBayes(X=X, y=y, scaler=scaler, encoder=encoder,
                        nb_clf=ComplementNB(alpha=0.0001, norm=False))
        nb_clf.predict()


    # Ensembles


    # Logistic Regression
    if LR_CLASSIFIER:
        lr_clf = LRegression(X=X, y=y, scaler=scaler, encoder=encoder, lr_clf=LinearRegression())
