import sys

from sklearn.neighbors import KNeighborsClassifier
import Classification.data as data  # custom utility
import Classification.adsa_utils as ad
import Classification.plots as plts  # custom utility
from Classification.knn import KNN
from Classification.processdata import ProcessData
import pandas as pd
import csv
from Classification.decision_tree import DTree
from sklearn.tree import DecisionTreeClassifier, plot_tree


PROCESS = 'ADULT'
GRID_SEARCH = False
KNN = False
DT = True
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
    elif PROCESS =='STUDENT':
        processed_data, class_label = process_student_dataset()
    else:
        # Error
        sys.exit()


    df = processed_data.process_data()  # Output data statistics
    print(f"DEBUG: \n {df.head()}")
    if CORRELATION_MATRIX:
        processed_data.plot_correlation_matrix(class_label)  # correlation matrix

    df_scaled, scaler = processed_data.scale_data(df)  # scale the data
    df_encoded_unscaled, df_predictive_unscaled, encoder_unscaled = processed_data.encode_data(df,
                                                                                               class_label)  # Unscaled Encoded
    df_encoded_scaled, df_predictive, encoder = processed_data.encode_data(df_scaled, class_label)  # Scaled Encoded

    # Copy unscaled data for usage in model
    X = df_encoded_unscaled.copy()
    y = df_predictive_unscaled.copy()

    # KNN
    if KNN:
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

    if DT:
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

    # Ensembles

    # Logistic Regression
