from sklearn.compose import ColumnTransformer

import adsa_utils as ad
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


class KNN:
    def __init__(self, X, y, scaler, encoder, knn_clf):
        self.X = X
        self.y = y
        self.scaler = scaler
        self.encoder = encoder
        self.knn_clf = knn_clf
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=43)
        self.validate("train")


    def train(self):
        # X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=43)
        # ad.custom_crossvalidation(self.X_train, self.y_train, self.knn_clf)
        pass

    def predict(self):
        num_columns = self.X_train.select_dtypes(include=['number']).columns.tolist()
        cat_columns = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', self.scaler, num_columns),  # scale numerical features
            ('cat', self.encoder, cat_columns)  # encode categorical columns
        ])
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', self.knn_clf)
        ])
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        print(f"Prediction Report\n:")
        self.validate("test")

    def validate(self, type):
        print("Validate")
        if type == "train":
            print(f"Training dataset evaluation:")
            ad.custom_crossvalidation(self.X_train, self.y_train, self.knn_clf)
        else:
            print(f"Test dataset evaluation:")
            ad.custom_crossvalidation(self.X_test, self.y_test, self.knn_clf)


    def grid_search(self, **params):
        # grid_search = GridSearchCV(KNeighborsClassifier(), param_grid=params)
        # grid_search.fit(X_train, y_train)
        # print(grid_search.best_params_)
        # print(grid_search.best_score_)
        pass