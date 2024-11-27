from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import adsa_utils as ad
from sklearn.pipeline import Pipeline
import plots as plts  # custom utility


class DTree:

    def __init__(self, X, y, encoder, dt_clf):
        self.X = X
        self.y = y
        self.encoder = encoder
        self.dt_clf = dt_clf
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=43)
        self.validate("train")

    def validate(self, type):
        print("Validate")
        if type == "train":
            print(f"Training dataset evaluation:")
            plts.custom_plot_tree(self.dt_clf, self.X_train)
            ad.custom_crossvalidation(self.X_train, self.y_train, self.dt_clf)
        else:
            print(f"Test dataset evaluation:")
            plts.custom_plot_tree(self.dt_clf, self.X_test)
            ad.custom_crossvalidation(self.X_test, self.y_test, self.dt_clf)

    def predict(self):
        #num_columns = self.X_train.select_dtypes(include=['number']).columns.tolist()
        cat_columns = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        preprocessor = ColumnTransformer([
            #('num', self.scaler, num_columns),  # scale numerical features
            ('cat', self.encoder, cat_columns)  # encode categorical columns
        ])
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', self.dt_clf)
        ])
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        # Check if I need this...
        print(f"Predication Classification Report\n: {classification_report(self.y_test, y_pred)}")
        print(f"Prediction Report\n:")
        self.validate("test")


