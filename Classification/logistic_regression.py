from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import adsa_utils as ad
from sklearn.pipeline import Pipeline
import plots as plts  # custom utility


class LRegression:

    def __init__(self, X, y, scaler, encoder, lr_clf):
        self.X = X
        self.y = y
        self.scaler = scaler
        self.encoder = encoder
        self.lr_clf = lr_clf
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=43)
        #self.lr_clf.fit(self.X_train, self.y_train)
        self.validate("train")

    def validate(self, type, y_pred=None):
        if type == "train":
            print(f"Training dataset evaluation:")
            #ad.custom_crossvalidation(self.X_train, self.y_train, self.lr_clf)
            ad.xval_regress(self.lr_clf, self.X_train, self.y_train, lin=True)
        else:
            print(f"Test dataset evaluation:")
            ad.custom_crossvalidation(self.X_test, y_pred, self.lr_clf)
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(self.y_test, y_pred),
                                          display_labels=self.lr_clf.classes_)
            disp.plot()


    def predict(self):
        """
        custom function to predict labels for Decision Tree
        Note that the datasets are already encoded
        :return:
        """
        num_columns = self.X_train.select_dtypes(include=['number']).columns.tolist()

        preprocessor_steps = []
        # preprocessor_steps.append(('cat', self.encoder, cat_columns))
        if self.scaler is not None:
            preprocessor_steps.append(('num', self.scaler, num_columns))
            preprocessor = ColumnTransformer(preprocessor_steps)

            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('classifier', self.lr_clf)
            ])
        else:
            pipeline = Pipeline([
                ('classifier', self.lr_clf)
            ])

        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)

        # Make predictions and display results
        print(f"Prediction Classification Report:\n {classification_report(self.y_test, y_pred)}")
        print(f"Prediction Report:\n")
        self.validate("test", y_pred)


