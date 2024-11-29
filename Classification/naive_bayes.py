from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import adsa_utils as ad
from sklearn.pipeline import Pipeline
import plots as plts  # custom utility
import Classification.data as data  # custom utility
import numpy as np

class NBayes:

    def __init__(self, X, y, scaler, encoder, nb_clf, show_confusion=True):
        self.X = X
        self.y = y
        self.scaler = scaler
        self.encoder = encoder
        self.nb_clf = nb_clf
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        # For additional data analysis given the output of the confusion matrix when using default hyperparameters
        # data.data_out(self.y_train, "dataout/adult/y_train.csv")
        # data.data_out(self.X_train, "dataout/adult/X_train.csv")#
        # data.data_out(self.y_test, "dataout/adult/y_test.csv")
        # data.data_out(self.X_test, "dataout/adult/X_test.csv")
        self.nb_clf.fit(self.X_train, self.y_train)
        self.validate("train", show_confusion=show_confusion)

    def validate(self, type, y_pred=None, show_confusion=True):
        if type == "train":
            print(f"Training dataset evaluation:")
            if show_confusion:
                ad.custom_crossvalidation(self.X_train, self.y_train, self.nb_clf)
            else:
                if self.y_train.select_dtypes(include=['object', 'category']).empty is False:
                    #y_train, ohe = data.encode_categorical_data_ohe(self.y_train)
                    y_train, le = data.encode_categorical_data_le(self.y_train)
                else:
                    y_train = self.y_train

                # Get predictions
                #y_pred_train = self.nb_clf.predict(self.X_train)
                print(f"Prediction Classification Report:\n {classification_report(self.X_train, y_train)}")
            # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(self.X_train, self.y_train),
            #                               display_labels=self.nb_clf.classes_).plot()
            # plt.show()

        else:
            print(f"Test dataset evaluation:")
            if self.y_test.select_dtypes(include=['object', 'category']).columns.any(): #encoding required
                #y_test_encoded, ohe = data.encode_categorical_data_ohe(self.y_test)
                y_test_encoded, le = data.encode_categorical_data_le(self.y_test)
                if show_confusion:
                    ad.custom_crossvalidation(y_test_encoded, y_pred, self.nb_clf)
                else:
                    print(f"Prediction Classification Report:\n {classification_report(y_train, y_pred)}")

                # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_encoded, y_pred),
                #                               display_labels=self.nb_clf.classes_)
            else:
                if show_confusion:
                    ad.custom_crossvalidation(self.y_test, y_pred, self.nb_clf)
                else:
                    print(f"Prediction Classification Report:\n {classification_report(self.y_test, y_pred)}")
            #     disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(self.y_test, y_pred),
            #                                   display_labels=self.nb_clf.classes_)
            #
            # disp.plot()
            # plt.show()


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
                ('classifier', self.nb_clf)
            ])
        else:
            pipeline = Pipeline([
                ('classifier', self.nb_clf)
            ])

        print(f"X_train {self.X_train.head()}")
        print(f"y_train {self.y_train.head()}")
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)

        # Make predictions and display results
        #print(f"Prediction Classification Report:\n {classification_report(self.y_test, y_pred)}")
        #print(f"Prediction Report:\n")
        self.validate("test", y_pred)

    def custom_grid_search(self):
        print("Starting grid search...")
        params = {
            'alpha': np.arange(0.0001, 1, 0.001),
            'norm': [True, False]
        }
        grid_search = GridSearchCV(self.nb_clf, param_grid=params, scoring='f1_macro')
        grid_search.fit(self.X_train, self.y_train)
        print(f"Best Params: {grid_search.best_params_}")
        print(f"Best Score: {grid_search.best_score_}")



