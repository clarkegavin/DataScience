import dbconnect
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
import adsa_utils as ad


import pandas as pd

# Load data
engine = dbconnect.connect()
#df = dbconnect.get_top100_data(engine)
df = dbconnect.get_all_data(engine)

# separate the class label from the regular attributes
y = df['inter1']
X = df.drop(['inter1'], axis=1)
X = X.drop(columns=['event_id_cnty'])

#split the dataset into 2 portions, 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43
)

#get categorical columns and encode them
categorical_columns = X.select_dtypes(include=['object']).columns
X_train[categorical_columns] = OrdinalEncoder().fit_transform(X_train[categorical_columns])
print(X_train.info())

#scale the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

#train the model
model = ComplementNB()
# model.fit(X_train, y_train)
print(f"Training dataset evaluation:")
ad.custom_crossvalidation(X_train, y_train, model)
