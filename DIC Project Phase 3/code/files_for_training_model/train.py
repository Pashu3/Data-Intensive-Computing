import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle

data = pd.read_csv('files_for_training_model/transaction_dataset.csv')

def predict(data):
    data.drop(['Unnamed: 0', 'Address', 'Index'], axis=1, inplace=True)
    col = data.select_dtypes(['object', 'category'])
    data.drop(col, axis=1, inplace=True)
    data[data.columns] = data[data.columns].apply(pd.to_numeric, errors='coerce')
    data.fillna(data.median(), inplace=True)
    X = data.drop('FLAG', axis=1)
    y = data['FLAG']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    print(xgb.score(X_train, y_train))
    pickle.dump(xgb, open("models/finalized_model.pkl", 'wb'))
    prediction_test = xgb.predict(X_test)
    return None
predict(data)

test_data = []
def test_model (test_data):
    model = pickle.load(open('models/finalized_model.pkl', 'rb'))
    prediction_test = model.predict(test_data)
    return prediction_test[0]
test_model(test_data)
