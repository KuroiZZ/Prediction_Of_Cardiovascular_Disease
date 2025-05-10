import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("processed.cleveland.csv", na_values=['?'])
clean_data = data.dropna()

target = "num"

correlation = clean_data.corr(numeric_only=True)[target]

#print(correlation.sort_values(ascending=False))


X = clean_data[['exang', 'cp', 'oldpeak', 'thalach', 'ca', 'slope']]
Y = (clean_data['num'] > 0).astype(int)

for x in range(1, 10):
    tesT_size = x/10
    X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=tesT_size, random_state=0)

    log_regression = LogisticRegression(max_iter=1000, solver="liblinear")

    log_regression.fit(X_train, Y_train)

    Y_pred = log_regression.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    class_report = classification_report(Y_test, Y_pred)

    print(f'{x/10} Accuracy: {accuracy}')


