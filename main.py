import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("processed.cleveland.csv", na_values=['?'])

clean_data = data.fillna(data.median(numeric_only=True))
#clean_data = data.dropna()

clean_data['num_binary'] = (clean_data['num'] > 0).astype(int)
clean_data = pd.get_dummies(clean_data, columns=["cp", "slope", "thal", "restecg"], drop_first=True)
clean_data["thalach"] = -clean_data["thalach"]


correlation = clean_data.corr(numeric_only=True)['num_binary'].sort_values(ascending=False)
#print(correlation)
strong_corr = correlation[correlation > 0.3].drop(labels=['num', 'num_binary'], errors='ignore')
#print(strong_corr_2)


X = clean_data[strong_corr.index]
Y = (clean_data['num_binary'])

for x in range(1, 10):
    tesT_size = x/10

    accuracies = []
    for seed in range(50):
        X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=tesT_size, random_state=seed)

        log_regression = LogisticRegression(max_iter=1000, solver="liblinear")

        log_regression.fit(X_train, Y_train)

        Y_pred = log_regression.predict(X_test)

        #accuracy = accuracy_score(Y_test, Y_pred)
        accuracies.append(accuracy_score(Y_test, Y_pred))
        #conf_matrix = confusion_matrix(Y_test, Y_pred)
        #class_report = classification_report(Y_test, Y_pred)

    #print(f'{x/10} Accuracy: {accuracy}')
    print(f"{tesT_size} Accuracy: {np.mean(accuracies)}")


