import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv("processed.cleveland.csv", na_values=['?'])

clean_data = data.fillna(data.median(numeric_only=True))
#clean_data = data.dropna()

clean_data['num_binary'] = (clean_data['num'] > 0).astype(int)
clean_data = pd.get_dummies(clean_data, columns=["cp", "slope", "thal", "restecg"], drop_first=True)
clean_data["thalach"] = -clean_data["thalach"]


correlation = clean_data.corr(numeric_only=True)['num_binary'].sort_values(ascending=False)
strong_corr = correlation[correlation > 0.3].drop(labels=['num', 'num_binary'], errors='ignore')


X = clean_data[strong_corr.index]
Y = (clean_data['num_binary'])
mean_accuracies = []

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

    mean_accuracy = np.mean(accuracies)
    mean_accuracies.append(mean_accuracy)

    print(f"{tesT_size} Accuracy: {np.mean(accuracies)}")


test_sizes = [1-(x/10) for x in range(1, 10)]
plt.figure(figsize=(8, 5))
plt.plot(test_sizes, mean_accuracies, marker='o', linestyle='-', color='b')
plt.title('Test Seti Oranı vs Ortalama Doğruluk')
plt.xlabel('Eğitim Seti Oranı')
plt.ylabel('Ortalama Doğruluk')
plt.grid(True)
plt.xticks(test_sizes)
plt.ylim(0.8, 0.88)
plt.show()


