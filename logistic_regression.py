import pandas as pd
import numpy as np
from scipy.optimize import fmin_tnc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/Statology/Python-Guides/main/default.csv"
# data = pd.read_csv("raw_state.txt")
data = pd.read_csv(url)

X = data[[ 'income']]
y = data['default']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
#
# log_regres = LogisticRegression()
# log_regres.fit(x_train, y_train)
#
# y_pred = log_regres.predict(x_test)
#
# matrix = metrics.confusion_matrix(y_test, y_pred)
#
# print(matrix)
# print("Точность:", metrics.accuracy_score(y_test, y_pred))


# X = feature values, all the columns except the last column
# X = data.iloc[:, :-1]
#
# # y = target values, last column of the data frame
# y = data.iloc[:, -1]

# filter out the applicants that got admitted
admitted = data.loc[y == 1]

# filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.legend()
plt.show()


X = np.c_[np.ones((X.shape[0], 1)), X]
y = y[:, np.newaxis]
theta = np.zeros((X.shape[1], 1))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression()
model.fit(X, y)
predicted_classes = model.predict(X)
accuracy = accuracy_score(y.flatten(), predicted_classes)
parameters = model.coef_
# print(parameters)
# input()

x_values = [np.min(X[:, 1] - 5), np.max(X[:, 2] + 5)]
y_values = - (parameters[0][0] + np.dot(parameters[0][1], x_values)) / parameters[0][2]

plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()