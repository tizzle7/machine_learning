# alogithm_debugging.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# load the breast cancer dataset
bc = pd.read_csv("wdbc.data")

X = bc.iloc[:, 2:].values
y = bc.iloc[:, 1].values

# transform the class labels from strings to integers (M = 1, B = 0)
le = LabelEncoder()
y = le.fit_transform(y)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

# define the pipeline
pipe_lr = Pipeline([("scl", StandardScaler()),
                    ("clf", LogisticRegression(C=10, penalty="l2",
                                               random_state=0))])

# use the learning curve function to evaluate the model
train_sizes, train_scores, test_scores =\
             learning_curve(estimator=pipe_lr,
                            X=X_train,
                            y=y_train,
                            train_sizes=np.linspace(0.1, 1.0, 10),
                            cv=10,
                            n_jobs=1)

print(train_scores)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig1, ax = plt.subplots()
ax.plot(train_sizes, train_mean, color="blue", marker="o", markersize=5,
        label="Training accuracy")
ax.fill_between(train_sizes, train_mean + train_std, train_mean - train_std,
                color="blue", alpha=0.15)

ax.plot(train_sizes, test_mean, color="green", marker="s", markersize=5,
        label="Validation accuracy")
ax.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
                color="green", alpha=0.15)

ax.set(xlabel="Number of training samples", ylabel="Accuracy", ylim=[0.8, 1.0])
ax.legend(loc="lower right")
ax.grid()

# create a validation curve using the implemented scikit function
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=pipe_lr, X=X_train,
                                             y=y_train, param_name="clf__C",
                                             param_range=param_range, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig2, ax = plt.subplots()
ax.plot(param_range, train_mean, color="blue", marker="o", markersize=5,
        label="Training accuracy")
ax.fill_between(param_range, train_mean + train_std, train_mean - train_std,
                color="blue", alpha=0.15)

ax.plot(param_range, test_mean, color="green", marker="s", markersize=5,
        label="Validation accuracy")
ax.fill_between(param_range, test_mean + test_std, test_mean - test_std,
                color="green", alpha=0.15)

ax.set(xlabel="Parameter C", ylabel="Accuracy", ylim=[0.8, 1.0])
ax.legend(loc="lower right")
plt.xscale("log")
ax.grid()


plt.show()


