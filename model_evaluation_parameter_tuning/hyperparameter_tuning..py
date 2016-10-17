# hyperparameter_tuning.py

import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

# tuning the SVM models hyperparameters using grid search
pipe_svc = Pipeline([("scl", StandardScaler()),
                     ("clf", SVC(random_state=1))])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# create a parameter grid where we only tune the inverse regularization
# parameter for the linear kernel SVM but both C and gamma for the RBF kernel
# SVM
param_grid = [{"clf__C": param_range,
               "clf__kernel": ["linear"]},
              {"clf__C": param_range,
               "clf__gamma": param_range,
               "clf__kernel": ["rbf"]}]

gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring="accuracy",
                  cv=10,
                  n_jobs=1)

gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

# estimate the performance of the best SVM on the test dataset
best_svc = gs.best_estimator_
best_svc.fit(X_train, y_train)

print("Test accuracy: {}".format(best_svc.score(X_test, y_test)))

# algorithm selection through nested cross-validation
gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring="accuracy",
                  cv=10, n_jobs=1)
scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)

print("CV accuracy SVC: {} +/- {}".format(np.mean(scores), np.std(scores)))

# tune the depth parameter of a decision tree algorithm to compare its
# performance to the SVM model
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{"max_depth": [1, 2, 3, 4, 5, 6, 7, None]}],
                  scoring="accuracy", cv=5)
scores = cross_val_score(gs, X_train, y_train, scoring="accuracy", cv=5)

print("CV accuracy Tree: {} +/- {}".format(np.mean(scores), np.std(scores)))
                  

