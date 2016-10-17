# cross_validation.py

import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# chain data scaling, PCA, and modeling trough logistic regression into a single
# pipeline
pipe_lr = Pipeline([("scl", StandardScaler()),
                    ("pca", PCA(n_components=2)),
                    ("clf", LogisticRegression(random_state=1))])

# use the stratified k-fold cross-validation method for parameter tuning
strat_kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)

scores = []

for k, (train, test) in enumerate(strat_kfold):
    pipe_lr.fit(X_train[train], y_train[train])

    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)

    print("Fold: {}, Class prop.: {}, Score: {}".format((k + 1),
                                                        np.bincount(y_train[train]),
                                                        score))

print("Cross-validation accuracy: {} +- {}".format(np.mean(scores),
                                                   np.std(scores)))

# use the scikit k-fold cross-validation scorer for more efficient computing,
# n_jobs sets the number of cpus used for calculation
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10,
                         n_jobs=1)

print("Cross-validation accuracy scores: {}".format(scores))
print("Cross-validation accuracy: {} +- {}".format(np.mean(scores),
                                                   np.std(scores)))

