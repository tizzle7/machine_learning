# pipelines.py

import pandas as pd
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

pipe_lr.fit(X_train, y_train)

print("Test Accuracy: {}".format(pipe_lr.score(X_test, y_test)))
