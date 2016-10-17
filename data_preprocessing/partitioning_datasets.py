# partitioning datasets.py

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split


wine = pd.read_csv("wine.data", header=None)
wine.columns = ["Class label", "Alcohol", "Malic acid", "Ash",
                "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
                "Nonflavanoid phenols", "Proanthocyanins", "Color intensity",
                "Hue", "OD280/OD315 of diluted wines", "Proline"]

print(wine.head())

# class labels belong to different grape types
print("Class labels: {}".format(np.unique(wine["Class label"])))

# use train_test_split function to generaze separate training and test sets,
# assign 70% of the data to the training set and 30% to the test set
X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)


