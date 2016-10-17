# feature_importance.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# import the wine dataset
wine = pd.read_csv("wine.data", header=None)
wine.columns = ["Class label", "Alcohol", "Malic acid", "Ash",
                "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
                "Nonflavanoid phenols", "Proanthocyanins", "Color intensity",
                "Hue", "OD280/OD315 of diluted wines", "Proline"]

# split the data into a training and test set
X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

# fit the random forest classifier using 10000 trees on the wine data using
# all available cores
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

# print out list of the features, starting with the most important ones
features = wine.columns[1:]
feature_importances = forest.feature_importances_
# argsort returns a list containing the indices ordered from the smallest to
# the largest value in the specified list
indices = np.argsort(feature_importances)[::-1] # order from largest to smallest

for f in range(0, X_train.shape[1]):
    print("{}: {} {}".format(f + 1, features[f],
                                feature_importances[indices[f]]))

fig, ax = plt.subplots()
ax.bar(range(0, X_train.shape[1]), feature_importances[indices],
       color="lightblue", align="center")
ax.set(title="Feature Importance")
ax.set_xticks(range(0, X_train.shape[1]))
ax.set_xticklabels(features.values, rotation=90)
ax.set_xlim([-1, X_train.shape[1]])
fig.tight_layout()

# only select the three most important features found by the random forest
# classifier
X_important = forest.transform(X_train, threshold=0.15)
print(X_important.shape)

plt.show()




