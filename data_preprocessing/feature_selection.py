# feature_selection.py

from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# import the wine dataset
wine = pd.read_csv("wine.data", header=None)
wine.columns = ["Class label", "Alcohol", "Malic acid", "Ash",
                "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
                "Nonflavanoid phenols", "Proanthocyanins", "Color intensity",
                "Hue", "OD280/OD315 of diluted wines", "Proline"]

# use train_test_split function to generaze separate training and test sets,
# assign 70% of the data to the training set and 30% to the test set
X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)


# standardize the features using the corresponding scikit-learn scaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# L1 regularization leads to sparse solutions, meaning that most feature weights
# will be zero, indicating which features are influencing the class labels the
# most
lr = LogisticRegression(penalty="l1", C=0.1)
lr.fit(X_train_std, y_train)

# both training and test accuracies do not indicate any overfitting
print("Training accuracy: {}".format(lr.score(X_train_std, y_train)))
print("Test accuracy: {}".format(lr.score(X_test_std, y_test)))

# print out the intercept and weight arrays of the fitted model, the weight
# vectors are sparse and as a result of the L1 regularization the model is
# robust to the potentially irrelevant features in the dataset
print(lr.intercept_)
print()
print(lr.coef_)

# plot the weight coefficients as a function of different regularization
# strengths, it shows that all the feature weights will be zero with a high
# regularization parameter
fig1, ax = plt.subplots(1, 1, figsize=(10, 7))
colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "pink",
          "lightgreen", "lightblue", "gray", "indigo", "orange"]

params, weights = [], []
for c in np.arange(-6, 6):
    lr = LogisticRegression(penalty="l1", C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)

    params.append(10**c)
    weights.append(lr.coef_[1]) # weights for first class label

weights = np.array(weights)

for i, color in zip(range(weights.shape[1]), colors):
    ax.plot(params, weights[:, i], label=wine.columns[i + 1], color=color)

ax.set(xlabel="C", xscale="log", xlim=[10**(-5), 10**5],
       ylabel="weight coefficient")
ax.axhline(0, color="black", linestyle="--", linewidth=2)

# implement the Sequential Backward Selection algorithm to select the most
# important features of the dataset
class SBS():

    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.estimator = clone(estimator)
        self.k_features = k_features # desired number of features to return
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """Remove the feature not contained in the best performing subset until
        k is reached.
        """
        X_train, X_test, y_train, y_test = \
                 train_test_split(X, y, test_size=self.test_size,
                                  random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim)) # column indices of final features
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score] # accuracy scores of best subset

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=(dim - 1)):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            self.scores_.append(scores[best])

            dim -= 1

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        """Return a new data array containing only the best scoring feature
        columns.
        """
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)

        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)

        return score

# apply the SBS alhorithm to a KNN classifier
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plot the classification accuracy of the KNN model based on the number of
# features used
k_features = [len(k) for k in sbs.subsets_]

fig2, ax = plt.subplots()
ax.plot(k_features, sbs.scores_, marker="o")
ax.set(xlabel="Number of features", ylabel="Accuracy", ylim=[0.7, 1.1])
ax.grid()

# check which features lead to a accuracy of 100%
k5 = list(sbs.subsets_[8]) # sample with 5 features is the 9th data point from the righ
print(wine.columns[1:][k5])

# performance of the KNN classifier on the whole dataset
knn.fit(X_train_std, y_train)
print("Training accuracy: {}".format(knn.score(X_train_std, y_train)))
print("Test accuracy: {}".format(knn.score(X_test_std, y_test)))

# performance of the classifier on the determined 5-feature subset
knn.fit(X_train_std[:, k5], y_train)
print("Training accuracy: {}".format(knn.score(X_train_std[:, k5], y_train)))
print("Test accuracy: {}".format(knn.score(X_test_std[:, k5], y_test)))

plt.show()
