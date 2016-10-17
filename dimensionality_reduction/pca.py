# pca.py

import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_tools import plot_decision_regions
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# import and process the wine data
wine = pd.read_csv("wine.data")
X, y = wine.iloc[:, 1:].values, wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# compute the covariance matrix and its eigenvectors (directions of maximum
# variance) and the corresponding eigenvalues (magnitude of variance
# direction)
cov_matrix = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)

# plot the variance explained ratios of the eigenvectors
eigen_vals_sorted = sorted(eigen_vals, reverse=True)
var_exp = [(i / sum(eigen_vals)) for i in eigen_vals_sorted]
cum_var_exp = np.cumsum(var_exp)

fig1, ax = plt.subplots()
ax.bar(range(1, 14), var_exp, label="explained variance")
ax.step(range(1, 14), cum_var_exp, label="cumulative explained variance")
ax.set(xlabel="Principal components", ylabel="Explained variance ratio")
ax.legend(loc="best")

# create eigenpairs and sort them by decreasing order of the eigenvalues
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(0, len(eigen_vals))]
eigen_pairs.sort(reverse=True)

# select the two eigenvectors that correspond to the largest variances (about
# 60%) in this dataset, create the 13x2 projection matrix
w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))

# transform the training data onto the reduced two-dimensional subspace
X_train_pca = X_train_std.dot(w)

# visualize the transformed dataset in a two-dimensional plot
fig2, ax = plt.subplots()
colors = ["red", "blue", "green"]
markers = ["s", "x", "o"]

for lab, col, mark in zip(np.unique(y_train), colors, markers):
    ax.scatter(X_train_pca[y_train==lab, 0],
               X_train_pca[y_train==lab, 1],
               color=col, label=lab, marker=mark)

ax.set(xlabel="PC 1", ylabel="PC 2")
ax.legend(loc="top left")

# use the scikit-learn PCA for dimensionality reduction and classify the
# transformed samples using logistic regression
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression()
lr.fit(X_train_pca, y_train)

# plot the decision regions for the training data
fig3, ax = plot_decision_regions(X_train_pca, y_train, classifier=lr)
ax.set(xlabel="PC 1", ylabel="PC 2")
ax.legend(loc="lower left")

# plot the decision regions for the test set
fig4, ax = plot_decision_regions(X_test_pca, y_test, classifier=lr)
ax.set(xlabel="PC 1", ylabel="PC 2")
ax.legend(loc="lower left")

# just calculate the explained variance ratios
pca = PCA(n_components=None)
X_train_pca = pca.fit_transform(X_train_std)

print(pca.explained_variance_ratio_)

plt.show()
