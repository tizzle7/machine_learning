# lda.py

import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_tools import plot_decision_regions
from sklearn.cross_validation import train_test_split
from sklearn.lda import LDA
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

# calculate the mean vectors of the data
mean_vecs = []
for lab in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == lab], axis=0))

# calculate within-class matrix as the sum of all the individual scatter
# matrices (since the class labels are not uniformly distributed in the data,
# the individual scatter matrices need to be scaled which makes them identical
# to the covariance matrices)
S_W = np.zeros((X_train_std.shape[1], X_train_std.shape[1]))
for lab, mv in zip(range(1, 4), mean_vecs):
    S_i = np.cov(X_train_std[y_train == lab].T)
    S_W += S_i

# calculate the between-class scatter matrix
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(X_train_std.shape[1], 1)

S_B = np.zeros((X_train_std.shape[1], X_train_std.shape[1]))
for lab, mv in zip(range(1, 4), mean_vecs):
    n = X[y == lab, :].shape[0]
    mv = mv.reshape(X_train_std.shape[1], 1)

    S_B += n * (mv - mean_overall).dot((mv - mean_overall).T)

# solve the eigenvalue problem of (S_W)⁽⁻¹⁾*S_B
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# form the eigenpairs and sort them in descending order
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
                for i in range(0, len(eigen_vals))]

eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# plot the "discriminability" ratios
eigen_vals_sorted = sorted(eigen_vals.real, reverse=True)
discr = [(i / sum(eigen_vals.real)) for i in eigen_vals_sorted]
cum_discr = np.cumsum(discr)

fig1, ax = plt.subplots()
ax.bar(range(1, 14), discr, label="individual discriminability")
ax.step(range(1, 14), cum_discr, label="cumulative discriminability")
ax.set(xlabel="Linear discriminants", ylabel="discriminability ratio",
       ylim=[-0.1, 1.1])
ax.legend(loc="best")

# create the transformation matrix
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
               eigen_pairs[1][1][:, np.newaxis].real))

# project the data set onto the new feature space and plot it
X_train_lda = X_train_std.dot(w)

fig2, ax = plt.subplots()
colors = ["red", "blue", "green"]
markers = ["s", "x", "o"]

for lab, col, mark in zip(np.unique(y_train), colors, markers):
    ax.scatter(X_train_lda[y_train == lab, 0],
               X_train_lda[y_train == lab, 1],
               color=col, label=lab, marker=mark)

ax.set(xlabel="LDA 1", ylabel="LDA 2")
ax.legend(loc="upper right")

# use LDA via scikit-learn and fit a logitics regression model for
# classification and plot the corresponding decision regions
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

lr = LogisticRegression()
lr.fit(X_train_lda, y_train)

fig3, ax = plot_decision_regions(X_train_lda, y_train, classifier=lr)
ax.set(xlabel="LD 1", ylabel="LD 2")
ax.legend(loc="best")

fig4, ax = plot_decision_regions(X_test_lda, y_test, classifier=lr)
ax.set(xlabel="LD 1", ylabel="LD 2")
ax.legend(loc="best")


plt.show()
