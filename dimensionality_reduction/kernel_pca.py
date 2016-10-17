# kernel_pca.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import exp
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_circles, make_moons
from sklearn.decomposition import PCA, KernelPCA

def rbf_kernel_pca(X, gamma, n_components):
    """Radial Basis Function kernel PCA implementation.

    Arguments:
    X -- data array containing all features for each sample
    gamma -- tuning parameter for the rbf kernel
    n_components -- number of principal components to return

    Returns:
    X_kern_pca -- projected dataset
    eigen_vals_K -- list of eigenvalues of the transformation matrix K
    """
    
    # calculate the pairwise squared distances between the data points
    sq_dists = pdist(X, "sqeuclidean")

    # convert the distances into a square matrix
    mat_sq_dists = squareform(sq_dists)

    # calculate the kernel matrix
    K = exp(-gamma * mat_sq_dists)

    # center the kernel matrix
    N = K.shape[0]
    one_N = np.ones((N, N)) / N
    K = K - one_N.dot(K) - K.dot(one_N) + one_N.dot(K).dot(one_N)

    # obtain the eigenpairs in sorted order
    eigen_vals, eigen_vecs = eigh(K)

    # collect the top n eigenvectors (projected samples)
    X_kern_pca = np.column_stack((eigen_vecs[:, -i] for i in range(1, n_components + 1)))

    # collect the corresponding eigenvalues
    eigen_vals_K = [eigen_vals[-i] for i in range(1, n_components + 1)]
    
    return X_kern_pca, eigen_vals_K

# create a nonlinear example dataset representing two half-moon shapes
X, y = make_moons(n_samples=100, random_state=123)

fig1, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="^", alpha=0.5)
ax.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="o", alpha=0.5)

# project the dataset using standard PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig2, [ax1, ax2] = plt.subplots(1, 2)
ax1.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color="red", marker="^",
            alpha=0.5)
ax1.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color="blue", marker="o",
            alpha=0.5)
ax1.set(xlabel="PC 1", ylabel="PC 2")
            
ax2.scatter(X_pca[y == 0, 0], np.zeros((50, 1)), color="red", marker="^",
            alpha=0.5)
ax2.scatter(X_pca[y == 1, 0], np.zeros((50, 1)), color="blue", marker="o",
            alpha=0.5)
ax2.set(xlabel="PC 1", yticks=[])

# apply the RBF kernel PCA function to project the dataset
X_kern_pca, eigen_vals = rbf_kernel_pca(X, gamma=15, n_components=2)

fig3, [ax1, ax2] = plt.subplots(1, 2)
ax1.scatter(X_kern_pca[y == 0, 0], X_kern_pca[y == 0, 1], color="red",
            marker="^", alpha=0.5)
ax1.scatter(X_kern_pca[y == 1, 0], X_kern_pca[y == 1, 1], color="blue",
            marker="o", alpha=0.5)
ax1.set(xlabel="PC 1", ylabel="PC 2")

ax2.scatter(X_kern_pca[y == 0, 0], np.zeros((50, 1)), color="red", marker="^",
            alpha=0.5)
ax2.scatter(X_kern_pca[y == 1, 0], np.zeros((50, 1)), color="blue", marker="o",
            alpha=0.5)
ax2.set(xlabel="PC 1", yticks=[])

# create a dataset consisting of two concentric circles
X, y = make_circles(n_samples=1000, random_state=123, noise=0.1, factor=0.2)

fig4, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], color="red", marker="^", alpha=0.5)
ax.scatter(X[y == 1, 0], X[y == 1, 1], color="blue", marker="o", alpha=0.5)

# apply standard PCA to the circle dataset
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig5, [ax1, ax2] = plt.subplots(1, 2)
ax1.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color="red", marker="^",
            alpha=0.5)
ax1.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color="blue", marker="o",
            alpha=0.5)
ax1.set(xlabel="PC 1", ylabel="PC 2")
            
ax2.scatter(X_pca[y == 0, 0], np.zeros((500, 1)), color="red", marker="^",
            alpha=0.5)
ax2.scatter(X_pca[y == 1, 0], np.zeros((500, 1)), color="blue", marker="o",
            alpha=0.5)
ax2.set(xlabel="PC 1", yticks=[])

# use the RBF kernel method for the data projection
X_kern_pca, eigen_vals = rbf_kernel_pca(X, gamma=15, n_components=2)

fig6, [ax1, ax2] = plt.subplots(1, 2)
ax1.scatter(X_kern_pca[y == 0, 0], X_kern_pca[y == 0, 1], color="red",
            marker="^", alpha=0.5)
ax1.scatter(X_kern_pca[y == 1, 0], X_kern_pca[y == 1, 1], color="blue",
            marker="o", alpha=0.5)
ax1.set(xlabel="PC 1", ylabel="PC 2")

ax2.scatter(X_kern_pca[y == 0, 0], np.zeros((500, 1)), color="red", marker="^",
            alpha=0.5)
ax2.scatter(X_kern_pca[y == 1, 0], np.zeros((500, 1)), color="blue", marker="o",
            alpha=0.5)
ax2.set(xlabel="PC 1", yticks=[])

# project new data points that were not part of the training set
def project_x(x_new, X, gamma, X_kern_pca, eigen_vals):
    pair_dist = np.array([np.sum((x_new - row)**2) for row in X])
    K = np.exp(-gamma * pair_dist)

    return K.dot(X_kern_pca / eigen_vals)

X, y = make_moons(n_samples=100, random_state=123)
X_kern_pca, eigen_vals = rbf_kernel_pca(X, gamma=15, n_components=1)

x_new = X[25]
x_proj = project_x(x_new, X, 15, X_kern_pca, eigen_vals)

print(x_proj, X_kern_pca[25]) # are identical

# use the built-in scikit-learn method for kernel PCA
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
X_kern_pca = kpca.fit_transform(X)

fig7, ax = plt.subplots()
ax.scatter(X_kern_pca[y==0, 0], X_kern_pca[y==0, 1], color="red", marker="^",
           alpha=0.5)
ax.scatter(X_kern_pca[y==1, 0], X_kern_pca[y==1, 1], color="blue", marker="o",
           alpha=0.5)
ax.set(xlabel="PC 1", ylabel="PC 2")


plt.show()
