# scikit-learn_svm.py

import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
from plot_tools import plot_decision_regions
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]    # only select petal length and width
y = iris.target

# split the data into separate training and test datasets (70% training data
# and 30% test data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

# standardize the features
sc = StandardScaler()
sc.fit(X_train) # estimate sample mean and standard deviation for each feature
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# create and train the support vector machine model
svm1 = SVC(kernel="linear", C=1.0, random_state=0)
svm1.fit(X_train_std, y_train)

# plot the decision regions
X_std = np.vstack((X_train_std, X_test_std))
y_std = np.hstack((y_train, y_test))

fig1, ax = plot_decision_regions(X_std, y_std, svm1, test_idx=range(105, 150))
ax.set(xlabel="Petal lenght [standardized]", ylabel="Petal width [standardized]")
ax.legend(loc="upper left")

# use the SGDClassifier implementation that also supports online learning for
# large datasets
svm2 = SGDClassifier(loss="hinge")
svm2.fit(X_train_std, y_train)

fig2, ax = plot_decision_regions(X_std, y_std, svm2, test_idx=range(105, 150))
ax.set(xlabel="Petal lenght [standardized]", ylabel="Petal width [standardized]")
ax.legend(loc="upper left")

# create an XOR dataset with random noise to simulate a nonlinear classification
# problem
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

fig3, ax = plt.subplots(1, 1)
ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c="b", marker="x",
           label="1")
ax.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1], c="r", marker="s",
           label="-1")
ax.legend(loc="top right")

# the kernel method transforms the two-dimensional dataset onto a new three-
# dimensional feature space where the classes become linearly separable, train
# a kernel SVM to create nonlinear decision boundaries
svm3 = SVC(kernel="rbf", random_state=0, gamma=0.10, C=10.0)
svm3.fit(X_xor, y_xor)

fig4, ax = plot_decision_regions(X_xor, y_xor, svm3)
ax.set(xlabel="X1", ylabel="X2")
ax.legend(loc="upper left")

# apply the a kernel SVM to the iris dataset, use a low gamma value to create
# a relatively soft decision boundary
svm4 = SVC(kernel="rbf", random_state=0, gamma=0.2, C=1.0)
svm4.fit(X_train_std, y_train)

fig5, ax = plot_decision_regions(X_std, y_std, svm4, test_idx=range(105, 150))
ax.set(xlabel="Petal length [standardized]", ylabel="Petal width [standardized]")
ax.legend(loc="upper left")

# train a model with a high gamma value to create very tight decision boundaries
svm5 = SVC(kernel="rbf", random_state=0, gamma=100.0, C=1.0)
svm5.fit(X_train_std, y_train)

fi6, ax = plot_decision_regions(X_std, y_std, svm5, test_idx=range(105, 150))
ax.set(xlabel="Petal length [standardized]", ylabel="Petal width [standardized]")
ax.legend(loc="upper left")

plt.show()

