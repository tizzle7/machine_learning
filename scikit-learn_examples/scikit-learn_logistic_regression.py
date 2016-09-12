# scikit-learn_logistic_regression.py

import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
from plot_tools import plot_decision_regions
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# plot of the sigmoid function from -7 to 7
z = np.arange(-7, 7, 0.1)
phi = sigmoid(z)

fig1, ax = plt.subplots(1, 1)
ax.plot(z, phi)
ax.axvline(0.0, color="k")
ax.axhspan(0.0, 1.0, facecolor="1.0", alpha=1.0, ls="dotted")
ax.axhline(y=0.5, ls="dotted", color="k")

ax.set(xlabel="z", ylabel="$\phi$ (z)", ylim=(-0.1, 1.1),
       yticks=[0.0, 0.5, 1.0])

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

# create and train a logistic regression model
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# plot the decision regions
X_std = np.vstack((X_train_std, X_test_std))
y_std = np.hstack((y_train, y_test))

fig2, ax = plot_decision_regions(X_std, y_std, lr, test_idx=range(105, 150))
ax.set(xlabel="Petal lenght [standardized]", ylabel="Petal width [standardized]")
ax.legend(loc="upper left")

# predict the class-membership probabilities
prob = lr.predict_proba(X_test_std)
print(prob)

# plot the L2 regularization path
weights, params = [], []

for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights) # create a concise list

fig3, ax = plt.subplots(1, 1)
ax.plot(params, weights[:, 0], label="Petal length")
ax.plot(params, weights[:, 1], linestyle="--", label="Petal width")
ax.set(xlabel="C", ylabel="Weight coefficient", xscale="log")
ax.legend(loc="upper left")

plt.show()
