# iris_perceptron.py

# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adaline import AdalineGD, AdalineSGD
from plot_tools import plot_decision_regions

data = pd.read_csv("iris.data", header=None)

# set column headers
data.columns = ["Sepal length", "Sepal width", "Petal length", "Petal width",
                "Class label"]

# only select the setosa and versicolor species
data = data.loc[(data["Class label"] == "Iris-setosa") |
                (data["Class label"] == "Iris-versicolor")]

# convert species labels to integers, -1 for setosa and 1 for versicolor
data["Class label"] = np.where(data["Class label"]=="Iris-setosa", -1, 1)

# split data into feature matrix (only "Sepal length" and "Petal length" are
# used) and target vector
X = data[["Sepal length", "Petal length"]]
y = data["Class label"]

# create a plot of the cost function against the iteration number for learning
# rates of 0.01 and 0.0001
iris_ada1 = AdalineGD(n_iterations=10, eta=0.01)
iris_ada1.fit(X.values, y.values)

iris_ada2 = AdalineGD(n_iterations=10, eta=0.0001)
iris_ada2.fit(X.values, y.values)

fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

ax[0].plot(range(1, iris_ada1.n_iterations + 1), np.log10(iris_ada1.cost_),
           marker="o")
ax[0].set(xlabel="Iterations", ylabel="log(SSE)",
          title="Adaline - Learning rate 0.01")

ax[1].plot(range(1, iris_ada2.n_iterations + 1), iris_ada2.cost_,
           marker="o")
ax[1].set(xlabel="Iterations", ylabel="SSE",
          title="Adaline - Learning rate 0.0001")

# standardize the features to increase the performance of the gradient descent
# convergence method, convert data points to normal distribution
X_std = np.copy(X.values)
X_std[:, 0] = (X.values[:, 0] - X.values[:, 0].mean()) / X.values[:, 0].std()
X_std[:, 1] = (X.values[:, 1] - X.values[:, 1].mean()) / X.values[:, 1].std()

# train Adaline classifier on the standardized data using a learning rate of
# 0.01, plot the change of the cost function over the iterations and show the
# decision regions
iris_ada3 = AdalineGD(n_iterations=15, eta=0.01)
iris_ada3.fit(X_std, y.values)

fig2, ax = plot_decision_regions(X_std, y.values, iris_ada3)
ax.set(xlabel="Sepal length [standardized]",
       ylabel="Petal length [standardized]",
       title="AdalineGD")
ax.legend(loc="upper left")

fig3, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(range(1, iris_ada3.n_iterations + 1), iris_ada3.cost_, marker="o")
ax.set(xlabel="Iterations", ylabel="SSE", xlim=(1, iris_ada3.n_iterations),
       title="AdalineGD - Learning rate 0.01, standardized")

# show the decision regions for thestochastic gradient descent Adaline
# classifier
iris_ada4 = AdalineSGD(n_iterations=15, eta=0.01, random_state=1)
iris_ada4.fit(X_std, y.values)
fig4, ax = plot_decision_regions(X_std, y.values, iris_ada4)
ax.set(xlabel="Sepal length [standardized]",
       ylabel="Petal length [standardized]",
       title="AdalineSGD")
ax.legend(loc="upper left")

# plot the change of the cost function over the iterations for the trained
# SGDAdaline object
fig5, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(range(1, iris_ada4.n_iterations + 1), iris_ada4.cost_, marker="o")
ax.set(xlabel="Iterations", ylabel="SSE",
       title="AdalineSGD - Learning rate 0.01, standardized")

plt.show()
