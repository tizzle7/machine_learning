# iris_perceptron.py

# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adaline import Adaline
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
iris_ada1 = Adaline(n_iterations=10, eta=0.01)
iris_ada1.fit(X.values, y.values)

iris_ada2 = Adaline(n_iterations=10, eta=0.0001)
iris_ada2.fit(X.values, y.values)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

ax[0].plot(range(1, iris_ada1.n_iterations + 1), np.log10(iris_ada1.cost_),
           marker="o")
ax[0].set(xlabel="Epochs", ylabel="log(SSE)",
          title="Adaline - Learning rate 0.01")

ax[1].plot(range(1, iris_ada2.n_iterations + 1), iris_ada2.cost_,
           marker="o")
ax[1].set(xlabel="Epochs", ylabel="log(SSE)",
          title="Adaline - Learning rate 0.0001")

plt.show()
