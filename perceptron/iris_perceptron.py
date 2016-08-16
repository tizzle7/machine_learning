# iris_perceptron.py

# dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from perceptron import Perceptron
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

# plot the sepal lenght versus the petal length for both iris species
fig1 = plt.figure()
ax = fig1.add_subplot(1, 1, 1)

# setosa iris
ax.scatter(X["Sepal length"].iloc[:50], X["Petal length"].iloc[:50],
           color="red", marker="o", label="Iris setosa")

# versicolor iris
ax.scatter(X["Sepal length"].iloc[50:], X["Petal length"].iloc[50:],
           color="blue", marker="o", label="Iris versicolor")

ax.set(xlabel="Sepal lenght", ylabel="Petal length")
ax.legend(loc="upper left")

# train the perceptron classifier
iris_ppn = Perceptron()
iris_ppn.fit(X.values, y.values)

# plot the numbers of errors for each perceptron iteration
fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)

ax.plot(range(1, iris_ppn.n_iterations + 1), iris_ppn.errors_, marker="o")

ax.set(xlabel="Iteration", ylabel="Number of missclassifications")

# plot the decision boundaries for the 2D dataset
fig3, ax = plot_decision_regions(X.values, y.values, iris_ppn)

ax.set(xlabel="sepal length [cm]", ylabel= "petal length [cm]")
ax.legend(loc="upper left")

plt.show()
