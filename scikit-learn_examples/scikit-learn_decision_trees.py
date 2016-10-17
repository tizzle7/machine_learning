# scikit-learn_decision_trees.py

import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
from plot_tools import plot_decision_regions
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# observe the impurity indices for a probability range between 0 and 1
def gini_index(p):
    """Gini index for a binary classification problem."""
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

def entropy(p):
    """Entropy for a binary classification problem."""
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def class_error(p):
    """Classification error for a binary classification problem."""
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
ent_sc = [e / 2.0 if e else None for e in ent]  # scaled entropy (ent / 2)
gin = gini_index(x)
err = [class_error(p) for p in x]

print(err)

fig1, ax = plt.subplots(1, 1)

labels = ["Gini index", "Entropy", "Entropy (scaled)", "Classification error"]
linestyles = ["--", "-", "-", "-."]
colors = ["red", "black", "lightgray", "green"]

for i , lab, ls, col, in zip([gin, ent, ent_sc, err], labels, linestyles,
                             colors):
    
    ax.plot(x, i, label=lab, linestyle=ls, color=col)

ax.set(xlabel="p(i=1)", ylabel="Impurity Index", ylim=([0, 1.1]))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3,
          fancybox=True, shadow=False)
ax.axhline(y=0.5, linewidth=1, color="k", linestyle="--")
ax.axhline(y=1.0, linewidth=1, color="k", linestyle="--")

# load the iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]    # only select petal length and width
y = iris.target

# split the data into separate training and test datasets (70% training data
# and 30% test data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

# train a decision tree with a maximum depth of 3 using entropy as impurity
# criterion
tree = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0)
tree.fit(X_train, y_train)

# plot the decision regions
X = np.vstack((X_train, X_test))
y = np.hstack((y_train, y_test))

fig2, ax = plot_decision_regions(X, y, tree, test_idx=range(105, 150))
ax.set(xlabel="Petal length [cm]", ylabel="Petal width [cm]")
ax.legend(loc="upper left")

# export the decision tree as a .dot file that can be visualized by GraphViz
export_graphviz(tree, out_file="tree.dot", feature_names=["petal length",
                                                          "petal width"])

# train a random forest classifier with 10 trees using two cores of our
# computer with the iris dataset and plot its decision boundaries
forest = RandomForestClassifier(criterion="entropy", n_estimators=10,
                                random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

fig3, ax = plot_decision_regions(X, y, forest, test_idx=range(105, 150))
ax.set(xlabel="Petal length [cm]", ylabel="Petal width [cm]")
ax.legend(loc="upper left")

plt.show()


