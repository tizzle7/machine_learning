# scikit-learn_svm.py

import sys
sys.path.append("/home/titian/Desktop/machine_learning")

import matplotlib.pyplot as plt
import numpy as np
from plot_tools import plot_decision_regions
from sklearn import datasets
from sklearn.cross_validation import train_test_split
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
svm = SVC(kernel="linear", C=1.0, random_state=0)
svm.fit(X_train_std, y_train)

# plot the decision regions
X_std = np.vstack((X_train_std, X_test_std))
y_std = np.hstack((y_train, y_test))

fig1, ax = plot_decision_regions(X_std, y_std, svm, test_idx=range(105, 150))
ax.set(xlabel="Petal lenght [standardized]", ylabel="Petal width [standardized]")
ax.legend(loc="upper left")

plt.show()

