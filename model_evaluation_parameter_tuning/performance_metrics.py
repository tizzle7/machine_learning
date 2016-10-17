# performance_metrics.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# load the breast cancer dataset
bc = pd.read_csv("wdbc.data")

X = bc.iloc[:, 2:].values
y = bc.iloc[:, 1].values

# transform the class labels from strings to integers (M = 1, B = 0)
le = LabelEncoder()
y = le.fit_transform(y)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

# define the SVM model
pipe_svc = Pipeline([("scl", StandardScaler()),
                     ("clf", SVC(random_state=1))])

# fit the model and create a confusion matrix to represent the results
pipe_svc.fit(X_train, y_train)

y_pred = pipe_svc.predict(X_test)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

print(conf_matrix)

# create confusion matrix illustration using matplotlibs matshow function
fig1, ax = plt.subplots()
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
ax.set(xlabel="Predicted label", ylabel="True label")

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center")

# use implemented scikit functions to assess the model performance
print("Precision score: {}".format(precision_score(y_true=y_test,
                                                   y_pred=y_pred)))
print("Recall score: {}".format(recall_score(y_true=y_test, y_pred=y_pred)))
print("F1-score: {}".format(f1_score(y_true=y_test, y_pred=y_pred)))

# positive class is always defined as the one labeled as class 1, to change
# this one can construct an own scorer
scorer = make_scorer(f1_score, pos_label=1)

# plot on ROC curve for the classifier using only two feature of the dataset
X_train2 = X_train[:, [4, 14]]
cv = StratifiedKFold(y_train, n_folds=3, random_state=1)
pipe_lr = Pipeline([("scl", StandardScaler()),
                    ("clf", LogisticRegression(random_state=1))])

fig2, ax = plt.subplots()

mean_tpr = 0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

# plot the roc curves for all 3 folds
for i, (train, test) in enumerate(cv):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(
        X_train2[test])

    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)

    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0

    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, lw=1, label="ROC fold {} (area = {})".format(
        i + 1, roc_auc))

# plot a diagonal representing random class guessing
ax.plot([0, 1], [0, 1], linestyle="--", color=(0.6, 0.6, 0.6),
        label="Random guessing")

# plot the average roc curve
mean_tpr /= len(cv)
mean_auc = auc(mean_fpr, mean_tpr)

ax.plot(mean_fpr, mean_tpr, linestyle="--", color="k",
        label="Mean ROC (area = {})".format(mean_auc), lw=2)

# plot the ROC of a perfect classifier
ax.plot([0, 0, 1], [0, 1, 1], lw=2, linestyle=":", color="k",
        label="Perfect performance")

ax.set(xlabel="False positive rate", xlim=[-0.05, 1.05],
       ylabel="True positive rate", ylim=[-0.05, 1.05])

ax.legend(loc="best")

# AUC score can also be directly calculated
pipe_svc.fit(X_train2, y_train)
y_pred2 = pipe_svc.predict(X_test[:, [4, 14]])

print("ROC AUC: {}".format(roc_auc_score(y_true=y_test, y_score=y_pred2)))


plt.show()

