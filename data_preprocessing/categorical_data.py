# categorical_data.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# create a data frame containing nominal and ordinal features
df = pd.DataFrame([
    ["green", "M", 10.1, "class1"],
    ["red", "L", 13.5, "class2"],
    ["blue", "XL", 15.3, "class1"]])
df.columns = ["color", "size", "price", "classlabel"]

# convert the categorical size strings into integers
size_mapping = {
    "XL": 3,
    "L": 2,
    "M": 1}
df["size"] = df["size"].map(size_mapping)

# convert the class labels to integer values
class_mapping = {label: i for i, label in enumerate(np.unique(df["classlabel"]))}
df["classlabel"] = df["classlabel"].map(class_mapping)

# reverse the class mapping
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df["classlabel"] = df["classlabel"].map(inv_class_mapping)

# use the scikit-learn LabelEncoder to convert the class labels
class_le = LabelEncoder()
df["classlabel"] = class_le.fit_transform(df["classlabel"].values)

# encoding the color values may lead to a problem since the algorithm will assum
# that a certain color is "larger" than another one, use one-hot encoding to
# avoid this problem, create new dummy features for each color value
X = df[["color", "size", "price"]].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

print(X)

ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()

print(X)

# pandas get_dummies method only convert string columns and leaves all other
# columns unchanged
df = pd.get_dummies(df)

print(df)
