# missing_data.py

import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

# simple csv data file
csv_data = """A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,"""

# convert to pd data frame
df = pd.read_csv(StringIO(csv_data))

# number of missing values
print(df.isnull().sum())

# drop rows with missing values
print(df.dropna())

# drop columns containing missing values
print(df.dropna(axis=1))

# only drop rows where all values are missing
print(df.dropna(how="all"))

# drop only rows with more than 4 missing values
print(df.dropna(thresh=4))

# only drop rows where missing values appear in specific columns
print(df.dropna(subset=["C"]))

# replace missing values by the mean value of the entire feature column
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)

print(imputed_data)



