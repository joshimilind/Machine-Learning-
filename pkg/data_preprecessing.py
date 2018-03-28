import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reading data
dataset = pd.read_csv(
    '/home/synerzip/nlp/nlp_workspace/Machine Learning/pkg/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Data.csv')
# dataset = pd.read_csv('./Data.csv')
print(dataset)

# matrix of feature
X = dataset.iloc[:, :-1].values
print('x\n', X)

Y = dataset.iloc[:, 3].values
print('y\n', Y)

from sklearn.preprocessing import Imputer

np.set_printoptions(threshold=np.nan)
# handling missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# imputer = Imputer(missing_values='NaN', strategy='median', axis=0)

imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print('x\n', X)

# Encoding categorical variable
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

print('\nfit transform\n', X)
