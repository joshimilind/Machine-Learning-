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
print('X >>\n', X)

Y = dataset.iloc[:, 3].values
print('Y >>\n', Y)

from sklearn.preprocessing import Imputer

np.set_printoptions(threshold=np.nan)
# handling missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

# imputer = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
# imputer = Imputer(missing_values='NaN', strategy='median', axis=0)

imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print('X >>\n', X)

# Encoding categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

print('\nX fit transform >>\n', X)

onehotencoder = OneHotEncoder(categorical_features=[0])
np.set_printoptions(threshold=np.nan)
X = onehotencoder.fit_transform(X).toarray()
print('\nX onehotencoder fit transform >>\n', X)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

print('Y after label encoding>>\n', Y)

