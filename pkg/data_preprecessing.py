# Data pre processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reading data

dataset = pd.read_csv('./Data.csv')
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

# splitting data into test data and train data

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train)

# feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
