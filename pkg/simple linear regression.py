# Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reading data
dataset = pd.read_csv('./Data.csv')
np.set_printoptions(threshold=np.nan)

# matrix of feature
X = dataset.iloc[:, :-1].values

# dependent variable vector
Y = dataset.iloc[:, 1].values

# splitting data into test data and train data

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3, random_state=0)

# fitting simple Linear Regression to training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
