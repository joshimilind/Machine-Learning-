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
