import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Perceptron import Perceptron

df = pd.read_csv('dataset.csv')

# Loading data set X
X_train = df.iloc[:, [1, 2]].values

# Break through Train and Test
X_test = X_train[:40]
X_train = X_train[40:]

# Loading data set Y
y_train = df.iloc[:, 4].values
y_train = np.where(y_train=='Iris-setosa', -1, 1)

# Break through Train and Test
y_test = y_train[:40]
y_train = y_train[40:]

ppn = Perceptron(eta=0.1, epochs=10)
ppn.fit(X_train, y_train)

ppn.test(X_test, y_test)
