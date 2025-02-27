from sympy import*
from load import *
X = np.c_[X, classLabels]

print(X[0])
print(raw_data[0])

X_matrix = Matrix(X)
print(X_matrix)