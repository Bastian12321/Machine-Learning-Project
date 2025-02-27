from load import *
import importlib_resources
import numpy as npy
from scipy.linalg import svd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
#X = np.c_[X, classLabels]

#centering X
for i in range(M):
    col = X[:, i]
    X[:, i] = (col - np.mean(col)) / np.std(col, ddof=1)
    
Y = X.astype(float)

U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T

rho = (S * S) / (S * S).sum()

# Project data onto principal component space
Z = Y @ V

# Plot variance explained
plt.figure()
plt.plot(rho, "o-")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained value")

f = plt.figure()
plt.title("pixel vectors of handwr. digits projected on PCs")
for c in classLabels:
    # select indices belonging to class c:
    class_mask = classLabels == c
    plt.plot(Z[class_mask, 0], Z[class_mask, 1], 'o',
             color='blue' if c == 0 else 'red')
plt.legend(classNames)
plt.xlabel("PC1")
plt.ylabel("PC2")

# output to screen
plt.show()
