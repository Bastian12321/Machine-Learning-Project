from load import *
import importlib_resources
import numpy as npy
from scipy.linalg import svd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import sympy as sp
    
U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T
rho = (S * S) / (S * S).sum()
threshold = 0.9

plt.figure()
plt.plot(range(1, len(rho) + 1), rho, "x-")
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-")
plt.plot([1, len(rho)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()
