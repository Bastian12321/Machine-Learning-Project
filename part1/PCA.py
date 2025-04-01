from load import *
import importlib_resources
import numpy as np
from scipy.linalg import svd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier

U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T

Z = Y @ V
plt.figure()
plt.title("chd data projected onto PC space")
colors = ['blue','red']
for c in classLabels:
    class_mask = classLabels == c
    plt.plot(Z[class_mask, 0], Z[class_mask, 1], 'o', color=colors[c], label=classNames[c])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
