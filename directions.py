from load import *
import importlib_resources
import numpy as npy
from scipy.linalg import svd 
import matplotlib.pyplot as plt 
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
import sympy as sp

fig, ax = plt.subplots(figsize=(8, 6))

# Scatter plot of PCA projections
for c in np.unique(classLabels):
    ax.scatter(Z[classLabels == c, 0], Z[classLabels == c, 1], 
               color=colors[c], label=classNames[c], alpha=0.6)

# Plot eigenvectors (arrows)
for i in range(V.shape[1]):  
    plt.arrow(0, 0, V[i, 0], V[i, 1], color='black', alpha=0.7, head_width=0.05)
    plt.text(V[i, 0]*1.15, V[i, 1]*1.15, attributeNames[i], color='black')

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA Biplot for CHD Data")
plt.legend()
plt.show()