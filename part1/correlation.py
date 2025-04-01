from load import *
import numpy as np
import seaborn as sn
from dtuimldmtools import similarity
import matplotlib.pyplot as plt
import sympy as sp

for i in range(M):
    for j in range(i + 1, M):
        x = np.array([X[:, i]], dtype=np.float64)
        y = np.array([X[:, j]], dtype=np.float64)
        sim = similarity(x, y, "cor")
        print(f"({attributeNames[i]}, {attributeNames[j]}): {sim[0][0]:.2f}")

#Heat map
corrMatrix = np.corrcoef((X.astype(float)).T, dtype=float)
corr = sp.Matrix(corrMatrix)
heatmap = sn.heatmap(data = corrMatrix, cmap='coolwarm', annot=True)
heatmap.set_yticklabels(attributeNames)
heatmap.set_xticklabels(attributeNames)
plt.show()
