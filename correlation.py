from load import *
import numpy as np
from dtuimldmtools import similarity

for i in range(M):
    for j in range(i + 1, M):
        x = np.array([X[:, i]], dtype=np.float64)
        y = np.array([X[:, j]], dtype=np.float64)
        sim = similarity(x, y, "cor")
        print(f"({attributeNames[i]}, {attributeNames[j]}): {sim[0][0]:.2f}")
