import numpy as np
from load import *
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 7))
u = np.floor(np.sqrt(M))
v = np.ceil(float(M) / u)


for i in range(M):
    plt.subplot(int(u), int(v), i + 1)
    plt.hist(X[:, i], color=((1 - (i * 0.1), i / M, 0.1 * i)))
    plt.xlabel(attributeNames[i])
    plt.ylim(0, N / 2)

plt.show()