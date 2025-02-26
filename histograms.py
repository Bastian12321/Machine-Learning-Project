import numpy as np
from splitData import *
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 7))
u = int(np.floor(np.sqrt(M0)))
v = int(np.ceil(float(M0) / u))

i = 0
for lbl in labels:
    plt.subplot(u, v, i + 1)
    i += 1
    # Plot histograms for both groups
    plt.hist(D0[lbl], bins=20, color='blue', alpha=0.5)
    plt.hist(D1[lbl], bins=20, color='red', alpha=0.5)
    
    plt.xlabel(lbl)

plt.figlegend(["chd = 0", "chd = 1"], loc="lower right", fontsize=12, frameon=True)
plt.tight_layout()
plt.show()