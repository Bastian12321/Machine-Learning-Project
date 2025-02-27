import numpy as np
from splitData import *
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 7))
u = int(np.floor(np.sqrt(M_plot)))
v = int(np.ceil(float(M_plot) / u))

i = 0
for lbl in plot_labels:
    plt.subplot(u, v, i + 1)
    i += 1
    # Plot histograms for both groups
    plt.hist(D0_plot[lbl], bins=20, color='blue', alpha=0.5)
    plt.hist(D1_plot[lbl], bins=20, color='red', alpha=0.5)
    
    plt.xlabel(lbl)

plt.figlegend(["Negative", "Positive"], loc="lower right", fontsize=12, frameon=True)
plt.tight_layout()
plt.show()

"""
attributes_to_logtransform = ['sbp', 'ldl', 'obesity']
for attribute in attributes_to_logtransform:
    D0_plot[attribute] = np.log(D0_plot[attribute])
    D1_plot[attribute] = np.log(D1_plot[attribute])
    
plt.figure(figsize=(8, 7))
u = int(np.floor(np.sqrt(M_plot)))
v = int(np.ceil(float(M_plot) / u))
i = 0

for lbl in plot_labels:
    plt.subplot(u, v, i + 1)
    i += 1
    # Plot histograms for both groups
    plt.hist(D0_plot[lbl], bins=20, color='blue', alpha=0.5)
    plt.hist(D1_plot[lbl], bins=20, color='red', alpha=0.5)
    
    plt.xlabel(lbl)

plt.figlegend(["Negative", "Positive"], loc="lower right", fontsize=12, frameon=True)
plt.tight_layout()
plt.show()
"""