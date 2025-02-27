from load import *
import matplotlib.pyplot as plt

#Delete famhist
X = np.delete(X, 4, axis=1)
attributeNames = np.delete(attributeNames, 4)
M = M - 1

colors = ["blue", "red"]
plt.figure(figsize=(12, 10))
for m1 in range(M):
    for m2 in range(M):
        plt.subplot(M, M, m1 * M + m2 + 1)
        for c in range(C):
            class_mask = classLabels == c
            plt.plot(np.array(X[class_mask, m2]), np.array(X[class_mask, m1]), ".", color=colors[c], markersize=3)
            if m1 == M - 1:
                plt.xlabel(attributeNames[m2])
            else:
                plt.xticks([])
            if m2 == 0:
                plt.ylabel(attributeNames[m1])
            else:
                plt.yticks([])

plt.tight_layout()
plt.legend(classNames)
plt.show()

