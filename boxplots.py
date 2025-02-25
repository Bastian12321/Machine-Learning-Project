from load import *
import matplotlib.pyplot as plt

plt.figure()
plt.boxplot(X)
plt.xticks(range(0, M), attributeNames)
plt.ylabel("cm")
plt.title("Fisher's Iris data set - boxplot")
plt.show()