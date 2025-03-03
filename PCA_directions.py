from PCA import *
print(S)
print(V)

for att in range(V.shape[1]):
    plt.arrow(0, 0, V[att, 0], V[att, 1])
    plt.text(V[att, 0], V[att, 1], attributeNames[att])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
plt.xlabel("PC" + str(1))
plt.ylabel("PC" + str(2))
plt.grid()
plt.plot(
    np.cos(np.arange(0, 2 * np.pi, 0.01)), np.sin(np.arange(0, 2 * np.pi, 0.01))
)
plt.axis("equal")
plt.show()