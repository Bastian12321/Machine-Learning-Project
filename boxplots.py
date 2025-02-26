from splitData import *
import matplotlib.pyplot as plt

D0_standardized = (D0 - D0.mean()) / D0.std(ddof=1)
D1_standardized = (D1 - D1.mean()) / D1.std(ddof=1)

positions_0 = np.arange(1, M0 + 1) * 2 - 1 
positions_1 = np.arange(1, M0 + 1) * 2

plt.figure(figsize=(12, 6))

plt.boxplot(D0_standardized.values, positions=positions_0, widths=0.6, patch_artist=True, 
            boxprops=dict(facecolor="blue", alpha=0.5), 
            medianprops=dict(color="black"),
            flierprops=dict(color="blue", markeredgecolor="blue"),
            whiskerprops=dict(color="blue"))
plt.boxplot(D1_standardized.values, positions=positions_1, widths=0.6, patch_artist=True, 
            boxprops=dict(facecolor="red", alpha=0.5),
            whiskerprops=dict(color="red"),
            flierprops=dict(color="red", markeredgecolor="red"),
            medianprops=dict(color="black"))

plt.xticks(np.arange(1, M0 + 1) * 2 - 0.5, labels, rotation=45, ha="right")

plt.title("Boxplots of standardized data")

plt.tight_layout()
plt.show()