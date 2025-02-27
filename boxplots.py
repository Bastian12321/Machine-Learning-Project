from splitData import *
import matplotlib.pyplot as plt

D0_standardized = (D0_plot - D0_plot.mean()) / D0_plot.std(ddof=1)
D1_standardized = (D1_plot - D1_plot.mean()) / D1_plot.std(ddof=1)

positions_0 = np.arange(1, M_plot + 1) * 2 - 1 
positions_1 = np.arange(1, M_plot + 1) * 2

plt.figure(figsize=(12, 6))

#https://stackoverflow.com/questions/41997493/python-matplotlib-boxplot-color
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

plt.xticks(np.arange(1, M_plot + 1) * 2 - 0.5, plot_labels, rotation=45, ha="right")

plt.tight_layout()
plt.show()