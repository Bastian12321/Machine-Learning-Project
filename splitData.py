from load import *

D0 = df[df['chd'] == 0].drop(columns=['chd'])
D1 = df[df['chd'] == 1].drop(columns=['chd'])

#Famhist is removed from data to be plottet as histogram and boxplot
D0_plot = D0.drop(columns=['famhist'])
D1_plot = D1.drop(columns=['famhist'])
plot_labels = D0_plot.columns
M_plot = len(plot_labels)

print("Data has been split succesfully!")