from load import *

D0 = df[df['chd'] == 0].drop(columns=['famhist', 'chd'])
D1 = df[df['chd'] == 1].drop(columns=['famhist', 'chd'])
labels = D0.columns
M0 = len(labels)