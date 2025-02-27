from splitData import *
D0['famhist'] = (D0['famhist'] == 'Present').astype(int)
D1['famhist'] = (D1['famhist'] == 'Present').astype(int)

print("chd Negative:")
print(D0.describe().to_latex())
print("chd Positive:")
print(D1.describe().to_latex())