#Taken from chat GPT
import sys
import os
part1_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../part1"))
sys.path.append(part1_path)
from load import *

#Feature transformation of famhist
df['famhist'] = (df['famhist'] == 'Present').astype(int)

#Standardization
D_standardized = (df - df.mean()) / df.std(ddof=1)
X_standardized = D_standardized.values