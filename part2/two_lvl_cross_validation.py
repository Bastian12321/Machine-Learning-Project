from sklearn import model_selection
from standardize import *
from ANN_func import train_ANN
from regression_func import train_ridge_regression
#Remove redundant attributes and the attribute we want to predict from dataset
unwanted_attributes = ['obesity', 'alcohol', 'typea', 'ldl', 'famhist', 'chd']
D = df.drop(columns=(unwanted_attributes))
X = D.values
y = df['ldl'].values
N, M = X.shape
attributeNames = list(df.drop(columns=['ldl', 'obesity', 'alcohol', 'typea', 'famhist', 'chd']).columns)

K1 = 10
lin_reg = []
ANN = []
Baseline = []
for K in range(2, K1 + 1):
    CV = model_selection.KFold(K, shuffle=True)
    lin_reg.append(train_ridge_regression(K, X, y, M, attributeNames, CV))
    ANN.append(train_ANN(K, X, y, M, CV))
    print(K)
print(lin_reg)
print(ANN)
print("done")


