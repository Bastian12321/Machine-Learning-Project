import importlib_resources
import numpy as np
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection
from dtuimldmtools import rlr_validate

def train_ridge_regression(K, X, y, M, attributeNames, CV):
    # Add offset attribute
    X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
    attributeNames = ["Offset"] + attributeNames
    M = M + 1

    # Values of lambda
    lambdas = np.power(10.0, range(-5, 9))

    # Initialize variables
    # T = len(lambdas)
    Error_train = np.empty((K, 1))
    Error_test = np.empty((K, 1))
    Error_train_rlr = np.empty((K, 1))
    Error_test_rlr = np.empty((K, 1))
    Error_train_nofeatures = np.empty((K, 1))
    Error_test_nofeatures = np.empty((K, 1))
    w_rlr = np.empty((M, K))
    mu = np.empty((K, M - 1))
    sigma = np.empty((K, M - 1))
    w_noreg = np.empty((M, K))

    k = 0
    for train_index, test_index in CV.split(X, y):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10

        (
            opt_val_err,
            opt_lambda,
            mean_w_vs_lambda,
            train_err_vs_lambda,
            test_err_vs_lambda,
        ) = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)

        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train

        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = (
            np.square(y_train - y_train.mean()).sum(axis=0) / y_train.shape[0]
        )
        Error_test_nofeatures[k] = (
            np.square(y_test - y_test.mean()).sum(axis=0) / y_test.shape[0]
        )

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0, 0] = 0  # Do no regularize the bias term
        w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = (
            np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
        )
        Error_test_rlr[k] = (
            np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]
        )

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:, k] = np.linalg.solve(XtX, Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[k] = (
            np.square(y_train - X_train @ w_noreg[:, k]).sum(axis=0) / y_train.shape[0]
        )
        Error_test[k] = (
            np.square(y_test - X_test @ w_noreg[:, k]).sum(axis=0) / y_test.shape[0]
        )
        k += 1
         
    return [opt_lambda, np.mean(Error_test)]
        