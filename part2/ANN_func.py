import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection

from dtuimldmtools import draw_neural_net, train_neural_net

# Normalize data
def train_ANN(K, X, y, M, CV):
    X = stats.zscore(X)

    hidden_units = [1, 2, 50] # range of hidden layers to test
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 10000
    
    optimal_h = hidden_units[0]
    lowest_mse = 0
    
    fold_errors = []
    for h in hidden_units:
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, h),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(h, 1),  # n_hidden_units to 1 output neuron
            # no final transfer function, i.e. "linear output"
        )
        loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

        print(f"Training model with {h} hidden units")
        errors = []  # make a list for storing generalization error in each loop
        for k, (train_index, test_index) in enumerate(CV.split(X, y)):
            print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[train_index, :])
            y_train = torch.Tensor(y[train_index]).unsqueeze(1)
            X_test = torch.Tensor(X[test_index, :])
            y_test = torch.Tensor(y[test_index]).unsqueeze(1)

            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(
                model,
                loss_fn,
                X=X_train,
                y=y_train,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )
            # Determine estimated class labels for test set
            y_test_est = net(X_test)

            # Determine errors
            se = (y_test_est.float() - y_test.float()) ** 2  # squared error
            mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean squared error
            errors.append(mse)  # store error rate for current CV fold

        # Print the average classification error rate for current hidden units (h)
        mean_mse = np.mean(errors)
        fold_errors.append(mean_mse)
        
        mse_val = mse[0]
        if h == 1:
            lowest_mse = mse_val
        elif mse <= lowest_mse:
            optimal_h = h
            lowest_mse = mse_val
            
    return [optimal_h, lowest_mse]