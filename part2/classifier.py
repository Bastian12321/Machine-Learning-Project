import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from standardize import *

def compute_baseline_error(y_train, y_test):
    # Determine the majority class in y_train
    majority_class = 1 if np.sum(y_train) > len(y_train) / 2 else 0
    
    # Create a prediction array with all entries set to the majority class
    y_pred_baseline = np.full(shape=len(y_test), fill_value=majority_class, dtype=int)
    
    # Compute classification error
    error = np.mean(y_pred_baseline != y_test)
    return error

def evaluate_log_model(X_train, y_train, X_test, y_test, l):
    model = LogisticRegression(C=l, max_iter=2000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    misclassified = np.sum(y_test != y_pred)
    n_test = len(y_test)
    error_score = misclassified / n_test
    
    return error_score
        
def evaluate_ANN_model(X_train, y_train, X_test, y_test, h):
    model = MLPClassifier(hidden_layer_sizes=(h,), alpha=1, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    misclassified = np.sum(y_test != y_pred)
    n_test = len(y_test)
    error_score = misclassified / n_test
            
    return error_score
    

D = df.drop(columns=['chd'])
X = D.values
y = df['chd'].values
N, M = X.shape

K_outer = 10 
K_inner = 10
CV_outer = model_selection.KFold(K_outer, shuffle=True, random_state=27)

hidden_layers = [1, 5, 10, 50, 100, 150, 200, 250]
lambdas = np.power(10.0, range(-5, 9))
result_columns = ['ANN-hiddenlayers', 'ANN-error', 'log-lambda', 'log-error', 'Baseline-error']
results = []
iteration = 1

for train_outer_index, test_outer_index in CV_outer.split(X, y):
    # extract training and test set for current CV fold
    X_outer_train = X[train_outer_index]
    y_outer_train = y[train_outer_index]
    X_outer_test = X[test_outer_index]
    y_outer_test = y[test_outer_index]
    
    CV_inner = model_selection.KFold(K_inner, shuffle=True, random_state=27)
    total_log_errors = []
    total_ANN_errors = []

    for train_inner_index, test_inner_index in CV_inner.split(X_outer_train, y_outer_train):
        log_errors = []
        ANN_errors = []
        
        X_inner_train = X[train_inner_index]
        y_inner_train = y[train_inner_index]
        X_inner_test = X[test_inner_index]
        y_inner_test = y[test_inner_index]
        
        for l in lambdas:
            log_errors.append(evaluate_log_model(X_inner_train, 
                                                 y_inner_train,
                                                 X_inner_test,
                                                 y_inner_test,
                                                 l))
        
        for h in hidden_layers:
            ANN_errors.append(evaluate_ANN_model(X_inner_train,  
                                                 y_inner_train,
                                                 X_inner_test,
                                                 y_inner_test,
                                                 h))
            
        total_log_errors.append(log_errors)
        total_ANN_errors.append(ANN_errors)

    log_errors_df = pd.DataFrame(total_log_errors, columns=lambdas)
    mean_errors = log_errors_df.mean(axis=0)
    optimal_lambda = mean_errors.idxmin()
    lowest_mean_error_log = mean_errors.min()
    
    ANN_errors_df = pd.DataFrame(total_ANN_errors, columns=hidden_layers)
    mean_errors = ANN_errors_df.mean(axis=0)
    optimal_h = mean_errors.idxmin()
    lowest_mean_error_ANN = mean_errors.min()
    
    outer_log_model_error = evaluate_log_model(X_outer_train,
                                               y_outer_train,
                                               X_outer_test,
                                               y_outer_test,
                                               optimal_lambda)
    
    outer_ANN_model_error = evaluate_ANN_model(X_outer_train, 
                                               y_outer_train, 
                                               X_outer_test, 
                                               y_outer_test, 
                                               optimal_h)
    
    results.append([optimal_h, 
                    outer_ANN_model_error, 
                    optimal_lambda, outer_log_model_error, 
                    compute_baseline_error(y_outer_train, y_outer_test)])

    iteration += 1

results_df = pd.DataFrame(results, columns=result_columns)
print(results_df)
    

        
        
        
        
        

    
    
    
    
