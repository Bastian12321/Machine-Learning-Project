import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

def evaluate_best_log_model(X_train, y_train, X_test, y_test, lambda_known, provided_lambda):
    if (lambda_known):
        lambdas = [provided_lambda]
    else:
        lambdas = np.power(10.0, range(-5, 9))
        
    lowest_error_score = 0
    optimal_lambda = lambdas[0]
    for l in lambdas:
        model = LogisticRegression(C=l, max_iter=2000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        misclassified = np.sum(y_test != y_pred)
        n_test = len(y_test)
        error_score = misclassified / n_test
        
        if l == lambdas[0]:
            lowest_error_score = error_score
        elif error_score >= lowest_error_score:
            lowest_error_score = error_score
            optimal_lambda = l
            
    return [optimal_lambda, lowest_error_score]
        
def evaluate_best_ANN_model():
    if hidden_layers_known:
        hidden_layers = [provided_hidden_layer]
    else:
        hidden_layers = [1, 5, 50, 100, 200, 500, 1000]
    
    lowest_error_score = 0
    optimal_hidden_layer = hidden_layers[0]
    for h in hidden_layers:
        model = MLPClassifier(hidden_layer_sizes=(h,), alpha=1, max_iter=10000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        misclassified = np.sum(y_test != y_pred)
        n_test = len(y_test)
        error_score = misclassified / n_test
        
        if hidden_layers == hidden_layers[0]:
            lowest_error_score = error_score
        elif error_score >= lowest_error_score:
            lowest_error_score = error_score
            optimal_hidden_layer = h
            
    return [optimal_hidden_layer, lowest_error_score]
    
