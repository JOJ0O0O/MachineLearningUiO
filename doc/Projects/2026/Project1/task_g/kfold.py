import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from config import CONFIG, BLUE
from data import runge_data, optimise_adam, grad_lasso_subgradient, cost_mse, grad_ols, grad_ridge

'''
AI declaration: AI was only used to print the results in a easy understandable way and to safe time. 
It was not used on algorithms but for fixing problems.
'''
np.set_printoptions(precision=4, suppress=True)

def kfold_function():
    degree = CONFIG["degree"]
    lam = CONFIG["lam"]
    k_folds = CONFIG["k_folds"]
    gamma = CONFIG["gamma"]
    
    x_raw, X_raw, y_raw = runge_data(n=CONFIG["n_samples"], degree=degree)

    #choosing k folds = 5, so every data point is used for 4 training times
    #each point gets used exactly once for validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=CONFIG["random_state"])

    #storing th mse results
    mse_results = {
        'OLS': {'train': [], 'test': []},
        'Ridge': {'train': [], 'test': []},
        'Lasso_GD': {'train': [], 'test': []},
        'Lasso_SK': {'train': [], 'test': []}
    }

    for train_idx, test_idx in kf.split(X_raw):
        # Split raw data
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]
        
        #scale the data to prevent leakage 
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_train_norm = (X_train - X_mean) / X_std
        X_test_norm = (X_test - X_mean) / X_std

        #centering the y data 
        y_mean = y_train.mean()
        y_train_centered = y_train - y_mean
        y_test_centered = y_test - y_mean

        theta_init = np.zeros(degree)

        #setting different fitting models 
        theta_ols = optimise_adam(lambda th: grad_ols(th, X_train_norm, y_train_centered), theta_init, gamma=gamma)
        theta_ridge = optimise_adam(lambda th: grad_ridge(th, X_train_norm, y_train_centered, lam), theta_init, gamma=gamma)
        theta_lasso = optimise_adam(lambda th: grad_lasso_subgradient(th, X_train_norm, y_train_centered, lam), theta_init, gamma=gamma)
        
        sk_lasso = Lasso(alpha=lam / 2.0, fit_intercept=False, max_iter=10000)
        sk_lasso.fit(X_train_norm, y_train_centered)
        theta_sk = sk_lasso.coef_

        # evaluate and store the reults in a dict
        models = {
            'OLS': theta_ols, 
            'Ridge': theta_ridge, 
            'Lasso_GD': theta_lasso, 
            'Lasso_SK': theta_sk
        }
        
        for name, theta in models.items():
            mse_results[name]['train'].append(cost_mse(theta, X_train_norm, y_train_centered))
            mse_results[name]['test'].append(cost_mse(theta, X_test_norm, y_test_centered))

    #print the results, output was formatted by ai to get a clean print output
    print("="*75)
    print(f"{'Method':<25} | {'CV Train MSE (std)':<20} | {'CV Test MSE (std)':<20}")
    print("="*75)

    csv_data = [["Method", "CV_Train_MSE", "CV_Train_STD", "CV_Test_MSE", "CV_Test_STD"]]

    for name in mse_results.keys():
        train_mean = np.mean(mse_results[name]['train'])
        train_std = np.std(mse_results[name]['train'])
        test_mean = np.mean(mse_results[name]['test'])
        test_std = np.std(mse_results[name]['test'])
        
        train_str = f"{train_mean:.6f} ({train_std:.4f})"
        test_str = f"{test_mean:.6f} ({test_std:.4f})"
        
        print(f"{name:<25} | {train_str:<20} | {test_str:<20}")
        csv_data.append([name, train_mean, train_std, test_mean, test_std])
    print("="*75)

    if CONFIG.get("save_csv", False):
        filename = f"kfold_results_deg{degree}_lam{lam}_kf{k_folds}.csv"
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
