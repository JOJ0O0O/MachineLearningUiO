import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from data import runge_data, optimiser_step, optimise, lasso_subgradient, cost_mse, grad_ols, grad_ridge, grad_lasso_subgradient, optimiser_step_adam, optimise_adam
from config import BLUE, RED, YELLOW, GREEN, GREY, GREY, CONFIG

def main_fit(save_figure=True):
    degree = CONFIG["degree"]
    lam = CONFIG["lam"]
    gamma = CONFIG["gamma"]
    
    #split the data
    x_raw, X, y = runge_data(n=CONFIG["n_samples"], degree=degree)
    X_train, X_test, y_train, y_test, x_train_raw, x_test_raw = train_test_split(
        X, y, x_raw, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
    )
    #calculating the thetas 
    theta_init = np.zeros(degree)

    # doing OLS with adam and gradident descent 
    theta_ols_gd = optimise_adam(lambda th: grad_ols(th, X_train, y_train), theta_init, gamma=gamma)

    # doing Ridge with adam and gradient descent, additionally adding the penalty terms 
    theta_ridge_gd = optimise_adam(lambda th: grad_ridge(th, X_train, y_train, lam=lam), theta_init, gamma=gamma)

    #doing lasso, as an iterative method with a subgradient  and penalty terms (better explanation is in the report)
    theta_lasso_gd = optimise_adam(lambda th: grad_lasso_subgradient(th, X_train, y_train, lam=lam), theta_init, gamma=gamma)

    # setting up scikit-Learn Lasso using alpha = lambda / 2 to match 1/n convention
    sklearn_lasso = Lasso(alpha=lam / 2.0, fit_intercept=False, max_iter=10000)
    sklearn_lasso.fit(X_train, y_train)
    theta_sklearn_lasso = sklearn_lasso.coef_

    #evaluating the results and printing them
    mse_ols_train, mse_ols_test = cost_mse(theta_ols_gd, X_train, y_train), cost_mse(theta_ols_gd, X_test, y_test)
    mse_ridge_train, mse_ridge_test = cost_mse(theta_ridge_gd, X_train, y_train), cost_mse(theta_ridge_gd, X_test, y_test)
    mse_lasso_train, mse_lasso_test = cost_mse(theta_lasso_gd, X_train, y_train), cost_mse(theta_lasso_gd, X_test, y_test)
    mse_sk_train, mse_sk_test = cost_mse(theta_sklearn_lasso, X_train, y_train), cost_mse(theta_sklearn_lasso, X_test, y_test)

    #print and plot where fully done by ai, to save time and get a nice layout for the plots and outputs 
    print("="*65)
    print(f"{'Method':<25} | {'Train MSE':<15} | {'Test MSE':<15}")
    print("="*65)
    print(f"{'OLS (Adam GD)':<25} | {mse_ols_train:<15.6f} | {mse_ols_test:<15.6f}")
    print(f"{'Ridge (Adam GD)':<25} | {mse_ridge_train:<15.6f} | {mse_ridge_test:<15.6f}")
    print(f"{'Custom Lasso (Subgrad GD)':<25} | {mse_lasso_train:<15.6f} | {mse_lasso_test:<15.6f}")
    print(f"{'Scikit-Learn Lasso':<25} | {mse_sk_train:<15.6f} | {mse_sk_test:<15.6f}")
    print("="*65)

    if CONFIG.get("save_csv", False):
        filename = f"fitting_results_deg{degree}_lam{lam}_n{CONFIG['n_samples']}.csv"
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Train MSE', 'Test MSE'])
            writer.writerow(['OLS (Adam GD)', mse_ols_train, mse_ols_test])
            writer.writerow(['Ridge (Adam GD)', mse_ridge_train, mse_ridge_test])
            writer.writerow(['Custom Lasso (Subgrad GD)', mse_lasso_train, mse_lasso_test])
            writer.writerow(['Scikit-Learn Lasso', mse_sk_train, mse_sk_test])

    # plotting two subplots, one with coefficients comparison and the other one with model fits vs. the raw data 
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi = 300)

    #Coefficient Comparison Across Degrees
    degrees = np.arange(1, degree + 1)
    axes[0].plot(degrees, theta_ols_gd, 's--', color=BLUE, label='OLS (GD)')
    axes[0].plot(degrees, theta_ridge_gd, '^--', color=GREEN, label=f'Ridge GD ($\lambda={lam}$)')
    axes[0].plot(degrees, theta_lasso_gd, 'o-', color=RED, label=f'Custom Lasso GD ($\lambda={lam}$)')
    axes[0].plot(degrees, theta_sklearn_lasso, 'x:', color=YELLOW, label=f'Sklearn Lasso ($\lambda={lam}$)')
    axes[0].axhline(0, color=GREY, linestyle=':', alpha=0.7)
    axes[0].set_xlabel('Polynomial Degree Feature Index', fontsize=11)
    axes[0].set_ylabel('Coefficient Weight ($\\theta$)', fontsize=11)
    axes[0].set_title('Coefficient Profiles across Methods', fontsize=12)
    axes[0].set_xticks(degrees)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    #Model Fits vs Raw Data Points
    # Sort x_test for clean line plots
    sort_idx = np.argsort(x_test_raw)
    x_sorted = x_test_raw[sort_idx]
    X_test_sorted = X_test[sort_idx]

    axes[1].scatter(x_train_raw, y_train, color=GREY, alpha=0.4, label='Train Data')
    axes[1].scatter(x_test_raw, y_test, color='black', alpha=0.7, label='Test Data', marker='x')
    axes[1].plot(x_sorted, X_test_sorted @ theta_ols_gd, color=BLUE, label='OLS Fit')
    axes[1].plot(x_sorted, X_test_sorted @ theta_ridge_gd, color=GREEN, label='Ridge Fit')
    axes[1].plot(x_sorted, X_test_sorted @ theta_lasso_gd, color=RED, label='Custom Lasso Fit')
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('Centered y', fontsize=11)
    axes[1].set_title('Test Set Predictions', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_figure:
        plt.savefig('customLasso_Ridge_OLS_sklearn.png')
    plt.show()