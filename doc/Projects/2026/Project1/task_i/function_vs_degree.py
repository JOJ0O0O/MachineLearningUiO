import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from data import runge_data
from config import BLUE, RED, YELLOW, GREY, CONFIG, SAVE_FIGURE
from kfold_sklearn import kfold_function

def plot_ols_cv(Save_figure = False):
    degrees = CONFIG["degree"]
    cv_mses = []
    
    for degree in degrees:
        x, X, y = runge_data(
            n=CONFIG["n_samples"],
            degree=degree,
            noise=CONFIG.get("noise", 0.1),
            seed=CONFIG["random_state"],
        )
        
        model = LinearRegression(fit_intercept=False)
        cv_mse = kfold_function(X, y, model)
        cv_mses.append(cv_mse)
        
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, cv_mses, marker='o', color=BLUE, label="OLS CV MSE")
    ax.set_yscale('log')
    ax.set_xlabel("Degree")
    ax.set_ylabel("Mean Validation MSE")
    ax.set_title("OLS CV MSE vs Polynomial Degree")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    
    if SAVE_FIGURE:
        fig.savefig("ols_cv_vs_degree.png", bbox_inches="tight")
    #plt.show()

    return cv_mses 

def plot_ridge_lasso_cv(cv_mses, Save_Figure = False):
    degrees = CONFIG["degree"]
    lambdas = CONFIG["lam"]
    
    ridge_best_mses = []
    ridge_best_lams_list = []
    lasso_best_mses = []
    lasso_best_lams_list = []
    
    best_ridge_lams = {}
    best_lasso_lams = {}
    
    for degree in degrees:
        x, X, y = runge_data(
            n=CONFIG["n_samples"],
            degree=degree,
            noise=CONFIG.get("noise", 0.1),
            seed=CONFIG["random_state"],
        )
        
        best_ridge_mse = float('inf')
        best_ridge_lam = None
        best_lasso_mse = float('inf')
        best_lasso_lam = None
        
        for lam in lambdas:
            ridge_model = Ridge(alpha=lam, fit_intercept=False)
            mse_ridge = kfold_function(X, y, ridge_model)
            if mse_ridge < best_ridge_mse:
                best_ridge_mse = mse_ridge
                best_ridge_lam = lam
                
            lasso_model = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
            mse_lasso = kfold_function(X, y, lasso_model)
            if mse_lasso < best_lasso_mse:
                best_lasso_mse = mse_lasso
                best_lasso_lam = lam
                
        best_ridge_lams[degree] = best_ridge_lam
        best_lasso_lams[degree] = best_lasso_lam
        
        ridge_best_mses.append(best_ridge_mse)
        ridge_best_lams_list.append(best_ridge_lam)
        lasso_best_mses.append(best_lasso_mse)
        lasso_best_lams_list.append(best_lasso_lam)
        
    fig, (ax_mse, ax_lam) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax_mse.plot(degrees, cv_mses, marker='^', color=BLUE, label="OLS CV MSE")
    ax_mse.plot(degrees, ridge_best_mses, marker='o', color=RED, label="Ridge CV MSE")
    ax_mse.plot(degrees, lasso_best_mses, marker='s', color=YELLOW, label="Lasso CV MSE")
    ax_mse.set_yscale('log')
    
    ax_mse.set_ylabel("Best Validation MSE")
    ax_mse.set_title("Best CV MSE and Chosen Lambda vs Degree")
    ax_mse.grid(True, alpha=0.3, linestyle="--")
    ax_mse.legend()
    
    ax_lam.semilogy(degrees, ridge_best_lams_list, marker='o', color=RED, label="Ridge Lambda")
    ax_lam.semilogy(degrees, lasso_best_lams_list, marker='s', color=YELLOW, label="Lasso Lambda")
    ax_lam.set_xlabel("Degree")
    ax_lam.set_ylabel("Chosen Lambda")
    ax_lam.grid(True, alpha=0.3, linestyle="--")
    ax_lam.legend()
    
    fig.tight_layout()
    
    if SAVE_FIGURE:
        fig.savefig("ridge_lasso_cv_vs_degree.png", bbox_inches="tight")
    #plt.show()
    
    return best_ridge_lams, best_lasso_lams


def plot_overfitting_fits(best_ridge_lams, best_lasso_lams):
    degrees_to_plot = CONFIG["degrees_to_plot"]
    lambdas = CONFIG["lam"]
    fig, axes = plt.subplots(len(degrees_to_plot), 1, figsize=(9, 13), sharex=True)
    
    grid = np.linspace(-1.0, 1.0, 500)
    
    for ax, degree in zip(axes, degrees_to_plot):
        x_train, X_train, y_train = runge_data(
            n=CONFIG["n_samples"],
            degree=degree,
            noise=CONFIG.get("noise", 0.1),
            seed=CONFIG["random_state"],
        )
        
        ols_model = LinearRegression(fit_intercept=False)
        ols_model.fit(X_train, y_train)
        
        if degree in best_ridge_lams and degree in best_lasso_lams:
            best_ridge_lam = best_ridge_lams[degree]
            best_lasso_lam = best_lasso_lams[degree]
        else:
            best_r_mse = float('inf')
            best_l_mse = float('inf')
            for lam in lambdas:
                r_model = Ridge(alpha=lam, fit_intercept=False)
                mse_r = kfold_function(X_train, y_train, r_model)
                if mse_r < best_r_mse:
                    best_r_mse = mse_r
                    best_ridge_lam = lam
                    
                l_model = Lasso(alpha=lam, fit_intercept=False, max_iter=10000)
                mse_l = kfold_function(X_train, y_train, l_model)
                if mse_l < best_l_mse:
                    best_l_mse = mse_l
                    best_lasso_lam = lam
        
        best_ridge_model = Ridge(alpha=best_ridge_lam, fit_intercept=False)
        best_lasso_model = Lasso(alpha=best_lasso_lam, fit_intercept=False, max_iter=10000)
        
        best_ridge_model.fit(X_train, y_train)
        best_lasso_model.fit(X_train, y_train)
        
        X_raw_train = np.column_stack([x_train**k for k in range(1, degree + 1)])
        X_mean = X_raw_train.mean(axis=0)
        X_std = X_raw_train.std(axis=0)
        
        X_grid_raw = np.column_stack([grid**k for k in range(1, degree + 1)])
        X_grid = (X_grid_raw - X_mean) / X_std
        
        y_pred_ols = ols_model.predict(X_grid)
        y_pred_ridge = best_ridge_model.predict(X_grid)
        y_pred_lasso = best_lasso_model.predict(X_grid)
        
        rng = np.random.default_rng(CONFIG["random_state"])
        raw_y_train = (1.0 / (1.0 + 25.0 * x_train**2) 
                       + CONFIG.get("noise", 0.1) * rng.standard_normal(CONFIG["n_samples"]))
        y_mean_val = raw_y_train.mean()
        runge_centered = 1.0 / (1.0 + 25.0 * grid**2) - y_mean_val
        
        ax.scatter(x_train, y_train, s=18, alpha=0.5, color=GREY, label="Training data")
        ax.plot(grid, runge_centered, color="black", linewidth=2, label="Actual Runge")
        ax.plot(grid, y_pred_ols, color=BLUE, linewidth=1.5, label="OLS Fit")
        ax.plot(grid, y_pred_ridge, color=RED, linewidth=1.5, label="Ridge Fit")
        ax.plot(grid, y_pred_lasso, color=YELLOW, linewidth=1.5, label="Lasso Fit")
        
        ax.set_title(f"Fits vs Actual Runge (Degree {degree})")
        ax.set_ylabel("Centered y")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    
    if SAVE_FIGURE:
        fig.savefig(f"overfitting_fits{degrees_to_plot}.png", bbox_inches="tight")
    #plt.show()


if __name__ == "__main__":
    cv_mses = plot_ols_cv()
    best_r_lams, best_l_lams = plot_ridge_lasso_cv(cv_mses)
    plot_overfitting_fits(best_r_lams, best_l_lams)