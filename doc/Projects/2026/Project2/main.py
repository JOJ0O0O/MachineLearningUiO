import NeuralNet as NN
import numpy as np
import Activation_Cost_funcs as act
import Optimizers as opt
import matplotlib.pyplot as plt
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve,
                                     train_test_split)

plt.rcParams.update({
    "figure.figsize": (3.35, 2.5),
    "figure.constrained_layout.use": True,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

"""Set up data from Project 1"""

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept=True):
    # polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T

def Scaledata(Xtrain, Xtest, ytrain, ytest):
    """
    Scales and centers the data using the standard deviation and the mean values of the train set
    Applies the same transformations to the test set as well (based on train values)

        Parameters 
    ----------------
    XTrain: np.array shape: (n,p)
        Training data input
    Xtest: np.array shape: (n,p)
        Test data
        
    ytrain: np.array shape: (n,1)
        Vector containing the data to be fitted
    ytest: np.array shape: (n,1)
        Vector containing the test data

    Retrun
    ---------------
    XTrain_norm: np.array shape: (n,p)
        Normed training data
    XTest_norm: np.array shape: (n,p)
        Normed test data 
        
    ytrain_centered: np.array shape: (n,1)
        Vector containing the centered train data
    ytest_centered: np.array shape: (n,1)
        Vector containing the centered test data
    """
    X_mean, X_std = Xtrain.mean(axis=0), Xtrain.std(axis=0)
    XTrain_norm = (Xtrain - X_mean) / X_std
    XTest_norm = (Xtest - X_mean) / X_std
    ytrain_centered = ytrain - ytrain.mean()
    ytest_centered = ytest - ytrain.mean()
    return XTrain_norm, XTest_norm, ytrain_centered, ytest_centered

rng = np.random.default_rng(2026)
n = 100
sigma = 0.1
x = np.sort(rng.uniform(-1, 1, n)).reshape(-1,1)
y = runge(x) + rng.normal(0, sigma, n).reshape(-1,1)
ymean= y.mean()
"""
AI-Declaration:
I used Chat-GPT6 Luna in VSCode in order to find an error as i did not know that I needed to reshape my data.
Four lines below are therefore AI generated.
This is equal to usage level 2.
27.10.2026
Johannes Schäfers
"""

xtrain = x
ytrain = y
xx = np.linspace(-1,1,1000)
xx = xx.reshape(-1,1)

rngtest = np.random.default_rng(2025)
xtest = np.sort(rngtest.uniform(-1, 1, 30)).reshape(-1,1)
ytest = runge(xtest) + rngtest.normal(0, sigma, 30).reshape(-1,1)

""" Set up done """

def BuildTestNets():
    """
    Test function to see if Neural Network implementation works
    Calls both Jax and non-Jax implementation of backpropagation
    """
    TestOpti = opt.AdamOptimizer
    TestNet = NN.NeuralNetwork(x.shape[1],
                            [30,30,1], 
                            [act.ReLU, act.sigmoid, act.sigmoid], 
                            act.mse, 
                            optimizer = TestOpti,
                            regularization="L2",
                            lam=0.001,
                            gamma=0.01,
                            batch_size=16)

    TestOptiJax = opt.AdamOptimizer
    TestNetJax = NN.NeuralNetwork(x.shape[1],
                            [30,30,1], 
                            [act.ReLU, act.sigmoid, act.sigmoid], 
                            act.mse, 
                            optimizer = TestOptiJax,
                            regularization = "L2",
                            lam=0.001,
                            gamma = 0.01,
                            batch_size=16)

    TestNet.train(x, y, 1000, sgd=False)
    TestNetJax.train_jax(x, y, 1000, sgd=False)

    """Build simple model to predict a range of x values to see if jax and own gradient produces the same"""
    ypred = TestNet.predict(x)
    yxx = TestNet.predict(xx)
    yxxjax = TestNetJax.predict(xx)

    fig, axis = plt.subplots()
    axis.plot(xx,yxx, label="Handwritten")
    axis.plot(xx,yxxjax, label="Jax")
    axis.legend()
    plt.show()
    plt.close(fig)

def task_b():
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    Optimizers = [opt.PlainGDOptimizer, opt.AdamOptimizer, opt.RMSPropOptimizer]
    optimizer_names = {
        opt.PlainGDOptimizer: "Stochastic Gradient Descent",
        opt.AdamOptimizer: "Adam",
        opt.RMSPropOptimizer: "RMSProp",
    }
    gammas = [0.001, 0.003, 0.01, 0.03, 0.07, 0.1, 0.15, 0.2,0.25,0.3]
    layers = [(50,1),(50,50,1),(50,50,50,1),(100,1),(100,100,1),(100,100,100,1)]

    param_grid = {"neuralnetwork__gamma": gammas,
                    "neuralnetwork__optimizer": Optimizers,
                    "neuralnetwork__layer_output_sizes": layers}
    net = make_pipeline(StandardScaler(),
                        NN.NeuralNetwork(network_input_size=x.shape[1],
                        regularization=None,
                        cost_fun=act.mse,
                        layer_output_sizes=(50,1),
                        activation_funcs=[act.sigmoid,act.sigmoid,act.sigmoid,act.identity],
                        optimizer=opt.AdamOptimizer,
                        batch_size=16))
    grid = GridSearchCV(net, param_grid, cv=cv, scoring="neg_mean_squared_error", return_train_score=True, n_jobs=-1)
    grid.fit(xtrain,ytrain.ravel())
    results = grid.cv_results_
    mse = np.full((len(Optimizers), len(layers), len(gammas)), np.nan)
    for params, score in zip(results["params"], results["mean_test_score"]):
        optimizer_index = Optimizers.index(params["neuralnetwork__optimizer"])
        layer_index = layers.index(params["neuralnetwork__layer_output_sizes"])
        gamma_index = gammas.index(params["neuralnetwork__gamma"])
        mse[optimizer_index, layer_index, gamma_index] = -score

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    print(best_params, best_model)
    network_models = {}
    network_scores = {}
    for layer in layers:
        candidates = [
            (index, params)
            for index, params in enumerate(results["params"])
            if params["neuralnetwork__layer_output_sizes"] == layer
            and np.isfinite(results["mean_test_score"][index])
        ]
        best_index, params = max(
            candidates, key=lambda item: results["mean_test_score"][item[0]]
        )
        if params == best_params:
            model = best_model
        else:
            model = clone(net).set_params(**params)
            model.fit(xtrain, ytrain.ravel())
        name = " x ".join(str(width) for width in layer[:-1])
        network_models[name] = model
        network_scores[name] = {
            "cv_mse": -results["mean_test_score"][best_index],
            "cv_se": results["std_test_score"][best_index] / np.sqrt(cv.get_n_splits()),
            "test_mse": np.mean((model.predict(xtest) - ytest.ravel()) ** 2),
            "optimizer": params["neuralnetwork__optimizer"].__name__,
            "gamma": params["neuralnetwork__gamma"],
        }

    fig, axes = plt.subplots(1, len(Optimizers), figsize=(16, 2.7),
                             constrained_layout=True)
    """
    AI Declaration:
    I have used AI to fix the below code, as my first written code had a bug inside of it
    Tool: GPT6-Luna in VSCode
    This corresponds to usage Level 1 as I have written, tested and verified the code myself and had only used AI as a small help.
    29.10.2026
    """
    finite_mse = mse[np.isfinite(mse)]
    if finite_mse.size == 0:
        raise ValueError("Cross-validation produced no finite MSE scores.")
    mse_min = finite_mse.min()
    mse_max = np.percentile(finite_mse, 90)
    if mse_max <= mse_min:
        mse_max = finite_mse.max()
    for i, (optimizer, axis) in enumerate(zip(Optimizers, axes)):
        im = axis.imshow(np.ma.masked_invalid(mse[i]), cmap="viridis_r",
                         origin="lower", aspect="auto",
                         vmin=mse_min, vmax=mse_max)
        for layer_index in range(len(layers)):
            for gamma_index in range(len(gammas)):
                value = mse[i, layer_index, gamma_index]
                if not np.isfinite(value):
                    continue
                text_color = "white" if im.norm(value) >= 0.5 else "black"
                axis.text(gamma_index, layer_index, f"{value:.3g}",
                          ha="center", va="center", fontsize=7, color=text_color)
        best_layer, best_gamma = np.unravel_index(
            np.nanargmin(mse[i]), mse[i].shape
        )
        axis.scatter(
            best_gamma,
            best_layer,
            marker="*",
            s=110,
            facecolors="none",
            edgecolors="red",
            linewidths=1.2,
            label="Best CV score",
        )
        axis.set_xticks(range(len(gammas)))
        axis.set_xticklabels([f"{gamma:g}" for gamma in gammas], fontsize=7)
        axis.set_yticks(range(len(layers)))
        axis.set_yticklabels(
            [" x ".join(str(width) for width in layer[:-1]) for layer in layers],
            fontsize=8,
        )
        axis.set_xlabel("$\\gamma$")
        axis.set_ylabel("Hidden-layer widths")
        axis.set_title(optimizer_names[optimizer], fontsize=10)
    fig.suptitle("Red stars mark each optimizer's best CV score")
    fig.colorbar(im, ax=axes, label="5-fold validation MSE (color clipped at 95th percentile)")
    output_path = Path(__file__).resolve().parent / "Task_b Layers vs Gammas.pdf"
    fig.savefig(output_path)
    print(f"Saved figure to {output_path}")
    plt.close(fig)
    """
    AI Declaration:
    I have used AI to create the below code to plot the best CV MSE for each optimizer
    Tool: GPT6-Luna in VSCode
    This corresponds to usage Level 3 as I have only tested and verified the code myself.
    08.11.2026
    """
    best_mse_by_optimizer = []
    for optimizer in Optimizers:
        candidates = [
            index
            for index, params in enumerate(results["params"])
            if params["neuralnetwork__optimizer"] is optimizer
            and np.isfinite(results["mean_test_score"][index])
        ]
        if not candidates:
            raise ValueError(
                f"Cross-validation produced no finite scores for "
                f"{optimizer_names[optimizer]}."
            )
        best_index = max(
            candidates, key=lambda index: results["mean_test_score"][index]
        )
        best_mse_by_optimizer.append(-results["mean_test_score"][best_index])

    fig, axis = plt.subplots(figsize=(6.3, 3.5), constrained_layout=True)
    bars = axis.bar(
        [optimizer_names[optimizer] for optimizer in Optimizers],
        best_mse_by_optimizer,
        color=["#4477AA", "#228833", "#CC6677"],
    )
    axis.bar_label(bars, fmt="%.4f", padding=3)
    axis.set_ylabel("Best CV MSE (lower is better)")
    axis.set_title("Best CV MSE by optimizer")
    axis.tick_params(axis="x", labelrotation=15)
    axis.grid(axis="y", alpha=0.2)
    output_path = (
        Path(__file__).resolve().parent / "Task_b Best Optimizer Comparison.pdf"
    )
    fig.savefig(output_path)
    print(f"Saved figure to {output_path}")
    plt.close(fig)
    return network_models, network_scores

def Polyfit():
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(), LinearRegression())
    search_poly = GridSearchCV(poly, {"polynomialfeatures__degree": range(1, 16)}, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)
    search_poly.fit(xtrain, ytrain.ravel())
    results = search_poly.cv_results_          # a dict of arrays, one entry per grid point
    mse = -results["mean_test_score"]
    se = results["std_test_score"] / np.sqrt(cv.get_n_splits())
    degrees = np.arange(1, 16)
    best_index = int(np.argmin(mse))
    best_degree = int(degrees[best_index])
    fig, ax = plt.subplots()
    ax.errorbar(degrees, mse, yerr=se, marker="o", capsize=3, linewidth=1.5)
    ax.scatter(best_degree, mse[best_index], color="crimson", zorder=3,
               label=f"Selected degree {best_degree}")
    ax.set_xticks(degrees)
    ax.set_title("Polynomial OLS: degree selection by training CV")
    ax.set_ylabel("5-fold validation MSE")
    ax.set_xlabel("Polynomial degree")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    output_path = Path(__file__).resolve().parent / "PolynomialRegression.pdf"
    fig.savefig(output_path)
    print(f"Saved figure to {output_path}")
    plt.close(fig)
    ols_model = search_poly.best_estimator_
    ypred = ols_model.predict(xtest)
    TestMSE = np.mean((ypred - ytest.ravel()) ** 2)
    print(f"CV-selected polynomial degree: {best_degree}")
    print(f"Selected OLS 5-fold CV MSE: {mse[best_index]:.6f} +/- {se[best_index]:.6f}")
    print(f"Selected OLS held-out test MSE: {TestMSE:.6f}")
    return ols_model, mse, se, best_degree

def _cv_score_grids(results, alpha_parameter, alphas, degrees, n_splits):
    mse_by_alpha_degree = np.empty((len(alphas), len(degrees)))
    se_by_alpha_degree = np.empty_like(mse_by_alpha_degree)
    alpha_indices = {alpha: index for index, alpha in enumerate(alphas)}
    degree_indices = {degree: index for index, degree in enumerate(degrees)}

    for result_index, params in enumerate(results["params"]):
        alpha_index = alpha_indices[params[alpha_parameter]]
        degree_index = degree_indices[params["polynomialfeatures__degree"]]
        mse_by_alpha_degree[alpha_index, degree_index] = -results["mean_test_score"][result_index]
        se_by_alpha_degree[alpha_index, degree_index] = (
            results["std_test_score"][result_index] / np.sqrt(n_splits)
        )

    return mse_by_alpha_degree, se_by_alpha_degree


def _plot_cv_contour(mse_plot, degrees, alphas, title, output_path):
    degrees_grid, alphas_grid = np.meshgrid(degrees, alphas)
    min_index = np.unravel_index(np.argmin(mse_plot), mse_plot.shape)

    fig, ax = plt.subplots()
    contour = ax.contourf(degrees_grid, alphas_grid, mse_plot, levels=20)
    fig.colorbar(contour, ax=ax, label="5-fold validation MSE")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(title)
    ax.set_xticks(degrees[::3])
    ax.set_xticklabels([str(degree) for degree in degrees[::3]])
    ax.set_yscale("log")
    ax.scatter(
        degrees_grid[min_index],
        alphas_grid[min_index],
        color="red",
        marker="x",
        s=100,
        label=r"Min. $MSE$",
    )
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    return min_index


def Ridgefit():
    """
    AI-Declaration:
    I used Chat-GPT6 Luna in VSCode in order to fix this function so I could properly use Ridge Regression.
    The general outline of this function was written by me, but some of the plotting / data formatting was AI generated.
    This is equal to usage level 3.
    03.11.2026
    Johannes Schäfers
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    poly = make_pipeline(
        PolynomialFeatures(include_bias=False), StandardScaler(), Ridge()
    )
    lambdas = np.logspace(-8, 0, 10)
    cv_train_size = len(next(cv.split(xtrain))[0])
    cv_alphas = lambdas * cv_train_size
    search_poly = GridSearchCV(
        poly,
        {
            "polynomialfeatures__degree": range(1, 31),
            "ridge__alpha": cv_alphas,
        },
        cv=cv,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        refit=False,
    )
    search_poly.fit(xtrain, ytrain.ravel())
    results = search_poly.cv_results_
    degrees = np.arange(1, 31)
    mse_plot, se_plot = _cv_score_grids(
        results, "ridge__alpha", cv_alphas, degrees, cv.get_n_splits()
    )
    alpha_indices = np.argmin(mse_plot, axis=0)
    degree_indices = np.arange(len(degrees))
    mse = mse_plot[alpha_indices, degree_indices]
    se = se_plot[alpha_indices, degree_indices]
    output_path = Path(__file__).resolve().parent / "PolynomialRidge.pdf"
    best_alpha_index, best_degree_index = _plot_cv_contour(
        mse_plot, degrees, lambdas, "Polynomial Ridge: 5-fold validation MSE",
        output_path
    )
    best_lambda = lambdas[best_alpha_index]
    best_degree = int(degrees[best_degree_index])
    best_alpha = len(xtrain) * best_lambda
    print(f"Best Ridge degree: {best_degree}, lambda: {best_lambda}")
    print(f"Saved figure to {output_path}")
    ridge_model = clone(poly).set_params(
        polynomialfeatures__degree=best_degree,
        ridge__alpha=best_alpha,
    )
    ridge_model.fit(xtrain, ytrain.ravel())
    ypred = ridge_model.predict(xtest)
    TestMSE = np.mean((ypred - ytest.ravel()) ** 2)
    print(f"CV-selected polynomial degree: {best_degree}")
    print(f"Selected Ridge 5-fold CV MSE: {mse[best_degree_index]:.6f} +/- {se[best_degree_index]:.6f}")
    print(f"Selected Ridge held-out test MSE: {TestMSE:.6f}")
    return ridge_model, mse, se, best_degree

def Lassofit():
    """
    AI-Declaration:
    I used Chat-GPT6 Luna in VSCode in order to fix this function so I could properly use Lasso Regression.
    The general outline of this function was written by me, but some of the plotting / data formatting was AI generated.
    This is equal to usage level 3.
    03.11.2026
    Johannes Schäfers
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    poly = make_pipeline(
        PolynomialFeatures(include_bias=False),
        StandardScaler(),
        Lasso(max_iter=50000, random_state=2026),
    )
    degrees = np.arange(1, 31)
    lambdas = np.logspace(-8, 0, 10)
    alphas = lambdas / 2
    search_poly = GridSearchCV(
        poly,
        {"polynomialfeatures__degree": degrees, "lasso__alpha": alphas},
        cv=cv,
        scoring="neg_mean_squared_error",
        return_train_score=True,
        refit=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        search_poly.fit(xtrain, ytrain.ravel())
    results = search_poly.cv_results_
    mse_plot, se_plot = _cv_score_grids(
        results, "lasso__alpha", alphas, degrees, cv.get_n_splits()
    )
    alpha_indices = np.argmin(mse_plot, axis=0)
    degree_indices = np.arange(len(degrees))
    mse = mse_plot[alpha_indices, degree_indices]
    se = se_plot[alpha_indices, degree_indices]
    output_path = Path(__file__).resolve().parent / "PolynomialLasso.pdf"
    best_alpha_index, best_degree_index = _plot_cv_contour(
        mse_plot, degrees, lambdas, "Polynomial Lasso: 5-fold validation MSE",
        output_path
    )
    best_lambda = lambdas[best_alpha_index]
    best_degree = int(degrees[best_degree_index])
    best_alpha = best_lambda / 2
    print(f"Best Lasso degree: {best_degree}, lambda: {best_lambda}")
    print(f"Saved figure to {output_path}")
    lasso_model = clone(poly).set_params(
        polynomialfeatures__degree=best_degree,
        lasso__alpha=best_alpha,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lasso_model.fit(xtrain, ytrain.ravel())
    ypred = lasso_model.predict(xtest)
    TestMSE = np.mean((ypred - ytest.ravel()) ** 2)
    print(f"CV-selected polynomial degree: {best_degree}")
    print(f"Selected Lasso 5-fold CV MSE: {mse[best_degree_index]:.6f} +/- {se[best_degree_index]:.6f}")
    print(f"Selected Lasso held-out test MSE: {TestMSE:.6f}")
    return lasso_model, mse, se, best_degree

def PlotComparison(
    network_models, network_scores, ols_model, best_degree, ols_cv_mse, ols_cv_se
):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 8),
                             constrained_layout=True)
    axes[0].plot(xx, runge(xx), color="#004488", linewidth=2, label="Runge function")
    axes[0].plot(xx, ols_model.predict(xx), color="#BB5566", linewidth=2,
                  label=f"OLS polynomial (degree {best_degree})")
    colors = ("#228833", "#AA3377", "#CC6677", "#4477AA")
    best_nn_name = min(
        network_scores, key=lambda name: network_scores[name]["cv_mse"]
    )
    best_nn_model = network_models[best_nn_name]
    axes[0].plot(
        xx,
        best_nn_model.predict(xx),
        color=colors[0],
        label=f"Best NN {best_nn_name}",
    )
    axes[0].scatter(xtrain, ytrain, s=18, color="0.5", alpha=0.65,
                    label="training observations")
    axes[0].scatter(xtest, ytest, s=32, marker="x", color="black",
                    label="held-out observations")
    axes[0].set_xlabel("$x$", fontsize=15)
    axes[0].set_ylabel("$y$", fontsize=15)
    axes[0].tick_params(axis="both", labelsize=13)
    axes[0].set_title("Fits on the shared Runge data split", fontsize=17)
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(
        frameon=False,
        fontsize=9,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )

    names = [f"NN {name}" for name in network_scores]
    names.insert(0, f"OLS degree {best_degree}")
    cv_errors = [ols_cv_mse]
    cv_errors.extend(network_scores[name]["cv_mse"] for name in network_scores)
    cv_standard_errors = [ols_cv_se]
    cv_standard_errors.extend(
        network_scores[name]["cv_se"] for name in network_scores
    )
    bar_colors = ["#BB5566", *colors[:len(network_scores)]]
    bars = axes[1].bar(
        names,
        cv_errors,
        yerr=cv_standard_errors,
        capsize=4,
        color=bar_colors,
    )
    axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=12)
    axes[1].set_ylabel(
        "Mean 5-fold CV MSE (error bars: standard error)", fontsize=15
    )
    axes[1].set_title("Cross-validation error by model", fontsize=17)
    axes[1].tick_params(axis="x", labelrotation=25, labelsize=12)
    axes[1].tick_params(axis="y", labelsize=12)
    axes[1].grid(axis="y", alpha=0.2)
    output_path = Path(__file__).resolve().parent / "Comparison_NN_vs_Poly.pdf"
    fig.savefig(output_path)
    print(f"Saved figure to {output_path}")
    plt.close(fig)


def main():
    BuildTestNets()
    network_models, network_scores = task_b()       #Test different neural network architectures and hyperparameters using only the sigmoid function
    Best_Poly, poly_mse, poly_se, best_degree = Polyfit()        #Test different polynomial degrees using OLS regression
    Best_Ridge, _, _, ridge_degree = Ridgefit()     #Compute Ridge regression for different polynomial degrees and regularization parameters
    Best_Lasso, _, _, lasso_degree = Lassofit()     #Compute Lasso regression for different polynomial degrees and regularization parameters
    print("\nBest configuration for each neural-network architecture:")
    print(
        f"{'Hidden widths':<18} {'Optimizer':<20} {'gamma':>8} "
        f"{'CV MSE':>12} {'CV SE':>12} {'Test MSE':>12}"
    )
    for name, scores in network_scores.items():
        print(
            f"{name:<18} {scores['optimizer']:<20} {scores['gamma']:>8g} "
            f"{scores['cv_mse']:>12.6f} {scores['cv_se']:>12.6f} "
            f"{scores['test_mse']:>12.6f}"
        )
    PlotComparison(
        network_models,
        network_scores,
        Best_Poly,
        best_degree,
        poly_mse[best_degree - 1],
        poly_se[best_degree - 1],
    ) #Compare the best neural network models with the best polynomial model

    from exercise_e import exercise_e
    exercise_e()


if __name__ == "__main__":
    results_path = Path(__file__).resolve().parent / "Project2_results.txt"
    with results_path.open("w", encoding="utf-8") as results_file:
        with redirect_stdout(results_file):
            main()