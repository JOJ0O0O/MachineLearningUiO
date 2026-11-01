import NeuralNet as NN
import numpy as np
import Activation_Cost_funcs as act
import Optimizers as opt
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve,
                                     train_test_split)


from exercise_e import exercise_e

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
x = np.sort(rng.uniform(-1, 1, n))
y = runge(x) + rng.normal(0, sigma, n)
ymean= y.mean()
"""
AI-Declaration:
I used Chat-GPT6 Luna in VSCode in order to find an error as i did not know that I needed to reshape my data.
Four lines below are therefore AI generated.
This is equal to usage level 2.
27.10.2026
Johannes Schäfers
"""

x = x.reshape(-1,1)
y = y.reshape(-1,1)
xx = np.linspace(-1,1,1000)
xx = xx.reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=2026
)

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
                            gamma=0.01)

    TestOptiJax = opt.AdamOptimizer
    TestNetJax = NN.NeuralNetwork(x.shape[1],
                            [30,30,1], 
                            [act.ReLU, act.sigmoid, act.sigmoid], 
                            act.mse, 
                            optimizer = TestOptiJax,
                            regularization = "L2",
                            lam=0.001,
                            gamma = 0.01)

    TestNet.train(x, y, 1000)
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
                        optimizer=opt.AdamOptimizer))
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
            "test_mse": np.mean((model.predict(xtest) - ytest.ravel()) ** 2),
            "optimizer": params["neuralnetwork__optimizer"].__name__,
            "gamma": params["neuralnetwork__gamma"],
        }

    fig, axes = plt.subplots(1, len(Optimizers), figsize=(18, 6),
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
        axis.set_title(f"{optimizer.__name__}", fontsize=10)
    fig.suptitle("Red stars mark each optimizer's best CV score")
    fig.colorbar(im, ax=axes, label="5-fold validation MSE (color clipped at 95th percentile)")
    output_path = Path(__file__).resolve().parent / "Task_b Layers vs Gammas.pdf"
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
    fig, ax = plt.subplots(figsize=(8, 5))
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


def PlotComparison(network_models, network_scores, ols_model, best_degree):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(xx, runge(xx), color="#004488", linewidth=2, label="Runge function")
    axes[0].plot(xx, ols_model.predict(xx), color="#BB5566", linewidth=2,
                  label=f"OLS polynomial (degree {best_degree})")
    colors = ("#228833", "#AA3377", "#CC6677", "#4477AA")
    for (name, model), color in zip(network_models.items(), colors):
        axes[0].plot(xx, model.predict(xx), color=color, label=f"NN {name}")
    axes[0].scatter(xtrain, ytrain, s=18, color="0.5", alpha=0.65,
                    label="training observations")
    axes[0].scatter(xtest, ytest, s=32, marker="x", color="black",
                    label="held-out observations")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title("Fits on the shared Runge data split")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )

    names = [f"NN {name}" for name in network_scores]
    names.insert(0, f"OLS degree {best_degree}")
    test_errors = [np.mean((ols_model.predict(xtest) - ytest.ravel()) ** 2)]
    test_errors.extend(network_scores[name]["test_mse"] for name in network_scores)
    bar_colors = ["#BB5566", *colors[:len(network_scores)]]
    bars = axes[1].bar(names, test_errors, color=bar_colors)
    axes[1].bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].set_ylabel("Held-out MSE")
    axes[1].set_title("Held-out test error by model")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(axis="y", alpha=0.2)
    fig.subplots_adjust(bottom=0.3, top=0.9, wspace=0.3)
    output_path = Path(__file__).resolve().parent / "Comparison_NN_vs_Poly.pdf"
    fig.savefig(output_path)
    print(f"Saved figure to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    #BuildTestNets()
    network_models, network_scores = task_b()
    Best_Poly, _, _, best_degree = Polyfit()
    print("\nBest configuration for each neural-network architecture:")
    print(f"{'Hidden widths':<18} {'Optimizer':<20} {'gamma':>8} {'CV MSE':>12} {'Test MSE':>12}")
    for name, scores in network_scores.items():
        print(
            f"{name:<18} {scores['optimizer']:<20} {scores['gamma']:>8g} "
            f"{scores['cv_mse']:>12.6f} {scores['test_mse']:>12.6f}"
        )
    PlotComparison(network_models, network_scores, Best_Poly, best_degree)
