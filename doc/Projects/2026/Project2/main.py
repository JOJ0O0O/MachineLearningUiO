import NeuralNet as NN
import numpy as np
import sklearn
import Activation_Cost_funcs as act
import Optimizers as opt
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve)


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
x = x.reshape(-1,1)
y = y.reshape(-1,1)
xx = np.linspace(-1,1,1000)

"""
AI-Declaration:
I used Chat-GPT6 Luna in VSCode in order to find an error as i did not know that I needed to reshape my data.
The four lines below are therefore AI generated.
This is equal to usage level 2.
Johannes Schäfers
"""
xx = xx.reshape(-1,1)

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

    plt.plot(xx,yxx, label="Handwritten")
    plt.plot(xx,yxxjax, label="Jax")
    plt.legend()
    plt.show()

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
    grid.fit(x,y)
    results = grid.cv_results_
    mse = np.empty((len(Optimizers), len(layers), len(gammas)))
    for params, score in zip(results["params"], results["mean_test_score"]):
        optimizer_index = Optimizers.index(params["neuralnetwork__optimizer"])
        layer_index = layers.index(params["neuralnetwork__layer_output_sizes"])
        gamma_index = gammas.index(params["neuralnetwork__gamma"])
        mse[optimizer_index, layer_index, gamma_index] = -score

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    print(best_params, best_model)
    fig, axes = plt.subplots(1, len(Optimizers), figsize=(25, 7),
                             constrained_layout=True)
    """
    AI Declaration:
    I have used AI to fix the below code, as my first written code had a bug inside of it
    Tool: GPT6-Luna in VSCode
    This corresponds to usage Level 1 as I have written, tested and verified the code myself and had only used AI as a small help
    """
    mse_min, mse_max = np.min(mse), np.max(mse)
    for i, (optimizer, axis) in enumerate(zip(Optimizers, axes)):
        im = axis.imshow(mse[i], cmap="viridis_r", origin="lower", aspect="equal",
                         vmin=mse_min, vmax=mse_max)
        for layer_index in range(len(layers)):
            for gamma_index in range(len(gammas)):
                value = mse[i, layer_index, gamma_index]
                text_color = "white" if im.norm(value) >= 0.5 else "black"
                axis.text(gamma_index, layer_index, f"{value:.3g}",
                          ha="center", va="center", fontsize=7, color=text_color)
        axis.set_xticks(range(len(gammas)))
        axis.set_xticklabels([f"{gamma:g}" for gamma in gammas], fontsize=7)
        axis.set_yticks(range(len(layers)))
        axis.set_yticklabels([str(layer) for layer in layers], fontsize=7)
        axis.set_xlabel("$\\gamma$")
        axis.set_ylabel("Network architecture")
        axis.set_title(f"{optimizer.__name__}: CV MSE", fontsize=9)
    fig.colorbar(im, ax=axes, label="MSE")
    output_path = Path(__file__).resolve().parent / "Task_b Layers vs Gammas.pdf"
    fig.savefig(output_path)
    print(f"Saved figure to {output_path}")

def Polyfit():
    cv = KFold(n_splits=5, shuffle=True, random_state=2026)
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(), LinearRegression())
    search_poly = GridSearchCV(poly, {"polynomialfeatures__degree": range(1, 16)}, cv=cv, scoring="neg_mean_squared_error", return_train_score=True)
    search_poly.fit(x, y)
    results = search_poly.cv_results_          # a dict of arrays, one entry per grid point
    mse = -results["mean_test_score"]
    se = results["std_test_score"] / np.sqrt(cv.get_n_splits() - 1)
    print(mse, " +/- ", se)
    plt.errorbar(range(1,16),mse,yerr=se)
    plt.title("MSE for Polynomial Regression", fontsize = 12)
    plt.ylabel("MSE", fontsize = 9)
    plt.xlabel("Degree of Polynomial", fontsize = 9)
    output_path = Path(__file__).resolve().parent / "PolynomialRegression.pdf"
    plt.savefig(output_path)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    #BuildTestNets()
    #task_b()
    Polyfit()