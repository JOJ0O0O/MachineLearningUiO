"""
Exercise e: multiclass classification of the MNIST digits with a feed-forward neural network.

By default (submission setting) the expensive steps are not repeated: the grid-search results
and the weights of the final network are loaded from the files handed in with the submission.
The script then re-plots the grid-search heatmaps, evaluates the network on the test set and
plots the confusion matrix.

    python exercise_e.py                                  # load grid search and weights (default)
    python exercise_e.py --train                          # load grid search, retrain the network
    python exercise_e.py --grid-search --train --n-jobs 12   # redo everything
    python exercise_e.py --quick-test                     # small grid search on 6000 samples (minutes)

LLM-assisted: see the "LLM-assisted" section of every function.
"""
import argparse
import ast
import os
import urllib.request
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FormatStrFormatter

from sklearn.metrics import (accuracy_score, mean_squared_error, make_scorer,
                             ConfusionMatrixDisplay, confusion_matrix)
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

import NeuralNet as NN
import Activation_Cost_funcs as act
import Optimizers as opt


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results", "exercise_e")


# ----------------------------------------------------------------------------------------
# Data and weights
# ----------------------------------------------------------------------------------------

def load_mnist(filename=os.path.join(BASE, "mnist.npz"), url=MNIST_URL, n_classes=10):
    """
    Loads MNIST (downloads it first if the file does not exist), flattens the images to
    vectors and scales the pixel values to [0, 1].

    Returns
    -------
    x_train : array (60000, 784), float32
    y_train : array (60000,), integer training labels
    y_train_onehot : array (60000, n_classes), one-hot encoded training labels
    x_test : array (10000, 784), float32
    y_test : array (10000,), integer test labels

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026 / Nov 2026)
    Role: Oct 2026: identified a shape mismatch between the softmax output (N, 10) and the
    integer labels (N,); one-hot encoding implemented by the author.
    Nov 2026 (clean-up): moved the author's loading code into this function; float32 data;
    file path relative to the script folder.
     
    """
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
    with np.load(filename) as data:
        x_train = (data["x_train"].reshape(len(data["x_train"]), -1) / 255.0).astype(np.float32)
        x_test = (data["x_test"].reshape(len(data["x_test"]), -1) / 255.0).astype(np.float32)
        y_train, y_test = data["y_train"], data["y_test"]
    y_train_onehot = np.eye(n_classes, dtype=np.float32)[y_train]
    return x_train, y_train, y_train_onehot, x_test, y_test


def save_weights(layers, epochs, filename):
    """
    Saves the weights and biases of all layers and the number of training epochs to an .npz file.

    Parameters
    ----------
    layers : list of tuples (W, b), weights and biases of each layer
    epochs : int, number of epochs the network was trained for
    filename : str, path of the .npz file

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Provided the function; stores each array under its own key (W0, b0, W1, ...)
    to avoid saving a ragged list with np.savez.
    Modifications: Integrated and tested by the author (save/load round trip).
    """
    arrays = {"epochs": epochs}
    for i, (W, b) in enumerate(layers):
        arrays[f"W{i}"] = np.asarray(W)
        arrays[f"b{i}"] = np.asarray(b)
    np.savez(filename, **arrays)


def load_weights(filename):
    """
    Loads the weights and biases saved by save_weights.

    Returns
    -------
    layers : list of tuples (W, b), weights and biases of each layer
    epochs : int, number of epochs the network was trained for

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Provided the function; reconstructs the layer list from the W{i}/b{i} keys.
    Modifications: Integrated and tested by the author (save/load round trip).
    """
    with np.load(filename) as data:
        n_layers = (len(data.files) - 1) // 2          # W{i} and b{i} per layer, plus "epochs"
        layers = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)]
        return layers, int(data["epochs"])


# ----------------------------------------------------------------------------------------
# Network and scoring
# ----------------------------------------------------------------------------------------

def softmax_cross_entropy_error(predict, target):
    """
    Error of the output layer for softmax activation combined with the cross-entropy loss,
    where the derivative simplifies to (a - y).

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Moved the author's nested output_error function to module level, so that the grid
    search can pass it to the network as well (and it can be sent to the parallel workers).
    Modifications: Integrated and tested by the author.
    """
    return predict - target


def activation_list(n_layers, hidden=act.ReLU, output=act.softmax):
    """
    One activation function per layer: the hidden activation for every hidden layer and the
    output activation for the last layer.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function (clean-up: replaces the same list expression used in
    several places).
     
    """
    return [hidden] * (n_layers - 1) + [output]


def build_network(input_size, layer_output_sizes, gamma, lam, seed, epochs=10):
    """
    Feed-forward network with ReLU hidden layers, softmax output, cross-entropy loss,
    the Adam optimizer and L2 regularization (switched off for lam = 0).

    Parameters
    ----------
    input_size : int, number of input features (784 for MNIST)
    layer_output_sizes : sequence of int, neurons per layer including the output layer
    gamma : float, learning rate
    lam : float, L2 regularization strength
    seed : int, random seed
    epochs : int, default number of training epochs (used by GridSearchCV's fit)

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function (clean-up: one place to set up the network for the grid
    search and the final model).
     
    """
    return NN.NeuralNetwork(network_input_size=input_size,
                            layer_output_sizes=list(layer_output_sizes),
                            activation_funcs=activation_list(len(layer_output_sizes)),
                            cost_fun=act.cross_entropy,
                            output_error=softmax_cross_entropy_error,
                            optimizer=opt.AdamOptimizer,
                            gamma=gamma,
                            epochs=epochs,
                            regularization="L2" if lam > 0 else None,
                            lam=lam,
                            seed=seed)


def softmax_accuracy(y_true, y_pred):
    """
    Accuracy for networks with a softmax output: one-hot targets and predicted class
    probabilities are converted to class labels with argmax. Returns NaN if the predictions
    contain NaN/inf (diverged training), so such fits show up as failed instead of getting a
    meaningless score (np.argmax silently returns 0 for NaN rows).

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function; scoring="accuracy" failed for every fit because sklearn
    cannot compare one-hot targets with continuous softmax outputs.
     
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not np.all(np.isfinite(y_pred)):
        return np.nan
    return accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))


def softmax_mse(y_true, y_pred):
    """
    Mean squared error between one-hot targets and softmax outputs. Returns NaN instead of
    raising for diverged predictions (sklearn's MSE raises 'Input contains NaN', which would
    abort the grid search because of error_score="raise").

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function (NaN-safe MSE scorer for the grid search).
     
    """
    y_pred = np.asarray(y_pred)
    if not np.all(np.isfinite(y_pred)):
        return np.nan
    return mean_squared_error(np.asarray(y_true), y_pred)


# ----------------------------------------------------------------------------------------
# Grid search
# ----------------------------------------------------------------------------------------

def report_failed_fits(results, score_key="mean_test_accuracy"):
    """
    Prints how many parameter combinations ended with a NaN score (diverged training),
    counted separately for every hyperparameter value, e.g. to spot learning rates that
    always diverge.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function.
     
    """
    scores = np.asarray(results[score_key], dtype=float)
    failed = [params for params, score in zip(results["params"], scores) if np.isnan(score)]
    print(f"{len(failed)} of {len(scores)} parameter combinations have a NaN {score_key}.")
    for key in (results["params"][0] if failed else []):
        total = Counter(str(params[key]) for params in results["params"])
        counts = Counter(str(params[key]) for params in failed)
        print(f"  NaN by {key}: " + ", ".join(f"{value}: {counts.get(value, 0)}/{n}" for value, n in total.items()))


def grid_search_hyperparameters(
        x,
        y,
        layers=[(50, 10), (50, 50, 10), (50, 50, 50, 10), (100, 10), (100, 100, 10), (100, 100, 100, 10)],
        gammas=[0.0003, 0.001, 0.003],
        epochs=[10, 20, 50],
        lams=[0, 0.001, 0.01],
        kfolds=3,
        n_jobs=4,
        seed=2026,
        results_filename=os.path.join(RESULTS_DIR, "grid_search_results.npz")
):
    """
    Cross-validated grid search (Adam optimizer) over the architecture, the learning rate,
    the number of epochs and the L2 regularization strength. Every fit is scored with the
    accuracy (used to select the best parameters) and the MSE between the one-hot targets
    and the softmax outputs. The results are saved to results_filename.

    Parameters
    ----------
    x : array (n_samples, n_features), training inputs
    y : array (n_samples, n_classes), one-hot encoded training targets
    layers : list of tuples, neurons per layer including the output layer (10 for MNIST)
    gammas : list of float, learning rates
    epochs : list of int, numbers of training epochs
    lams : list of float, L2 regularization strengths
    kfolds : int, number of cross-validation folds
    n_jobs : int, number of parallel worker processes
    seed : int, random seed (network initialisation and fold split)
    results_filename : str, path of the .npz file with the results

    Returns
    -------
    dict, the saved results as returned by load_grid_search_results

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026 / Nov 2026)
    Role: Oct 2026: fixed the author's version (parameter names without pipeline prefix,
    cross-entropy cost and output error passed to the network, NaN-initialised result array,
    refit=False, results stored without pickled objects).
    Nov 2026: NaN-safe accuracy and MSE scorers (scoring="accuracy" failed for every fit);
    error_score="raise"; lambda kept as a separate axis of the result arrays; best parameters
    chosen by accuracy with nanargmax; restriction to Adam by the author.
    Nov 2026 (clean-up): plotting moved to plot_grid_search_results, the saved file is returned
    in the same form as when loaded. (The network is set up here and not with build_network,
    because the activation list must fit the deepest architecture of the grid.)
     
    """
    activation_funcs = activation_list(max(len(layer) for layer in layers))
    net = NN.NeuralNetwork(network_input_size=x.shape[1],
                           layer_output_sizes=layers[0],
                           activation_funcs=activation_funcs,      # long enough for the deepest network
                           cost_fun=act.cross_entropy,
                           output_error=softmax_cross_entropy_error,
                           optimizer=opt.AdamOptimizer,
                           epochs=epochs[0],
                           lam=lams[0],
                           regularization="L2",
                           seed=seed)
    param_grid = {"layer_output_sizes": layers, "gamma": gammas, "epochs": epochs, "lam": lams}
    scoring = {"accuracy": make_scorer(softmax_accuracy),
               "mse": make_scorer(softmax_mse, greater_is_better=False)}
    grid = GridSearchCV(net, param_grid, cv=KFold(n_splits=kfolds, shuffle=True, random_state=seed),
                        scoring=scoring, n_jobs=n_jobs, refit=False, error_score="raise")
    grid.fit(x, y)
    results = grid.cv_results_
    report_failed_fits(results)

    # result arrays with the axes (layers, gamma, epochs, lam)
    shape = (len(layers), len(gammas), len(epochs), len(lams))
    accuracy = np.full(shape, np.nan)
    mse = np.full(shape, np.nan)
    for params, acc_score, mse_score in zip(results["params"], results["mean_test_accuracy"], results["mean_test_mse"]):
        index = (layers.index(params["layer_output_sizes"]), gammas.index(params["gamma"]),
                 epochs.index(params["epochs"]), lams.index(params["lam"]))
        accuracy[index] = acc_score
        mse[index] = -mse_score            # make_scorer(greater_is_better=False) stores -MSE

    # refit=False with several metrics: GridSearchCV has no best_params_, so pick it here
    mean_accuracy = np.asarray(results["mean_test_accuracy"], dtype=float)
    if np.all(np.isnan(mean_accuracy)):
        raise RuntimeError("Every parameter combination has a NaN accuracy (all fits diverged).")
    best = results["params"][int(np.nanargmax(mean_accuracy))]

    os.makedirs(os.path.dirname(results_filename), exist_ok=True)
    np.savez(results_filename,
             accuracy=accuracy, mse=mse,
             layers=np.array([str(layer) for layer in layers]),
             gammas=np.array(gammas), epochs=np.array(epochs), lam=np.array(lams),
             axes=np.array(["layers", "gamma", "epochs", "lam"]),
             optimizer="AdamOptimizer",
             best_layers=np.array(best["layer_output_sizes"]), best_gamma=best["gamma"],
             best_epochs=best["epochs"], best_lam=best["lam"],
             best_score=np.nanmax(mean_accuracy))
    print(f"Saved grid-search results to {results_filename}")
    return load_grid_search_results(results_filename)


def load_grid_search_results(filename):
    """
    Loads the grid-search results written by grid_search_hyperparameters.

    Returns
    -------
    dict with
        accuracy, mse : arrays, shape (n_layers, n_gammas, n_epochs, n_lams), CV means
        layers : list of tuples (neurons per layer including the output layer)
        gammas, epochs, lams : lists of the grid values
        best_layers, best_gamma, best_epochs, best_lam, best_score : best configuration by accuracy

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function. Clean-up: best_layers is read as a list of integers (the
    stored array cannot be parsed with ast.literal_eval); clear message if the file is missing.
     
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Run with --grid-search to create it.")
    with np.load(filename) as data:
        return {
            "accuracy": np.array(data["accuracy"], dtype=float),
            "mse": np.array(data["mse"], dtype=float),
            "layers": [tuple(ast.literal_eval(str(layer))) for layer in data["layers"]],
            "gammas": [float(g) for g in data["gammas"]],
            "epochs": [int(e) for e in data["epochs"]],
            "lams": [float(l) for l in data["lam"]],
            "best_layers": tuple(int(n) for n in data["best_layers"]),
            "best_gamma": float(data["best_gamma"]),
            "best_epochs": int(data["best_epochs"]),
            "best_lam": float(data["best_lam"]),
            "best_score": float(data["best_score"]),
        }


# ----------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------

def plot_grid_search(scores, layers, gammas, epochs, filename=None,
                     score_label="CV MSE", lower_is_better=True, log_scale=True):
    """
    Shows the grid search (architecture x learning rate x epochs) for one regularization
    strength as a single heatmap sized for one column of a two-column report. Rows are the
    architectures; every learning rate is a block of columns, one column per number of epochs
    (explained by the small key on top). The best cell is marked with a star, failed fits
    (NaN) are grey.

    Parameters
    ----------
    scores : array, shape (n_layers, n_gammas, n_epochs), e.g. the CV accuracy for one lambda
    layers : list of tuples, layer sizes including the output layer
    gammas : list of float, learning rates
    epochs : list of int, numbers of epochs
    filename : str or None, path of the saved figure (not saved if None)
    score_label : str, colour-bar label
    lower_is_better : bool, True for errors (MSE), False for accuracies
    log_scale : bool, logarithmic colour scale

    Returns
    -------
    fig : matplotlib Figure

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026 / Nov 2026)
    Role: Provided the function (Oct 2026: nested-block heatmap with optimizer rows).
    Nov 2026: rewritten for a single optimizer (no optimizer axis and key), compact
    colour-bar labels on a log scale (they were cut off), thin lines between architectures.
     
    """
    n_lay, n_gam, n_ep = scores.shape
    image = scores.reshape(n_lay, n_gam * n_ep)        # column = gamma * n_ep + epochs

    cmap = plt.get_cmap("viridis_r" if lower_is_better else "viridis").copy()
    cmap.set_bad("lightgray")
    finite = image[np.isfinite(image)]
    if log_scale:
        finite = finite[finite > 0]
    if finite.size > 0:
        norm = (LogNorm if log_scale else Normalize)(vmin=finite.min(), vmax=finite.max())
    else:
        norm = Normalize(vmin=0, vmax=1)    # everything failed: grey image, dummy colour range

    fig = plt.figure(figsize=(3.4, 3.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.09, 1], width_ratios=[1, 0.045],
                          hspace=0.16, wspace=0.06, left=0.19, right=0.8, bottom=0.12, top=0.9)
    ax, cax = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    key_ax = fig.add_subplot(gs[0, 0].subgridspec(1, 3, width_ratios=[0.6, 1, 0.6])[1])

    # heatmap: white lines separate the learning-rate blocks and the architectures
    im = ax.imshow(image, cmap=cmap, norm=norm, origin="lower", aspect="auto", interpolation="nearest")
    for col in range(1, n_gam):
        ax.axvline(col * n_ep - 0.5, color="white", lw=1.6)
    for row in range(1, n_lay):
        ax.axhline(row - 0.5, color="white", lw=0.6)
    ax.set_xticks([col * n_ep + (n_ep - 1) / 2 for col in range(n_gam)])
    ax.set_xticklabels([f"{gamma:g}" for gamma in gammas])
    ax.set_yticks(range(n_lay))
    hidden = [layer[:-1] for layer in layers]      # e.g. (100, 100) -> "100x2"
    ax.set_yticklabels([f"{h[0]}$\\times${len(h)}" if len(set(h)) == 1 else "-".join(map(str, h)) for h in hidden])
    ax.tick_params(length=0, labelsize=7)
    ax.set_xlabel(r"Learning rate $\gamma$", fontsize=8)
    ax.set_ylabel("Hidden layers (neurons)", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # best configuration
    if np.any(np.isfinite(image)):
        best = np.nanargmin(image) if lower_is_better else np.nanargmax(image)
        best_row, best_col = np.unravel_index(best, image.shape)
        ax.plot(best_col, best_row, marker="*", markersize=9, color="red",
                markeredgecolor="white", markeredgewidth=0.6)

    # colour bar; on a log scale compact labels (0.04 instead of 4x10^-2, which was cut off)
    cbar = fig.colorbar(im, cax=cax)
    if log_scale:
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        cbar.ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
    cbar.ax.tick_params(labelsize=6, which="both")
    cbar.set_label(score_label, fontsize=8)

    # key: the columns inside one learning-rate block
    key_ax.imshow(np.zeros((1, n_ep)), cmap="Greys", vmin=0, vmax=1, aspect="auto")
    for col in range(n_ep + 1):
        key_ax.axvline(col - 0.5, color="black", lw=0.5)
    key_ax.axhline(-0.5, color="black", lw=0.5)
    key_ax.axhline(0.5, color="black", lw=0.5)
    key_ax.set_xticks(range(n_ep))
    key_ax.set_xticklabels([str(n) for n in epochs])
    key_ax.set_yticks([])
    key_ax.xaxis.tick_top()
    key_ax.xaxis.set_label_position("top")
    key_ax.tick_params(length=0, labelsize=6, pad=1)
    key_ax.set_xlabel("epochs per block", fontsize=6, labelpad=2)
    for spine in key_ax.spines.values():
        spine.set_visible(False)

    if filename is not None:
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename)
        print(f"Saved figure to {filename}")
    return fig


def plot_grid_search_results(results, stem):
    """
    One accuracy and one MSE heatmap (plot_grid_search) per regularization strength,
    saved as {stem}_accuracy_lam{lambda}.pdf and {stem}_mse_lam{lambda}.pdf.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function (clean-up: the plotting loop previously inside
    grid_search_hyperparameters, now also used for loaded results).
     
    """
    for i, lam in enumerate(results["lams"]):
        for metric, label, lower_is_better, log_scale in [("accuracy", "CV accuracy", False, False),
                                                           ("mse", "CV MSE", True, True)]:
            fig = plot_grid_search(results[metric][..., i], results["layers"], results["gammas"], results["epochs"],
                                   filename=f"{stem}_{metric}_lam{lam:g}.pdf",
                                   score_label=f"{label} ($\\lambda$ = {lam:g})",
                                   lower_is_better=lower_is_better, log_scale=log_scale)
            plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """
    Saves the confusion matrix of the test-set predictions.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Clean-up: moved the author's confusion-matrix plot into its own function.
     
    """
    fig, ax = plt.subplots(figsize=(9, 3.6))
    confusion_matrix_ = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix_, display_labels=list(range(10))).plot(ax=ax, colorbar=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved figure to {filename}")

    return confusion_matrix_

def precision(confusion_matrix):
    """
    Precision per class from a confusion matrix (rows = true class, columns = predicted class):
    the fraction of images predicted as class k that really are class k, TP_k / (TP_k + FP_k).

    Parameters
    ----------
    confusion_matrix : array (n_classes, n_classes), e.g. from sklearn.metrics.confusion_matrix

    Returns
    -------
    array (n_classes,), precision of every class (0 for a class that is never predicted)
    array (n_classes,), uncertainty of precision of every class

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function.
  
    """
    confusion_matrix = np.asarray(confusion_matrix, dtype=float)
    predicted = confusion_matrix.sum(axis=0)                      # column sums: TP + FP
    precision = np.divide(np.diag(confusion_matrix), predicted, out=np.zeros(len(confusion_matrix)), where=predicted > 0)
    # Calculate the uncertainty of precision (standard error of the proportion)
    uncertainty = np.sqrt(precision * (1 - precision) / predicted)
    return precision, uncertainty


def recall(confusion_matrix):
    """
    Recall per class from a confusion matrix (rows = true class, columns = predicted class):
    the fraction of images of class k that are recognised as class k, TP_k / (TP_k + FN_k).

    Parameters
    ----------
    confusion_matrix : array (n_classes, n_classes), e.g. from sklearn.metrics.confusion_matrix

    Returns
    -------
    array (n_classes,), recall of every class (0 for a class without test images)
    array (n_classes,), uncertainty of recall of every class

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function.
  
    """
    confusion_matrix = np.asarray(confusion_matrix, dtype=float)
    actual = confusion_matrix.sum(axis=1)                         # row sums: TP + FN
    recall = np.divide(np.diag(confusion_matrix), actual, out=np.zeros(len(confusion_matrix)), where=actual > 0)
    # Calculate the uncertainty of recall (standard error of the proportion)
    uncertainty = np.sqrt(recall * (1 - recall) / actual)
    return recall, uncertainty



def f1_score(confusion_matrix):
    """
    F1 score per class from a confusion matrix: the harmonic mean of precision and recall,
    2 P R / (P + R).

    Parameters
    ----------
    confusion_matrix : array (n_classes, n_classes), e.g. from sklearn.metrics.confusion_matrix

    Returns
    -------
    array (n_classes,), F1 score of every class (0 if precision and recall are both 0)

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function.
  
    """
    confusion_matrix = np.asarray(confusion_matrix, dtype=float)
    precision_val, precision_uncertainty = precision(confusion_matrix)
    recall_val, recall_uncertainty = recall(confusion_matrix)
    F1 = np.divide(2 * precision_val * recall_val, precision_val + recall_val, out=np.zeros(len(precision_val)), where=(precision_val + recall_val) > 0)

    F1_uncertainty = 2 / np.square(precision_val + recall_val) * np.sqrt(np.square(np.square(precision_val) * recall_uncertainty) +
                            np.square(np.square(recall_val) * precision_uncertainty))  # gaussian error propagation

    return F1, F1_uncertainty


def latex_table_accuracy_precision_recall_f1(confusion_matrix, filename):
    """
    Saves a LaTeX table with the accuracy, precision, recall and F1 score of every class.

    Parameters
    ----------
    confusion_matrix : array (n_classes, n_classes), e.g. from sklearn.metrics.confusion_matrix
    filename : str, path of the .tex file

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function.
  
    """
    acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    acc_uncertainty = np.sqrt(acc * (1 - acc) / confusion_matrix.sum())
    precision_val, precision_uncertainty = precision(confusion_matrix)
    recall_value, recall_uncertainty = recall(confusion_matrix)
    f1, f1_uncertainty = f1_score(confusion_matrix)

    with open(filename, "w") as f:
        f.write("\\begin{tabular}{l|c|c|c}\n")
        f.write("\\hline\n")
        f.write("Class &  Precision & Recall & F1 score \\\\\n")
        f.write("\\hline\n")
        for i in range(len(confusion_matrix)):
            f.write(f"{i} & $({(precision_val[i] * 100):.2f} \\pm {(precision_uncertainty[i] * 100):.2f})\\%$ & $({(recall_value[i] * 100):.2f} \\pm {(recall_uncertainty[i] * 100):.2f})\\%$ & $({(f1[i] * 100):.2f} \\pm {(f1_uncertainty[i] * 100):.2f})\\%$ \\\\\n")
        f.write("\\hline\n")
        f.write(f"Accuracy & $({(acc * 100):.2f} \\pm {acc_uncertainty * 100:.2f})\\%$ & - & - \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
    print(f"Saved LaTeX table to {filename}")


# ----------------------------------------------------------------------------------------
# Final network and exercise e
# ----------------------------------------------------------------------------------------

def multiclassifier(x_train, y_train_onehot, layer_output_sizes, gamma, lam, epochs,
                    weight_file, train=False, seed=2026):
    """
    Returns the final multiclass classifier with the given hyperparameters: either trained on
    the full training set (train=True, the weights are then saved to weight_file) or with the
    weights loaded from weight_file (default, submission setting).

    Parameters
    ----------
    x_train, y_train_onehot : training inputs and one-hot targets
    layer_output_sizes : sequence of int, neurons per layer including the output layer
    gamma, lam : float, learning rate and L2 regularization strength
    epochs : int, number of training epochs (only used if train=True)
    weight_file : str, .npz file the weights are saved to / loaded from
    train : bool, train the network instead of loading the weights
    seed : int, random seed

    Returns
    -------
    classifier : NN.NeuralNetwork
    epochs : int, number of epochs the network was trained for

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026 / Nov 2026)
    Role: Oct 2026: classifier built after the grid search from its best parameters; loaded
    weights checked against the layer sizes.
    Nov 2026 (clean-up): data loading, grid search and evaluation moved to exercise_e; the
    terminal questions replaced by the train argument (command line); the best lambda is used.
     
    """
    classifier = build_network(x_train.shape[1], layer_output_sizes, gamma, lam, seed)

    if train:
        print(f"Training the network for {epochs} epochs...")
        classifier.train(x_train, y_train_onehot, epochs=epochs)
        save_weights(classifier.get_weights(), epochs, weight_file)
        print(f"Saved weights to {weight_file}")
        return classifier, epochs

    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"{weight_file} not found. Run with --train to create it.")
    weights, epochs = load_weights(weight_file)
    if list(layer_output_sizes) != [W.shape[1] for W, _ in weights]:
        raise ValueError(f"The weights in {weight_file} do not match the layer sizes {list(layer_output_sizes)}.")
    classifier.set_weights(weights)
    print(f"Loaded weights ({epochs} training epochs) from {weight_file}")
    return classifier, epochs


def parse_args():
    """
    Command-line options. Without options the submission setting is used: grid-search results
    and weights are loaded from file instead of being recomputed.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Provided the function. Clean-up: submission defaults (load instead of recompute),
    the interactive y/n questions replaced by the --grid-search and --train flags.
     
    """
    parser = argparse.ArgumentParser(description="Exercise e: MNIST classification with a feed-forward network. "
                                                 "By default, results and weights are loaded from file.")
    parser.add_argument("--grid-search", action="store_true",
                        help="redo the grid search (about a day on a laptop) instead of loading the results")
    parser.add_argument("--train", action="store_true",
                        help="retrain the final network instead of loading its weights")
    parser.add_argument("--quick-test", action="store_true",
                        help="only a small grid search on 6000 samples to check the set-up (minutes)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="epochs for retraining the final network (default: best from the grid search)")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="parallel grid-search workers (default: 4)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="random seed (default: 2026)")
    parser.add_argument("--results-file", default=os.path.join(RESULTS_DIR, "grid_search_results.npz"),
                        help="grid-search results (.npz)")
    parser.add_argument("--weight-file", default=os.path.join(BASE, "weights_exercise_e_grid_search.npz"),
                        help="weights of the final network (.npz)")
    return parser.parse_args()


def exercise_e(args = parse_args()):
    """
    Exercise e: grid search (or loading its results), final network (trained or loaded),
    evaluation on the MNIST test set and plots.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Nov 2026)
    Role: Clean-up of the author's restructuring: the steps of exercise e are called here in
    order, each implemented in its own function; quick-test mode.
     
    """
    x_train, y_train, y_train_onehot, x_test, y_test = load_mnist()

    # quick check of the whole grid-search pipeline on a small subset; real results stay untouched
    if args.quick_test:
        subset = np.random.default_rng(args.seed).choice(len(x_train), size=6000, replace=False)
        quick_file = os.path.join(RESULTS_DIR, "quick_test", "grid_search_quick_test.npz")
        results = grid_search_hyperparameters(x_train[subset], y_train_onehot[subset],
                                              layers=[(50, 10), (100, 100, 10)], epochs=[1, 2], lams=[0, 0.01],
                                              kfolds=2, n_jobs=args.n_jobs, seed=args.seed,
                                              results_filename=quick_file)
        plot_grid_search_results(results, quick_file.replace(".npz", ""))
        return

    # 1. hyperparameters: grid search or results from file
    if args.grid_search:
        results = grid_search_hyperparameters(x_train, y_train_onehot, n_jobs=args.n_jobs, seed=args.seed,
                                              results_filename=args.results_file)
    else:
        results = load_grid_search_results(args.results_file)
    print(f"Best CV accuracy {results['best_score']:.4f}: hidden layers {results['best_layers'][:-1]}, "
          f"gamma = {results['best_gamma']:g}, epochs = {results['best_epochs']}, lambda = {results['best_lam']:g}")
    plot_grid_search_results(results, args.results_file.replace(".npz", ""))

    # 2. final network with the best hyperparameters: trained or loaded
    classifier, epochs = multiclassifier(x_train, y_train_onehot,
                                         layer_output_sizes=results["best_layers"],
                                         gamma=results["best_gamma"], lam=results["best_lam"],
                                         epochs=args.epochs or results["best_epochs"],
                                         weight_file=args.weight_file, train=args.train, seed=args.seed)
    print(f"Training loss (cross entropy): {classifier.cost(x_train, y_train_onehot):.4f}")

    # 3. evaluation on the test set
    y_pred = np.argmax(classifier.predict(x_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy (MNIST, {epochs} epochs): {accuracy:.4f}")

    hidden = "x".join(str(n) for n in results["best_layers"][:-1])

    confusion_matrix_own_implementation = plot_confusion_matrix(y_test, y_pred,
                          title=f"Multiclass Classification Confusion Matrix (MNIST) - Epochs: {epochs}",
                          filename=os.path.join(RESULTS_DIR, f"confusion_matrix_{epochs}_epochs_hidden_{hidden}.pdf"))

    latex_table_accuracy_precision_recall_f1(confusion_matrix_own_implementation, os.path.join(RESULTS_DIR, f"latex_table_{epochs}_epochs_hidden_{hidden}.tex"))

    # 4. verify that scikit learn NN gives similar accuracy and confusion matrix (for the same architecture, learning rate, epochs and L2)
    # LLM-assisted (Claude Opus 5.5, Nov 2026): alpha converted to the lambda convention of lecture
    # (sklearn: grad += alpha*W/B, NeuralNet: grad += 2*lam*W  =>  alpha = 2*lam*B); same batch size;
    # no early stop, so that both train for the same number of epochs.
    batch_size = 16                      # same mini-batch size as NeuralNet
    sk_MLP = MLPClassifier(hidden_layer_sizes=results["best_layers"][:-1],
                        activation="relu", solver="adam",
                        alpha=2 * results["best_lam"] * batch_size,
                        batch_size=batch_size,
                        learning_rate_init=results["best_gamma"],
                        max_iter=epochs, n_iter_no_change=epochs,
                        shuffle=False,
                        random_state=args.seed)

    # AI-assisted (Claude): integer labels -> softmax + cross-entropy, like NeuralNet
    sk_MLP.fit(x_train, np.argmax(y_train_onehot, axis=1))
    sk_y_pred = sk_MLP.predict(x_test)
    sk_accuracy_score = accuracy_score(y_test, sk_y_pred)
    print(f"scikit-learn's MLPClassifier test accuracy (MNIST, {epochs} epochs): {sk_accuracy_score:.4f}")

    confusion_matrix_sklearn = plot_confusion_matrix(y_test, sk_y_pred,
                          title=f"scikit-learn's MLPClassifier Confusion Matrix (MNIST) - Epochs: {epochs}",
                          filename=os.path.join(RESULTS_DIR, f"confusion_matrix_sklearn_{epochs}_epochs_hidden_{hidden}.pdf"))

    print(f"scikit-learn's MLPClassifier confusion matrix:\n{confusion_matrix_sklearn}")

    latex_table_accuracy_precision_recall_f1(confusion_matrix_sklearn, os.path.join(RESULTS_DIR, f"latex_table_sklearn_{epochs}_epochs_hidden_{hidden}.tex"))

    # 5. compare to logistic regression
    model = LogisticRegression(solver="saga", max_iter=1000, C=np.inf, random_state=args.seed)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test) 
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    confusion_matrix_logistic = plot_confusion_matrix(y_test, y_pred,
                          title=f"Logistic Regression Confusion Matrix (MNIST)",
                          filename=os.path.join(RESULTS_DIR, f"confusion_matrix_logistic_regression.pdf"))
    print(f"Logistic Regression test accuracy (MNIST): {accuracy:.4f}")
    print(f"Logistic Regression confusion matrix:\n{confusion_matrix_logistic}")

    latex_table_accuracy_precision_recall_f1(confusion_matrix_logistic, os.path.join(RESULTS_DIR, f"latex_table_logistic_regression_{epochs}_epochs_hidden_{hidden}.tex"))


if __name__ == "__main__":
    exercise_e()
