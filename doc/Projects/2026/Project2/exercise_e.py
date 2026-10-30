from re import match
import urllib.request
import os

import NeuralNet as NN

import numpy as np

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix as sk_confusion_matrix
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (KFold, StratifiedKFold, cross_val_score, cross_validate,
                                     cross_val_predict, GridSearchCV, validation_curve, learning_curve)



import Activation_Cost_funcs as act
import Optimizers as opt
import matplotlib.pyplot as plt

from tqdm import tqdm


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def save_weights(layers, epochs, filename):
    """
    Save the weights and biases of all layers to an .npz file.

    Parameters
    ----------
    layers : list of tuples (W, b), weights and biases of each layer
    filename : str, path of the .npz file

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Provided the function; stores each array under its own key (W0, b0, W1, ...)
    to avoid saving a ragged list with np.savez.
    Modifications: Integrated and tested by the author (save/load round trip).
    """
    arrays = {}
    for i, (W, b) in enumerate(layers):
        arrays[f"W{i}"] = np.asarray(W)
        arrays[f"b{i}"] = np.asarray(b)
    arrays["epochs"] = epochs
    np.savez(filename, **arrays)


def load_weights(filename):
    """
    Load the weights and biases saved by save_weights.

    Parameters
    ----------
    filename : str, path of the .npz file

    Returns
    -------
    list of tuples (W, b), weights and biases of each layer

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Provided the function; reconstructs the layer list from the W{i}/b{i} keys.
    Modifications: Integrated and tested by the author (save/load round trip).
    """
    with np.load(filename) as data:
        n_layers = len(data.files) // 2
        return [(data[f"W{i}"], data[f"b{i}"]) for i in range(n_layers)], data["epochs"]


def yes_no(question):
    while True:
        answer = input(question + " (y/n): ").strip().lower()
        if answer in ['y', 'n']:
            return answer == 'y'
        print("Invalid input. Please enter 'y' or 'n'.")
    
def choose_load_weights_from_file(epochs, filename):
    """
    Asks whether to load saved weights. Returns (weights, epochs stored in the file) on 'y'
    and (None, epochs) on 'n'.

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Identified that the retry after invalid input called this function with the wrong
    arguments and discarded its result; replaced the recursion by the author's yes_no loop.
    Modifications: Integrated and tested by the author.
    """
    if not yes_no(f"Do you want to load weights and biases from {os.path.basename(filename)}?"):
        return None, epochs
    print(f"\rLoading weights and biases from {os.path.basename(filename)}...", end="")
    weights, epochs = load_weights(filename)
    print(f"\rWeights and biases loaded from {os.path.basename(filename)}.")
    return weights, int(epochs)


def softmax_cross_entropy_error(predict, target):
    """
    Computes the error for the output layer when using softmax activation and cross-entropy loss.
    This is the special case where the derivative of the output layer simplifies to (a - y).

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Moved the author's nested output_error function to module level, so that the grid
    search can pass it to the network as well (and it can be sent to the parallel workers).
    Modifications: Integrated and tested by the author.
    """
    return predict - target

def choose_save_weights_to_file(weights, epochs, filename):
    choice = input(f"Do you want to save the trained weights and biases to {os.path.basename(filename)}? (y/n)").strip().lower()
    match choice:
        case 'y':
            print(f"\rSaving weights and biases to {os.path.basename(filename)}...", end="")
            save_weights(weights, epochs, filename)
            print(f"\rWeights and biases saved to {os.path.basename(filename)}.")
        case 'n':
            pass
        case _:
            print("Invalid choice. Please enter 'y' or 'n'.")
            choose_save_weights_to_file(weights, epochs, filename)  # Recursive call until valid input is received
    return


def grid_search_hyperparameters(
        NN,
        x,
        y,
        kfolds=5,
        gammas=[0.001, 0.01, 0.03, 0.1, 0.3],
        optimizers=[opt.MomentumOptimizer, opt.AdamOptimizer],
        layers=[(50,10),(50,50,10),(50,50,50,10),(100,10),(100,100,10),(100,100,100,10)],
        epochs=[10, 20, 50],
        activation_funcs=[act.ReLU, act.ReLU, act.softmax],
        cost_fun=act.cross_entropy,
        output_error=softmax_cross_entropy_error,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        seed=2026,
        save_results_to_file=True,
        results_filename="results/exercise_e/grid_search_results.npz"
):
    """
    Perform a grid search over the specified hyperparameters for the neural network.

    Parameters
    ----------
    NN : module containing the NeuralNetwork class
        The neural network module to be used for training.
    x : array-like, shape (n_samples, n_features)
        The input data for training.
    y : array-like, shape (n_samples, n_classes)
        One-hot encoded target values for training.
    kfolds : int, optional
        The number of folds for cross-validation. Default is 5.
    gammas : list of float, optional
        List of learning rates to be tested. Default is [0.001, 0.01, 0.03, 0.1, 0.3].
    optimizers : list of Optimizer classes, optional
        List of optimizer classes to be tested. Default is [MomentumOptimizer, AdamOptimizer].
    layers : list of tuples, optional
        List of layer configurations to be tested, be careful, the output layer size must match the number of classes in the target variable. Default is [(50,10),(50,50,10),(50,50,50,10),(100,10),(100,100,10),(100,100,100,10)].
    activation_funcs : list of callable activation functions, optional
        The first entry is used for all hidden layers, the last one for the output layer. Default is [ReLU, ReLU, softmax].
    cost_fun : callable, optional
        Cost function of the network. Default is cross_entropy.
    output_error : callable or None, optional
        Output-layer error passed to the network. Default is softmax_cross_entropy_error (a - y).
    scoring : str, optional
        The scoring metric to be used for evaluation. Default is "neg_mean_squared_error".
    n_jobs : int, optional
        The number of parallel jobs to be used for the grid search. Default is -1 (all available cores).
    seed : int, optional
        The random seed for reproducibility. Default is 2026.
    epochs : int, optional
        The number of epochs to train the network for. Default is 20.
    save_results_to_file : bool, optional
        Whether to save the grid search results to a file. Default is True.
    results_filename : str, optional
        The filename for saving the grid search results. Default is "results/exercise_e/grid_search_results.npz".
    Returns
    -------
    best_params : dict, best parameter combination (keys as in GridSearchCV.best_params_)

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Fixed the author's version: parameter names without the "neuralnetwork__" prefix
    (no pipeline any more), cross-entropy cost and output error passed to the network,
    activation list built for the deepest architecture, NaN-initialised result array,
    refit=False, results stored without pickled objects, figure saved next to the results
    file, best parameters returned.
    Modifications: Integrated and tested by the author.
    """

    # one activation per layer of the deepest network: hidden activation repeated, output activation last
    max_layers = max(len(layer) for layer in layers)
    activation_funcs = [activation_funcs[0]] * (max_layers - 1) + [activation_funcs[-1]]

    cv = KFold(n_splits=kfolds, shuffle=True, random_state=seed)
    param_grid = {
        "gamma": gammas,
        "optimizer": optimizers,
        "layer_output_sizes": layers,
        "epochs": epochs
        }

    net = NN.NeuralNetwork(network_input_size=x.shape[1],
                        regularization=None,
                        cost_fun=cost_fun,
                        output_error=output_error,
                        layer_output_sizes=layers[0],
                        activation_funcs=activation_funcs,
                        optimizer=optimizers[0],
                        epochs=epochs[0],
                        seed=seed)

    grid = GridSearchCV(net, param_grid, cv=cv, scoring=scoring, return_train_score=True, n_jobs=n_jobs, refit=False)
    grid.fit(x, y)
    results = grid.cv_results_

    mse = np.full((len(optimizers), len(layers), len(gammas), len(epochs)), np.nan)
    for params, score in zip(results["params"], results["mean_test_score"]):
        optimizer_index = optimizers.index(params["optimizer"])
        layer_index = layers.index(params["layer_output_sizes"])
        gamma_index = gammas.index(params["gamma"])
        epoch_index = epochs.index(params["epochs"])
        mse[optimizer_index, layer_index, gamma_index, epoch_index] = -score

    best_params = grid.best_params_
    best_score = grid.best_score_

    if save_results_to_file:
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        np.savez(results_filename,
                 mse=mse,
                 gammas=np.array(gammas),
                 optimizers=np.array([optimizer.__name__ for optimizer in optimizers]),
                 layers=np.array([str(layer) for layer in layers]),
                 epochs=np.array(epochs),
                 best_optimizer=best_params["optimizer"].__name__,
                 best_gamma=best_params["gamma"],
                 best_layers=np.array(best_params["layer_output_sizes"]),
                 best_epochs=best_params["epochs"],
                 best_score=best_score)

    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")

    mse_min, mse_max = np.nanmin(mse), np.nanmax(mse)
    for epoch_index, n_epochs in enumerate(epochs):
        fig, axes = plt.subplots(1, len(optimizers), figsize=(25, 7),
                                 constrained_layout=True, squeeze=False)
        axes = axes[0]
        for i, (optimizer, axis) in enumerate(zip(optimizers, axes)):
            im = axis.imshow(mse[i, :, :, epoch_index], cmap="viridis_r", origin="lower", aspect="equal",
                             vmin=mse_min, vmax=mse_max)
            for layer_index in range(len(layers)):
                for gamma_index in range(len(gammas)):
                    value = mse[i, layer_index, gamma_index, epoch_index]
                    text_color = "white" if im.norm(value) >= 0.5 else "black"
                    axis.text(gamma_index, layer_index, f"{value:.3g}",
                              ha="center", va="center", fontsize=7, color=text_color)
            axis.set_xticks(range(len(gammas)))
            axis.set_xticklabels([f"{gamma:g}" for gamma in gammas], fontsize=7)
            axis.set_yticks(range(len(layers)))
            axis.set_yticklabels([str(layer) for layer in layers], fontsize=7)
            axis.set_xlabel("$\\gamma$")
            axis.set_ylabel("Network architecture")
            axis.set_title(f"{optimizer.__name__}: CV MSE, {n_epochs} epochs", fontsize=9)
        fig.colorbar(im, ax=axes, label="MSE")

        if save_results_to_file:
            output_path = results_filename.replace(".npz", f"_{n_epochs}_epochs.pdf")
            fig.savefig(output_path)
            print(f"Saved figure to {output_path}")
        plt.close(fig)

    return best_params


def multiclassifier(
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        architecture : dict[str, list[int] | list[callable]] = {
            "layer_output_sizes": [64, 64, 10],  # 2 hidden layers with 64 neurons each, and an output layer with 10 neurons (for 10 classes of digits)
            "activations": [act.ReLU, act.ReLU, act.softmax],   # activation functions for each layer (ReLU for hidden layers, softmax for output layer)
        },
        seed = 2026,
        weight_file = "weights_exercise_e.npz",
        training_epochs = 1000,
        confusion_matrix_filename = "confusion_matrix_exercise_e_1000_epochs_2_hidden_layer_64x64.pdf"
):
    """
    Method executes exercise e (Multiclass classification using neural networks)

    Parameters
    ----------
    url : String, url link to the used data set; default: MNIST data set (handwritten numbers from 1 to 10)
    architecture : dict, defines the architecture of the used neural network to classify the data set
        - "layer_output_sizes": list of integers, defines the number of neurons in each layer (excluding input layer)
        - "activations": list of callable activation functions, defines the activation function for each layer (excluding input layer)
    seed : int, random seed for reproducibility; default: 2026
    weight_file : String, name of the .npz file to save/load the weights and biases of the trained model; default: "weights_exercise_e.npz"
    training_epochs : int, number of epochs to train the model; default: 1000
    confusion_matrix_filename : String, name of the file to save the confusion matrix plot; default: "confusion_matrix_exercise_e_1000_epochs_2_hidden_layer_64x64.pdf"

    Returns
    -------
    None

    LLM-assisted
    ------------
    Tool: Claude (Claude Opus 5.5, claude.ai, Oct 2026)
    Role: Fixed the model set-up: the classifier was undefined after the grid search; it is now
    built after the optional grid search from its best parameters, the output error uses the
    module-level softmax_cross_entropy_error, and loaded weights are checked against the
    classifier's own layer sizes.
    Modifications: Integrated and tested by the author.
    """

    # LLM (Claude Opus 5.5, Oct 2026): suggested resolving the path relative to the script's
    # own folder so that it does not depend on the working directory; verified by the author.
    base = os.path.dirname(os.path.abspath(__file__))
    weight_file_path = os.path.join(base, weight_file)

    # verify that the mnist dataset exist and download otherwise
    if not os.path.exists("mnist.npz"):
        urllib.request.urlretrieve(url, "mnist.npz")

    #extract data and scale pixel values to [0,1] range
    with np.load("mnist.npz") as data:
        x_train = data["x_train"].reshape(len(data["x_train"]), -1) / 255.0
        y_train = data["y_train"]
        x_test = data["x_test"].reshape(len(data["x_test"]), -1) / 255.0
        y_test = data["y_test"]

    # LLM (Claude Opus 5.5, Oct 2026): identified a shape mismatch between the softmax output (N, 10)
    # and the integer labels (N,) in output_error. One-hot encoding implemented by the author;
    # shapes verified via print statements before training.
    n_classes = architecture["layer_output_sizes"][-1]
    y_train_onehot = np.eye(n_classes)[y_train]
    y_test_onehot = np.eye(n_classes)[y_test]

    """print("x_train shape:", x_train.shape)
    print("y_train_onehot shape:", y_train_onehot.shape)
    print("x_test shape:", x_test.shape)
    print("y_test_onehot shape:", y_test_onehot.shape)"""

    output_error = softmax_cross_entropy_error if architecture["activations"][-1] == act.softmax else None

    # default hyperparameters; replaced by the grid-search optimum if a grid search is run
    layer_output_sizes = list(architecture["layer_output_sizes"])
    activation_funcs = architecture["activations"]
    optimizer_class, gamma = opt.AdamOptimizer, 0.001

    if yes_no(f"Do you intend to grid search over hyperparameters for the architecture {architecture}? (Warning: This may take a long time)"):
        best_params = grid_search_hyperparameters(
            NN=NN,
            x=x_train,
            y=y_train_onehot,
            kfolds=5,
            gammas=[0.001, 0.01, 0.03, 0.1, 0.3],
            optimizers=[opt.MomentumOptimizer, opt.AdamOptimizer],
            layers=[(50,10),(50,50,10),(50,50,50,10),(100,10),(100,100,10),(100,100,100,10)],
            epochs=[10, 20, 50, 100],
            activation_funcs=activation_funcs,
            output_error=output_error,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            seed=seed,
            save_results_to_file=True,
            results_filename=os.path.join(base, "results", "exercise_e", "grid_search_results.npz")
        )
        optimizer_class = best_params["optimizer"]
        gamma = best_params["gamma"]
        layer_output_sizes = list(best_params["layer_output_sizes"])
        activation_funcs = [activation_funcs[0]] * (len(layer_output_sizes) - 1) + [activation_funcs[-1]]

    mc_classifier = NN.NeuralNetwork(
        network_input_size=x_train.shape[1],
        layer_output_sizes=layer_output_sizes,
        activation_funcs=activation_funcs,
        cost_fun=act.cross_entropy if activation_funcs[-1] == act.softmax else act.mse,
        optimizer=optimizer_class,
        gamma=gamma,
        output_error=output_error,
        regularization=None,
        lam=0.0,
        seed=seed
    )

    if os.path.exists(weight_file_path):
        weights, training_epochs = choose_load_weights_from_file(training_epochs, weight_file_path)

        if weights is not None:
            if layer_output_sizes != [w.shape[1] for w, _ in weights]:
                raise Exception("The loaded weights do not match the specified architecture. Please check the architecture or choose not to load weights.")
                
            mc_classifier.set_weights(weights)
        else:
            print("No pre-trained weights loaded. Training the model from scratch...")
            mc_classifier.train(x_train, y_train_onehot, epochs=training_epochs)
            choose_save_weights_to_file(mc_classifier.get_weights(), training_epochs, weight_file_path)

    else:
        print("No pre-trained weights found. Training the model from scratch...")
        mc_classifier.train(x_train, y_train_onehot, epochs=training_epochs)
        choose_save_weights_to_file(mc_classifier.get_weights(), training_epochs, weight_file_path)

    training_loss = mc_classifier.cost(x_train, y_train_onehot)
    print(f"Training Loss (MNIST) - Epochs: {training_epochs}: {training_loss:.4f}")

    # plot the multiclass confusion matrix for each run
    y_pred_multi = np.argmax(mc_classifier.predict(x_test), axis=1)

    cm_multi = sk_confusion_matrix(y_test, y_pred_multi)
    print(cm_multi)
    accuracy_multi = accuracy(y_test, y_pred_multi)
    print(f"Multiclass Classification Accuracy (MNIST) - Epochs: {training_epochs}: {accuracy_multi:.4f}")
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 3.6))

    ConfusionMatrixDisplay(cm_multi, display_labels=[i for i in range(10)]).plot(ax=ax, colorbar=False)

    print(f"Multiclass Classification Confusion Matrix (MNIST) - Epochs: {training_epochs}")

    fig.suptitle(f"Multiclass Classification Confusion Matrix (MNIST) - Epochs: {training_epochs}")
    fig.savefig(os.path.join(base, confusion_matrix_filename))
    plt.tight_layout(); plt.show()

    return 


def exercise_e():
    multiclassifier(weight_file = "weights_exercise_e_grid_search.npz",)

    """multiclassifier(
        architecture = {
            "layer_output_sizes": [32, 32, 10],  # 2 hidden layers with 32 neurons each, and an output layer with 10 neurons (for 10 classes of digits)
            "activations": [act.ReLU, act.ReLU, act.softmax],   # activation functions for each layer (ReLU for hidden layers, softmax for output layer)
        },
        seed = 2026,
        weight_file = "weights_exercise_e_2000_epochs_2_hidden_layer_32x32.npz",
        training_epochs = 2000,
        confusion_matrix_filename = "confusion_matrix_exercise_e_2000_epochs_2_hidden_layer_32x32.pdf"
    )

    multiclassifier(
        architecture = {
            "layer_output_sizes": [64, 64, 64, 10],  # 3 hidden layers with 64 neurons each, and an output layer with 10 neurons (for 10 classes of digits)
            "activations": [act.ReLU, act.ReLU, act.ReLU, act.softmax],   # activation functions for each layer (ReLU for hidden layers, softmax for output layer)
        },
        seed = 2026,
        weight_file = "weights_exercise_e_1000_epochs_3_hidden_layer_64x64x64.npz",
        training_epochs = 1000,
        confusion_matrix_filename = "confusion_matrix_exercise_e_1000_epochs_3_hidden_layer_64x64x64.pdf"
    )

    multiclassifier(
        architecture = {
            "layer_output_sizes": [10],  # no hidden layer, linear regression
            "activations": [act.softmax],   # activation functions for output layer
        },
        seed = 2026,
        weight_file = "weights_exercise_e_linear_regression.npz",
        training_epochs = 1000,
        confusion_matrix_filename = "confusion_matrix_exercise_e_linear_regression.pdf"
    )"""

    return


if __name__ == "__main__":
    exercise_e()

