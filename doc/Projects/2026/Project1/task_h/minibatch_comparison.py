import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import shared utilities and configs based on the project structure
from model_definition_data_generation import runge_data, closed_form, cost, sgd, optimise, gradient
from configuration import BLUE, RED, GREEN, YELLOW, GREY, COMMON, MINIBATCH

batch_sizes = MINIBATCH['batch_sizes']
n_epochs = MINIBATCH['n_epochs']
t0 = MINIBATCH['t0']
t1 = MINIBATCH['t1']

#defining differnt schedule types and storing them in a dict
schedules = {
    "Constant (gamma=0.01)": {"gamma": 0.01, "schedule": None},
    "Slow Decay (t0=1.0, t1=1000)": {"gamma": 0.1, "schedule": (1.0, 1000.0)},
    "Fast Decay (t0=10.0, t1=100)": {"gamma": 0.1, "schedule": (10.0, 100.0)},
}

def run_minibatch_experiment(save_figures = True):
    #generating the data not using the returnes x (_)
    x10_full, X10_full, y10_full = runge_data(degree=COMMON['degree'], noise=COMMON['noise'], seed=COMMON['seed'])

    #splitting the dat (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X10_full, y10_full, test_size=0.2, random_state=COMMON['seed']
    )
    n_train = len(y_train)

    #computing closed OLS form
    theta_ols_train = closed_form(X_train, y_train)
    c_ols_train = cost(theta_ols_train, X_train, y_train)
    c_ols_test = cost(theta_ols_train, X_test, y_test)  # Analytical baseline on Test Set

    #full gradient numerical baseline
    full_path = optimise(
        lambda theta: gradient(theta, X_train, y_train),
        np.zeros(COMMON['degree']),
        "adam",
        gamma=0.05,
        num_iters=n_epochs,
    )
    full_excess_train = np.array([cost(theta, X_train, y_train) - c_ols_train for theta in full_path])
    full_test_cost = np.array([cost(theta, X_test, y_test) for theta in full_path])
    full_evaluations = np.arange(len(full_excess_train)) * n_train

    #doing some experiments with stochastic gradient descent
    results = {}
    for b_size in batch_sizes:
        history = sgd(
            X_train, y_train,
            method="adam",
            n_epochs=n_epochs,
            batch_size=b_size,
            schedule=(t0, t1),
        )
        excess_train = np.array([cost(theta, X_train, y_train) - c_ols_train for theta in history])
        test_cost = np.array([cost(theta, X_test, y_test) for theta in history])
        evaluations = np.arange(len(excess_train)) * n_train
        results[b_size] = (evaluations, excess_train, test_cost, history[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    for b_size in batch_sizes:
        evaluations, excess_train, test_cost, _ = results[b_size]
        ax1.semilogy(
            evaluations, np.maximum(excess_train, 1e-16),
            linewidth=1.5, label=f"Minibatch (M={b_size})"
        )
        ax2.plot(evaluations, test_cost, linewidth=1.5, label=f"Minibatch (M={b_size})")

    ax1.axhline(0.0, color=GREY, linestyle="--")
    ax1.set_xlabel("Individual Gradient Evaluations")
    ax1.set_ylabel("Train Excess Cost")
    ax1.set_title("Training Cost by Batch Size")
    ax1.grid(True, which="both", alpha=0.3, linestyle="--")
    ax1.legend()

    ax2.axhline(c_ols_test, color=RED, linestyle=":", label="OLS Test MSE")
    ax2.set_xlabel("Individual Gradient Evaluations")
    ax2.set_ylabel("Test MSE")
    ax2.set_title("Test Cost by Batch Size")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend()

    fig.tight_layout()
    if save_figures:
        plt.savefig(f"BatchSizeComparison_M{'-'.join(map(str, batch_sizes))}_{n_epochs}epochs.png",
        bbox_inches="tight")

    for batch in batch_sizes:
        # resetting the dict for each batch dsize
        schedule_results = {}
        
        for name, params in schedules.items():
            #use x and y train instead of X10 and y10
            history = sgd(
                X_train, y_train,
                method="adam",
                n_epochs=n_epochs,
                batch_size=batch,
                gamma=params["gamma"],
                schedule=params["schedule"],
            )
            #calculate cost against training data and training OLS baseline
            excess_train = np.array([cost(theta, X_train, y_train) - c_ols_train for theta in history])
            
            #multiply by n_train instead of n10 (make it coherent with x and y data)
            evaluations = np.arange(len(excess_train)) * n_train
            schedule_results[name] = (evaluations, excess_train)

        #plotting the current batch size
        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        for name, (evals, excess_train) in schedule_results.items():
            ax.semilogy(evals, np.maximum(excess_train, 1e-16), linewidth=2, label=name)

        #include full gradient baseline for better comparison
        ax.semilogy(full_evaluations, np.maximum(full_excess_train, 1e-16), color=GREY, linestyle="--", label="Full-Gradient Adam")

        ax.set_xlabel("Individual Gradient Evaluations", fontsize=11, fontweight='bold')
        ax.set_ylabel(r"Train Excess Cost: $C_{train}(\theta_k) - C_{train}(\hat{\theta}_{OLS})$", fontsize=11, fontweight='bold')
        ax.set_title(f"Schedule Comparison on Train Data (M = {batch}, {n_epochs} epochs)", fontsize=13, pad=12)
        ax.grid(True, which="both", alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        fig.tight_layout()
        
        #save the fig with the match size in the file name, one run stores every batch size for the current epoch, feel free to change the epoch above 
        if save_figures:
            plt.savefig(f'new_Minibatch_{batch}_Comparing_Learning_Rates_Train_{n_epochs}epochs.png')

    # polynomial fits plot compared to the Runge function
    x_grid = np.linspace(-1.0, 1.0, 500)
    raw_features = np.column_stack([x10_full**k for k in range(1, COMMON['degree'] + 1)])
    grid_features = np.column_stack([x_grid**k for k in range(1, COMMON['degree'] + 1)])
    grid_features = (grid_features - raw_features.mean(axis=0)) / raw_features.std(axis=0)

    rng = np.random.default_rng(COMMON['seed'])
    rng.uniform(-1.0, 1.0, len(x10_full))
    y_offset = np.mean(
        1.0 / (1.0 + 25.0 * x10_full**2)
        + COMMON['noise'] * rng.standard_normal(len(x10_full))
    )

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.scatter(x10_full, y10_full + y_offset, s=18, alpha=0.5, label="Noisy data")
    ax.plot(
        x_grid, 1.0 / (1.0 + 25.0 * x_grid**2),
        color="black", linewidth=2, label="Runge function"
    )
    ax.plot(
        x_grid, grid_features @ theta_ols_train + y_offset,
        color=YELLOW, linestyle = '--', linewidth=2, label="OLS fit"
    )

    for b_size in batch_sizes:
        theta = results[b_size][3]
        ax.plot(
            x_grid, grid_features @ theta + y_offset,
            linewidth=1.2, label=f"Minibatch fit (M={b_size})"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("new_Polynomial Fits Compared with the Runge Function")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()

    if save_figures: 
        plt.savefig('Polynomialfits_vs_Rungefunction.png')
    plt.show()

if __name__ == "__main__":
    run_minibatch_experiment()