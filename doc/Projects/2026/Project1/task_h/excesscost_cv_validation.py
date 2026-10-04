import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import KFold

#import shared utilities and configs based on the project structure
from model_definition_data_generation import runge_data, closed_form, cost, sgd, optimise, gradient
from configuration import RED, GREY, COMMON, CROSS_VALIDATION

batch_sizes = CROSS_VALIDATION['batch_sizes']
n_epochs_list = CROSS_VALIDATION['n_epochs_list']
t0 = CROSS_VALIDATION['t0']
t1 = CROSS_VALIDATION['t1']
n_splits = CROSS_VALIDATION['n_splits']

def run_cv_experiment(save_figures = True):
    for n_epochs in n_epochs_list:
        #generating the data not using the returnes x (_)
        x10_full, X10_full, y10_full = runge_data(degree=COMMON['degree'], noise=COMMON['noise'], seed=COMMON['seed'])

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=COMMON['seed'])

        #data Structures for accumulating fold results, creating empty lists to store it later
        cv_ols_val_costs = []
        cv_full_train_curves = []
        cv_full_val_curves = []
        cv_sgd_train_curves = {b: [] for b in batch_sizes}
        cv_sgd_val_curves = {b: [] for b in batch_sizes}

        n_train_cv = 0 #will dynamically update based on CV fold size, just stating zero as a dummy value

        for fold, (train_idx, val_idx) in enumerate(kf.split(X10_full)):
            X_tr, X_val = X10_full[train_idx], X10_full[val_idx]
            y_tr, y_val = y10_full[train_idx], y10_full[val_idx]
            n_train_cv = len(y_tr)
            
            #old reference
            theta_ols = closed_form(X_tr, y_tr)
            c_ols_tr = cost(theta_ols, X_tr, y_tr)
            cv_ols_val_costs.append(cost(theta_ols, X_val, y_val))
            
            #full grad adam baseline
            full_path = optimise(
                lambda theta: gradient(theta, X_tr, y_tr),
                np.zeros(COMMON['degree']), "adam", gamma=0.05, num_iters=n_epochs
            )
            cv_full_train_curves.append([cost(th, X_tr, y_tr) - c_ols_tr for th in full_path])
            cv_full_val_curves.append([cost(th, X_val, y_val) for th in full_path])
            
            # minibatch sgd
            for b_size in batch_sizes:
                history = sgd(
                    X_tr, y_tr, method="adam", n_epochs=n_epochs, 
                    batch_size=b_size, schedule=(t0, t1)
                )
                cv_sgd_train_curves[b_size].append([cost(th, X_tr, y_tr) - c_ols_tr for th in history])
                cv_sgd_val_curves[b_size].append([cost(th, X_val, y_val) for th in history])

        #cross validation results 
        mean_ols_val = np.mean(cv_ols_val_costs)
        std_ols_val = np.std(cv_ols_val_costs)

        mean_full_train = np.mean(cv_full_train_curves, axis=0)
        mean_full_val = np.mean(cv_full_val_curves, axis=0)
        evals_full = np.arange(len(mean_full_train)) * n_train_cv

        mean_sgd_train = {}
        mean_sgd_val = {}
        evals_sgd = {}

        for b in batch_sizes:
            mean_sgd_train[b] = np.mean(cv_sgd_train_curves[b], axis=0)
            mean_sgd_val[b] = np.mean(cv_sgd_val_curves[b], axis=0)
            evals_sgd[b] = np.arange(len(mean_sgd_train[b])) * n_train_cv

        #plotting two figures in one plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=300)
        colors = cm.viridis(np.linspace(0, 0.9, len(batch_sizes)))

        #first subolot CV mean train excess Cost
        ax1.semilogy(
            evals_full, np.maximum(mean_full_train, 1e-16),
            color=GREY, linewidth=2.5, alpha=0.7, label=f"Full-Grad Adam (M={n_train_cv})"
        )

        for b_size, color in zip(batch_sizes, colors):
            ax1.semilogy(
                evals_sgd[b_size], np.maximum(mean_sgd_train[b_size], 1e-16),
                color=color, linewidth=1.5, alpha=0.85, label=f"Minibatch (M={b_size})"
            )

        ax1.set_xlim(0, evals_full[-1])
        ax1.set_xlabel("Individual Gradient Evaluations", fontsize=11, fontweight='bold')
        ax1.set_ylabel(r"CV Mean Train Excess Cost: $C_{train}(\theta_k) - C_{train}(\hat{\theta}_{OLS})$", fontsize=11, fontweight='bold')
        ax1.set_title(f"Training Convergence (CV Averaged, {n_epochs} Epochs)", fontsize=13, pad=10)
        ax1.grid(True, which="both", alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right', frameon=True, fontsize=9)

        #second subplot, CV mean validation generalization error
        ax2.plot(
            evals_full, mean_full_val,
            color=GREY, linewidth=2.5, alpha=0.7, label=f"Full-Grad Adam (M={n_train_cv})"
        )

        for b_size, color in zip(batch_sizes, colors):
            ax2.plot(
                evals_sgd[b_size], mean_sgd_val[b_size],
                color=color, linewidth=1.5, alpha=0.85, label=f"Minibatch (M={b_size})"
            )

        ax2.axhline(
            y=mean_ols_val, color=RED, linestyle=':', linewidth=2,
            label=f"Analytical OLS CV Limit ({mean_ols_val:.4f})"
        )

        ax2.set_xlim(0, evals_full[-1])
        ax2.set_ylim(0.02, 0.08) 
        ax2.set_xlabel("Individual Gradient Evaluations", fontsize=11, fontweight='bold')
        ax2.set_ylabel("CV Mean Validation MSE", fontsize=11, fontweight='bold')
        ax2.set_title(f"Generalization on Unseen Folds (CV Averaged, {n_epochs} Epochs)", fontsize=13, pad=10)
        ax2.grid(True, which="both", alpha=0.3, linestyle='--')
        ax2.legend(loc='best', frameon=True, fontsize=9)

        fig.tight_layout()
        if save_figures:
            plt.savefig(f"Train_vs_CV_Minibatch_SGD_{n_epochs}epochs.png")
        
        plt.show()

        #print summary table, formated by ai
        print(f"{'Method / Batch Size':<22} | {'Final CV Train Excess':<25} | {'Final CV Val MSE (Bias)':<25} | {'CV Val Std (Variance)':<20}")
        print("-" * 100)
        for b_size in batch_sizes:
            final_tr_exc = mean_sgd_train[b_size][-1]
            final_val_mse = mean_sgd_val[b_size][-1]
            final_val_std = np.std([fold_curve[-1] for fold_curve in cv_sgd_val_curves[b_size]])
            print(f"Minibatch (M={b_size:<4})       | {final_tr_exc:<25.3e} | {final_val_mse:<25.4f} | {final_val_std:<20.4f}")

        final_full_tr = mean_full_train[-1]
        final_full_val = mean_full_val[-1]
        final_full_val_std = np.std([fold_curve[-1] for fold_curve in cv_full_val_curves])

        print(f"Full-Grad Adam (M={n_train_cv:<2}) | {final_full_tr:<25.3e} | {final_full_val:<25.4f} | {final_full_val_std:<20.4f}")
        print(f"Analytical OLS         | {'0.000e+00 (Exact)':<25} | {mean_ols_val:<25.4f} | {std_ols_val:<20.4f}")


if __name__ == "__main__":
    run_cv_experiment()