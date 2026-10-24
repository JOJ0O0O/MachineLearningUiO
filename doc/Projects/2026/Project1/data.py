import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from config import CONFIG

n_samples = CONFIG["n_samples"]
degree = CONFIG["degree"]
lam = CONFIG["lam"]
test_size = CONFIG["test_size"]
random_starter = CONFIG["random_state"]
lr = CONFIG["lr"]
num_iters = CONFIG["num_iters"]

np.set_printoptions(precision=4, suppress=True)


#generating the data
def runge_data(n=100, degree=10, noise=0.1, seed=2026):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n) # setting the range for x 
    y = 1.0 / (1.0 + 25.0 * x**2) + noise * rng.standard_normal(n) #defining the function with noise 
    
    # Store raw x for plotting functions
    X = np.column_stack([x**k for k in range(1, degree + 1)])
    
    # Standardise design matrix and centre target (y) 
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    y_centered = y - y.mean()
    return x, X_norm, y_centered

#optimizer from e and f
def optimiser_step(method, theta, g, state, t, gamma, beta1=0.9, beta2=0.999, eps=1e-8):
    """One update of theta from the gradient g at step t."""
    if method == "adam":
        m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g       
        r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g   
        state["m"], state["r"] = m, r
        m_hat = m / (1.0 - beta1**t)                              
        r_hat = r / (1.0 - beta2**t)
        return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state   
    raise ValueError(f"unknown method {method}")

def optimise(grad, theta0, method, gamma, num_iters=1000, tol=1e-6):
    """Run optimiser from theta0 with the full gradient."""
    theta, state = np.array(theta0, dtype=float), {}
    history = [theta.copy()]
    for t in range(1, num_iters + 1):
        g = grad(theta)
        theta, state = optimiser_step(method, theta, g, state, t, gamma)
        history.append(theta.copy())
        if np.linalg.norm(g) < tol:
            break
    return np.array(history)

# lasso subgradient
def lasso_subgradient(theta, X, y, lam):
    """
    Computes the subgradient for Lasso: (2/n) X^T (X*theta - y) + lambda * sign(theta)
    """
    n = len(y)
    grad_mse = (2.0 / n) * X.T @ (X @ theta - y)
    grad_l1 = lam * np.sign(theta)
    return grad_mse + grad_l1

#using the cost function from the tuesday week 38
def cost_mse(theta, X, y):
    return np.mean((X @ theta - y) ** 2)

#using the optimizer from e and f 
def grad_ols(theta, X, y):
    return (2.0 / len(y)) * X.T @ (X @ theta - y)

def grad_ridge(theta, X, y, lam):
    return (2.0 / len(y)) * X.T @ (X @ theta - y) + 2.0 * lam * theta

def grad_lasso_subgradient(theta, X, y, lam):
    return (2.0 / len(y)) * X.T @ (X @ theta - y) + lam * np.sign(theta)

def optimiser_step_adam(theta, g, state, t, gamma=0.01, beta1=0.9, beta2=0.999, eps=1e-8): #gamma and beta arent fix yet, we will use a different
    #gamma later 
    m = beta1 * state.get("m", 0.0) + (1.0 - beta1) * g       
    r = beta2 * state.get("r", 0.0) + (1.0 - beta2) * g * g   
    state["m"], state["r"] = m, r
    m_hat = m / (1.0 - beta1**t)                              
    r_hat = r / (1.0 - beta2**t)
    return theta - gamma * m_hat / (np.sqrt(r_hat) + eps), state

def optimise_adam(grad, theta0, gamma=CONFIG.get("gamma", 0.01), num_iters=CONFIG.get("num_iters", 10000)):
    theta, state = np.array(theta0, dtype=float), {}
    for t in range(1, num_iters + 1):
        g = grad(theta)
        theta, state = optimiser_step_adam(theta, g, state, t, gamma=gamma)
    return theta


