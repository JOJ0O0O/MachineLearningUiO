import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Generate sparse, high-dimensional data (p > n)
# 100 samples, 200 features, but only 5 features actually matter.
np.random.seed(42)
n_samples, n_features = 100, 200
X = np.random.randn(n_samples, n_features)

true_coef = np.zeros(n_features)
true_coef[:5] = [10.0, -5.0, 8.0, -3.5, 2.0]  # The "True Signal"

# Target variable with small Gaussian noise
y = X @ true_coef + np.random.randn(n_samples)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Train Ridge and Lasso using Cross-Validation
alphas = np.logspace(-4, 4, 100)
ridge = RidgeCV(alphas=alphas).fit(X_train, y_train)
lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X_train, y_train)

# 3. Evaluate models
ridge_y_pred = ridge.predict(X_test)
lasso_y_pred = lasso.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_y_pred)
lasso_mse = mean_squared_error(y_test, lasso_y_pred)

ridge_nonzeros = np.sum(np.abs(ridge.coef_) > 1e-7)
lasso_nonzeros = np.sum(np.abs(lasso.coef_) > 1e-7)

# 4. Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi = 300)

# Left plot: Feature Coefficient Comparison
axes[0].stem(
    range(n_features),
    ridge.coef_,
    linefmt="r-",
    markerfmt="ro",
    basefmt="k-",
    label="Ridge Coefficients",
)
axes[0].stem(
    range(n_features),
    lasso.coef_,
    linefmt="y-",
    markerfmt="yo",
    basefmt="k-",
    label="Lasso Coefficients",
)
axes[0].stem(
    range(n_features),
    true_coef,
    linefmt="g--",
    markerfmt="gx",
    basefmt="k-",
    label="True Coefficients",
)
axes[0].set_title("Coefficient Estimates per Feature")
axes[0].set_xlabel("Feature Index")
axes[0].set_ylabel("Coefficient Weight")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Right plot: Predictions vs Actual Test Values
axes[1].scatter(y_test, ridge_y_pred, color="red", alpha=0.6, label="Ridge")
axes[1].scatter(y_test, lasso_y_pred, color="orange", alpha=0.8, label="Lasso")
axes[1].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--",
    lw=2,
    label="Perfect Prediction",
)
axes[1].set_title("Predicted vs. Actual Targets (Test Set)")
axes[1].set_xlabel("Actual y")
axes[1].set_ylabel("Predicted y")
axes[1].legend(loc="upper left")
axes[1].grid(True, linestyle="--", alpha=0.5)

# --- ADD MSE RESULTS BOX TO THE PLOT ---
results_text = (
    f"Ridge MSE: {ridge_mse:.2f}\n"
    f"  Active Features: {ridge_nonzeros}/{n_features}\n\n"
    f"Lasso MSE: {lasso_mse:.2f}\n"
    f"  Active Features: {lasso_nonzeros}/{n_features}"
)

# Render formatted textbox in lower-right corner of subplot 2
axes[1].text(
    0.95,
    0.05,
    results_text,
    transform=axes[1].transAxes,
    fontsize=10,
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
)

plt.tight_layout()
plt.savefig('Lasso_better_than_ridge.png')
plt.show()