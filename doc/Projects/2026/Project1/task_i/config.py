import numpy as np

CONFIG = {
    "n_samples": 150,
    "degree": list(range(1,31)),
    "lam": np.logspace(-8, 2, 50).tolist(),
    "lam_fix": 0.001,
    "test_size": 0.2,
    "random_state": 42,
    "lr": 0.01,
    "gamma": 0.01,
    "num_iters": 10000,
    "k_folds": 5,
    "save_csv": True,
    "save_img": True,
    "degrees_to_plot": [4,8, 15]
}

BLUE, RED, GREEN, YELLOW, GREY = "#004488", "#BB5566", "#228833", "#DDAA33", "#777777"
GRAY = GREY 

RUN_KFOLD = True
RUN_JAX = False
RUN_FIT = True

SAVE_FIGURE = True 