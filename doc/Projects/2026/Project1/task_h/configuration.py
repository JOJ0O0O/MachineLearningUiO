import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#using same definition as in week 38 notebook
BLUE, RED, GREEN, YELLOW, GREY = "#004488", "#BB5566", "#228833", "#DDAA33", "#777777"
np.set_printoptions(precision=4, suppress=True)

RUN_MINIBATCH = True
RUN_CV = True

COMMON = {
    "seed": 2026,
    "degree": 10,
    "noise": 0.1,
}

MINIBATCH = {
    "batch_sizes": [1, 5, 10, 25, 50, 100],
    "n_epochs": 1000,
    "t0": 10.0,
    "t1": 100.0,
}

CROSS_VALIDATION = {
    "batch_sizes": [1, 5, 10, 25, 50, 80],
    "n_epochs_list": [100,200,500,1000,2000],
    "n_splits": 5,
    "t0": 10.0,
    "t1": 100.0,
}