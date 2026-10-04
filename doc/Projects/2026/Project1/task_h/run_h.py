from configuration import (
    COMMON,
    CROSS_VALIDATION,
    MINIBATCH,
    RUN_CV,
    RUN_MINIBATCH,
)

from minibatch_comparison import run_minibatch_experiment
from excesscost_cv_validation import run_cv_experiment

SAVE_FIGURES = True 

if __name__ == "__main__":
    if RUN_MINIBATCH:
        run_minibatch_experiment()

    if RUN_CV:
        run_cv_experiment()