# Project 2 Description

This repository contains the code for Project 2 in the course FYS-STK3155/FYS4155 at the University of Oslo. The project is developed by:

- Johannes Schäfers
- Ferdinand Nilius
- Benedikt Schneider
- Steffen Pablo Krieg

In this project we numerically implement a handbuilt FFNN and compare its performance on a regression example on the Runge function and on a classification example on the MNIST dataset. 
The points of comparison is either OLS, Ridge or Lasso regression for the regression or Logistic Regression for the classification example.

## Requirements

The required packages are listed in requirements.txt.
Install them using e.g.:

```
pip install -r ./doc\Projects\2026\Project2\requirements.txt
```
Note that you might have to paste a different path to the requirements.txt file instead of just the relative path as listed above.

## Repository structure
The repository is organised into subdirectories and code files as follows:

```
├── Project2/
│   ├── results/
│   │   ├── exercise_b_c_d/
│   │   ├── exercise_e/
│   ├── Activation_Cost_funcs.py
│   ├── exercise_e.py
│   ├── main.py
│   ├── NeuralNet.py
│   ├── NeuralNet_nobiaspen.py
│   ├── Optimizers.py
│   ├── Project2.ipynb
│   ├── Project2_nobiaspen.ipynb
└────── README.md
```

- `results/`: Contains some of the Figures and results produced for the report split into the two subdirectories. For the rest of the figures see `Project2.ipynb`.
- `Activation_Cost_funcs.py`: All used activation and cost functions, with 
- `main.py`: Task b) (Testing of different Optimizers), comparison with OLS, creation of Ridge and Lasso models. This file will create part of the figures and a Project2_results.txt file, which will contain selected results of the run.
- `NeuralNet.py`: The Neural Network class as a whole
- `Optimizers.py`: The Code for all used Optimizers
- `Project2.ipynb`: Results and code for task c) and d)
- `exercise_e.py`: Code for the classification tasks
- `Project2_nobiaspen.ipynb`: Contains the results for the same investigations as in `Project2.ipynb` with the fixed network. This is not contained in the report to stay consistent in throughout the report, as the code for task e) is too computationally expensive to re run. 

To re-generate all results of the Project the files `main.py` and `Project2.ipynb` (all cells, which contain code not provided from the start) need to be executed.

# Results

For the produced Figures please see the Figures directory.
Selected results are saved in the file `Project2_results.txt`(This is regenerated upon a rerun of main.py).
For results from task c and d please view the file `Project2.ipynb`.
