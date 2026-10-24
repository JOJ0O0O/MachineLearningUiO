import NeuralNet as NN
import numpy as np
import sklearn
import Activation_Cost_funcs as act
import Optimizers as opt
import matplotlib.pyplot as plt

"""Set up data from Project 1"""

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def design_matrix(x, degree, intercept=True):
    # polynomial features [1, x, x^2, ..., x^degree] (drop the 1 if intercept=False)
    start = 0 if intercept else 1
    return np.vstack([x**p for p in range(start, degree + 1)]).T

def Scaledata (Xtrain, Xtest, ytrain, ytest):
    """
    Scales and centers the data using the standard deviation and the mean values of the train set
    Applies the same transformations to the test set as well (based on train values)

        Parameters 
    ----------------
    XTrain: np.array shape: (n,p)
        Training data input
    Xtest: np.array shape: (n,p)
        Test data
        
    ytrain: np.array shape: (n,1)
        Vector containing the data to be fitted
    ytest: np.array shape: (n,1)
        Vector containing the test data

    Retrun
    ---------------
    XTrain_norm: np.array shape: (n,p)
        Normed training data
    XTest_norm: np.array shape: (n,p)
        Normed test data 
        
    ytrain_centered: np.array shape: (n,1)
        Vector containing the centered train data
    ytest_centered: np.array shape: (n,1)
        Vector containing the centered test data
    """
    X_mean, X_std = Xtrain.mean(axis=0), Xtrain.std(axis=0)
    XTrain_norm = (Xtrain - X_mean) / X_std
    XTest_norm = (Xtest - X_mean) / X_std
    ytrain_centered = ytrain - ytrain.mean()
    ytest_centered = ytest - ytrain.mean()
    return XTrain_norm, XTest_norm, ytrain_centered, ytest_centered

rng = np.random.default_rng(2026)
n = 100
sigma = 0.1                                  # noise level: explore it!
x = np.sort(rng.uniform(-1, 1, n))
y = runge(x) + rng.normal(0, sigma, n)

xtrain, xTest, ytrain, ytest = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=2026)

xtrain, xTest, ytrain, ytest = Scaledata(xtrain, xTest, ytrain, ytest)
xx = np.linspace(-1,1,1000)
"""
AI-Declaration:
I used Chat-GPT6 Luna in VSCode in order to find an error as i did not know that I needed to reshape my data.
The four lines below are therefore AI generated.
This is equal to usage level 2.
Johannes Schäfers
"""

xtrain = xtrain.reshape(-1, 1)
xTest = xTest.reshape(-1, 1)
ytrain = ytrain.reshape(-1, 1)
ytest = ytest.reshape(-1, 1)
xx = xx.reshape(-1,1)

""" Set up done """

TestAdam = opt.RMSPropOptimizer(gamma=0.01)
TestNet = NN.NeuralNetwork(xtrain.shape[1],
                        [30,30,1], 
                        [act.ReLU, act.sigmoid, act.sigmoid], 
                        act.mse, 
                        optimizer=TestAdam,
                        regularization="L2",
                        lam=0.001)

TestAdamJax = opt.RMSPropOptimizer(gamma=0.01)
TestNetJax = NN.NeuralNetwork(xtrain.shape[1],
                        [30,30,1], 
                        [act.ReLU, act.sigmoid, act.sigmoid], 
                        act.mse, 
                        optimizer=TestAdamJax,
                        regularization="L2",
                        lam=0.001)

TestNet.train(xtrain, ytrain, 1000)
TestNetJax.train_jax(xtrain, ytrain, 1000)


"""Build simple model to predict a range of x values to see if jax and own gradient produces the same"""
ypred = TestNet.predict(xTest)
yxx = TestNet.predict(xx)
yxxjax = TestNetJax.predict(xx)

plt.plot(xx,yxx)
plt.plot(xx,yxxjax)
plt.show()