import numpy as np
from sklearn.model_selection import KFold
from config import CONFIG

'''
AI delcaration: AI was used for chaning our own kfold implementation to the sklearn implementation, as it runs natively in C/Cython and is 
therefore much faster than our own implementation. This was necessary, because during our 
first run with our own Lasso Code we could not receivve any results!
'''
np.set_printoptions(precision=4, suppress=True)

def kfold_function(X_raw, y_raw, model):
    k_folds = CONFIG["k_folds"]
    
    #choosing k folds = 5, so every data point is used for 4 training times
    #each point gets used exactly once for validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=CONFIG["random_state"])

    #storing th mse results
    test_mses = []

    for train_idx, test_idx in kf.split(X_raw):
        # Split raw data
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]
        
        #scale the data to prevent leakage 
        X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
        X_train_norm = (X_train - X_mean) / X_std
        X_test_norm = (X_test - X_mean) / X_std

        #centering the y data 
        y_mean = y_train.mean()
        y_train_centered = y_train - y_mean
        y_test_centered = y_test - y_mean

        #setting different fitting models 
        model.fit(X_train_norm, y_train_centered)
        y_pred = model.predict(X_test_norm)
        
        # evaluate and store the reults in a dict
        test_mses.append(np.mean((y_pred - y_test_centered)**2))

    return np.mean(test_mses)
