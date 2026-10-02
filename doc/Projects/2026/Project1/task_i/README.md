To run this code, you have to execute file function_vs_degree.py, however you can change the settings in the file config.py. It contains information 
about the size of the loop used for degrees and for lambda, but also whether to save outputs or not. Furthermore, it defines the k for k_fold.py 
and the number of data points. The script kfold.py totally relies on sklearn, the data.py is the same file used before (so it contains every data
function used for this fit). We decided to use sklearn for both, cross validation and for lasso and ridge, as it uses coordinate descent which 
runs much faster as it uses analytical solutions. For 30 degrees and 50 lambdas, there was no way to use our own implementation with adam!
