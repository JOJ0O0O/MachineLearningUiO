To receive the same results as shown in the report, this task is divided in six scripts. 
The script data.py contains every definition that was either implemented in previous tasks or from the Tuesday notebooks. 
The main tasks are divided in three scripts:
  fitting.py contains the code for plotting our own Lasso Regression and compare it to sklearn, closed form and Ridge Regression.
  jax_ad.py contains code for doing a jax auto differentiation, this is only used for comparison to our own implementation, however, as computing times can increase it is recommended to turn it off in the config.py file.
  kfolds.py runs a k fold cross validation for evaluating the produced fits
To set every hyperparameter and to decide which script to run there is a configuration file named config.py. The current settings were used for the plots and .csv files shown in the report. 
It is possible to turn saving of figures and tables on or off in run_g.py script. You only have to run this script, but feel free to explore the code and to change the hyperparameters in config.py
