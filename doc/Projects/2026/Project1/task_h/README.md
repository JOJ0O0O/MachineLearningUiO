To recreate the results described in the report, we decided to put the code into five scripts. 
model_definition_data_generation.py includes every definition used from parts above as well as functions from the Tuesday notebook.
Then their are two main scripts (minibatch_comparison.py and excesscost_cv_validation.py) that run the defined functions and plot the excess_cost and compare the minibatch sizes. 
There is no need to change anything in those three scripts, as we have the script (configuration.py), where you can change the main setttings. 
The dictionary MINIBATCH changes the settings for script minibatch_comparison.py and the dictionary CROSS_VALIDATION changes the hyperparameter for the script excesscost_cv_validation.py. Lists are storing the values that will be run in a loop to print every output either as a .csv file or a .png file. 
The file also contains information which script to run, by setting them to False, they will not run! 
Finally there is the run_h.py script, you only have to execute this file, it will import the functions from the other scripts. In case you do not want to save the figures and just want to open them for a single view, you can turn SAVE_FIGURES to false.
Feel free to change variables to explore the code, but the current setting was used to get the same plots and results shown in the report.
