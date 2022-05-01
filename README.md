# weather-prediction

### Objective and ideas

#### _Objective_
Build a modular machine learning engineering framework where a data scientist can supply through a YAML file the following:
- URI of data set
- names of training functions for various models (stored in a single file)
- locations of cleaning feature extraction scripts for various models
- other info like any specs for target construction
- training parameters/hyperparameters for various model
- metrics on which to pick best model

The framework then run experiments using ML Flow on various models using the supplied specs by 
- cleaning data and engineering features (based on feature engineering script specified for method)
- training/cross-validating the various models and using their respective functions and hyperparameters
- picks the best model, which can served through a REST API endpoint, at the user's discretion.  


To add another method type, all the data scientist has to do is 
- create the relevant cleaning, feature extraction, and training functions to the relevant files and add their details to the YAML file. 

The project is under development and in its current state represents the framework customised for a certain task, to find the best temperature forecasting model. I've picked this task because it's one that can be approached using various methods:  
- regression using various conventional machine learning and neural network tools, and 
- time series forecasting using various libraries such as Prophet or Darts

Thus this dataset can a good example of how the framework an orchestrate the various experiments. 

The elements can still be examined, however. 
- the yaml file lays out the specification
- the get_best_model.py script contains cross-validation or training functions for different methods that return the best tuned model for the relevant method.
- the multivariate_fe.py for feature extraction for this specific problem



Key additional features to be implemented:
- a command line tool that can set up the basic file structure with instructions on how to populate the files
- ability to upsert tracking data to any URI, including one on any cloud service
- parallelly run various cross-validations through multiprocessing
- generating performance and explainability charts and for each model type
- packaging as a docker container that can be run from a data-scientist's machine or in the cloud


_Ignore_  
Instructions  
- populate yaml file  
- populate imports.py