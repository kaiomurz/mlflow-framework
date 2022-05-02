# Modular ML Experiment Framework


#### _Objective_
Build a modular machine learning engineering framework built on top of ML FLow that can 
- orchestrate training and hyperparameter tuning experiments  
- on various ML methods  
- using various cleaning and feature extraction pipelines. 
The modularity results from the ease with which the data scientist can add, remove, or swap out the various models, hyperparameter sets, or feature extraction pipelines by simply reconfiguring a YAML file. 

The framework then compares the best tuned model of each ML method and offers it up as a REST API. 

Such a framework could be useful to data scientists because of the ease with which the data science pipeline could be set up. It could also be useful to ML engineers who can orchestrate a periodic training and deployment cycle. 

__Details_
The user can supply through a YAML file the following:  
- URI of data set and ML Flow tracking server  
- names of training/cross-validation functions for various models stored in a designated file  
- locations of cleaning feature extraction scripts for various models  
- training parameters/hyperparameters for various models  
- the metric on which to pick best model, etc.   

The framework then runs experiments using ML Flow on various models using the supplied specs by   
- applying the chosen data cleaning and feature engineering pipelines to each model and  
- training/cross-validating the various models and using their respective functions and hyperparameters.  
At the end of all the runs, it picks the best model which can served through a REST API endpoint at the user's discretion.  


The framework is modular because to add another method type, all the data scientist has to do is add   
- the relevant cleaning, feature extraction, and training functions to the relevant files,  
- their details to the `Specs.yaml` file, and  
- add the necessary import statements to the  `train.py` file.

#### _Example use and Current status_
The project is under development and in its current state represents the framework customised for a specific task: to find the best temperature forecasting model for a given dataset. I've picked this task because it's one that can be approached using various methods:  
- regression using various conventional machine learning and neural network tools, and   
- time series forecasting using various libraries such as Prophet or Darts

Thus, this dataset can a good example of how the framework an orchestrate in a modular fashion ML Flow experiments using the various machine learning methods. 

To get a sense of how the eventual framework will work, please install the dependencies as listed in `requirements.txt` and run the `train.py` file. It will  
- read the the specifications for various models provided in `Specs.yaml`  
- engineer features using the `multivariate_fe.py` created for this specific problem  
- run the respective cross-validation or training functions various models in `get_best_model.py`  
- pick the best model of the best method.   


#### _Key additional features to be implemented_
- a command line tool that can set up the basic file structure with instructions on how to populate the files  
- ability to upsert tracking data to any URI, including one on any cloud service  
- parallelly run ML Flow experiments various methods through multiprocessing  
- generating performance and explainability charts and for each model type  
- facility to package a project created on the framework as a docker container so that can be part of a CI-CD pipeline for periodic retraining. 
 


