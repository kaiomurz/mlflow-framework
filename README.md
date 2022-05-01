# ML Flowork


#### _Objective_
Build a _modular_ machine learning engineering framework based on ML Flow where a data scientist can supply through a YAML file the following:  
- URI of data set  
- names of training functions for various models (stored in a single file)  
- locations of cleaning feature extraction scripts for various models  
- other info like any specs for target construction  
- training parameters/hyperparameters for various model  
- metrics on which to pick best model   

The framework then runs experiments using ML Flow on various models using the supplied specs by   
- cleaning data and engineering features (based on feature engineering script specified for method),  
- training/cross-validating the various models and using their respective functions and hyperparameters,  
and picks the best model which can served through a REST API endpoint at the user's discretion.  


The framework is modular because to add another method type, all the data scientist has to do is add   
- the relevant cleaning, feature extraction, and training functions to the relevant files,  
- their details to the `Specs.yaml` file, and 
- add the necessary import statements to the  `train.py` file.

#### _Example use and Current status_
The project is under development and in its current state represents the framework customised for a specific task: to find the best temperature forecasting model for a given dataset. I've picked this task because it's one that can be approached using various methods:  
- regression using various conventional machine learning and neural network tools, and   
- time series forecasting using various libraries such as Prophet or Darts

Thus this dataset can a good example of how the framework an orchestrate ML Flow experiments using the various machine learning methods. 

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
- packaging as a docker container that can be run from a data-scientist's machine or in the cloud. 
 


