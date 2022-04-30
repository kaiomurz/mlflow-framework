
import pandas as pd
from sklearn.model_selection import train_test_split
from yaml import load
from yaml import CLoader as Loader
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import recall_score, precision_score, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
import sys
import numpy as np
import mlflow
import pprint
from cml_models import get_best_model
# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras import losses


import multivariate_fe as mfe


target = 'temp'

features = ['pressure', 'temp', 'dew_temp', 'rel_hum', 'max_vap_press',\
            'actual_vap_press', 'spec_hum', 'wv_conc','air_density',\
            'wind_speed', 'max_wind_speed', 'wind_direction', 'month_sine',\
            'month_sine_slant', 'day_sine', 'day_sine_slant', 'hour_sine',\
            'hour_sine_slant', 'minute_sine', 'minute_sine_slant',\
            'wind_direction_sine','wind_direction_sine_slant']

train_data_length = 300000 # convert to % of data size. Pull from yaml
ticks = 72 # pull from yaml
metric_name = "MAE"
metric = mean_absolute_error

learners = {
    "Linear Regression" : LinearRegression(),
    "Lasso Regression": Lasso(),
    "Ridge Regression": Ridge(),
    "XG Boost Regression": xgb.XGBRegressor()
}
def retrieve_specs(specs_file='Specs.yaml'):
    with open(specs_file,'rb') as f:
        specs = load(f, Loader=Loader)
    return specs




def main():
    print(learners)
    specs = retrieve_specs()
    pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(specs)
    # sys.exit()

    #get imports

    # extract general specs
    ticks = specs['problem-specific specs']['hours']*6
    methods = specs['methods']
    train_test_split = specs['train-test split']
    # pp.pprint(methods)


    data = pd.read_csv(specs['data URI'])
    data = mfe.engineer(data)
    # print(data.head())
    print(data.shape)

# remove methods for which run == False


########################################################################
### modularise this ###

### collate all methods based on clean module and train them in order of clean module in order to 
### avoid repitition in cleaning.
    X, y = mfe.create_target(data, features, target)

    X = mfe.scale(X)   

    X_train, y_train, X_test, y_test = mfe.get_train_test_splits(X, y, train_test_split, ticks)
   

    print(X_train.shape, len(y_train))
    print(X_test.shape, len(y_test))

########################################################################

    #move training of each model type to separate module
    # create module for mlflow.run() to run experiment and log results, 
    # that can be called on each training model from the train.py file.
    # i.e. iterate over model_type:module dictionary
    # the train.py file will contain all the parameters for each model that it gets from MLproject. 
    # it can then pass on the relevant parameters to each ML method module
    # find best model for each ticks value
    
    # 
 

    # add progress bar



    # for flavor in specs['flavors']: #change to flavors
    best_model = None
    best_perf = np.inf
    best_run_id = None
    best_method = None

    for method in specs['methods']:
        # model, run_id = scikit_learn.get_best_model(X_train, y_train, specs, method, learners)
        model, run_id = get_best_model.get_best_model(X_train, y_train, specs, method, learners)

        y_true = y_test
        y_pred = model.predict(X_test)
        perf = metric(y_true, y_pred)
        
        print(f"\n\n{method}     {-perf:.3f}\n\n")

        if perf < best_perf:
            best_model = model
            best_perf = perf
            best_run_id = run_id
            best_method = method
        print(f"\n\nPerformance: {perf}  Best Performance: {best_perf}    Best Method: {best_method}\n\n")

    print(best_run_id)    
            #log method and metric and 

        # mlflow.pyfunc.save_model(best_model)

# https://www.programcreek.com/python/example/121632/mlflow.pyfunc


# models and paths directory/list


if __name__=="__main__":
    main()
