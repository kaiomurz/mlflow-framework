
import pandas as pd
# from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.metrics import recall_score, precision_score, mean_absolute_error, mean_absolute_percentage_error

import mlflow

from cml_models import scikit_learn

import multivariate_fe as mfe


target = 'temp'

features = ['pressure', 'temp', 'dew_temp', 'rel_hum', 'max_vap_press',\
            'actual_vap_press', 'spec_hum', 'wv_conc','air_density',\
            'wind_speed', 'max_wind_speed', 'wind_direction', 'month_sine',\
            'month_sine_slant', 'day_sine', 'day_sine_slant', 'hour_sine',\
            'hour_sine_slant', 'minute_sine', 'minute_sine_slant',\
            'wind_direction_sine','wind_direction_sine_slant']

train_data_length = 300000 # convert to 80% of data size
ticks = 72 # convert to list
metric_name = "MAE"
metric = mean_absolute_error


def main():
    data = pd.read_csv('jena_climate_2009_2016.csv')# from URL
    data = mfe.engineer(data)
    # print(data.head())
    print(data.shape)

########################################################################
### modularise this ###
    X, y = mfe.create_target(data, features, target)

    X = mfe.scale(X)   

    X_train, y_train, X_test, y_test = mfe.get_train_test_splits(X, y, train_data_length, ticks)
   

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
    frameworks = {
        "scikit_learn": [
            "Linear Regression",
            "Lasso",
            "Ridge",
            "Decision Tree"
        ]
    }

    # add progress bar



    for framework in frameworks:
        for method in frameworks[framework]:
            clf = scikit_learn.get_best_model(X_train, y_train, method)
            y_true = y_test
            y_pred = clf.predict(X_test)
            perf = metric(y_true, y_pred)
            perf = metric(y_true, y_pred)
            print(f"\n\n{method}     {-perf:.3f}\n\n")

            #log method and metric

    

# models and paths directory/list


if __name__=="__main__":
    main()
