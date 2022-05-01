"""
code for SKlearn cross validation
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses


import mlflow




# train on subset of data and find best hyperparameters. then train a model using all the data on those
# parameters.return that as best model

def get_sk_learn_best_model(X_train, y_train, specs, method, learners):

    # from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    # from sklearn.tree import DecisionTreeRegressor
    # from sklearn.ensemble import RandomForestRegressor
    # import xgboost as xgb
    # from sklearn.metrics import mean_absolute_error
    # from sklearn.model_selection import GridSearchCV

    with mlflow.start_run() as run:#run_id="Linear_Regression"
        mlflow.autolog()
        
        # lr = LinearRegression()
        # regressor = specs['methods'][method]['learner']
        regressor = learners[method]
        param_grid = specs['methods'][method]['param_grid']
        clf_gs = GridSearchCV(regressor, param_grid=param_grid, scoring='neg_median_absolute_error', n_jobs=10, verbose=2)
        clf_gs.fit(X_train, y_train)
        print(run.info.run_id, clf_gs.best_params_)
    
    return(clf_gs.best_estimator_, run.info.run_id)

def get_keras_01_best_model(X_train, y_train, specs, method, learners):

    # import tensorflow as tf
    # from tensorflow.keras import layers
    # from tensorflow.keras import losses

    input_size = X_train.shape[1]
    model = tf.keras.Sequential([
        layers.Dense(2*input_size, activation='relu', input_shape=(input_size,)),
        layers.Dense(20, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.MeanAbsoluteError())#, metrics=[tf.keras.metrics.Recall()])
    with mlflow.start_run() as run:#run_id="Linear_Regression"
        mlflow.autolog()
        model.fit(X_train, y_train, epochs=10)#, validation_split=0.2)
    
    return(model, run.info.run_id)


#add tags for mlfow tracking server
# some way to multiprocess decsion trees

def get_best_model(X_train, y_train, specs, method, learners):
    print("method: ", method)

    #choose best model function based on on method i.e. if method in [list of ]
    fit_function = specs['methods'][method]['fit-function']
    
    #fit functions
    # sk_learn_methods = ['Linear Regression', 'Lasso Regression', 'Ridge Regression','XG Boost Regression']
    # keras_methods = []


    if fit_function == "scikit learn":
        print("fit-function", fit_function, "\n\n")
        return get_sk_learn_best_model(X_train, y_train, specs, method, learners)    
    elif fit_function == "keras 01":
        print("fit-function", fit_function)
        return get_keras_01_best_model(X_train, y_train, specs, method, learners)    

    