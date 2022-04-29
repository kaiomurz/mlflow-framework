"""
code for SKlearn cross validation
"""

# from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import mlflow




# create abstract class for best model maker. model makers for each method inherit from that
# put import statements in function?

# train on subset of data and find best hyperparameters. then train a model using all the data on those
# parameters.return that as best model
# def get_best_params(X_train, y_train):
#     pass

# def train_on_best_params(X_train, y_train, best_params):
#     pass

def get_sk_learn_best_model(X_train, y_train, specs, method, learners):
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


#add tags for mlfow tracking server
# some way to multiprocess decsion trees

def get_best_model(X_train, y_train, specs, method, learners):
    print("method: ", method)

    #choose best model function based on on method i.e. if method in [list of ]
    sk_learn_methods = ['Linear Regression', 'Lasso Regression', 'Ridge Regression','XG Boost Regression']
    keras_methods = []
    if method in sk_learn_methods:
        return get_sk_learn_best_model(X_train, y_train, specs, method, learners)    

    # with mlflow.start_run() as run:#run_id="Linear_Regression"
    #     mlflow.autolog()
        
    #     # lr = LinearRegression()
    #     # regressor = specs['methods'][method]['learner']
    #     regressor = learners[method]
    #     param_grid = specs['methods'][method]['param_grid']
    #     clf_gs = GridSearchCV(regressor, param_grid=param_grid, scoring='neg_median_absolute_error', n_jobs=10, verbose=2)
    #     clf_gs.fit(X_train, y_train)
    #     print(run.info.run_id, clf_gs.best_params_)


    
    # return(clf_gs.best_estimator_, run.info.run_id)
