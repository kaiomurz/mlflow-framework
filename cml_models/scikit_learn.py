"""
code for SKlearn cross validation
"""

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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

# param_grids = {  ### read params for grid search from ML project file or dedicated yaml file that contains grid search parameters
#     "Linear Regression": {
#         "fit_intercept": [True, False], 
#         "positive": [True, False]
#     },
#     "Lasso":{
#         "alpha":[0.1, 0.5, 1],
#         "fit_intercept": [True, False], 
#         "positive": [True, False]    
#     }
# }

methods = {
    "Linear Regression": {
        "regressor": LinearRegression(),
        "param_grid": {
            "fit_intercept": [True, False], 
            "positive": [True, False]
            }
    },
    "Lasso": {
        "regressor": Lasso(),
        "param_grid": {
            "alpha":[0.1, 0.5, 1],
            "fit_intercept": [True, False], 
            "positive": [True, False]
            }
    },
    "Ridge": {
        "regressor": Ridge(),
        "param_grid": {
            "alpha":[0.1, 0.5, 1],
            "fit_intercept": [True, False], 
            "positive": [True, False]
            }
    },
    "Decision Tree": {
        "regressor": DecisionTreeRegressor(),
        "param_grid": {
            "criterion": [ "friedman_mse", "absolute_error" ],#"poisson" "squared_error",
            "max_depth": [3, 5, 10],
            
        }


    }
    
}

# regressors = {
#     "Linear Regression":LinearRegression(),
#     "Lasso": Lasso()
# }


#add tags for mlfow tracking server
# some way to multiprocess decsion trees

def get_best_model(X_train, y_train, method):
    
    with mlflow.start_run():#run_id="Linear_Regression"
        mlflow.autolog()
        
        # lr = LinearRegression()
        regressor = methods[method]["regressor"]
        param_grid = methods[method]["param_grid"]
        clf_gs = GridSearchCV(regressor, param_grid=param_grid, scoring='neg_median_absolute_error', n_jobs=10, verbose=2)
        clf_gs.fit(X_train, y_train)
        print(clf_gs.best_params_)

    
    return(clf_gs.best_estimator_)
