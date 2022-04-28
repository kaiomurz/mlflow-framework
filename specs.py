specs = {
    'ticks': 72,\
    'flavors':{
        'scikit_learn':{
            'Linear Regression':{
                'regressor': LinearRegression(),
                'param_grid': {
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            },
            'Lasso':{
                'regressor': Lasso(),
                'param_grid': {
                    'alpha': [0.1, 0.5, 1],
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            },
            'Ridge':{
                'regressor': Ridge(),
                'param_grid':{
                    'alpha': [0.1, 0.5, 1],
                    'fit_intercept': [True, False],
                    'positive': [True, False]
                }
            },
            'XG Boost':{
                'regressor': xgb.XGBRegressor(),
                'param_grid': {'max_depth': [3, 5]
                }
            }
        }
    }
}