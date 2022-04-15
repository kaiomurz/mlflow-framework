#######################################################################


# create features and choose target

features = ['pressure', 'temp', 'dew_temp', 'rel_hum', 'max_vap_press', 'month', 'day', 'hour', 'minute', 'month_sine', 'month_sine_slant',\
            'day_sine', 'day_sine_slant', 'hour_sine', 'hour_sine_slant', 'minute_sine', 'minute_sine_slant', 'wind_direction_sine',\
            'wind_direction_sine_slant', 'precipitates'] 
            
target = 'precipitates'
target = 'temp'

#######################################################################

# Choose metric and esstablish a baseline for predictions as the value 
# 24 hours before the time being predicted.

metric = mean_absolute_error
metric_name = "MAE"

ticks_dict = {1:'10 min', 6:'1 hr', 36:'6 hrs',72:'12 hrs',108:'18 hrs',144:'24 hrs'} #10 minutes, 1hr, 6hrs, 12hrs

# Establish a baseline performance 

print(f"Time\tTicks\t{metric_name}")#\tFalse Percentage")
for ticks in ticks_dict:
    y_pred = data[target][ticks:-144]
    y_true = data[target][144+ticks:]

    performance = metric(y_true, y_pred)
    print(f"{ticks_dict[ticks]}\t{ticks}\t{performance:.5f}")   

#######################################################################

# Create an X and y for the ML models based on the chosen target. 

X = data[features]
y = data[target]

#######################################################################

# Standardise the np array X and split out the train and test sets.

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#######################################################################

# Split out the train and test sets based on lenght and number of ticks.

train_data_length = 300000
ticks = 72

X_train = X[:train_data_length-ticks]
y_train = y[ticks:train_data_length]

X_test = X[train_data_length-ticks:-ticks]
y_test = y[train_data_length:]

print(X_train.shape[0], len(y_train))
print(X_test.shape[0], len(y_test))

#######################################################################

### create a classifier/regressor and train it

# clf = DecisionTreeClassifier(max_depth=2)

# clf = RandomForestClassifier(max_depth=5)
# clf = GradientBoostingClassifier()
# clf = LogisticRegression()

# clf = LogisticRegression()#penalty='l1', solver='liblinear')
# clf = LogisticRegression(penalty='l2', solver='liblinear')

clf = LinearRegression()
# clf = RandomForestRegressor()
# clf = Ridge()
# clf = Lasso()
# clf = DecisionTreeRegressor()
# clf = AdaBoostRegressor()


# clf.fit(X_train,y_train)

#######################################################################

### create a cross-validator and find best model


# rf_param_grid = {
#     'n_estimators': [100, 200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [3,4,5,6,7,8]
# }

# CV_clf = cross_validate(X_train, y_train, scoring='neg_mean_absolute_error')
CV_clf = GridSearchCV(estimator=clf, scoring='mae', n_jobs=4, verbose=4)

CV_clf.fit(X_train,y_train)

# clf = CV_clf.best_estimator_

#######################################################################

### assess the classifier/regressor basd on relevant metric

y_true = y_test
y_pred = clf.predict(X_test)
# y_pred = X_test['precipitates']
# recall = recall_score(y_test, y_pred)

print(metric(y_true, y_pred))