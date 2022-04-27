"""
Common script for feature extraction for multivariate  models.

drop_columns(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame

rename_columns(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame

add_datetime_elements(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame

circularise(series: pd.core.series.Series) -> Tuple[pd.core.series.Series, pd.core.series.Series]

add_circular_features(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame

"""
from typing import Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


#list functions in docstring

# Engineer target
# data['precipitates'] = np.abs((data['temp']-data['dew_temp']) / (273+data['dew_temp']) )< 0.001
# then shift the column by the number of ticks
# leave 'precipitates' in as a target too.
# scale features

def drop_columns(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Docstring
    """
    data = data.drop('Tpot (K)', axis=1)
    print("data dropped")
    return data

def rename_columns(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Dosctring
    """
    renamer = {
        'Date Time': "date_time",
        'p (mbar)': "pressure", # air pressure in millibars
        'T (degC)': "temp", # air temperature in degrees Celsius
        'Tdew (degC)': "dew_temp", # dew temperature in degrees Celsius
        'rh (%)': "rel_hum", # relative humidity
        'VPmax (mbar)': "max_vap_press", # saturation vapour pressure in millibars
        'VPact (mbar)': "actual_vap_press", # actual vapour pressure in millibars
        'VPdef (mbar)': "vap_press_deficit", # vapour pressure deficit in millibars
        'sh (g/kg)': "spec_hum", # specific humidity g of water in kg of air
        'H2OC (mmol/mol)': "wv_conc", # water vapour concentration millimoles of water/mole of air
        'rho (g/m**3)': "air_density", # density of air
        'wv (m/s)': "wind_speed",
        'max. wv (m/s)': "max_wind_speed",
        'wd (deg)': "wind_direction"
    }

    data.columns = [renamer[column] for column in data.columns]
    print("columns renamed")

    return data

def add_datetime_elements(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:    
    """
    Docstring
    """
    data['date_time'] = pd.to_datetime(data['date_time'], format='%d.%m.%Y %H:%M:%S')

    data['year'] = [value.year for value in data['date_time']]
    data['month'] = [value.month for value in data['date_time']]
    data['day'] = [value.day for value in data['date_time']]
    data['hour'] = [value.hour for value in data['date_time']]
    data['minute'] = [value.minute for value in data['date_time']]

    return data

def circularise(series: pd.core.series.Series) -> Tuple[pd.core.series.Series, pd.core.series.Series]:
    """
    Docstring
    """
    # shrink to 0 to 2*pi
    min = np.min(series)
    max = np.max(series)
    series = series - min
    series = series*2*np.pi/max

    sine = np.sin(series)
    cosine_sign = np.cos(series)

    return (sine, cosine_sign)

def add_circular_features(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Doctring
    """
    circular_features = ['month', 'day', 'hour', 'minute', 'wind_direction']

    for feature in circular_features:
        sine, cosine_sign = circularise(data[feature])
        data[f'{feature}_sine'] = sine
        data[f'{feature}_sine_slant'] = np.sign(cosine_sign)

    return data

def create_target(data, features, target):
    X = data[features]
    y = data[target]

    return X, y

def scale(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X

def get_train_test_splits(X,y, train_data_length, ticks):
    X_train = X[:train_data_length-ticks]
    y_train = y[ticks:train_data_length]

    X_test = X[train_data_length-ticks:-ticks]
    y_test = y[train_data_length:]

    return X_train, y_train, X_test, y_test

def ohe_slants(data: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    pass

def engineer(data):
    data = drop_columns(data)
    data = rename_columns(data)
    data = add_datetime_elements(data)
    data = add_circular_features(data)
    #data = ohe_slants(data)
    # print(data.head())
    return data

def main():
    data = pd.read_csv('jena_climate_2009_2016.csv')# from URL
    data = engineer(data)
    # save engineered data

    

if __name__ == "__main__":
    main()

