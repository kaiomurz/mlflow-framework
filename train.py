
import pandas as pd

import multivariate_fe as mfe


target = 'temp'

features = ['pressure', 'temp', 'dew_temp', 'rel_hum', 'max_vap_press',\
            'actual_vap_press', 'spec_hum', 'wv_conc','air_density',\
            'wind_speed', 'max_wind_speed', 'wind_direction', 'month_sine',\
            'month_sine_slant', 'day_sine', 'day_sine_slant', 'hour_sine',\
            'hour_sine_slant', 'minute_sine', 'minute_sine_slant',\
            'wind_direction_sine','wind_direction_sine_slant']

def main():
    data = pd.read_csv('jena_climate_2009_2016.csv')# from URL
    data = mfe.engineer(data)
    # print(data.head())
    print(data.shape)

    X, y = mfe.create_target(data, features, target)

    X = mfe.scale(X)
    print(X[:10])
    print(X.shape)
    print(y.shape)

if __name__=="__main__":
    main()
