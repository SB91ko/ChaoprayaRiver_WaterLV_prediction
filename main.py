import pandas as pd
import re,glob
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import itertools


def test_stationarity(series_input):
    stats = ['Test Statistic','p-value','Lags','Observations']
    df_test = adfuller(series_input, autolag='AIC')
    df_results = pd.Series(df_test[0:4], index=stats)
    for key,value in df_test[4].items():
        df_results['Critical Value (%s)'%key] = value
    ##extra
    if df_results[1] <= 0.05:
        print("strong evidence against the null hypothesis(H0), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a  unit root, indicating it is non-stationary")


class ARIMA_MODEL:
    def __init__(self,df_target_col,train_percent):       
        self.series_input = df_target_col.values
        self.train_percent = train_percent
        self.train
        self.test
        self.best_pdq
    
    def __split_train_test__(self,series_input,train_percent):
        size = int(len(X) * 0.66)
        train, test = X[0:size-1], X[size:len(X)]
        return self.train,self.test

    def __findARIMAparameter__(self):
        p=q=range(0,10)
        d=range(1,4) # fix d = 1 or 2
        pdq = itertools.product(p,d,q)
        
        aic_result =999999
        for param in pdq:
            try:
                model_arima = ARIMA(train,order=param)
                model_arima_fit = model_arima.fit()
                if model_arima_fit.aic < aic_result:
                    aic_result = model_arima_fit.aic
                    best_pqd = pdq
                    print(param,model_arima_fit.aic)
            except:
                continue
        return self.best_pqd


def batch_data(df,target,shift_day):
    """
    df : input dataframe
    target : predict target
    shift_day : time window, no. of lookahead data
    """
    x_data = df.values[:-shift_day]
    print(type(x_data))
    print("Shape:",x_data.shape)
    print("*"*20)
    y_data = df[target].values[:-shift_day]
    print(type(y_data))
    print("Shape:", y_data.shape)
    return x_data,y_data

if __name__ == "__main__":
    #cleaning input data

    # re-scale
    rain_min_max_scaler = MinMaxScaler()
    water_min_max_scaler = MinMaxScaler()

    #set lable data/ time window / batch data
    batch_data()
    
