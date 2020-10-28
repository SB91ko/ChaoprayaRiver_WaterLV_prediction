import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import numpy as np

class Arima_model:

    def __init__(self,input_series,interpolate=3):
        self.input_series = input_series
        self.interpolate = interpolate
        self.X_data = input_series.interpolate(method='pad', limit=interpolate).dropna().values
        
        self.check_na()
        
        # self.test_stationarity()

        
    # def test_stationarity(self):
    #     stats = ['Test Statistic','p-value','Lags','Observations']
    #     df_test = adfuller(self.input_series, autolag='AIC')
    #     df_results = pd.Series(df_test[0:4], index=stats)
    #     for key,value in df_test[4].items():
    #         df_results['Critical Value (%s)'%key] = value        
    #     print(df_results)
        #extra
        # if df_results[1] <= 0.05:
        #     print("strong evidence against the null hypothesis(H0), reject the null hypothesis. Data has no unit root and is stationary")
        # else:
        #     print("weak evidence against null hypothesis, time series has a  unit root, indicating it is non-stationary")

    def check_na(self):
        print("Is there are NAN remain?:...... ",np.isnan(self.X_data.sum()))
        return

    def model_arima(self,pdq=(1,1,1)):
        size = int(len(self.X_data) * 0.7)
        train, test = self.X_data[0:size-1], self.X_data[size:len(self.X_data)]

        history = [x for x in train]
        validation = list()

        for t in range(len(test)):
            model = ARIMA(history, order=pdq) #edited pdq already
            model_fit = model.fit()             ###
            self.model = model_fit
            output = model_fit.forecast()       ###
            yhat = output[0]
            validation.append(yhat)            ###
            obs = test[t]
            history.append(obs)
            print(t,'....predicted=%f, expected=%f' % (yhat, obs))
        
        #print(self.model.summary())
        
        d = {'Test':test,'Predict':validation}
        data = pd.DataFrame(data=d,index= self.input_series.index[size:len(self.X_data)])
        return data
    
class WANN_model:
    pass # TO DO list