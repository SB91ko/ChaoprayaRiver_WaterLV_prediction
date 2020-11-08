import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import numpy as np

import pywt
import warnings


class prepTS:
    def __init__(self,input_series,interpolate=3):
        self.input_series = input_series
        self.interpolate = interpolate
        self.X_data = input_series.interpolate(method='pad', limit=interpolate).dropna().values
        self.check_na()
        self.test_stationarity()

    def check_na(self):
        print("Is there are NAN remain?:...... ",np.isnan(self.X_data.sum()))
        return

    def test_stationarity(self):
        stats = ['Test Statistic','p-value','Lags','Observations']
        df_test = adfuller(self.input_series.dropna(), autolag='AIC')
        df_results = pd.Series(df_test[0:4], index=stats)
        for key,value in df_test[4].items():
            df_results['Critical Value (%s)'%key] = value        
        print(df_results)
        #extra
        if df_results[1] <= 0.05:
            print("strong evidence against the null hypothesis(H0), reject the null hypothesis. Data has no unit root and is stationary")
        else:
            print("weak evidence against null hypothesis, time series has a  unit root, indicating it is non-stationary")


class Arima_model:
    warnings.filterwarnings('ignore')
    def __init__(self,X_data,pdq=(1,1,1),flag_bestpdq=False):
        self.X_data = X_data 
        self.model= None
        self.pdq = pdq
        self.flag_bestpdq = flag_bestpdq
        
        self.size = int(len(self.X_data) * 0.7)
        self.train,self.test = self.X_data[0:self.size-1], self.X_data[self.size:len(self.X_data)]

        self.bestparameter()
               
        
    def model_result(self):
        history = [x for x in self.train]
        validation = list()
        for t in range(len(self.test)):
            model = ARIMA(history, order=self.pdq) #edited pdq already
            model_fit = model.fit()             ###
            self.model = model_fit
            output = model_fit.forecast()       ###
            yhat = output[0]
            validation.append(yhat)            ###
            obs = self.test[t]
            history.append(obs)
            #print(t,'....predicted=%f, expected=%f' % (yhat, obs))
        
        print(self.model.summary())
        
        d = {'Test':self.test,'Predict':validation}
        #data = pd.DataFrame(data=d,index= self.input_series.index[self.size:len(self.X_data)])
        data = pd.DataFrame(data=d)
        return data

    def bestparameter(self):
        if self.flag_bestpdq:
            import itertools
            
            #p=d=q=range(0,10); pdq = list(itertools.product(p,d,q))

            p=q=range(0,10)
            d=range(1,4)                # fix d = 1 or 2
            temp_pdq = itertools.product(p,d,q)

            aic_result =999999
            warnings.filterwarnings('ignore')
            for param in temp_pdq:
                try:
                    model_arima = ARIMA(self.train,order=param)
                    model_arima_fit = model_arima.fit()
                    if model_arima_fit.aic < aic_result:
                        aic_result = model_arima_fit.aic
                        print(param,model_arima_fit.aic)
                        self.pdq = param
                except:
                    continue
            print("final pdq",self.pdq)
            return 

class WANN_model:
    def __init__(self,X_data):
        self.X_data = X_data
        self.lv = int(np.log(len(self.X_data)))
        self.coeffs = None
        self.wa_X = self.wavelet_denoising()

    def wavelet_denoising(self, wavelet='db4'):
        #https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
        def madev(d, axis=None):
            """ Mean absolute deviation of a signal """
            return np.mean(np.absolute(d - np.mean(d, axis)), axis)

        coeff = pywt.wavedec(self.X_data, wavelet, mode="per")
        self.coeffs = coeff # just in case
        sigma = (1/0.6745) * madev(coeff[-self.lv])
        uthresh = sigma * np.sqrt(2 * np.log(len(self.X_data)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        return pywt.waverec(coeff, wavelet, mode='per')

    def wavelet_reconstruction(self):
        return pywt.waverec(self.coeffs,wavelet = self.wa_X)
       
    # def wavelet_denoising(self, wavelet='db4', level=1):
    #     #https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform
    #     def madev(d, axis=None):
    #         """ Mean absolute deviation of a signal """
    #         return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    #     coeff = pywt.wavedec(self.X_data, wavelet, mode="per")
    #     self.coeffs = coeff # just in case
    #     sigma = (1/0.6745) * madev(coeff[-level])
    #     uthresh = sigma * np.sqrt(2 * np.log(len(self.X_data)))
    #     coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    #     return pywt.waverec(coeff, wavelet, mode='per')
