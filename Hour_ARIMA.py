from DLtools.Data import instant_data,station_sel

from tqdm import tqdm
import time
import pmdarima as pm
import pandas as pd
import numpy as np

np.random.seed(42)

###########################################
loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
# df,mode = loading.daily_instant(),'day'

st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
if mode =='hour': n_past,n_future = 24*6,72
elif mode =='day': n_past,n_future = 60,30
split_date = '2016-11-01'
##########################################
save_path =host_path+'/ARIMA/'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)
#------------------------------------------    
# Define target
data = df[start_p:stop_p]
# interpolate 72 hour(3days data)| accept 3 day missing
data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
data_uni = data[target]

# # Train_test spilt
# ratio = int((data.shape[0])*.7)
# train,test = data[:ratio],data[ratio:]
# model = ARIMA(train, order=(1,1,0))
# model_fit = model.fit(disp=0)
# result = model.fit() 
# trainPredict = result.predict(0, len(train) , 
#                              typ = 'levels').rename("Predictions") 
# mse,nse,r2 = real_eva_error(train,trainPredict[:])

# data = df['2013-01-01':'2017-12-31'].interpolate(limit=72).apply(lambda x: x.fillna(x.mean()),axis=0).astype('float32')
# data = data[TARGET]


train,test = data_uni[:split_date],data_uni[split_date:]
#-----------------------------
arima = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(arima.summary())

history = [x for x in train]
testPredict,ytest = list(),list()
start_time = time.time()


#for t in tqdm(range(len(test))):
for t in tqdm(range(len(test))):
    output=arima.fit_predict(history,n_periods=72)
    # arima.update(history)
    # output = arima.predict(n_periods=72)
    
    yhat = output
    testPredict.append(yhat)
    obs = test[t]
    history.append(obs)
    ytest.append(obs)
    history.pop(0)
    
hour = ['{}h'.format(i+1) for i in range (72)]
result = pd.DataFrame(testPredict,columns=hour)
result.to_csv(save_path+'arimaoutput.csv')