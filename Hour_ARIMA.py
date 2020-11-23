from DLtools.Data import instant_data,station_sel
from DLtools.evaluation_rec import real_eva_error
from tqdm import tqdm

from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

###########################################
loading = instant_data()
# df,mode = loading.hourly_instant(),'hour'
df,mode = loading.daily_instant(),'day'

st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
if mode =='hour': n_past,n_future = 24*7,72
elif mode =='day': n_past,n_future = 60,30
##########################################
save_path =host_path+'/ARIMA/'

# Define target
data = df[start_p:stop_p]
# interpolate 72 hour(3days data)| accept 3 day missing
data['Day'] = data.index.dayofyear #add day
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

ratio = int((data.shape[0])*.7)
train,test = data[:ratio],data[ratio:]
history = [x for x in train]
testPredict = list()

# test=test[:int(len(test)/10)]
for t in tqdm(range(len(test))):
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    testPredict.append(yhat)
    obs = test[t]
    history.append(obs)
Tmse,Tnse,Tr2 = real_eva_error(testPredict,test)

yhat_series = pd.Series(testPredict, index=test.index)

plt.figure(figsize=(20,5))
plt.plot(train,label='training')
plt.plot(test, label='actual')
plt.plot(yhat_series, label='forecast')
plt.title('Water Level CPY015 ARIMA Forecast vs Actuals\n'+'Test MSE: %.3f | Test NSE: %.3f | R2 score: %.3f' % (Tmse,Tnse,Tr2))
plt.xlabel('Date')
plt.ylabel('Water level(mls)')
plt.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(save_path+'result_arima.png', dpi=300, bbox_inches='tight')



# df_Train = pd.DataFrame({'Train':train,'TrainPredict':trainPredict})
df_Test = pd.DataFrame({'Test':test,'TestPredict':testPredict})
# df_Train.to_csv(host_path+'Daily/ARIMA/train_predicton.csv')
df_Test.to_csv(host_path+'Daily/ARIMA/test_predicton.csv')

idx=['Modelname','Feature','n_in_time','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2']
col = ['ARIMA']
_df = pd.DataFrame(["ARIMA","None","None",'None','None', 'None','None',Tmse, Tnse,Tr2],index=idx,columns=col)
error = pd.concat([error,_df],axis=1)
error.to_csv(host_path+'Daily/eval.csv')
