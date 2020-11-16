from DLtools.Data import instant_data
from DLtools.evaluation_rec import real_eva_error
from sklearn.metrics import mean_squared_error,r2_score

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

loading = instant_data()
df =loading.hourly_instant()
TARGET= 'CPY015_wl' # target station

save_path='/home/song/Public/Song/Work/Thesis/output/Hourly/ARIMA/'
#load previous error rec
idx=['Modelname','Feature','n_in_time','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2']

try: error = pd.read_csv('/home/song/Public/Song/Work/Thesis/output/Hourly/eval.csv')
except: error = pd.DataFrame(index = idx)

# Define target
data = df['2013-01-01':'2017-12-31']

# interpolate 72 hour(3days data)| accept 3 day missing
data = data.interpolate(limit=72).apply(lambda x: x.fillna(x.mean()),axis=0).astype('float32')
data = data[TARGET]

# Train_test spilt
ratio = int((data.shape[0])*.7)
train,test = data[:ratio],data[ratio:]

history = [x for x in train]
predictions,confidents = list(),list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    conf = output[2]
    predictions.append(yhat)
    confidents.append(conf)
    obs = test[t]
    history.append(obs)
    
    #print('predicted=%f, expected=%f' % (yhat, obs))

mse,nse,r2 = real_eva_error(predictions,test)
print('Test MSE: %.3f Test NSE: %.3f' % (mse,nse))
print("R2 score: {}".format(r2))

yhat_series = pd.Series(predictions, index=test.index)
plt.figure(figsize=(20,5))
plt.plot(train,label='training')
plt.plot(test, label='actual')
plt.plot(yhat_series, label='forecast')
plt.title('Water Level CPY015 ARIMA Forecast vs Actuals\n'+'Test MSE: %.3f | Test NSE: %.3f | R2 score: %.3f' % (mse,nse,r2))
plt.xlabel('Date')
plt.ylabel('Water level(mls)')
plt.legend(loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(save_path+'result_arima.png', dpi=300, bbox_inches='tight')