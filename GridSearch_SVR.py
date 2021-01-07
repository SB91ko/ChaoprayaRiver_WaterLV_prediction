import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn import svm

from DLtools.Trial_evaluation_rec import record_list_result,real_eva_error
from DLtools.Data import instant_data,station_sel
from DLtools.feature_sel import call_mar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
#----------------------------------#
def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df
def to_supervise(data,target,n_out):
    data[target]=data[target].shift(-n_out)
    data = data.astype('float64').dropna()
    X = data.drop([target],axis=1)
    xlabels = list(X.columns)
    X = X
    y = data[target]
    return X,y,xlabels
#-------------------------------------------#
loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
#df,mode = loading.daily_instant(),'day'

if mode =='hour': n_past,n_future = 24*6,24
elif mode =='day': n_past,n_future = 30,14
else: n_future=None; print('incorrect input')

st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)

start_p = '2016-01-01'
stop_p ='2017-01-01'
#-----------------------------
data = df[start_p:stop_p]
split_date = int(len(data)*.7)
data = data.interpolate(limit=3000000000,limit_direction='both').astype('float32')
data['Day'] = data.index.dayofyear #add day
#-----------------------------
cutoff=.3
data_mar = call_mar(data,target,mode,cutoff=cutoff)
data_mar = move_column_inplace(data_mar,target,0)
n_features = len(data_mar.columns)
#----------------------------
out_t_step=1

X,Y,_ = to_supervise(data_mar,target,out_t_step)
#trainX, testX = X[:split_date].dropna(),X[split_date:].dropna()
#trainY, testY = Y[:split_date].dropna(),Y[split_date:].dropna()
trainX, testX = X.iloc[:split_date].dropna(),X.iloc[split_date:].dropna()
trainY, testY = Y.iloc[:split_date].dropna(),Y.iloc[split_date:].dropna()
#--------------------------------------------#


scaler=StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.transform(testX)
#param_grid = {'C': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,10**2,10**3,10**4,10**5,10**6], 'gamma': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,10**2,10**3,10**4,10**5,10**6],'kernel': ['rbf', 'linear']}
param_grid = {'C': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,10**2,10**3,10**4], 'gamma': [1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1,10,10**2,10**3,10**4,10**5,10**6],'kernel': ['rbf', 'linear']}
grid = GridSearchCV(svm.SVR(),param_grid,refit=True,verbose=2)
grid.fit(trainX,trainY)
print(grid.best_estimator_)
