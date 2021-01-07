from DLtools.Data import instant_data,station_sel
from DLtools.Trial_evaluation_rec import record_alone_result
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.tsa.vector_ar.var_model import VAR
from DLtools.Data import instant_data,station_sel
from DLtools.feature_sel import call_mar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
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
syn='var'

#stop_p = '2015-06-01'
#split_date = '2015-05-01'
##########################################
save_path =host_path+'/VAR_trial144/'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)
#------------------------------------------
# Define target
data = df[start_p:stop_p]
# interpolate 72 hour(3days data)| accept 3 day missing
data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()

def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df

cutoff=.3
data_mar = call_mar(data,target,mode,cutoff=cutoff)
data_mar = move_column_inplace(data_mar,target,0)
train,test = data_mar[:split_date],data_mar[split_date:]

#-------------------#
idx = test.index    #
#-------------------#
val = list()

history = [x for x in train]
testPredict=list()

for t in tqdm(range(len(test))):
    history = pd.concat([train,test.iloc[:t,:]])

    mod = VAR(history)
    result = mod.fit(maxlags=144,ic='aic')

    yhat = result.forecast(history.values,72)
    testPredict.append(yhat[:,0])


    y_temp = test.iloc[t:t+72,0].values
    val.append(y_temp)

    #testPredict = pd.concat([testPredict,pd.Series(yhat,index=idx[t:t+72])],axis=1)
    #Ytest=pd.concat([Ytest,pd.Series(save,index=idx[t:t+72])])

#------------- Dataframe -------------------#
testY = pd.DataFrame(val,index=idx)
#----------#
#hour = ['{}h'.format(i+1) for i in range (72)]


VAR_result = pd.DataFrame(testPredict)#,columns=hour)
VAR_result.to_csv(save_path+'VARoutput.csv')

# for i,col in enumerate(VAR_result.columns):
#   VAR_result[col]=VAR_result[col].shift(i).dropna()
#   testY[i]=testY[i].shift(i).dropna()
#   record_alone_result(syn,i,None,testY,None,testPredict,target,0,save_path,n_past=0,n_features=0)
