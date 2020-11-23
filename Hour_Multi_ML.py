import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn import svm
from sklearn import linear_model
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from DLtools.Data import instant_data,intersection,station_sel
from DLtools.evaluation_rec import record_alone_result,nashsutcliffe
from DLtools.feature_sel import call_mar

def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df
def high_corr(data,threshold=.95):
    """Eliminate first columns with high corr"""
    corr_matrix = data.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop
def corr_w_Y(data,target,threshold= 0.8):
    # correlation
    corr_test = data.corr(method='pearson')[target]
    corr_test = corr_test[(corr_test> threshold) | (corr_test< -threshold) ]
    corr_test = corr_test.sort_values(ascending=False)
    #corr_test =corr_test[1:] # eliminate Target it own
    return corr_test
def corr_select(data,target):
    col_feature = corr_w_Y(data,target,0.5).index
    data = data[col_feature]
    high_col = high_corr(data.iloc[:,1:]) #exclude target it own
    data.drop(columns=high_col,inplace=True)
    return data
def plot_corr(data,syn):
    global out_t_step
    ##Display / save
    plt.subplots(figsize=(10,10))
    mask = np.triu(data.corr())
    sns.heatmap(data.corr(), annot = True, vmin=-1, vmax=1, center= 0,mask=mask)
    plt.savefig(save_path+'Corr_{}lag{}.png'.format(syn,out_t_step), bbox_inches='tight')
    return

def linear():
    global trainX,trainY,testX,testY,syn
    start_time = time.time()
    regr = linear_model.LinearRegression()
    regr.fit(trainX,trainY)
    trainPredict = regr.predict(trainX)
    testPredict = regr.predict(testX)

    trainPredict = pd.Series(data=(trainPredict),index=trainY.index)
    testPredict = pd.Series(data=(testPredict),index=testY.index)
    time_ = time.time() - start_time
    return trainPredict,testPredict,time_

def svr():
    global trainX,trainY,testX,testY,syn
    start_time = time.time()
    svr = svm.SVR(kernel='rbf',C=1e3)
    scale = StandardScaler()
    pipe = Pipeline([('scaler', scale), ('svr', svr)])
    
    pipe.fit(trainX, trainY)
    trainPredict = pipe.predict(trainX)
    testPredict = pipe.predict(testX)

    trainPredict = pd.Series(data=(trainPredict),index=trainY.index)
    testPredict = pd.Series(data=(testPredict),index=testY.index)
    time_ = time.time() - start_time
    return trainPredict,testPredict,time_

def forecast_accuracy(forecast, actual,title):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    mse = np.mean((forecast - actual)**2)       #MSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    nse = nashsutcliffe(actual,forecast)
    try: r2 = r2_score(actual, forecast)
    except: r2 = np.NaN
    result = {'MSE':mse,'rmse':rmse,'R2':r2,'NSE':nse,'mape':mape,  'mae': mae,
            'mpe': mpe, 'corr':corr}
    result =  pd.Series(result,name=title)
    return result
###########################################
loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
# df,mode = loading.daily_instant(),'day'

st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
if mode =='hour': n_past,n_future = 24*7,72
elif mode =='day': n_past,n_future = 60,30
###########################################

if __name__ == "__main__":  
    for out_t_step in (range(1,n_future+1)):
        data = df[start_p:stop_p]
        data['Day'] = data.index.dayofyear #add day
        data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
        data[target]=data[target].shift(-out_t_step)
        data=data.dropna()
        
        #### MAR selection ##
        data = call_mar(data,target,mode,cutoff=0.3)
        #### Corr selection##
        # data = corr_select(data,target)
        # save_path =host_path+'/Linear/'
        # plot_corr(data,'mar_corr')
        ##################################
        data = move_column_inplace(data,target,0)
        
        
        #### plot ###
        # plot_corr(data,'ML')
        X = data.drop(columns=[target])
        Y = data[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
        print(trainX.shape,trainY.shape,testX.shape,testY.shape)
        ######################################
        save_path =host_path+'/Linear/'
        syn = 'linear'+str(out_t_step)
        trainPredict,testPredict,use_t = linear()
        batch_size = use_t
        n_features = 'Mars'
        n_past='all'
        print(out_t_step,'  time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future=1,)
# ###################################################
#     for out_t_step in (range(1,n_future+1)):
        # data = df[start_p:stop_p]
        # data['Day'] = data.index.dayofyear #add day
        # data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
        # data[target]=data[target].shift(-out_t_step)
        # data=data.dropna()
       
        # data = move_column_inplace(data,target,0)

        # X = data.drop(columns=[target])
        # Y = data[target]
        # trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
        
        save_path =host_path+'/SVR/'
        syn = 'SVR'+str(out_t_step)
        trainPredict,testPredict,use_t = svr()
        batch_size = use_t
        n_features = 'Mars'
        n_past='all'
        print(out_t_step,'  time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future=1)
        ###################################