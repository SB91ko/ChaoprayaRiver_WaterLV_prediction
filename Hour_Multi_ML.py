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
from sklearn.ensemble import RandomForestRegressor 
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from DLtools.Data import instant_data,intersection,station_sel
from DLtools.evaluation_rec import record_alone_result,nashsutcliffe
from DLtools.feature_sel import call_mar,hi_corr_select

def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df
def plot_corr(data,syn):
    global out_t_step
    ##Display / save
    plt.subplots(figsize=(10,10))
    mask = np.triu(data.corr())
    sns.heatmap(data.corr(), annot = False, vmin=-1, vmax=1, center= 0,mask=mask)
    plt.savefig(save_path+'{}lag{}.png'.format(syn,out_t_step), bbox_inches='tight')
    return

def linear():
    global trainX,trainY,testX,testY,syn
    start_time = time.time()
    steps = [('scale',StandardScaler()),('pca', PCA(n_components = n_pca)), ('lr', linear_model.LinearRegression())]
    pipe = Pipeline(steps=steps)
    pipe.fit(trainX,trainY)
    trainPredict = pipe.predict(trainX)
    testPredict = pipe.predict(testX)

    trainPredict = pd.Series(data=(trainPredict),index=trainY.index)
    testPredict = pd.Series(data=(testPredict),index=testY.index)
    time_ = time.time() - start_time
    return trainPredict,testPredict,time_

def svr():
    global trainX,trainY,testX,testY,syn
    start_time = time.time()
    svr = svm.SVR(kernel='rbf',C=1e3)
    steps = [('scale',StandardScaler()),('pca', PCA(n_components = n_pca)), ('svr', svr)]
    pipe = Pipeline(steps=steps)

    pipe.fit(trainX, trainY)
    trainPredict = pipe.predict(trainX)
    testPredict = pipe.predict(testX)

    trainPredict = pd.Series(data=(trainPredict),index=trainY.index)
    testPredict = pd.Series(data=(testPredict),index=testY.index)
    time_ = time.time() - start_time
    return trainPredict,testPredict,time_
def rf():
    global trainX,trainY,testX,testY,syn
    start_time = time.time()

    rf = RandomForestRegressor(n_estimators = 100)
    steps = [('scale',StandardScaler()),('pca', PCA(n_components =n_pca)), ('rf', rf)]
    pipe = Pipeline(steps=steps)
    pipe.fit(trainX,trainY)

    trainPredict = pipe.predict(trainX)
    testPredict = pipe.predict(testX)
    trainPredict = pd.Series(data=(trainPredict),index=trainY.index)
    testPredict = pd.Series(data=(testPredict),index=testY.index)
    time_ = time.time() - start_time
    return trainPredict,testPredict,time_
def var(data):
    start_time_ = time.time()
    # train,test = data[:int(0.7*(len(data)))],data[int(0.7*(len(data))):]
    data = data.interpolate(limit=30000000,limit_direction='both').astype('float32')
    split_date = '2017-01-01'
    train,test = data[:split_date],data[split_date:]

    steps = [('scale',StandardScaler()),('pca', PCA(n_components =n_pca))]
    pipe = Pipeline(steps=steps)
    pipe.fit(data)

    train,test = data[:int(0.7*(len(data)))],data[int(0.7*(len(data))):]
    sc_train,sc_test = pipe.transform(train),pipe.transform(test)

    model = VAR(endog=sc_train)
    model_fit = model.fit(9)

    trainPredict = model_fit.forecast(sc_train, steps=len(sc_train))
    testPredict = model_fit.forecast(sc_test, steps=len(sc_test))
    try:
        trainPredict = pipe.inverse_transform(trainPredict)
        testPredict = pipe.inverse_transform(testPredict)

        trainPredict = pd.Series(data=(trainPredict[:,0]),index=train.index)
        testPredict = pd.Series(data=(testPredict[:,0]),index=test.index)
    except:
        trainPredict,testPredict = -999,-999
    trainY = pd.Series(data=(train.iloc[:,0]),index=train.index)
    testY = pd.Series(data=(test.iloc[:,0]),index=test.index)
    time_ = time.time() - start_time_
    return trainPredict,testPredict,time_,trainY,testY
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

def inti_data(df):
    global start_p,stop_p
    data = df[start_p:stop_p].astype('float32')
    # data['Day'] = data.index.dayofyear #add day
    data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
    data[target]=data[target].shift(-out_t_step)
    data.dropna(inplace=True)
    return data
###########################################
loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
# df,mode = loading.daily_instant(),'day'

st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
if mode =='hour': n_past,n_future = 24*7,72
elif mode =='day': n_past,n_future = 60,30
n_pca=5
###########################################

if __name__ == "__main__":  
    for out_t_step in (range(1,n_future+1)):    
        data = inti_data(df)
        cutoff=0.3
        #### MAR selection ####
        data = call_mar(data,target,mode,cutoff=cutoff)
        ##################################
        data = move_column_inplace(data,target,0)
        
        print(data.columns)
        #### plot #####
        save_path =host_path+'/corr/'
        if out_t_step==0: plot_corr(data,'mar{}_pca_'.format(cutoff))
        
        X = data.drop(columns=[target])
        Y = data[target]
        split_date = '2017-01-01'
        trainX, testX = X[:split_date].dropna(),X[split_date:].dropna()
        trainY, testY = Y[:split_date].dropna(),Y[split_date:].dropna()
        
        # trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
        # print(trainX.shape,trainY.shape,testX.shape,testY.shape)
        # ############ LINEAR ##################
        # save_path =host_path+'/Linear/'
        # syn = 'linear_pca_{}_{}'.format(cutoff,str(out_t_step))
        # trainPredict,testPredict,use_t = linear()
        # use_time = use_t
        # n_features = 'MarsPca_{}'.format(cutoff)
        # n_past='all'
        # print(cutoff,out_t_step,'  LR time......',use_t)
        # record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1,)
        
        # ######### VAR ################
        # data = inti_data(df)
        # X = data.drop(columns=[target])
        # Y = data[target]
        # split_date = '2017-01-01'
        # trainX, testX = X[:split_date],X[split_date:]
        # trainY, testY = Y[:split_date],Y[split_date:]

        # save_path =host_path+'/VAR/'
        # syn = 'VAR_pca_{}_{}'.format(cutoff,str(out_t_step))
        # trainPredict,testPredict,time_,train,test =var(data)
        
        # n_features = 'MarsPca_{}'.format(cutoff)
        # n_past='all'
        # print(cutoff,out_t_step,'  VAR time......',time_)
        # record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,time_,save_path,n_past,n_features,n_future=1)
        #### SVR ################
    
        # data = inti_data(df)
        # X = data.drop(columns=[target])
        # Y = data[target]
        # split_date = '2017-01-01'
        # trainX, testX = X[:split_date],X[split_date:]
        # trainY, testY = Y[:split_date],Y[split_date:]
        save_path =host_path+'/SVR/'
        syn = 'SVR_pca{}_{}'.format(cutoff,str(out_t_step))
        trainPredict,testPredict,use_t = svr()
        use_time = use_t
        n_features = 'MarsPca_{}'.format(cutoff)
        n_past='all'
        print(cutoff,out_t_step,'  SVR time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)
        
        ####### RF ################
        # data = inti_data(df)
        # # X = data.drop(columns=[target])
        # # Y = data[target]
        # # split_date = '2017-01-01'
        # # trainX, testX = X[:split_date],X[split_date:]
        # # trainY, testY = Y[:split_date],Y[split_date:]
        # save_path =host_path+'/RF/'
        # syn = 'RF_pca_{}_{}'.format(cutoff,str(out_t_step))
        # trainPredict,testPredict,use_t = rf()
        # use_time = use_t
        # n_features = 'MarsPca_{}'.format(cutoff)
        # n_past='all'
        # print(cutoff,out_t_step,'  RF time......',use_t)
        # record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)
# ############### MAR CORR ########################
#     for out_t_step in (range(1,n_future+1)):
#         data = df[start_p:stop_p]
#         data['Day'] = data.index.dayofyear #add day
#         data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
#         data[target]=data[target].shift(-out_t_step)
#         data=data.dropna(inplace=True)
        
#         #### MAR selection ##
#         data = call_mar(data,target,mode,cutoff=0.3)
#         #### Corr selection##
#         data = hi_corr_select(data,target)
#         save_path =host_path+'/corr/'
#         plot_corr(data,'mar_corr_')
#         ####################
#         data = move_column_inplace(data,target,0)
#         if data.shape[1]>1: 
                
#             X = data.drop(columns=[target])
#             Y = data[target]
#             split_date = '2017-01-01'
#             trainX, testX = X[:split_date],X[split_date:]
#             trainY, testY = Y[:split_date],Y[split_date:]

#             # ############ LINEAR ##################
#             save_path =host_path+'/Linear/'
#             syn = 'linear_Mars_CORR'+str(out_t_step)
#             trainPredict,testPredict,use_t = linear()
#             use_time = use_t
#             n_features = 'Mars_CORR'
#             n_past='all'
#             print(out_t_step,'  LR time......',use_t)
#             record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1,)
#             ########### VAR ################
#             save_path =host_path+'/VAR/'
#             syn = 'VAR_Mars_CORR'+str(out_t_step)
#             trainPredict,testPredict,time_,train,test =var(data)
#             use_time_ = use_t
            
#             n_past='all'
#             print(out_t_step,'  VAR CORR time......',use_time_)
#             record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time_,save_path,n_past,n_features,n_future=1)
#             ######## SVR ################
#             save_path =host_path+'/SVR/'
#             syn = 'SVR_Mars_CORR'+str(out_t_step)
#             trainPredict,testPredict,use_t = svr()
#             use_time = use_t
#             n_features = 'Mars_CORR'
#             n_past='all'
#             print(out_t_step,'  SVR CORR time......',use_t)
#             record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)
#             ######## RF ################
#             save_path =host_path+'/RF/'
#             syn = 'RF_Mars_CORR'+str(out_t_step)
#             trainPredict,testPredict,use_t = rf()
#             use_time = use_t
#             n_features = 'Mars_CORR'
#             n_past='all'
#             print(out_t_step,'  RF CORR time......',use_t)
#             record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)
#         else: pass
############### CORR ########################
    # for out_t_step in (range(1,n_future+1)):
    #     data = df[start_p:stop_p]
    #     data['Day'] = data.index.dayofyear #add day
    #     data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
    #     data[target]=data[target].shift(-out_t_step)
    #     data.dropna(inplace=True)
        
    #     data_corr = hi_corr_select(data,target)
    #     save_path =host_path+'/corr/'
    #     plot_corr(data,'corr_')
    #     ###################
    #     data = move_column_inplace(data,target,0)
       
    #     X = data.drop(columns=[target])
    #     Y = data[target]
    #     trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
    #     split_date = '2017-01-01'
    #     trainX, testX = X[:split_date],X[split_date:]
    #     trainY, testY = Y[:split_date],Y[split_date:]

    #     ########## LINEAR ##################
    #     save_path =host_path+'/Linear/'
    #     syn = 'linear_CORR'+str(out_t_step)
    #     trainPredict,testPredict,use_t = linear()
    #     use_time = use_t
    #     n_features = 'CORR'
    #     n_past='all'
    #     print(out_t_step,'  LR time......',use_t)
    #     record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1,)
    #     ########### VAR ################
    #     save_path =host_path+'/VAR/'
    #     syn = 'VAR_CORR'+str(out_t_step)
    #     trainPredict,testPredict,time_,train,test =var(data)
    #     use_time_ = use_t
        
    #     n_past='all'
    #     print(out_t_step,'  VAR CORR time......',use_time_)
    #     record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time_,save_path,n_past,n_features,n_future=1)
    #     ###### SVR ################
    #     save_path =host_path+'/SVR/'
    #     syn = 'SVR_CORR'+str(out_t_step)
    #     trainPredict,testPredict,use_t = svr()
    #     use_time = use_t
    #     n_features = 'CORR'
    #     n_past='all'
    #     print(out_t_step,'  SVR CORR time......',use_t)
    #     record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)
    #     ######## RF ################
    #     save_path =host_path+'/RF/'
    #     syn = 'RF_CORR'+str(out_t_step)
    #     trainPredict,testPredict,use_t = rf()
    #     use_time = use_t
    #     n_features = 'CORR'
    #     n_past='all'
    #     print(out_t_step,'  RF CORR time......',use_t)
    #     record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)