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
def corr_select(data,target):
    def corr_w_Y(data,target,threshold= 0.8):
        # correlation
        corr_test = data.corr(method='pearson')[target]
        corr_test = corr_test[(corr_test> threshold) | (corr_test< -threshold) ]
        corr_test = corr_test.sort_values(ascending=False)
        #corr_test =corr_test[1:] # eliminate Target it own
        print(corr_test)
        return corr_test
    def high_corr_RM(data,threshold=.95):
        """Eliminate first columns with high corr"""
        corr_matrix = data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return to_drop
    col_feature = corr_w_Y(data,target,0.8).index
    data = data[col_feature]
    high_col = high_corr_RM(data.iloc[:,1:]) #exclude target it own
    data = data.drop(columns=high_col)
    return data
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

def var(data):
    start_time_ = time.time()
    train,test = data[:int(0.7*(len(data)))],data[int(0.7*(len(data))):]

    sc_data = StandardScaler()
    sc_data.fit(data)

    sc_train,sc_test = sc_data.transform(train),sc_data.transform(test)
    # scale_data = sc_data.fit_transform(data)
    # train,test = scale_data[:int(0.7*(len(scale_data)))],scale_data[int(0.7*(len(scale_data))):]
    model = VAR(endog=sc_train)
    model_fit = model.fit(9)

    trainPredict = model_fit.forecast(sc_train, steps=len(sc_train))
    testPredict = model_fit.forecast(sc_test, steps=len(sc_test))

    trainPredict = sc_data.inverse_transform(trainPredict)
    testPredict = sc_data.inverse_transform(testPredict)

    trainPredict = pd.Series(data=(trainPredict[:,0]),index=train.index)
    testPredict = pd.Series(data=(testPredict[:,0]),index=test.index)
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
        print(data.shape)
        data['Day'] = data.index.dayofyear #add day
        data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
        data[target]=data[target].shift(-out_t_step)
        data=data.dropna()
        
        #### MAR selection ##
        data = call_mar(data,target,mode,cutoff=0.1)
        ##################################
        data = move_column_inplace(data,target,0)
        
        #### plot ###
        save_path =host_path+'/corr/'
        plot_corr(data,'mar')
        # plot_corr(data,'ML')
        X = data.drop(columns=[target])
        Y = data[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
        print(trainX.shape,trainY.shape,testX.shape,testY.shape)
        
        ############ LINEAR ##################
        save_path =host_path+'/Linear/'
        syn = 'linear'+str(out_t_step)
        trainPredict,testPredict,use_t = linear()
        use_time = use_t
        n_features = 'Mars'
        n_past='all'
        print(out_t_step,'  LR time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1,)
        ########### VAR ################
        save_path =host_path+'/VAR/'
        syn = 'VAR'+str(out_t_step)
        trainPredict,testPredict,time_,train,test =var(data)
        use_time_ = use_t
        n_features = 'Mars'
        n_past='all'
        print(out_t_step,'  VAR time......',use_time_)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time_,save_path,n_past,n_features,n_future=1)

        ######### SVR ################
        save_path =host_path+'/SVR/'
        syn = 'SVR'+str(out_t_step)
        trainPredict,testPredict,use_t = svr()
        use_time = use_t
        n_features = 'Mars'
        n_past='all'
        print(out_t_step,'  SVR time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)
        
################ MAR CORR ########################
    for out_t_step in (range(1,n_future+1)):
        data = df[start_p:stop_p]
        data['Day'] = data.index.dayofyear #add day
        data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
        data[target]=data[target].shift(-out_t_step)
        data=data.dropna()
        
        #### MAR selection ##
        data = call_mar(data,target,mode,cutoff=0.3)
        #### Corr selection##
        data = corr_select(data,target)
        save_path =host_path+'/corr/'
        plot_corr(data,'mar_corr')
        ####################
        data = move_column_inplace(data,target,0)
       
        X = data.drop(columns=[target])
        Y = data[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
        print(trainX.shape,trainY.shape,testX.shape,testY.shape)
        ############ LINEAR ##################
        save_path =host_path+'/Linear/'
        syn = 'linear_CORR'+str(out_t_step)
        trainPredict,testPredict,use_t = linear()
        use_time = use_t
        n_features = 'Mars_CORR'
        n_past='all'
        print(out_t_step,'  LR time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1,)
        ########### VAR ################
        save_path =host_path+'/VAR/'
        syn = 'VAR_CORR'+str(out_t_step)
        trainPredict,testPredict,time_,train,test =var(data)
        use_time_ = use_t
        
        n_past='all'
        print(out_t_step,'  VAR CORR time......',use_time_)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time_,save_path,n_past,n_features,n_future=1)
        ######### SVR ################
        save_path =host_path+'/SVR/'
        syn = 'SVR_CORR'+str(out_t_step)
        trainPredict,testPredict,use_t = svr()
        use_time = use_t
        n_features = 'Mars_CORR'
        n_past='all'
        print(out_t_step,'  SVR CORR time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)
################ CORR ########################
    for out_t_step in (range(1,n_future+1)):
        data = df[start_p:stop_p]
        data['Day'] = data.index.dayofyear #add day
        data = data.interpolate(limit=30000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean()
        data[target]=data[target].shift(-out_t_step)
        data=data.dropna()
        

        #### Corr selection##
        data_corr = corr_select(data,target)
        save_path =host_path+'/corr/'
        plot_corr(data,'corr')
        ####################
        data = move_column_inplace(data,target,0)
       
        X = data.drop(columns=[target])
        Y = data[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
        print(trainX.shape,trainY.shape,testX.shape,testY.shape)
        ############ LINEAR ##################
        save_path =host_path+'/Linear/'
        syn = 'linear_CORR'+str(out_t_step)
        trainPredict,testPredict,use_t = linear()
        use_time = use_t
        n_features = 'Mars_CORR'
        n_past='all'
        print(out_t_step,'  LR time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1,)
        ########### VAR ################
        save_path =host_path+'/VAR/'
        syn = 'VAR_CORR'+str(out_t_step)
        trainPredict,testPredict,time_,train,test =var(data)
        use_time_ = use_t
        
        n_past='all'
        print(out_t_step,'  VAR CORR time......',use_time_)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time_,save_path,n_past,n_features,n_future=1)
        ######### SVR ################
        save_path =host_path+'/SVR/'
        syn = 'SVR_CORR'+str(out_t_step)
        trainPredict,testPredict,use_t = svr()
        use_time = use_t
        n_features = 'Mars_CORR'
        n_past='all'
        print(out_t_step,'  SVR CORR time......',use_t)
        record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1)