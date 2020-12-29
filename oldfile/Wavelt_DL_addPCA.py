from DLtools.evaluation_rec import record_list_result
from DLtools.Data import instant_data,station_sel
from DLtools.feature_sel import call_mar
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
# from keras.utils.vis_utils import plot_model

from keras.models import Model
import pywt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,LSTM,RepeatVector,TimeDistributed,Dropout,Conv1D,MaxPooling1D,Input
from tensorflow.keras.callbacks import EarlyStopping

from keras.optimizers import SGD
from keras.callbacks import Callback

from random import seed
seed(42)
#######################################
def Preprocess_pca(input):
    pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components =n_pca))])
    return pipe.fit_transform(input)
def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df
########################################
#Wave  Split XY
def Wavsplit_xy(data,n_past,n_future,pca):
    cAx,cDx,y = Wavsplit_series(data,n_past,n_future,pca)

    if pca==True:
        n_features = n_pca
    elif pca==False:
        n_features = data.shape[1]

    print('*original x,y:', cAx.shape,cDx.shape,y.shape)
    cAx = cAx.reshape((cAx.shape[0], cAx.shape[1],n_features))
    cDx = cDx.reshape((cDx.shape[0], cDx.shape[1],n_features))
    y = y[:,:,0]
    return cAx,cDx,y
def Wavsplit_series(series, n_past, n_future,pca):
    # n_past ==> no of past observations
    # n_future ==> no of future observations
    
    #### PCA IMPLEMENT ##################
    try: series = series.values
    except: pass

    if pca==True: 
        pc_series = Preprocess_pca(series)
    else:
        pc_series=series
    ##########################
    cAx,cDx, y = list(), list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = pc_series[window_start:past_end, :], series[past_end:future_end, :]
        ######################## Wavelet #########################
        N,M = np.zeros(past.shape),np.zeros(past.shape)
        N[:,:],M[:,:] = past[:,:],past[:,:]
        for i in range(past.shape[1]):
          coeff = pywt.swt(past[:,i],wavelet='db4',level=1) 
          cA, cD =coeff[0][0],coeff[0][1] 
          N[:,i] =  cA
          M[:,i] =  cD
        cAx.append(N)
        cDx.append(M)
        y.append(future)
    return np.array(cAx),np.array(cDx), np.array(y)

def Wavtrain_test_split_xy(input_df,pca):
    split_date = '2017-01-01'
    train,test = input_df[:split_date],input_df[split_date:]
    cAX_train,cDX_train, y_train = Wavsplit_xy(train,n_past,n_future,pca)
    cAX_test,cDX_test, y_test = Wavsplit_xy(test,n_past,n_future,pca)
    return  cAX_train,cDX_train, y_train, cAX_test,cDX_test, y_test

######################################
def split_xy(data,n_past,n_future,pca):
    x,y = split_series(data.values,n_past,n_future,pca)
    if pca==True:
        n_features = n_pca
    elif pca==False:
        n_features = data.shape[1]

    x = x.reshape((x.shape[0], x.shape[1],n_features))
    y = y[:,:,0]
    return x,y
def split_series(series, n_past, n_future,pca=False):
    # n_past ==> no of past observations
    # n_future ==> no of future observations 
    ######################
    try: series = series.values
    except: pass
    if pca==True: 
        pc_series = Preprocess_pca(series)
    elif pca==False: 
        pc_series=series
    ##########################
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = pc_series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)
def train_test_split_xy(input_df,pca):
    split_date = '2017-01-01'
    train,test = input_df[:split_date],input_df[split_date:]
    X_train, y_train = split_xy(train,n_past,n_future,pca)
    X_test, y_test = split_xy(test,n_past,n_future,pca)
    return  X_train, y_train, X_test, y_test
###### SETTING AREA ################
def datapreprocess():
    loading = instant_data()
    df,mode = loading.hourly_instant(),'hour'
    
    df = df[start_p:stop_p]
    data = df
    data = data.interpolate(limit=300000000,limit_direction='both').astype('float32')#interpolate neighbor first, for rest NA fill with mean() #.apply(lambda x: x.fillna(x.mean()),axis=0)
    data[target].plot()
    # # MARS
    mars_cutoff = 0.3
    data_mar = call_mar(data,target,mode,cutoff=mars_cutoff)
    data_mar = move_column_inplace(data_mar,target,0)
    # # SCALE
    # scaler = MinMaxScaler()
    # df_scaled = pd.DataFrame(scaler.fit_transform(data_mar), columns=data_mar.columns,index=data_mar.index)
    scaler_tar = MinMaxScaler()
    scaler_tar.fit(data[target].to_numpy().reshape(-1,1))
    return data_mar,scaler_tar
mode='hour'
# if mode =='hour': n_past,n_future = 24,7
if mode =='hour': n_past,n_future = 24,72 #NOTE chang to 24 in-72 out
elif mode =='day': n_past,n_future = 60,30
st = 'CPY012'

target,start_p,stop_p,host_path=station_sel(st,mode)

####################################
def flatten(X):
    '''
    Flatten a 3D array.
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.   
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1]-1), :]
    return(flattened_X)
def scale(X, scaler):
    '''
    Scale 3D array.
    Inputs
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize
    Output
    X            Scaled 3D array.
    '''
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :])                    
    return X
#######################################

############# HYBRID DEEP LEARNING ################################

def build_cnn_auto():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the LSTM for approximate
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu',name="cA_CNN")(inputA)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    
    #y is the EN_LSTM for detail  
    y = layers.LSTM(200, activation="relu",name="cD_LSTM")(inputD)
    y = layers.RepeatVector(n_future)(y)
    y = layers.LSTM(200, activation="relu", return_sequences=True,name="cD_LSTM")(inputD)
    y = layers.TimeDistributed(Dense(100, activation='relu'))(y)
    y = layers.TimeDistributed(Dense(n_features))(y)
    y = layers.LSTM(100, activation="sigmoid")(y)
    #combining 2 lstm
    com = layers.concatenate([x, y])
    z = Dense(n_future)(com)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    return model
def build_cnn_lstm():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the LSTM for approximate
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputA)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50, activation='relu')(x)  
    #y is the LSTM for detail  
    y = layers.LSTM(100, activation="sigmoid", return_sequences=False,name="cD_LSTM")(inputD)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(50)(y)
    
    #combining 2 lstm
    com = layers.concatenate([x, y])
    # z = LSTM(200, activation='relu', return_sequences=False)(com)
    # z = Dense(100, activation="relu")
    z = Dense(n_future)(com)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
def build_Waveann():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, n_features), name="cA")
    inputD = keras.Input(shape=(n_past, n_features), name="cD")
    #x for approximate
    x = layers.Flatten()(inputA)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    #y for detail  
    y = layers.Flatten()(inputD)
    y = layers.Dense(10, activation='sigmoid')(y)
    y = layers.Dropout(0.3)(y)
    #combining 2 lstm
    com = layers.concatenate([x, y])
    z = Dense(n_future)(com)
    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(optimizer=my_optimizer, loss='mse')    
    model.summary()
    return model

def run_code(model,batch_size,syn,minmaxscaler,flag_pca):
    global target,mode
    syn= syn+'_wcAcD'
    
    if flag_pca==True: syn= syn+'_PCA'
    else: syn= syn+'_NoPCA'
    start_time = time.time()
    ################### Scale #######################

    df_scaled,scaler_tar = datapreprocess()   
    cAX_train,cDX_train, y_train, cAX_test,cDX_test, y_test = Wavtrain_test_split_xy(df_scaled,pca=flag_pca)
    ##############Scale or not####################
    if minmaxscaler==True:
        scaler = MinMaxScaler().fit(flatten(cAX_train))
        cAX_train = scale(cAX_train, scaler)
        scaler = MinMaxScaler().fit(flatten(cDX_train))
        cDX_train = scale(cDX_train, scaler)
        scaler = MinMaxScaler().fit(flatten(cAX_test))
        cAX_test = scale(cAX_test, scaler)
        scaler = MinMaxScaler().fit(flatten(cDX_test))
        cDX_test = scale(cDX_test, scaler)
        syn= syn+'_MinMaxLayer'
    else: pass
    #################################################
    print('*'*20,syn,'*'*20)
    print("shape")
    print(cAX_test.shape,cDX_test.shape,y_test.shape)
    #################################################
    validataion = ([cAX_test, cDX_test],y_test)
    history = model.fit(x=[cAX_train, cDX_train], y=y_train,epochs=epochs,batch_size=batch_size,verbose=verbose,validation_data = validataion,callbacks=callbacks)
    time_ = time.time() - start_time
    ########### plot loss ########################
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(save_path+'loss_{}_{:.3}.png'.format(syn,time_), dpi=100, bbox_inches='tight') 
    plt.clf()
    #################################################
    trainPredict = model.predict([cAX_train, cDX_train]).astype('float32')
    testPredict = model.predict([cAX_test, cDX_test]).astype('float32')
    # y_train_ori = scaler_tar.inverse_transform(y_train)
    # trainPredict = scaler_tar.inverse_transform(trainPredict.reshape(y_train_ori.shape))
    # y_test_ori = scaler_tar.inverse_transform(y_test)
    # testPredict = scaler_tar.inverse_transform(testPredict.reshape(y_test_ori.shape))
    y_train_ori,y_test_ori =y_train,y_test
    model.save(save_path+'{}.h5'.format(syn))
    record_list_result(syn,df_scaled,mode,y_train_ori,y_test_ori,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
    
####### ORIGINAL DEEP LEARNING ##########################
def build_ann_original():
    global n_past,n_future,n_features
    model = Sequential()
    model.add(Input(shape=(n_past, n_features)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_future))
    model.compile(optimizer=my_optimizer, loss='mse')    
    model.summary()
    return model
def build_lstm_original():
    global n_past,n_future,n_features
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_past, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_future))
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
def build_autolstm_original():
    global n_past,n_future,n_features
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_past, n_features)))
    model.add(RepeatVector(n_future))                                  # Decoder 
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()    
    return model
def build_cnn1d_original():
    global n_past,n_future,n_features
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_past, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_future))
    model.compile(optimizer=my_optimizer, loss='mse')    
    model.summary()
    return model

def run_code_alone(model,batch_size,syn,cAcD,minmaxscaler,flag_pca):
    global target,mode,callbacks    
    df_scaled,scaler_tar = datapreprocess()

    ##############Wavelet or not############
    if cAcD==True:
        X_train,cDX_train, y_train, X_test,cDX_test, y_test = Wavtrain_test_split_xy(df_scaled,pca=flag_pca)
        if flag_pca==True:        syn= syn+'_wcA_PCA'
        else: syn=syn+'_wcA_noPCA'
    else: 
        X_train, y_train, X_test, y_test = train_test_split_xy(df_scaled,pca=flag_pca)
        if flag_pca==True:        syn= syn+'_PCA'
        else: syn=syn+'_noPCA'
    ##############Scale or not####################
    if minmaxscaler==True:
        scaler = MinMaxScaler().fit(flatten(X_train))
        X_train = scale(X_train, scaler)
        scaler = MinMaxScaler().fit(flatten(X_test))
        X_test = scale(X_test, scaler)
        syn= syn+'_MinMaxLayer'
    else: pass
     #####################################################   
    start_time = time.time()
    
    print('*'*20,syn,'*'*20)
    print("shape")
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)
    history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    time_ = time.time() - start_time
    ########### plot loss ########################
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(save_path+'loss_{}_{:.3}.png'.format(syn,time_), dpi=100, bbox_inches='tight') 
    plt.clf()
    #################################################
    trainPredict = model.predict(X_train).astype('float32')
    testPredict = model.predict(X_test).astype('float32')
    # y_train_ori = scaler_tar.inverse_transform(y_train)
    # trainPredict = scaler_tar.inverse_transform(trainPredict.reshape(y_train_ori.shape))
    # y_test_ori = scaler_tar.inverse_transform(y_test)
    # testPredict = scaler_tar.inverse_transform(testPredict.reshape(y_test_ori.shape))
    y_train_ori,y_test_ori =y_train.astype('float32'),y_test.astype('float32')
    model.save(save_path+'{}.h5'.format(syn))
    record_list_result(syn,df_scaled,mode,y_train_ori,y_test_ori,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
    return 
###############################################################
def run_yolo(call,batch,minmax=False,cAcD=True,flag_pca=True):
    batch_size= batch
   
    if call =='auto':   run_code_alone(build_autolstm_original(),batch_size,'Auto_Tin{}_b{}'.format(n_past,batch_size),cAcD=cAcD,minmaxscaler=minmax,flag_pca=flag_pca)
    elif call =='cnn': run_code_alone(build_cnn1d_original(),batch_size,'CNN_Tin{}_b{}'.format(n_past,batch_size),cAcD=cAcD,minmaxscaler=minmax,flag_pca=flag_pca)
    elif call=='ann': run_code_alone(build_ann_original(),batch_size,'ANN_Tin{}_b{}'.format(n_past,batch_size),cAcD=cAcD,minmaxscaler=minmax,flag_pca=flag_pca) 
    elif call=='lstm': run_code_alone(build_lstm_original(),batch_size,'LSTM_Tin{}_b{}'.format(n_past,batch_size),cAcD=cAcD,minmaxscaler=minmax,flag_pca=flag_pca) 
    
    elif call=='cnnlstm':run_code(build_cnn_lstm(),batch_size,'wCNNLSTM_v2_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)
    elif call=='annann':run_code(build_Waveann(),batch_size,'wANNANN_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)
    elif call=='cnnauto':run_code(build_cnn_auto(),batch_size,'wCNNauto_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)
#####################################################
my_optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]
verbose, epochs = 1, 100
######################################
flag_pca=False
if flag_pca==True: 
    n_pca = 5
    n_features = n_pca
    save_path =host_path+'/wDL_test24_PCA5/' #NOTE chang to 24 in-72 out
elif flag_pca==False:
    n_features = 9
    save_path =host_path+'/wDL_test24_noPCA/' #NOTE chang to 24 in-72 out
##################################


#################### TRIAL IMPROVE #########################
# try: run_yolo('cnnauto',256,minmax=True,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# try: run_yolo('cnnauto',256,minmax=False,flag_pca=flag_pca)
# except KeyboardInterrupt: pass

# try: run_yolo('cnnauto',128,minmax=True,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# try: run_yolo('cnnauto',128,minmax=False,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# #############################################



################# BASE LINE ###################
# # # run_yolo('cnn',256,minmax=True,cAcD=False,flag_pca=flag_pca)
# # run_yolo('cnn',256,minmax=True,cAcD=True,flag_pca=flag_pca)
# # run_yolo('cnn',128,minmax=True,cAcD=True,flag_pca=flag_pca)
# # run_yolo('cnn',128,minmax=True,cAcD=False,flag_pca=flag_pca)

# # run_yolo('ann',256,minmax=True,cAcD=True,flag_pca=flag_pca)
# # run_yolo('ann',256,minmax=True,cAcD=False,flag_pca=flag_pca)
# # run_yolo('ann',128,minmax=True,cAcD=True,flag_pca=flag_pca)
# run_yolo('ann',128,minmax=True,cAcD=False,flag_pca=flag_pca)
# try: run_yolo('auto',256,minmax=True,cAcD=False,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# try: run_yolo('auto',128,minmax=True,cAcD=False,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# try: run_yolo('auto',256,minmax=True,cAcD=True,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# try: run_yolo('auto',128,minmax=True,cAcD=True,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
#################### NEW LSTM ########################
# try: run_yolo('lstm',256,minmax=True,cAcD=False,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# try: run_yolo('lstm',128,minmax=True,cAcD=False,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# try: run_yolo('lstm',256,minmax=True,cAcD=True,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
# try: run_yolo('lstm',128,minmax=True,cAcD=True,flag_pca=flag_pca)
# except KeyboardInterrupt: pass
############################################################


# run_yolo('annann',128,minmax=True,cAcD=True,flag_pca=flag_pca)
# run_yolo('annann',256,minmax=True,cAcD=True,flag_pca=flag_pca)
# run_yolo('cnnlstm',128,minmax=True,cAcD=True,flag_pca=flag_pca)
# run_yolo('cnnlstm',256,minmax=True,cAcD=True,flag_pca=flag_pca)


run_yolo('cnn',128,minmax=False,cAcD=True,flag_pca=flag_pca)
run_yolo('auto',128,minmax=False,cAcD=True,flag_pca=flag_pca)
run_yolo('auto',256,minmax=False,cAcD=True,flag_pca=flag_pca)
try: run_yolo('lstm',256,minmax=False,cAcD=True,flag_pca=flag_pca)
except KeyboardInterrupt: pass
try: run_yolo('lstm',128,minmax=False,cAcD=True,flag_pca=flag_pca)
except KeyboardInterrupt: pass


run_yolo('annann',256,minmax=False,cAcD=True,flag_pca=flag_pca)
run_yolo('cnnlstm',256,minmax=False,cAcD=True,flag_pca=flag_pca)
run_yolo('cnnlstm',128,minmax=False,cAcD=True,flag_pca=flag_pca)
