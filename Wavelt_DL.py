from DLtools.evaluation_rec import record_list_result
from DLtools.Data import instant_data,station_sel
from DLtools.feature_sel import call_mar,hi_corr_select
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

def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df
#Wave  Split XY
def Wavsplit_xy(data,n_past,n_future):
    cAx,cDx,y = Wavsplit_series(data,n_past,n_future)
    print('*original x,y:', cAx.shape,cDx.shape,y.shape)
    cAx = cAx.reshape((cAx.shape[0], cAx.shape[1],n_features))
    cDx = cDx.reshape((cDx.shape[0], cDx.shape[1],n_features))
    y = y[:,:,0]
    return cAx,cDx,y
def Wavsplit_series(series, n_past, n_future):
    # n_past ==> no of past observations
    # n_future ==> no of future observations
    # if type(series)== 'pandas.core.frame.DataFrame': 
    series = series.values
    
    cAx,cDx, y = list(), list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
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
def Wavtrain_test_split_xy(input_df):
#   split_date = int(len(input_df)*.7)
#   train,test = input_df[:split_date],input_df[split_date:]
  split_date = '2017-01-01'
  train,test = input_df[:split_date],input_df[split_date:]
  # check_pca(train,True,pca_cut)

  cAX_train,cDX_train, y_train = Wavsplit_xy(train,n_past,n_future)
  cAX_test,cDX_test, y_test = Wavsplit_xy(test,n_past,n_future)
  return  cAX_train,cDX_train, y_train, cAX_test,cDX_test, y_test

###### SETTING AREA ################
def datapreprocess():
    loading = instant_data()
    df,mode = loading.hourly_instant(),'hour'
    # df,mode = loading.daily_instant(),'day'
    
    df = df[start_p:stop_p]
    data = df
    data = data.interpolate(limit=300000000,limit_direction='both').astype('float32')#interpolate neighbor first, for rest NA fill with mean() #.apply(lambda x: x.fillna(x.mean()),axis=0)
    data[target].plot()
    # # MARS
    mars_cutoff = 0.2
    data_mar = call_mar(data,target,mode,cutoff=mars_cutoff)
    data_mar = move_column_inplace(data_mar,target,0)
    # # SCALE
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(data_mar), columns=data_mar.columns,index=data_mar.index)
    scaler_tar = MinMaxScaler()
    scaler_tar.fit(data[target].to_numpy().reshape(-1,1))
    return df_scaled,scaler_tar
mode='hour'
# if mode =='hour': n_past,n_future = 24,7
if mode =='hour': n_past,n_future = 24*7,72
elif mode =='day': n_past,n_future = 60,30
st = 'CPY012'

target,start_p,stop_p,host_path=station_sel(st,mode)
save_path =host_path+'/wDL/'
my_optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)


#######################################################
def build_lstmlstm():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the LSTM for approximate
    x = layers.LSTM(200, activation="relu", return_sequences=False,name="cA_LSTM")(inputA)
    x = layers.Dense(100)(x)
    x = layers.Dropout(0.2)(x)
    #y is the LSTM for detail  
    y = layers.LSTM(200, activation="relu", return_sequences=False,name="cD_LSTM")(inputD)
    y = layers.Dense(100)(y)
    y = layers.Dropout(0.2)(y)

    #combining 2 lstm
    com = layers.concatenate([x, y])
    # z = LSTM(200, activation='relu', return_sequences=False)(com)
    z = Dense(100, activation="relu")(com)
    z = Dense(n_future,activation = 'relu')(z)

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
    y = layers.LSTM(200, activation="relu", return_sequences=False,name="cD_LSTM")(inputD)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(20)(y)

    #combining 2 lstm
    com = layers.concatenate([x, y])
    # z = LSTM(200, activation='relu', return_sequences=False)(com)
    z = Dense(100, activation="relu")(com)
    z = Dense(n_future,activation = 'relu')(z)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    return model
def build_cnn_cnn():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the LSTM for approximate
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputA)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50, activation='relu')(x)
    #y is the LSTM for detail  
    y = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputD)
    y = layers.MaxPooling1D(pool_size=2)(y)
    y = layers.Flatten()(y)
    y = layers.Dense(50, activation='relu')(y)
    #combining 2 lstm
    com = layers.concatenate([x, y])

    z = Dense(100, activation="relu")(com)
    z = Dense(n_future,activation = 'relu')(z)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
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
####################################
# def flatten(X):
#     '''
#     Flatten a 3D array.
#     Input
#     X            A 3D array for lstm, where the array is sample x timesteps x features.   
#     Output
#     flattened_X  A 2D array, sample x features.
#     '''
#     flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
#     for i in range(X.shape[0]):
#         flattened_X[i] = X[i, (X.shape[1]-1), :]
#     return(flattened_X)
# def scale(X, scaler):
#     '''
#     Scale 3D array.
#     Inputs
#     X            A 3D array for lstm, where the array is sample x timesteps x features.
#     scaler       A scaler object, e.g., sklearn.preprocessing.StandardScaler, sklearn.preprocessing.normalize
#     Output
#     X            Scaled 3D array.
#     '''
#     for i in range(X.shape[0]):
#         X[i, :, :] = scaler.transform(X[i, :, :])                    
#     return X
#######################################
def run_code_alone(model,batch_size,syn,zoom=False):
    global target,mode,df_scaled    
    # scaler = MinMaxScaler().fit(flatten(cAX_train))
    # cAX_train_sc = scale(cAX_train, scaler)
    # scaler = MinMaxScaler().fit(flatten(cAX_test))
    # cAX_test_sc = scale(cAX_test, scaler)

    start_time = time.time()
    verbose, epochs = 1, 100
    
    history = model.fit(cAX_train,y_train,epochs=epochs,validation_data=(cAX_test,y_test),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    time_ = time.time() - start_time
    ########### plot loss ########################
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(save_path+'loss_{}_{:.3}.png'.format(syn,time_), dpi=300, bbox_inches='tight') 
    plt.clf()
    #################################################
    trainPredict = model.predict(cAX_train).astype('float32')
    testPredict = model.predict(cAX_test).astype('float32')
    y_train_ori = scaler_tar.inverse_transform(y_train)
    trainPredict = scaler_tar.inverse_transform(trainPredict.reshape(y_train_ori.shape))
    y_test_ori = scaler_tar.inverse_transform(y_test)
    testPredict = scaler_tar.inverse_transform(testPredict.reshape(y_test_ori.shape))

    record_list_result(syn,df_scaled,mode,y_train_ori,y_test_ori,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
    model.save(save_path+'{}.h5'.format(syn))
    return 

def run_code(model,batch_size,syn,zoom=False):
    global target,mode
    start_time = time.time()
    print('*'*50)
    print ( 'batch :', batch_size,'\nmodel :', syn)
    print('*'*50)
    verbose, epochs = 2, 100
    ################### Scale #######################
    # scaler = MinMaxScaler().fit(flatten(cAX_train))
    # cAX_train_sc = scale(cAX_train, scaler)
    # scaler = MinMaxScaler().fit(flatten(cAX_test))
    # cAX_test_sc = scale(cAX_test, scaler)
    # scaler = MinMaxScaler().fit(flatten(cDX_train))
    # cDX_train_sc = scale(cDX_train, scaler)
    # scaler = MinMaxScaler().fit(flatten(cDX_test))
    # cDX_test_sc = scale(cDX_test, scaler)
    #################################################
    validataion = ([cAX_test, cDX_test],y_test)
    history = model.fit(x=[cAX_train, cDX_train], y=y_train,epochs=epochs,batch_size=batch_size,verbose=verbose,validation_data = validataion,callbacks=callbacks)
    time_ = time.time() - start_time
    ########### plot loss ########################
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(save_path+'loss_{}_{:.3}.png'.format(syn,time_), dpi=300, bbox_inches='tight') 
    plt.clf()
    #################################################
    trainPredict = model.predict([cAX_train, cDX_train]).astype('float32')
    testPredict = model.predict([cAX_test, cDX_test]).astype('float32')
    y_train_ori = scaler_tar.inverse_transform(y_train)
    trainPredict = scaler_tar.inverse_transform(trainPredict.reshape(y_train_ori.shape))
    y_test_ori = scaler_tar.inverse_transform(y_test)
    testPredict = scaler_tar.inverse_transform(testPredict.reshape(y_test_ori.shape))

    # a=pd.DataFrame(trainPredict)
    # b=pd.DataFrame(testPredict)
    # pd.concat([a,b]).to_csv(save_path+'result_{}.csv'.format(syn))

    record_list_result(syn,df_scaled,mode,y_train_ori,y_test_ori,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
    model.save(save_path+'{}.h5'.format(syn))
#####################################################
# # PCA
# if mode=='day':n_com_pca = 5
# elif mode =='hour': n_com_pca = 8
df_scaled,scaler_tar = datapreprocess()

n_features = df_scaled.shape[1]
cAX_train,cDX_train, y_train, cAX_test,cDX_test, y_test = Wavtrain_test_split_xy(df_scaled)
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]
###################################################
print('sample x timesteps x features')
print('sample 0: 2 timestep : 5 feature \n',cAX_test[0,:2,:5])
print(
      '\n********XY train-test split********'
      '\ntrainX :',cAX_train.shape,           '     testX :',cAX_test.shape,
      '\ntrainY :',y_train.shape,           '     testY :',y_test.shape,)
# # ######################################
batch_size_list = [128,256]

for batch_size in batch_size_list:
    
    # run_code(build_cnn1D(),batch_size,'t_wCnn1d_{}'.format(batch_size))
    # run_code(build_cnn_cnn(),batch_size,'t_wCnnCnn_{}'.format(batch_size))
    # run_code(build_cnn_lstm(),batch_size,'t_wCNN_LSTM_{}'.format(batch_size))
    # run_code(build_lstmlstm(),batch_size,'t_wLSTMLSTM_{}'.format(batch_size))
    
    run_code_alone(build_autolstm_original(),batch_size,'t_ori_PCA_cA_wAuto_{}'.format(batch_size))
    run_code_alone(build_cnn1d_original(),batch_size,'t_ori_PCA_cA_wCNN_{}'.format(batch_size))