from DLtools.Trial_evaluation_rec import record_list_result
from DLtools.Data import instant_data,station_sel
from DLtools.feature_sel import call_mar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
import pywt
np.random.seed(42)
############# Keras ###################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

##--------------------------- SETTING AREA ------------------------------##
loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
# df,mode = loading.daily_instant(),'day'
if mode =='hour': n_past,n_future = 24*7,72
elif mode =='day': n_past,n_future = 60,30
st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
#------------ DL PARAMETER ---------------------#
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-5 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]
my_optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)

#--------------------------- 2 Yr Edit -----------------------------------#
host_path = './CPY012/2Yr_flood/'
start_p = '2016-01-01'
split_date = '2017-05-10'
stop_p = '2018-01-01'
n_pca = 7
#-----------------------Baseline / Hybrid -----------------------------------#
save_path =host_path+'Wavelet_exp3'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)
#----------------------------------------------------------#
#Split XY

def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df
def split_series(series, n_past, n_future):
    # n_past ==> no of past observations
    # n_future ==> no of future observations 
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)
def split_xy(data,n_past,n_future):
    x,y = split_series(data.values,n_past,n_future)
    x = x.reshape((x.shape[0], x.shape[1],n_features))
    y = y[:,:,0]
    return x,y
##--------------added Wavelet ---------------##
def SWT_seq_tran(signal2d):
    N,M = np.zeros(signal2d.shape),np.zeros(signal2d.shape)
    N[:,:],M[:,:] = signal2d[:,:],signal2d[:,:]
    for i in range(signal2d.shape[1]):
        coeff = pywt.swt(signal2d[:,i],wavelet='db4',level=1)  #NOTE exp swt
        # coeff = pywt.dwt(signal2d[:,i],wavelet='db4') #NOTE: exp2 change to dwt
        cA, cD =coeff[0][0],coeff[0][1] 
        N[:,i] =  cA
        M[:,i] =  cD
    return N,M
def SWTsplit_series_cAcD(series, n_past, n_future):
    # n_past ==> no of past observations
    # n_future ==> no of future observations
    cAx,cDx, cAy,cDy = list(), list(), list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = series[window_start:past_end, :], series[past_end:future_end, :]
        ######################## Wavelet #########################
        _cA,_cD = SWT_seq_tran(past)
        cA_,cD_ = SWT_seq_tran(future)

        cAx.append(_cA);cDx.append(_cD)
        cAy.append(cA_);cDy.append(cD_)
    return np.array(cAx),np.array(cDx), np.array(cAy), np.array(cDy)
def SWTsplit_xy(data,n_past,n_future):
    cAx,cDx,cAy,cDy = SWTsplit_series_cAcD(data.values,n_past,n_future)
    cAx = cAx.reshape((cAx.shape[0], cAx.shape[1],n_features))
    cDx = cDx.reshape((cDx.shape[0], cDx.shape[1],n_features))
    
    cAy= cAy[:,:,0]
    cDy= cDy[:,:,0]
    return cAx,cDx,cAy,cDy
#----------------------------------------------------------#
def build_lstm():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    # x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.CuDNNLSTM(200,input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
def build_ende_lstm():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    # x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.CuDNNLSTM(200,  input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.RepeatVector(n_future)(x)
    # x = layers.LSTM(200, activation='relu',return_sequences=True)(x)
    x = layers.CuDNNLSTM(200, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(100, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(1))(x)
    out = layers.Reshape((-1,n_future))(x)
    model = keras.Model(inputs=[input], outputs=out)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
       
    return model
def build_cnn1d():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    return model
#------------ Hybrid
def build_cnn_autolstm():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the CNN for approximate
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputA)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(50, activation='relu')(x)
    # x = layers.Dense(n_future)(x)
    #y is the LSTM for detail  
    y = layers.CuDNNLSTM(200, return_sequences=False)(inputD)
    y = layers.RepeatVector(n_future)(y)
    y = layers.CuDNNLSTM(200, return_sequences=True)(y)
    y = layers.TimeDistributed(layers.Dense(100, activation='relu'))(y)
    y = layers.TimeDistributed(layers.Dense(50))(y)
    y = layers.CuDNNLSTM(50)(y)
    # y = layers.Reshape((-1,50))(y)
    # y = layers.Dense(50,activation='sigmoid')(y)

    #combining 2 lstm
    com = layers.concatenate([x, y])
    # z = LSTM(200, activation='relu', return_sequences=False)(com)
    # z = Dense(100, activation="relu")
    z = layers.Dense(n_future)(com)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
def build_cnn_lstm():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the CNN for approximate
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputA)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(100, activation='relu')(x)


    #y is the LSTM for detail  
    y = layers.CuDNNLSTM(200,input_shape=(n_past, n_features),return_sequences=False)(inputD)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(100,activation = 'relu')(y)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10,activation='sigmoid')(x)
    #combining 2 lstm
    com = layers.concatenate([x, y])
    # z = LSTM(200, activation='relu', return_sequences=False)(com)
    # z = Dense(100, activation="relu")
    z = layers.Dense(n_future)(com)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
def build_cnn_cnn():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the CNN for approximate
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputA)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(50)(x)
    # x = layers.BatchNormalization()(x)


    #y is the CNN for detail  
    y = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputD)
    y = layers.MaxPooling1D(pool_size=2)(y)
    y = layers.Flatten()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(100, activation='relu')(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(50)(y)
    #combining 2 lstm
    com = layers.concatenate([x, y])
    z = layers.Dense(n_future)(com)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer='adam')
    return model   
def build_wann():
    inputA = keras.Input(shape=(n_past, n_features), name="cA")
    inputD = keras.Input(shape=(n_past, n_features), name="cD")
    com = layers.Concatenate(1)([inputA, inputD])
    x = layers.Flatten()(com)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(n_future)(x)
    
    model = keras.Model(inputs=[inputA,inputD], outputs=x)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def run_code(model,batch_size,syn):
    global target,mode,df,cX_train,cX_test,cy_train,cy_test
    verbose, epochs = 1, 100
    history = model.fit(cX_train[0],cy_train[0],epochs=epochs,validation_data=(cX_test[0],cy_test[0]),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    trainPredic_cA = model.predict(cX_train[0])
    testPredict_cA = model.predict(cX_test[0])
    
    history = model.fit(cX_train[1],cy_train[1],epochs=epochs,validation_data=(cX_test[1],cy_test[1]),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    trainPredict_cD = model.predict(cX_train[1])
    testPredict_cD = model.predict(cX_test[1])
    
    trainPredic_cA = scaler_cA.inverse_transform(trainPredic_cA)
    trainPredict_cD = scaler_cD.inverse_transform(trainPredict_cD)
    testPredict_cA = scaler_cAt.inverse_transform(testPredict_cA)
    testPredict_cD = scaler_cDt.inverse_transform(testPredict_cD)

    def iwave(cA,cD):
        iwave = list()
        for i in range(cA.shape[0]):
            # wave = pywt.idwt(cA[i,:],cD[i,:], 'db4')
            wave = pywt.iswt((cA[i,:],cD[i,:]), 'db4')
            iwave.append(wave)
        return np.array(iwave)
    trainPredict = iwave(trainPredic_cA,trainPredict_cD)
    testPredict = iwave(testPredict_cA,testPredict_cD)

    def history_plot(history_model,name):   
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot (history_model.history['loss'])
        ax.plot (history_model.history['val_loss'])
        ax.set_title ('model loss:{}'.format(name))
        ax.set_xlabel('epoch')
        ax.legend(['train','val'],loc='upper left')
        fig.savefig(save_path+'/loss_{}.png'.format(name), dpi=100, bbox_inches='tight') 
        fig.clear()
        plt.close(fig)
    # history_plot(history,syn)    


    #---------- Inverse ------------------#
    # y_train = scaler_tar.inverse_transform(y_train)
    # trainPredict = scaler_tar.inverse_transform(trainPredict.reshape(y_train.shape))
    # y_test = scaler_tar.inverse_transform(y_test)
    # testPredict = scaler_tar.inverse_transform(testPredict.reshape(y_test.shape))

    record_list_result(syn,df,mode,y_train,y_test,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)

#------------------------- Main ---------------------------------#
df = df[start_p:stop_p]
data = df
data['Day'] = data.index.dayofyear #add day
data = data.interpolate(limit=300000000,limit_direction='both').astype('float32')#interpolate neighbor first, for rest NA fill with mean() #.apply(lambda x: x.fillna(x.mean()),axis=0)

cutoff=.3
data_mar = call_mar(data,target,mode,cutoff=cutoff)
data_mar = move_column_inplace(data_mar,target,0)
n_features = len(data_mar.columns)
##----------- SCALE
def Preprocess_pca(input):
    
    pipe = Pipeline([('scaler', StandardScaler())])#, ('pca',PCA(n_components =n_pca))])
    scaler = pipe.fit(input)
    sc_input = pipe.transform(input)
    sc_input = pd.DataFrame(sc_input, index=input.index)
    return scaler,sc_input

# X data
_,sc_data = Preprocess_pca(data_mar)
# Y data
scaler_tar = MinMaxScaler()
scaler_tar.fit(data_mar[target].to_numpy().reshape(-1,1))
data_mar[target] = scaler_tar.transform(data_mar[target].to_numpy().reshape(-1,1))

##----------- train test split 
sc_train,sc_test = sc_data[:split_date],sc_data[split_date:]
train,test = data_mar[:split_date],data_mar[split_date:]
##--------- Keep original
_, y_train = split_xy(train,n_past,n_future)
_, y_test = split_xy(test,n_past,n_future)
##--------- Wavelet_traintest
# n_features=n_pca
cAxTrain,cDxTrain,cAyTrain,cDyTrain = SWTsplit_xy(sc_train,n_past,n_future)
cAxTest,cDxTest,cAyTest,cDyTest = SWTsplit_xy(sc_test,n_past,n_future)
#----------------- minmax scale for Deeplearning 
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
scaler = MinMaxScaler().fit(flatten(cAxTrain))
cAX_train = scale(cAxTrain, scaler)
scaler = MinMaxScaler().fit(flatten(cDxTrain))
cDX_train = scale(cDxTrain, scaler)
scaler = MinMaxScaler().fit(flatten(cAxTest))
cAX_test = scale(cAxTest, scaler)
scaler = MinMaxScaler().fit(flatten(cDxTest))
cDX_test = scale(cDxTest, scaler)

scaler_cA = MinMaxScaler()
cAyTrain = scaler_cA.fit_transform(cAyTrain)
scaler_cD = MinMaxScaler()
cDyTrain = scaler_cD.fit_transform(cDyTrain)
scaler_cAt = MinMaxScaler()
cAyTest = scaler_cAt.fit_transform(cAyTest)
scaler_cDt = MinMaxScaler()
cDyTest = scaler_cDt.fit_transform(cDyTest)
##----------- Run Experiment -----------------##
cX_train = [cAxTrain,cDxTrain]
cX_test = [cAxTest,cDxTest]
cy_train = [cAyTrain,cDyTrain]
cy_test = [cAyTest,cDyTest]
for batch_size in [32,64]:
    run_code(build_lstm(),batch_size,'wLSTMs_MAR{}_b{}_Tin{}'.format(cutoff,batch_size,n_past))
    run_code(build_cnn1d(),batch_size,'wCNNs_MAR{}_b{}_Tin{}'.format(cutoff,batch_size,n_past))
    # run_code(build_cnn_lstm(),batch_size,'wCNN-CuDNNLSTM_v3_MAR{}_b{}_Tin{}'.format(cutoff,batch_size,n_past))
    # run_code(build_cnn_autolstm(),batch_size,'wCNN-AutoCuDNNLSTM_MAR{}_b{}_Tin{}'.format(cutoff,batch_size,n_past))
