from DLtools.Trial_evaluation_rec import record_list_result
from DLtools.Data import instant_data,station_sel
from DLtools.feature_sel import call_mar

import numpy as np
import matplotlib.pyplot as plt
import os 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
# from keras.utils.vis_utils import plot_model

import pywt
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Flatten,LSTM,RepeatVector,TimeDistributed,Dropout,Conv1D,MaxPooling1D,Input
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import Callback

############ Tensorflow ##########################
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
############# Keras ###################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
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
    cAx,cDx,y = Wavsplit_series_y(data,n_past,n_future,pca)

    if pca:
        n_features = n_pca
    else:
        n_features = data.shape[1]

    print('*original x,y:', cAx.shape,cDx.shape,y.shape)
    cAx = cAx.reshape((cAx.shape[0], cAx.shape[1],n_features))
    cDx = cDx.reshape((cDx.shape[0], cDx.shape[1],n_features))
    y = y[:,:,0]
    return cAx,cDx,y
def Wavsplit_series_y(series, n_past, n_future,pca):
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
        _cA,_cD = wavelet_list(past)
        # cA_,cD_ = wavelet_list(future)

        cAx.append(_cA);cDx.append(_cD)
        y.append(future)
    return np.array(cAx),np.array(cDx), np.array(y)
def Wavsplit_series_cAcD(series, n_past, n_future,pca):
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
    cAx,cDx, cAy,cDy = list(), list(), list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        # slicing the past and future parts of the window
        past, future = pc_series[window_start:past_end, :], series[past_end:future_end, :]
        ######################## Wavelet #########################
        _cA,_cD = wavelet_list(past)
        cA_,cD_ = wavelet_list(future)

        cAx.append(_cA);cDx.append(_cD)
        cAy.append(cA_);cDy.append(cD_)
    return np.array(cAx),np.array(cDx), np.array(cAy), np.array(cDy)
def wavelet_list(signal2d):
    N,M = np.zeros(signal2d.shape),np.zeros(signal2d.shape)
    N[:,:],M[:,:] = signal2d[:,:],signal2d[:,:]
    for i in range(signal2d.shape[1]):
        coeff = pywt.swt(signal2d[:,i],wavelet='db4',level=1) 
        # coeff = pywt.dwt(past[:,i],wavelet='db4') #NOTE: exp2 change to dwt
        cA, cD =coeff[0][0],coeff[0][1] 
        N[:,i] =  cA
        M[:,i] =  cD
    return N,M
def Wavtrain_test_split_xy(input_df,pca):
    # split_date = int(len(input_df)*.7)
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
    # split_date = int(len(input_df)*.7)
    train,test = input_df[:split_date],input_df[split_date:]
    X_train, y_train = split_xy(train,n_past,n_future,pca)
    X_test, y_test = split_xy(test,n_past,n_future,pca)
    return  X_train, y_train, X_test, y_test
###### SETTING AREA ################
def call_data():
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
def history_plot(history_model,name):   
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot (history_model.history['loss'])
    ax.plot (history_model.history['val_loss'])
    ax.set_title ('model loss:{}'.format(name))
    ax.set_xlabel('epoch')
    ax.legend(['train','val'],loc='upper left')
    fig.savefig(save_path+'/loss_{}.png'.format(name), dpi=50, bbox_inches='tight') 
    fig.clear()
    plt.close(fig)
############# HYBRID DEEP LEARNING ################################
def build_cnn_auto():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the LSTM for approximate
    x = layers.Conv1D(filters=128, kernel_size=2, activation='relu',name="cA_CNN")(inputA)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation='relu')(x)
    
    #y is the EN_LSTM for detail  
    y = layers.LSTM(200, activation="relu",name="cD_LSTM")(inputD)
    y = layers.BatchNormalization()(y)
    y = layers.RepeatVector(n_future)(y)
    y = layers.LSTM(200, activation="relu", return_sequences=True)(y)
    y = layers.BatchNormalization()(y)
    y = layers.TimeDistributed(layers.Dense(100, activation='relu'))(y)
    y = layers.TimeDistributed(layers.Dense(100))(y)
    y = layers.LSTM(100, activation="sigmoid")(y)
    
    #combining 2 lstm
    com = layers.concatenate([x, y])
    z = layers.Dense(n_future)(com)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    return model
def build_cnn_auto_colab():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the LSTM for approximate
    x = layers.Conv1D(filters=128, kernel_size=2, activation='relu',name="cA_CNN")(inputA)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation='relu')(x)
    
    #y is the EN_LSTM for detail  
    y = layers.LSTM(200, activation="relu",name="cD_LSTM")(inputD)
    # y = layers.BatchNormalization()(y)
    y = layers.RepeatVector(n_future)(y)
    y = layers.LSTM(200, activation="relu", return_sequences=True)(y)
    # y = layers.BatchNormalization()(y)
    y = layers.TimeDistributed(layers.Dense(100, activation='relu'))(y)
    y = layers.TimeDistributed(layers.Dense(50))(y)
    y = layers.LSTM(100, activation="sigmoid")(y)
    
    #combining 2 lstm
    com = layers.concatenate([x, y])
    z = layers.Dense(n_future)(com)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    return model
def build_cAcD_auto():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, n_features), name="cA")
    inputD = keras.Input(shape=(n_past, n_features), name="cD")
    
    com = layers.Concatenate(1)([inputA, inputD])
    encode = layers.LSTM(200, activation="relu", return_sequences=False)(com)
    decode = layers.RepeatVector(n_future)(encode)
    decode = layers.LSTM(200, activation="relu", return_sequences=True)(decode)
    decode = layers.TimeDistributed(layers.Dense(100, activation='relu'))(decode)
    decode = layers.TimeDistributed(layers.Dense(1))(decode)
    out = layers.Reshape((-1,n_future))(decode)

    model = keras.Model(inputs=[inputA, inputD], outputs=out )
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
    z = layers.Dense(n_future)(com)

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
    z = layers.Dense(n_future)(com)
    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(optimizer=my_optimizer, loss='mse')    
    model.summary()
    return model
def build_cnn_cnn():
    global n_past,n_future,n_features
    inputA = keras.Input(shape=(n_past, int(n_features)), name="cA")
    inputD = keras.Input(shape=(n_past, int(n_features)), name="cD")
    #x is the CNN for approximate
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu',name="cA_CNN")(inputA)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(100, activation='relu')(x)

    #y is the CNN for detail  
    y = layers.Conv1D(filters=64, kernel_size=2, activation='relu',name="cD_CNN")(inputD)
    y = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(y)
    y = layers.MaxPooling1D(pool_size=2)(y)
    y = layers.Flatten()(y)
    y = layers.Dense(100, activation='relu')(y)
    y = layers.Dropout(0.2)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(100, activation='relu')(y)
    #combining 2 lstm
    com = layers.concatenate([x, y])
    z = layers.Dense(n_future)(com)

    model = keras.Model(inputs=[inputA, inputD], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    return model
def run_code(model,batch_size,syn,minmaxscaler,flag_pca):
    global target,mode
    syn= syn+'_wcAcD'
    
    if flag_pca==True: syn= syn+'_PCA'
    else: syn= syn+'_NoPCA'

    ################### Scale #######################
    df_mars,scaler_tar = call_data()   
    cAX_train,cDX_train, y_train, cAX_test,cDX_test, y_test = Wavtrain_test_split_xy(df_mars,pca=flag_pca)
    ##############Scale or not####################
    if minmaxscaler:
        scaler = MinMaxScaler().fit(flatten(cAX_train))
        cAX_train = scale(cAX_train, scaler)
        scaler = MinMaxScaler((0,1)).fit(flatten(cDX_train))
        cDX_train = scale(cDX_train, scaler)
        scaler = MinMaxScaler().fit(flatten(cAX_test))
        cAX_test = scale(cAX_test, scaler)
        scaler = MinMaxScaler((0,1)).fit(flatten(cDX_test))
        cDX_test = scale(cDX_test, scaler)

        scaler_ytrain = MinMaxScaler().fit(y_train)
        y_train = scaler_ytrain.transform(y_train)
        scaler_ytest = MinMaxScaler().fit(y_test)
        y_test = scaler_ytest.transform(y_test)
        syn= syn+'_MinMaxLayer(XY)'
    else: pass
    #################################################
    print('*'*20,syn,'*'*20)
    print("shape")
    print(cAX_test.shape,cDX_test.shape,y_test.shape)
    #################################################
    Xtrain_,Xtest_ = [cAX_train, cDX_train],[cAX_test, cDX_test]
    validataion = (Xtest_,y_test)
    history = model.fit(x=Xtrain_, y=y_train,epochs=epochs,batch_size=batch_size,verbose=verbose,validation_data = validataion,callbacks=callbacks)
    history_plot(history,syn)

    #################################################
    trainPredict = model.predict(Xtrain_).astype('float32')
    testPredict = model.predict(Xtest_).astype('float32')
    trainPredict,testPredict = trainPredict.reshape(y_train.shape),testPredict.reshape(y_test.shape)
    
    if minmaxscaler:
        y_train_ori = scaler_ytrain.inverse_transform(y_train)
        trainPredict = scaler_ytrain.inverse_transform(trainPredict)
        y_test_ori = scaler_ytest.inverse_transform(y_test)
        testPredict = scaler_ytest.inverse_transform(testPredict)
    else:    y_train_ori,y_test_ori =y_train,y_test

    model.save(save_path+'/{}.h5'.format(syn))
    record_list_result(syn,df_mars,mode,y_train_ori,y_test_ori,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
    
####### ORIGINAL DEEP LEARNING ##########################
def build_ann_original():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Flatten()(input)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(n_future)(x)
    
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.compile(optimizer=my_optimizer, loss='mse')    
    model.summary()
    return model
def build_lstm_original():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
def build_autolstm_original():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.RepeatVector(n_future)(x)
    x = layers.LSTM(200, activation='relu',return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(100, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(1))(x)
    x = layers.Reshape((-1,n_future))(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()    
    return model
def build_cnn1d_original():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu',name="cA_CNN")(input)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer=my_optimizer, loss='mse')    
    model.summary()
    return model
def run_code_alone(model,batch_size,syn,cAcD,minmaxscaler,flag_pca):
    global target,mode,callbacks    
    df_mars,scaler_tar = call_data()

    ##############Wavelet or not############
    if cAcD==True:
        X_train,cDX_train, y_train, X_test,cDX_test, y_test = Wavtrain_test_split_xy(df_mars,pca=flag_pca)
        if flag_pca==True:        syn= syn+'_wcA_PCA'
        else: syn=syn+'_wcA_noPCA'

    else: 
        X_train, y_train, X_test, y_test = train_test_split_xy(df_mars,pca=flag_pca)
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

    
    print('*'*20,syn,'*'*20)
    print("shape")
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)
    history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    ########### plot loss ########################
    history_plot(history,syn)
    #################################################
    trainPredict = model.predict(X_train).astype('float32')
    testPredict = model.predict(X_test).astype('float32')
    trainPredict,testPredict = trainPredict.reshape(y_train.shape),testPredict.reshape(y_test.shape)
    # y_train_ori = scaler_tar.inverse_transform(y_train)
    # trainPredict = scaler_tar.inverse_transform(trainPredict)
    # y_test_ori = scaler_tar.inverse_transform(y_test)
    # testPredict = scaler_tar.inverse_transform(testPredict)
    y_train_ori,y_test_ori =y_train.astype('float32'),y_test.astype('float32')
    model.save(save_path+'/{}.h5'.format(syn))
    record_list_result(syn,df_mars,mode,y_train_ori,y_test_ori,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
    return 
###############################################################
def run_yolo(call,batch,minmax=False,cAcD=True,flag_pca=True):
    batch_size= batch
    if call =='auto':   run_code_alone(build_autolstm_original(),batch_size,'Auto_Tin{}_b{}'.format(n_past,batch_size),cAcD=cAcD,minmaxscaler=minmax,flag_pca=flag_pca)
    elif call =='cnn': run_code_alone(build_cnn1d_original(),batch_size,'CNN_Tin{}_b{}'.format(n_past,batch_size),cAcD=cAcD,minmaxscaler=minmax,flag_pca=flag_pca)
    elif call=='ann': run_code_alone(build_ann_original(),batch_size,'ANN_Tin{}_b{}'.format(n_past,batch_size),cAcD=cAcD,minmaxscaler=minmax,flag_pca=flag_pca) 
    elif call=='lstm': run_code_alone(build_lstm_original(),batch_size,'LSTM_Tin{}_b{}'.format(n_past,batch_size),cAcD=cAcD,minmaxscaler=minmax,flag_pca=flag_pca) 
    
    elif call=='cnnlstm':run_code(build_cnn_lstm(),batch_size,'wsp_CNNLSTM_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)
    elif call=='annann':run_code(build_Waveann(),batch_size,'wANNANN_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)
    elif call=='cAcDauto':run_code(build_cAcD_auto(),batch_size,'wAUTO_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)
    
    elif call=='cnnauto':run_code(build_cnn_auto(),batch_size,'wCNN[bat]auto[sig]_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)
    elif call=='cnnauto_colab':run_code(build_cnn_auto_colab(),batch_size,'w_CNN[bat]auto[sig]_colab_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)

    elif call=='cnncnn':run_code(build_cnn_cnn(),batch_size,'wCNNCNN_Tin{}_b{}'.format(n_past,batch_size),minmaxscaler=minmax,flag_pca=flag_pca)
#####################################################
mode='hour'
if mode =='hour': n_past,n_future = 96,72 #NOTE chang to 24 in-72 out
elif mode =='day': n_past,n_future = 60,30
st = 'CPY012'
# #Full
# split_date = '2016-01-18'
# target,start_p,stop_p,host_path=station_sel(st,mode)
##################################
# *********************2 Yr trail**********************
split_date = '2015-06-11'
stop_p = '2016/02/01'
n_pca = 6
n_features = 15
target,start_p,_,host_path=station_sel(st,mode)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#################################
my_optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
# my_optimizer = 'adam'
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-5 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]
verbose, epochs = 1, 100
####################################################


# save_path =host_path+'/Baseline_{}-{}'.format(n_past,n_future)
save_path =host_path+'/Hybrid_Yscale'
if not os.path.exists(save_path):
    os.makedirs(save_path)
#####################################################
flag_pca=True
if flag_pca: 
    n_features = n_pca
    minmax=False
else:
    minmax=True
# ************* Trial  *************
run_yolo('cnnauto',128,minmax=minmax,flag_pca=flag_pca)
run_yolo('annann',128,minmax=minmax,flag_pca=flag_pca)
run_yolo('cnnlstm',128,minmax=minmax,flag_pca=flag_pca)

run_yolo('cnnauto',128,minmax=True,flag_pca=flag_pca)
run_yolo('annann',128,minmax=True,flag_pca=flag_pca)
run_yolo('cnnlstm',128,minmax=True,flag_pca=flag_pca)



# =================================================================
flag_pca=False
n_features = 15
if flag_pca: 
    n_features = n_pca
    minmax=False
else:
    minmax=True

run_yolo('cnnauto',128,minmax=minmax,flag_pca=flag_pca)
run_yolo('annann',128,minmax=minmax,flag_pca=flag_pca)
run_yolo('cnnlstm',128,minmax=minmax,flag_pca=flag_pca)

#################### DONT DELETE #####################
# ************* BASE LINE CNN/AUTOEN *************
# run_yolo('cnn',128,minmax=minmax,cAcD=False,flag_pca=flag_pca)
# # run_yolo('ann',128,minmax=minmax,cAcD=False,flag_pca=flag_pca)
# run_yolo('lstm',128,minmax=minmax,cAcD=False,flag_pca=flag_pca)
# run_yolo('auto',128,minmax=minmax,cAcD=False,flag_pca=flag_pca)
# # ************* TRIAL IMPROVE *************
# run_yolo('cnnauto',256,minmax=minmax,flag_pca=flag_pca)
# run_yolo('annann',256,minmax=minmax,cAcD=True,flag_pca=flag_pca)
# run_yolo('cnnlstm',256,minmax=minmax,cAcD=True,flag_pca=flag_pca)