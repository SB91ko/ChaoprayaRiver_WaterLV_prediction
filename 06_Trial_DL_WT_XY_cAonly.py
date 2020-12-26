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
from tensorflow.keras.utils import plot_model
import pywt
np.random.seed(42)
############# Keras ###################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #ignore cuDNN log
#---------------------------------------
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

##--------------------------- SETTING AREA ------------------------------##
loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
# df,mode = loading.daily_instant(),'day'
if mode =='hour': n_past,n_future = 24*6,72
elif mode =='day': n_past,n_future = 60,30
st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
DLtype = 'split_wave_cAonly'
#------------ DL PARAMETER ---------------------#
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-5 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]
#my_optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)

#--------------------------- 4 Yr Edit -----------------------------------#
# host_path = './CPY012/4Yr_flood/'
# start_p='2014-10-01'
# stop_p='2017-10-01'
split_date = '2016-11-01'
#n_pca = 4

syn=''
# Yscale = False # scaler Y before put in model
allscale = True # scale X before put in model
w_std = False # standardize before wavelet transform
#-----------------------Baseline / Hybrid -----------------------------------#
save_path =host_path+'06_wtXY_cAonly'
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
        # past= DWT_seq_tran(past)
        # future = DWT_seq_tran(future)
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)
def split_xy(data,n_past,n_future):
    x,y = split_series(data.values,n_past,n_future)
    x = x.reshape((x.shape[0], x.shape[1],n_features))
    y = y[:,:,0]
    return x,y

##--------------added Wavelet ---------------##
def wavelet_t(series,std):
    name = series.name
    idx = series.index
    signal = series.values
    if std: signal = (signal - signal.mean())/(signal.std())

    coeff = pywt.swt(signal,'db4',level=3)
    coeff = np.array(coeff)

    cA3,cD3 = coeff[0][0],coeff[0][1]
    # _,cD2 = coeff[1][0],coeff[1][1]
    # _,cD1 = coeff[2][0],coeff[2][1]
    #----------------------------
    dict_data = {
            # name:signal,
            '{}_cA3'.format(name): cA3,
            '{}_cD3'.format(name): cD3,
            # '{}_cD2'.format(name): cD2,
            # '{}_cD1'.format(name): cD1
            }
    wt = pd.DataFrame(dict_data,dtype='float32',index=idx)
    return wt
def df_wavelet(df,std):
    global syn
    syn = syn+'[wav]'
    wav_df = pd.DataFrame()
    for col in df:
        a = wavelet_t(df[col],std)
        wav_df = pd.concat([wav_df,a],axis=1)
    return wav_df
def inverse_WT(coeffs_list):
    """
    coeffs = [cA4,cD4,cD3....cD1]
    """
    wav=list()
    for i in range(coeffs_list[0].shape[0]):
        iwave = pywt.iswt([coeff[i,:] for coeff in coeffs_list],'db4')
        wav.append(iwave)
    return np.array(wav)
#---------------------- MODEL ------------------------------------#
def build_mod_cnn1d():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(input)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    #x = layers.BatchNormalization()(x) # added
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(500)(x)
    x = layers.Dense(200)(x)
    x = layers.Dense(n_future, activation='relu')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
    return model
def build_mod2_cnn1d():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(input)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x) # added
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(500)(x)
    x = layers.Dense(200)(x)
    x = layers.Dense(n_future, activation='relu')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
    return model

def build_orimod_cnn1d():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(input)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    #x = layers.BatchNormalization()(x) # added
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(500)(x)
    x = layers.Dense(200)(x)
    x = layers.Dense(n_future, activation='linear')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
    return model

def build_lstm():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    # x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.CuDNNLSTM(200,input_shape=(n_past, n_features),return_sequences=False)(input)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
    return model
def build_cnn1d():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(input)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
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
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
    return model
def build_ann():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Flatten()(input)
    x = layers.Dense(2000, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dense(n_future,activation='linear')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
    return model
def build_lstm_v2():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    # x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.CuDNNLSTM(300,return_sequences=True)(input)
    x = layers.CuDNNLSTM(300,return_sequences=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
    return model
#---------------------------#    
def run_code_editv2(model,X,Y,batch_size):
    global target,mode
    verbose, epochs = 1, 100
    history = model.fit(X[0],Y[0],epochs=epochs,validation_data=(X[1],Y[1]),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
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
    history_plot(history,syn)    

    trainPredict = model.predict(X[0])
    testPredict = model.predict(X[1])
    # # ---------- Inverse ------------------#
    # # if Yscale:
    # #     y_train = scaler_tar.inverse_transform(y_train)
    # #     trainPredict = scaler_tar.inverse_transform(trainPredict.reshape(y_train.shape))
    # #     y_test = scaler_tar.inverse_transform(y_test)
    # #     testPredict = scaler_tar.inverse_transform(testPredict.reshape(y_test.shape))
    # #---------------result-------------------- #
    # record_list_result(syn,df,mode,Y[0],Y[1],trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
    return trainPredict,testPredict
#------------------------- Main ---------------------------------#
df = df[start_p:stop_p]
data = df
data['Day'] = data.index.dayofyear #add day
data = data.interpolate(limit=300000000,limit_direction='both').astype('float32')#interpolate neighbor first, for rest NA fill with mean() #.apply(lambda x: x.fillna(x.mean()),axis=0)

cutoff=.3
data_mar = call_mar(data,target,mode,cutoff=cutoff)
data_mar = move_column_inplace(data_mar,target,0)
n_features = len(data_mar.columns)

if w_std: syn=syn+'[w_std]'
wdata = df_wavelet(data_mar,w_std)

def extract_target_signal(data,staion):
    def st_column_select(data,staion):
        data_col = [i.split("_") for i in data.columns]
        _col =list()
        for i in data_col:
            if i[0]==staion : 
                i='_'.join(i)
                _col.append(i)
        return data[_col]
    cpy = st_column_select(data,staion)
    #Put in function latter
    train,test = cpy[:split_date],cpy[split_date:]
    x,y = split_series(train.values,n_past,n_future)
    x = x.reshape((x.shape[0], x.shape[1],2))
    yA3_train = y[:,:,0]
    yD3_train = y[:,:,1]
    #yD2_train = y[:,:,2]
    #yD1_train = y[:,:,3]
    #--------------------------------------------
    x,y = split_series(test.values,n_past,n_future)
    x = x.reshape((x.shape[0], x.shape[1],2))
    yA3_test = y[:,:,0]
    yD3_test = y[:,:,1]
    #yD2_test = y[:,:,2]
    #yD1_test = y[:,:,3]
    #------------------------------------------
    yA3 = [yA3_train,yA3_test]
    yD3 = [yD3_train,yD3_test]
    #yD2 = [yD2_train,yD2_test]
    #yD1 = [yD1_train,yD1_test]
    return yA3,yD3
yA3_ori,yD3_ori= extract_target_signal(wdata,'CPY012')
#------------------------------------------------------
if allscale:
    syn = syn+'[X_sc]'  
    scaler = MinMaxScaler()
    wdata[wdata.columns] = scaler.fit_transform(wdata[wdata.columns])


##-------add ------- Select degree 
def syn_column_select(data,sel_word):
    data_col = [i.split("_") for i in data.columns]
    _col =list()
    for i in data_col:
        if i[2]==sel_word : 
            i='_'.join(i)
            _col.append(i)
    return data[_col]

data_cA3 = syn_column_select(wdata,'cA3')
data_cD3 =syn_column_select(wdata,'cD3')
#data_cD2 = syn_column_select(wdata,'cD2')
#data_cD1 = syn_column_select(wdata,'cD1')
#--------------------------
def autosplit(data):    
    ##----------- train test split 
    train,test = data[:split_date],data[split_date:]
    # ##--------- Wavelet_traintest
    X_train, y_train = split_xy(train,n_past,n_future)
    X_test, y_test= split_xy(test,n_past,n_future)
    return [X_train,X_test],[y_train,y_test]
#--------------------------------

# def run_code(model,batch_size,syn):
#     global target,mode,df,X_train,y_train,X_test,y_test,wX_train,wX_test
#     verbose, epochs = 1, 100
#     history = model.fit([X_train,wX_train],y_train,epochs=epochs,validation_data=([X_test,wX_test],y_test),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
#     def history_plot(history_model,name):   
#         fig, ax = plt.subplots(figsize=(6.4, 4.8))
#         ax.plot (history_model.history['loss'])
#         ax.plot (history_model.history['val_loss'])
#         ax.set_title ('model loss:{}'.format(name))
#         ax.set_xlabel('epoch')
#         ax.legend(['train','val'],loc='upper left')
#         fig.savefig(save_path+'/loss_{}.png'.format(name), dpi=100, bbox_inches='tight') 
#         fig.clear()
#         plt.close(fig)
#     history_plot(history,syn)    
    
#     trainPredict = model.predict([X_train,wX_train])
#     testPredict = model.predict([X_test,wX_test])
#     # ---------- Inverse ------------------#
#     if Yscale:
#         y_train = scaler_tar.inverse_transform(y_train)
#         trainPredict = scaler_tar.inverse_transform(trainPredict.reshape(y_train.shape))
#         y_test = scaler_tar.inverse_transform(y_test)
#         testPredict = scaler_tar.inverse_transform(testPredict.reshape(y_test.shape))
#     record_list_result(syn,df,mode,y_train,y_test,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)

def wav_dl_wav_run(model_list,batch,syn):
	X,_=autosplit(data_cA3)
	cA3ytrain,cA3ytest = run_code_editv2(model_list,X,yA3_ori,batch)
	_,Y=autosplit(data_mar)
	y_train,y_test = Y[0],Y[1]

	scaler_t = StandardScaler()
	y_train = scaler_t.fit_transform(y_train)
	y_train = scaler_t.inverse_transform(y_train)

	scaler = StandardScaler()
	trainPredict = scaler.fit_transform(cA3ytrain)
	trainPredict = scaler_t.inverse_transform(trainPredict)

	scaler_t = StandardScaler()
	y_test = scaler_t.fit_transform(y_test)
	y_test = scaler_t.inverse_transform(y_test)


	scaler = StandardScaler()
	testPredict = scaler.fit_transform(cA3ytest)
	testPredict = scaler_t.inverse_transform(testPredict)

	record_list_result(syn,df,DLtype,y_train,y_test,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
	return
##----------- Run Experiment -----------------##
for batch_size in [16,32]:
    # model = [build_cnn1d(),build_cnn1d(),build_cnn1d(),build_cnn1d()]
    model = build_cnn1d()
    wav_dl_wav_run(model,batch_size,'w(cA)CNN_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
    model = build_lstm()
    wav_dl_wav_run(model,batch_size,'w(cA)lstm_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
    #model = build_lstm_v2()
    #wav_dl_wav_run(model,batch_size,'w(cA)lstm(big)_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
    model = build_ann()
    wav_dl_wav_run(model,batch_size,'w(cA)ANN_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
