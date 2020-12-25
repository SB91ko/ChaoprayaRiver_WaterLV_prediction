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

syn='lstm'
Yscale = True # scaler Y before put in model 
allscale = True # scale X before put in model
w_std = False # standardize before wavelet transform
#-----------------------Baseline / Hybrid -----------------------------------#
save_path =host_path+'07_wave_XY_throw'
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
    _,cD2 = coeff[1][0],coeff[1][1]
    _,cD1 = coeff[2][0],coeff[2][1]
    #----------------------------
    dict_data = {
            # name:signal,
            '{}_cA3'.format(name): cA3,
            '{}_cD3'.format(name): cD3,
            '{}_cD2'.format(name): cD2,
            '{}_cD1'.format(name): cD1}
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
#----------------------------------------------------------#
def build_lstm():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    # x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.CuDNNLSTM(200,input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
def build_ende_lstm():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    # x = layers.CuDNNLSTM(200,  input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.RepeatVector(n_future)(x)
    x = layers.LSTM(200, activation='relu',return_sequences=True)(x)
    # x = layers.CuDNNLSTM(200, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(100, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(1))(x)
    out = layers.Reshape((-1,n_future))(x)
    model = keras.Model(inputs=[input], outputs=out)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
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
    return model
#-------------Hybrid

def build_cnnlstm3():
    global n_past,n_future,n_features
    inputA3 = keras.Input(shape=(n_past, n_features), name="cA3")
    inputD3 = keras.Input(shape=(n_past, n_features), name="cD3")
    inputD2 = keras.Input(shape=(n_past, n_features), name="cD2")
    inputD1 = keras.Input(shape=(n_past, n_features), name="cD1")
    #x is the CNN for approximate
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(inputA3)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(144, activation='relu')(x)
    x = layers.Dense(n_future)(x)

    #y is the LSTM for detail  
    b = layers.CuDNNLSTM(200,return_sequences=False)(inputD3)
    b = layers.Dropout(0.2)(b)
    b = layers.Dense(100)(b)
    b = layers.Dropout(0.2)(b)
    b = layers.Dense(n_future,activation='tanh')(b)
    # b = layers.LeakyReLU()(b)
    
    c = layers.CuDNNLSTM(200,return_sequences=False)(inputD2)
    c = layers.Dropout(0.2)(c)
    c= layers.Dense(100)(c)
    c = layers.Dropout(0.2)(c)
    c = layers.Dense(n_future,activation='tanh')(c)
    # c = layers.LeakyReLU()(c)

    d = layers.CuDNNLSTM(200,return_sequences=False)(inputD1)
    d = layers.Dropout(0.2)(d)
    d = layers.Dense(100)(d)
    d = layers.Dropout(0.2)(d)
    d = layers.Dense(n_future,activation='tanh')(d)
    # d = layers.LeakyReLU()(d)
    #combining 2 lstm
    com = layers.concatenate([x,b,c,d])
    # z = LSTM(200, activation='relu', return_sequences=False)(com)
    # z = Dense(100, activation="relu")
    z = layers.Dense(n_future)(com)
    model = keras.Model(inputs=[inputA3, inputD3,inputD2,inputD1], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
def build_lstm4():
    global n_past,n_future,n_features
    inputA3 = keras.Input(shape=(n_past, n_features), name="cA3")
    inputD3 = keras.Input(shape=(n_past, n_features), name="cD3")
    inputD2 = keras.Input(shape=(n_past, n_features), name="cD2")
    inputD1 = keras.Input(shape=(n_past, n_features), name="cD1")
    #x is the CNN for approximate
    a = layers.CuDNNLSTM(200,return_sequences=False)(inputA3)
    a = layers.Dropout(0.2)(a)
    a = layers.Dense(100,activation = 'relu')(a)
    a = layers.Dropout(0.2)(a)
    a = layers.Dense(n_future, activation='relu')(a)

    #y is the LSTM for detail  
    b = layers.CuDNNLSTM(200,return_sequences=False)(inputD3)
    b = layers.Dropout(0.2)(b)
    b = layers.Dense(100)(b)
    b = layers.Dropout(0.2)(b)
    b = layers.Dense(n_future,activation='tanh')(b)
    #b = layers.LeakyReLU()(b)
    
    c = layers.CuDNNLSTM(200,return_sequences=False)(inputD2)
    c = layers.Dropout(0.2)(c)
    c= layers.Dense(100)(c)
    c = layers.Dropout(0.2)(c)
    c = layers.Dense(n_future,activation='tanh')(c)
    

    d = layers.CuDNNLSTM(200,return_sequences=False)(inputD1)
    d = layers.Dropout(0.2)(d)
    d = layers.Dense(100)(d)
    d = layers.Dropout(0.2)(d)
    d = layers.Dense(n_future,activation='tanh')(d)
    #d = layers.LeakyReLU()(d)
    #combining 2 lstm
    com = layers.concatenate([a,b,c,d])
    # z = LSTM(200, activation='relu', return_sequences=False)(com)
    # z = Dense(100, activation="relu")
    z = layers.Dense(n_future)(com)
    model = keras.Model(inputs=[inputA3, inputD3,inputD2,inputD1], outputs=z)
    model.compile(loss='mse', optimizer=my_optimizer)
    model.summary()
    return model
#---------------------------#    
def run_code(model,batch_size,syn):
    global target,mode,df,X_train,y_train,X_test,y_test
    verbose, epochs = 1, 200
    history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
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
    
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    trainPredict = trainPredict.reshape(y_train.shape)
    trainPredict = testPredict.reshape(y_test.shape)
    # # ---------- Inverse ------------------#

    scaler_tar = MinMaxScaler()
    scaler_tar.fit(y_train)
    #y_train = scaler_tar.inverse_transform(y_train)
    trainPredict = scaler_tar.inverse_transform(trainPredict)
    scaler_tar = MinMaxScaler()
    scaler_tar.fit(y_test)
    #y_test = scaler_tar.inverse_transform(y_test)
    testPredict = scaler_tar.inverse_transform(trainPredict)
    print('*'*50,np.max(y_train),np.max(trainPredict))
    #---------------result-------------------- #
    record_list_result(syn,df,mode,Y[0],Y[1],trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
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
##----------- SCALE
# def Preprocess_pca(input):
#     global syn
#     syn = syn+'[pca_scminmax]'
#     pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components =n_pca)),('minmax',MinMaxScaler())])
#     scaler = pipe.fit(input)
#     sc_input = pipe.transform(input)
#     sc_input = pd.DataFrame(sc_input, index=input.index)
#     return scaler,sc_input

# #--- X data
# _,sc_data = Preprocess_pca(data_mar)

if w_std: syn=syn+'[w_std]'
wdata = df_wavelet(data_mar,w_std)
# scaler = MinMaxScaler()
# wdata[wdata.columns] = scaler.fit_transform(wdata[wdata.columns])
#--- Y data
# # SCALE
# if Yscale:
#     syn = syn+'[y_sc]'        
#     scaler_tar = MinMaxScaler()
#     scaler_tar.fit(data_mar[target].to_numpy().reshape(-1,1))
#     print(data_mar[target].to_numpy().reshape(-1,1).shape)
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
data_cD2 = syn_column_select(wdata,'cD2')
data_cD1 = syn_column_select(wdata,'cD1')

def autosplit(data):    
    ##----------- train test split 
    train,test = data[:split_date],data[split_date:]
    # ##--------- Wavelet_traintest
    X_train, y_train = split_xy(train,n_past,n_future)
    X_test, y_test= split_xy(test,n_past,n_future)
    return [X_train,X_test],[y_train,y_test]
#--------------------------------
batch_size=16
X_cA,_=autosplit(data_cA3)
X_cD3,_=autosplit(data_cD3)
X_cD2,_=autosplit(data_cD2)
X_cD1,_=autosplit(data_cD1)

X_train=[X_cA[0],X_cD3[0],X_cD2[0],X_cD1[0]]
X_test=[X_cA[1],X_cD3[1],X_cD2[1],X_cD1[1]]

_,Y=autosplit(data_mar)
y_train,y_test = Y[0],Y[1]


##----------- Run Experiment -----------------##
for batch_size in [16]:
    run_code(build_cnnlstm3(),batch_size,'wCNNCuDNNLSTMx3_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
    run_code(build_lstm4(),batch_size,'wLSTMx4_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
    