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
DLtype = '05_Lv3wave_PCA_w2'
split_date = '2016-11-01'
#------------ DL PARAMETER ---------------------#
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-5 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]
#my_optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)


#--------------------------- 2 Yr Edit -----------------------------------#
# host_path = './CPY012/2Yr_flood/'
# start_p = '2016-01-01'
# split_date = '2017-05-10'
# stop_p = '2018-01-01'

n_pca = 4
syn=''
# Yscale = False # scaler Y before put in model 
allscale = True # scale X before put in model
w_std = False # standardize before wavelet transform
#-----------------------Baseline / Hybrid -----------------------------------#
save_path =host_path+'06_wtXY_bestPCA4_allcoeff'
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
def split_xy(data,n_past,n_future,checkPCA):
    global n_features
    if checkPCA: n_features = n_pca
    
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
    cA2,cD2 = coeff[1][0],coeff[1][1]
    cA1,cD1 = coeff[2][0],coeff[2][1]
    #----------------------------
    dict_data = {
            # name:signal,
            '{}_cA3'.format(name): cA3,
            '{}_cA2'.format(name): cA2,
            '{}_cA1'.format(name): cA1,
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
def inverse_WT(coeffs_list):
    """
    coeffs = [cA4,cD4,cD3....cD1]
    """
    wav=list()
    for i in range(coeffs_list[0].shape[0]):
        iwave = pywt.iswt([coeff[i,:] for coeff in coeffs_list],'db4')
        wav.append(iwave)
    return np.array(wav)
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
    global syn
    syn = syn+'[pca]'
    pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components =n_pca))])#,('minmax',MinMaxScaler())])
    scaler = pipe.fit(input)
    sc_input = pipe.transform(input)
    colname = ['PC_{}'.format(i+1)for i in range(sc_input.shape[1])]

    sc_input = pd.DataFrame(sc_input, index=input.index,columns=colname)
    return scaler,sc_input
#--- X data
_,sc_data = Preprocess_pca(data_mar)

if w_std: syn=syn+'[w_std]'
wdata = df_wavelet(data_mar,w_std)

#----------- Y data
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
    x = x.reshape((x.shape[0], x.shape[1],6))
    yA3_train = y[:,:,0]
    yA2_train = y[:,:,1]
    yA1_train = y[:,:,2]
    yD3_train = y[:,:,3]
    yD2_train = y[:,:,4]
    yD1_train = y[:,:,5]
    #--------------------------------------------
    x,y = split_series(test.values,n_past,n_future)
    x = x.reshape((x.shape[0], x.shape[1],6))
    yA3_test = y[:,:,0]
    yA2_test = y[:,:,1]
    yA1_test = y[:,:,2]
    yD3_test = y[:,:,3]
    yD2_test = y[:,:,4]
    yD1_test = y[:,:,5]
    #------------------------------------------
    yA3 = [yA3_train,yA3_test]
    yA2 = [yA2_train,yA2_test]
    yA1 = [yA1_train,yA1_test]
    yD3 = [yD3_train,yD3_test]
    yD2 = [yD2_train,yD2_test]
    yD1 = [yD1_train,yD1_test]
    return yA3,yA2,yA1,yD3,yD2,yD1
yA3_ori,yA2_ori,yA1_ori,yD3_ori,yD2_ori,yD1_ori= extract_target_signal(wdata,'CPY012')

#------------ X data 
wdata = df_wavelet(sc_data,w_std)
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
data_cA2 = syn_column_select(wdata,'cA2')
data_cA1 = syn_column_select(wdata,'cA1')
data_cD3 =syn_column_select(wdata,'cD3')
data_cD2 = syn_column_select(wdata,'cD2')
data_cD1 = syn_column_select(wdata,'cD1')
#--------------------------
n_features = n_pca
def autosplit(data,chekcPCA=False):    
    ##----------- train test split 
    train,test = data[:split_date],data[split_date:]
    # ##--------- Wavelet_traintest
    X_train, y_train = split_xy(train,n_past,n_future,chekcPCA)
    X_test, y_test= split_xy(test,n_past,n_future,chekcPCA)
    return [X_train,X_test],[y_train,y_test]

#---------------------- MODEL ------------------------------------#
def build_mod_cnn1d():
    global n_past,n_future,n_features,n_pca
    input = keras.Input(shape=(n_past, n_pca))
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(input)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.BatchNormalization()(x) # added
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(2000, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1000, activation='relu')(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(n_future, activation='linear')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    plot_model(model, to_file='model_{}.png'.format(syn), show_shapes=True)
    return model
def build_lstm():
    global n_past,n_future,n_features,n_pca
    input = keras.Input(shape=(n_past, n_pca))
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
    plot_model(model, to_file='model_{}.png'.format(syn), show_shapes=True)
    return model
def build_cnn1d():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, n_pca))
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
    plot_model(model, to_file='model_{}.png'.format(syn), show_shapes=True)
    return model
#---------------------------#
# #---------------------- Deep MODEL ------------------------------------#
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
    x = layers.Dense(n_future, activation='linear')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    
    return model
def build_orimod_cnn1d():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(input)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    #x = layers.BatchNormalization()(x) # added
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.Conv1D(filters=64, kernel_size=2, activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1000)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(500)(x)
    x = layers.Dense(200)(x)
    x = layers.Dense(n_future, activation='linear')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    
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
    
    return model
def build_lstm_v2():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    # x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.CuDNNLSTM(400)(input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    
    return model
def build_tanh_lstm():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    # x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.CuDNNLSTM(1000)(input)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(500, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_future,activation='tanh')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model
def build_linear_lstm():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    # x = layers.LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=False)(input)
    x = layers.CuDNNLSTM(200,input_shape=(n_past, n_features),return_sequences=False)(input)
    # x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(n_future,activation='linear')(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model
#---------------------------#
def run_code_editv2(model,X,Y,batch_size):
    global target,mode
    verbose, epochs = 1, 100
    model.fit(X[0],Y[0],epochs=epochs,validation_data=(X[1],Y[1]),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    trainPredict = model.predict(X[0])
    testPredict = model.predict(X[1])

    return trainPredict,testPredict
def wav_dl_wav_run(model_list,batch,syn):
    global n_features
    check_PCA=True
    
    n_features = n_pca
    X,_=autosplit(data_cA3,check_PCA)
    cA3ytrain,cA3ytest = run_code_editv2(model_list[0],X,yA3_ori,batch)
    X,_=autosplit(data_cA2,check_PCA)
    cA2ytrain,cA2ytest = run_code_editv2(model_list[0],X,yA2_ori,batch)
    X,_=autosplit(data_cA1,check_PCA)
    cA1ytrain,cA1ytest = run_code_editv2(model_list[0],X,yA1_ori,batch)
    X,_=autosplit(data_cD3,check_PCA)
    cD3ytrain,cD3ytest = run_code_editv2(model_list[1],X,yD3_ori,batch)
    X,_=autosplit(data_cD2,check_PCA)
    cD2ytrain,cD2ytest = run_code_editv2(model_list[2],X,yD2_ori,batch)
    X,_=autosplit(data_cD1,check_PCA)
    cD1ytrain,cD1ytest = run_code_editv2(model_list[3],X,yD1_ori,batch)

    trainPredict_3 = inverse_WT([cA3ytrain,cD3ytrain,cD2ytrain,cD1ytrain])
    testPredict_3 = inverse_WT([cA3ytest,cD3ytest,cD2ytest,cD1ytest])

    trainPredict_2 = inverse_WT([cA2ytrain,cD2ytrain,cD1ytrain])
    testPredict_2 = inverse_WT([cA2ytest,cD2ytest,cD1ytest])
    trainPredict_1 = inverse_WT([cA1ytrain,cD1ytrain])
    testPredict_1 = inverse_WT([cA1ytest,cD1ytest])
    #-------------------------------
    n_features = len(data_mar.columns)
    _,Y=autosplit(data_mar)
    y_train,y_test = Y[0],Y[1]
    record_list_result('c3'+syn,df,DLtype,y_train,y_test,trainPredict_3,testPredict_3,target,batch_size,save_path,n_past,n_features,n_future)
    record_list_result('c2'+syn,df,DLtype,y_train,y_test,trainPredict_2,testPredict_2,target,batch_size,save_path,n_past,n_features,n_future)
    record_list_result('c1'+syn,df,DLtype,y_train,y_test,trainPredict_1,testPredict_1,target,batch_size,save_path,n_past,n_features,n_future)
    n_features = n_pca
##----------- Run Experiment -----------------##
# for batch_size in [16,32]:
#     model = [build_cnn1d(),build_cnn1d(),build_cnn1d(),build_cnn1d()]
#     wav_dl_wav_run(model,batch_size,'wCNN_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))

# for batch_size in [16,32]:
#     model_B = [build_lstm(),build_mod_cnn1d(),build_mod_cnn1d(),build_mod_cnn1d()]
#     wav_dl_wav_run(model_B,batch_size,'wLSTM_dCNNx4_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))

for batch_size in [16]:
        model_D = [build_lstm_v2(),build_tanh_lstm(),build_tanh_lstm(),build_tanh_lstm()]
        wav_dl_wav_run(model_D,batch_size,'w_LSTM(big)_(tanh)lstmx3_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
        model_A = [build_lstm_v2(),build_orimod_cnn1d(),build_orimod_cnn1d(),build_orimod_cnn1d()]
        wav_dl_wav_run(model_A,batch_size,'w_LSTM(big)_dCNN(tanh)x3_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))

