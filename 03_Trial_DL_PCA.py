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
np.random.seed(42)
############# Keras ###################
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

#---------------- SETTING AREA -------------------#
loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
# df,mode = loading.daily_instant(),'day'
if mode =='hour': n_past,n_future = 24*6,72
elif mode =='day': n_past,n_future = 60,30
st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
split_date = '2016-11-01'
#**************** DL PARAMETER **********************#
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-5 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]
my_optimizer = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)

# #---------------------- 2 Yr Edit -------------------------#
# host_path = './CPY012/2Yr_flood/'
# start_p = '2016-01-01'
# split_date = '2017-05-10'
# stop_p = '2018-01-01'
#----------------------------------------------------
n_pca = 4

syn=''
Yscale = True
# allscale = True
#-----------------------Baseline / Hybrid -----------------------------------#
save_path =host_path+'Baseline_PCA'
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
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
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
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)   
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
    plot_model(model, to_file=save_path+'model_{}.png'.format(syn), show_shapes=True)
    return model
def run_code(model,batch_size,syn):
    global target,mode,df,X_train,y_train,X_test,y_test
    verbose, epochs = 1, 100
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
    # ---------- Inverse ------------------#
    if Yscale:
        y_train = scaler_tar.inverse_transform(y_train)
        trainPredict = scaler_tar.inverse_transform(trainPredict.reshape(y_train.shape))
        y_test = scaler_tar.inverse_transform(y_test)
        testPredict = scaler_tar.inverse_transform(testPredict.reshape(y_test.shape))
    record_list_result(syn,df,'03_DL_PCA',y_train,y_test,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future)
#--------------------------------
def build_ann():
    global n_past,n_future,n_features
    input = keras.Input(shape=(n_past, int(n_features)))
    x = layers.Flatten()(input)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dense(n_future)(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')  
    plot_model(model, to_file=save_path+'modelANN_{}.png'.format(syn), show_shapes=True)  
    model.summary()
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
    x = layers.Dense(n_future)(x)
    x = layers.LeakyReLU()(x)
    model = keras.Model(inputs=[input], outputs=x)
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    plot_model(model, to_file=save_path+'modelCNN_{}.png'.format(syn), show_shapes=True)
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
    plot_model(model, to_file=save_path+'modelLSTM_{}.png'.format(syn), show_shapes=True)
    return model
#------------------------- Main ---------------------------------#
df = df[start_p:stop_p]
data = df
data['Day'] = data.index.dayofyear #add day
data = data.interpolate(limit=300000000,limit_direction='both').astype('float32')#interpolate neighbor first, for rest NA fill with mean() #.apply(lambda x: x.fillna(x.mean()),axis=0)


cutoff=.3
data_mar = call_mar(data,target,mode,cutoff=cutoff)
data_mar = move_column_inplace(data_mar,target,0)
n_features = len(data_mar.columns)

##----------- SCALE--------------##
def Preprocess_pca(input):
    global syn
    syn = syn+'[Xpca_scminmax]'
    pipe = Pipeline([('scaler', StandardScaler()), ('pca',PCA(n_components =n_pca)),('minmax',MinMaxScaler())])
    scaler = pipe.fit(input)
    sc_input = pipe.transform(input)
    sc_input = pd.DataFrame(sc_input, index=input.index)
    return scaler,sc_input
# X data
_,sc_data = Preprocess_pca(data_mar)
# Y data
if Yscale:
    syn = syn+'[y_sc]'        
    scaler_tar = MinMaxScaler()
    scaler_tar.fit(data_mar[target].to_numpy().reshape(-1,1))
    print(data_mar[target].to_numpy().reshape(-1,1).shape)
# if allscale:
#     syn = syn+'[X_sc]'  
#     scaler = MinMaxScaler()
#     data_mar[data_mar.columns] = scaler.fit_transform(data_mar[data_mar.columns])
#     print(data_mar.columns)

##----------- train test split -----------------##
sc_train,sc_test = sc_data[:split_date],sc_data[split_date:]
train,test = data_mar[:split_date],data_mar[split_date:]
#Keep original
_, y_train = split_xy(train,n_past,n_future)
_, y_test = split_xy(test,n_past,n_future)
# set all feature to PCA ( change n_feature as well )
n_features=n_pca
X_train, _ = split_xy(sc_train,n_past,n_future)
X_test, _ = split_xy(sc_test,n_past,n_future)

##----------- Run Experiment -----------------##
for batch_size in [16]:
        run_code(build_ann(),batch_size,'pca_ANN_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
        run_code(build_mod2_cnn1d(),batch_size,'pca_dCNN(linear)_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))
        run_code(build_lstm_v2(),batch_size,'pca_CuDNNLSTM(big)_MAR{}_b{}_Tin{}_{}'.format(cutoff,batch_size,n_past,syn))