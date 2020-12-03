from DLtools.evaluation_rec import record_list_result
from DLtools.Data import instant_data,station_sel
from DLtools.feature_sel import call_mar,hi_corr_select
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
# from keras.utils.vis_utils import plot_model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,LSTM,RepeatVector,TimeDistributed,Dropout,Conv1D,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)

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
###### SETTING AREA ################
loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
# df,mode = loading.daily_instant(),'day'
if mode =='hour': n_past,n_future = 24*7,72
elif mode =='day': n_past,n_future = 60,30

st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
save_path =host_path+'/DL/'
#######################################################
#Split XY
def split_xy(data,n_past,n_future):
    x,y = split_series(data.values,n_past,n_future)
    x = x.reshape((x.shape[0], x.shape[1],n_features))
    y = y[:,:,0]
    return x,y
def build_lstm():
    global n_past,n_future,n_features
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, activation='relu', return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_future))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model
def build_ende_lstm():
    global n_past,n_future,n_features
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_past, n_features)))
    model.add(RepeatVector(n_future))                                  # Decoder 
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.summary()    
    return model
def build_cnn1d():
    global n_past,n_future,n_features
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_past, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_future))
    model.compile(optimizer='adam', loss='mse')    
    model.summary()
    return model
def run_code(model,batch_size,syn,zoom=True):
    global target,mode,df
    start_time = time.time()
    verbose, epochs = 1, 120
    
    history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    time_ = time.time() - start_time
    ########### plot loss ########################
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(save_path+'loss_{}_{:.3}.png'.format(syn,time_), dpi=300, bbox_inches='tight') 
    plt.clf()
    #################################################
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    
    scale_y_test,scale_testPredict = record_list_result(syn,df,mode,y_train,y_test,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future,scaler_t=scaler_tar)
    model.save(save_path+'{}.h5'.format(syn))

    if zoom==True:
        mid_ = int(len(scale_y_test)/2)
        fig,ax = plt.subplots(2,1,figsize=(20,5))
        ax[0].plot(scale_y_test[mid_:mid_+500,1],label='Actual_test(first)')
        ax[0].plot(scale_testPredict[mid_:mid_+500,1], linestyle='dashed',label='Predict_test(first)')
        ax[0].legend()
        ax[1].plot(scale_y_test[mid_:mid_+500,-1], label='Actual_test(last)')
        ax[1].plot(scale_testPredict[mid_:mid_+500,-1], linestyle='dashed',label='Predict_test(last)')
        ax[1].legend()
        fig.suptitle('[{}] {}\n'.format(syn,mode)+'Water Level {} zoom 500 data points'.format(target))
        fig.savefig(save_path+'Plot_{}_zoom500data.png'.format(syn), dpi=300, bbox_inches='tight') 
        plt.clf()

callback_early_stopping = EarlyStopping(monitor='val_loss',patience=10, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]
######################################################
df = df[start_p:stop_p]
data = df

data['Day'] = data.index.dayofyear #add day
data = data.interpolate(limit=300000000,limit_direction='both').astype('float32')#interpolate neighbor first, for rest NA fill with mean() #.apply(lambda x: x.fillna(x.mean()),axis=0)
data[target].plot()

# for cutoff in [0.3]:
#     data_mar = call_mar(data,target,mode,cutoff=cutoff)
#     # Move Y to first row
#     data_mar = move_column_inplace(data_mar,target,0)
#     n_features = len(data_mar.columns)
#     # SCALE
#     scaler_tar = MinMaxScaler()
#     scaler_tar.fit(data_mar[target].to_numpy().reshape(-1,1))
#     print(data_mar[target].to_numpy().reshape(-1,1).shape)
#     scaler = MinMaxScaler()
#     data_mar[data_mar.columns] = scaler.fit_transform(data_mar[data_mar.columns])

#     # Train-Test split
#     # split_pt = int(data_mar.shape[0]*.7)
#     # train,test = data_mar.iloc[:split_pt,:],data_mar.iloc[split_pt:,:]
#     split_date = '2017-01-01'
#     train,test = data_mar[:split_date],data_mar[split_date:]

#     X_train, y_train = split_xy(train,n_past,n_future)
#     X_test, y_test = split_xy(test,n_past,n_future)

#     ######################################
#     batch_size_list = [32,64,128]
#     for batch_size in batch_size_list:
#         print(y_train.shape,y_test.shape)
#         try: run_code(build_lstm(),batch_size,'LSTM_MAR{}_{}'.format(cutoff,batch_size))
#         except KeyboardInterrupt: pass
#         try: run_code(build_cnn1d(),batch_size,'CNN_1D_MAR{}_{}'.format(cutoff,batch_size))
#         except KeyboardInterrupt: pass
#         try: run_code(build_ende_lstm(),batch_size,'En_Dec_LSTM_MAR{}_{}'.format(cutoff,batch_size))
#         except KeyboardInterrupt: pass
        

# # # # ###### SETTING ################
# # # # #### MAR selection ##
# # # # data = call_mar(data,target,mode,cutoff=0.3)
# # # # #### Corr selection##
# # # # data = hi_corr_select(data,target)
# # # # n_features = len(data.columns)
# # # # # Move Y to first row
# # # # data = move_column_inplace(data,target,0)
# # # # # SCALE
# # # # scaler_tar = MinMaxScaler()
# # # # scaler_tar.fit(data[target].to_numpy().reshape(-1,1))
# # # # scaler = MinMaxScaler()
# # # # data[data.columns] = scaler.fit_transform(data[data.columns])
# # # # # Train-Test split
# # # # # split_pt = int(data.shape[0]*.7)
# # # # # train,test = data.iloc[:split_pt,:],data.iloc[split_pt:,:]
# # # # split_date = '2017-01-01'
# # # # train,test = data[:split_date],data[split_date:]

# # # # #Split XY
# # # # X_train, y_train = split_xy(train,n_past,n_future)
# # # # X_test, y_test = split_xy(test,n_past,n_future)
# # # # #######################################
# # # # batch_size_list = [32,128,256,512]
# # # # for batch_size in batch_size_list:
# # # #     try:run_code(build_cnn1d(),batch_size,'CNN_1D_MAR_CORR_{}'.format(batch_size))
# # # #     except:pass
# # # #     try:run_code(build_ende_lstm(),batch_size,'En_Dec_LSTM_MAR_CORR_{}'.format(batch_size))
# # # #     except:pass
# # # #     try:run_code(build_lstm(),batch_size,'LSTM_MAR_CORR_{}'.format(batch_size))
# # # #     except:pass
############# ALL FEATURE ##########################
###### SETTING ################
n_features = len(data.columns)

# Move Y to first row
data = move_column_inplace(data,target,0)
# SCALE
scaler_tar = MinMaxScaler()
scaler_tar.fit(data[target].to_numpy().reshape(-1,1))
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Train-Test split
# split_pt = int(data.shape[0]*.7)
# train,test = data.iloc[:split_pt,:],data.iloc[split_pt:,:]
split_date = '2017-01-01'
train,test = data[:split_date],data[split_date:]

#Split XY
X_train, y_train = split_xy(train,n_past,n_future)
X_test, y_test = split_xy(test,n_past,n_future)
#######################################
batch_size_list = [128,256,64,32]
for batch_size in batch_size_list:
    # try:run_code(build_cnn1d(),batch_size,'CNN_1D_{}'.format(batch_size))
    # except KeyboardInterrupt: pass
    try:run_code(build_ende_lstm(),batch_size,'En_Dec_LSTM_{}'.format(batch_size))
    except KeyboardInterrupt: pass
    try:run_code(build_lstm(),batch_size,'LSTM_{}'.format(batch_size))
    except KeyboardInterrupt: pass