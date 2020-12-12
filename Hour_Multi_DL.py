from DLtools.evaluation_rec import record_list_result
from DLtools.Data import instant_data,station_sel
from DLtools.feature_sel import call_mar,hi_corr_select
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
# from keras.utils.vis_utils import plot_model
import pywt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,LSTM,RepeatVector,TimeDistributed,Dropout,Conv1D,MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df

#Split XY
def split_xy(data,n_past,n_future):
    x,y = split_series(data.values,n_past,n_future)
    x = x.reshape((x.shape[0], x.shape[1],n_features))
    y = y[:,:,0]
    return x,y

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

def train_test_split_xy(input_df):
    split_date = '2017-01-01'
    train,test = input_df[:split_date],input_df[split_date:]
    X_train, y_train = split_xy(train,n_past,n_future)
    X_test, y_test = split_xy(test,n_past,n_future)
    return  X_train, y_train, X_test, y_test
#################################################
def to_wave_x(input):
    cA, cD = pywt.dwt(input, 'db4')
    arr_cA = np.array(cA) #input 1 
    arr_cD = np.array(cD) #input 2
    input= np.concatenate([arr_cA,arr_cD],-1)
    
    return input
#PCA Split XY
def PCA_split_xy(data,n_past,n_future,n_pca_com):
    x,y = PCA_split_series(data.values,n_past,n_future)
        
    x = x.reshape((x.shape[0], x.shape[1],n_pca_com))
    y = y[:,:,0]
    return x,y

def PCA_split_series(series, n_past, n_future):
    # n_past ==> no of past observations
    # n_future ==> no of future observations 
    X, y = list(), list()

    pca = PCA(n_components =n_com_pca)
    pca_series=pca.fit_transform(series)

    for window_start in range(len(pca_series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(pca_series):
            break
        # slicing the past and future parts of the window
        past, future = pca_series[window_start:past_end, :], series[past_end:future_end, :]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)

def PCA_train_test_split_xy(input_df):
    split_date = '2017-01-01'
    train,test = input_df[:split_date],input_df[split_date:]
    X_train, y_train = PCA_split_xy(train,n_past,n_future,n_com_pca)
    X_test, y_test = PCA_split_xy(test,n_past,n_future,n_com_pca)
    return  X_train, y_train, X_test, y_test
########################################
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
    trainPredict = model.predict(X_train).astype('float32')
    testPredict = model.predict(X_test).astype('float32')
    
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

# # MARS
cutoff = 0.2
data_mar = call_mar(data,target,mode,cutoff=cutoff)
#####################################################
# # PCA
if mode=='day':n_com_pca = 5
elif mode =='hour': n_com_pca = 8
# pca_data = pipe.fit_transform(data_mar)
# pca_data = pd.DataFrame(pca_data, index=data_mar.index)
# pca_data = pd.concat([pca_data,data_mar[target]],axis=1)
# # Move Y to first row
# pca_data = move_column_inplace(pca_data,target,0)
data_mar = move_column_inplace(data_mar,target,0)
n_features = len(data_mar.columns)

# # SCALE
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(data_mar), columns=data_mar.columns,index=data_mar.index)

scaler_tar = StandardScaler()
scaler_tar.fit(data[target].to_numpy().reshape(-1,1))

X_train, y_train, X_test, y_test = PCA_train_test_split_xy(df_scaled)

cA, cD = pywt.dwt(X_train, 'db4')
seq_approx_train = np.array(cA) #input 1 
seq_detail_train = np.array(cD) #input 2
X_train= np.concatenate([seq_approx_train,seq_detail_train],-1)

cA, cD = pywt.dwt(X_test, 'db4')
seq_approx_test = np.array(cA) #input test 1
seq_detail_test = np.array(cD) #input test 2
X_test= np.concatenate([seq_approx_test,seq_detail_test],-1)

print('sample x timesteps x features')
print('sample 0: 2 datapoint : 5feature \n',X_train[0,:2,:5])
print('\n********XY train-test split********'
      '\ntrainX :',X_train.shape,           '     testX :',X_test.shape,
      '\ntrainY :',y_train.shape,           '     testY :',y_test.shape,
      '\n********Wavelet Transform********'
      '\ncA_train :',seq_approx_train.shape,      '   cD_train :',seq_detail_train.shape,
      '\ncA_test :',seq_approx_test.shape,      '     cD_test :',seq_detail_test.shape,)
# ######################################
batch_size_list = [256,512]
for batch_size in batch_size_list:
    
    try: run_code(build_lstm(),batch_size,'wLSTM_PCA{}_{}'.format(cutoff,batch_size))
    except KeyboardInterrupt: pass
    try: run_code(build_cnn1d(),batch_size,'wCNN_1D_PCA{}_{}'.format(cutoff,batch_size))
    except KeyboardInterrupt: pass
    try: run_code(build_ende_lstm(),batch_size,'wEn_Dec_LSTM_PCA{}_{}'.format(cutoff,batch_size))
    except KeyboardInterrupt: pass

# ############# ALL FEATURE ##########################
# ###### SETTING ################
# n_features = len(data.columns)

# # Move Y to first row
# data = move_column_inplace(data,target,0)
# # SCALE
# scaler_tar = MinMaxScaler()
# scaler_tar.fit(data[target].to_numpy().reshape(-1,1))
# scaler = MinMaxScaler()
# data[data.columns] = scaler.fit_transform(data[data.columns])

# # Train-Test split
# # split_pt = int(data.shape[0]*.7)
# # train,test = data.iloc[:split_pt,:],data.iloc[split_pt:,:]
# split_date = '2017-01-01'
# train,test = data[:split_date],data[split_date:]

# #Split XY
# X_train, y_train = split_xy(train,n_past,n_future)
# X_test, y_test = split_xy(test,n_past,n_future)
# #######################################
# batch_size_list = [256,128]
# for batch_size in batch_size_list:
#     try:run_code(build_cnn1d(),batch_size,'CNN_1D_{}'.format(batch_size))
#     except KeyboardInterrupt: pass
#     # try:run_code(build_ende_lstm(),batch_size,'En_Dec_LSTM_{}'.format(batch_size))
#     # except KeyboardInterrupt: pass
#     # try:run_code(build_lstm(),batch_size,'LSTM_{}'.format(batch_size))
#     # except KeyboardInterrupt: pass