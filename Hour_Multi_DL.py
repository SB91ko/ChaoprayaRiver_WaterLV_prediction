from DLtools.evaluation_rec import real_eva_error,error_rec,list_eva_error
from DLtools.Data import del_less_col,check_specific_col,instant_data,intersection,station_sel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils.vis_utils import plot_model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,LSTM,RepeatVector,TimeDistributed,Input,Dropout,Conv1D,MaxPooling1D
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
def record_result(y_train,y_test,trainPredict,testPredict,scaler_target,syn):
    global save_path,error
    global n_past,n_future,n_features,batch_size
    
    scale_Y = scaler_target.inverse_transform(y_train)
    scale_Yhat = scaler_target.inverse_transform(trainPredict.reshape(y_train.shape))
    scale_Y_t = scaler_target.inverse_transform(y_test)
    scale_Yhat_t = scaler_target.inverse_transform(testPredict.reshape(y_test.shape))

    mse, nse,r2 = list_eva_error(scale_Y, scale_Yhat,n_future)
    Tmse, Tnse,Tr2 = list_eva_error(scale_Y_t, scale_Yhat_t,n_future)

    for d in range(n_future):
        index = np.arange(len(y_train)+len(y_test))
        Y= pd.Series(data=scale_Y[:,d],index=index[:len(y_train)])
        Yhat = pd.Series(data=(scale_Yhat[:,d].ravel()),index=index[:len(y_train)])
        Y_t= pd.Series(data=scale_Y_t[:,d],index=index[-len(y_test):])
        Yhat_t = pd.Series(data=(scale_Yhat_t[:,d].ravel()),index=index[-len(y_test):])
        
        #Save fig only for hour 1,5,10,15...
        # if ((d==0)or(d+1%5==0)):
        plt.figure(figsize=(15,5))
        plt.plot(Y, label = "Actual")
        plt.plot(Yhat, label = "Predict")
        
        plt.plot(Y_t, label = "Actual_test")
        plt.plot(Yhat_t, label = "Predict_test")
        plt.title('[{}] Hours{}\n'.format(syn,d+1)+'Water Level CPY015 Forecast vs Actuals\n'+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse[d],nse[d],r2[d])+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse[d],Tnse[d],Tr2[d]))
        plt.legend()
        plt.savefig(save_path+'Plot_{}_h{}.png'.format(syn,d+1), dpi=300, bbox_inches='tight') 
        # plt.show()
        
        _df = pd.DataFrame(["{}_{}h".format(syn,str(d+1)),n_features,n_past,batch_size,mse[d], nse[d],r2[d],Tmse[d], Tnse[d],Tr2[d]],index=idx,columns=[syn])
        error = pd.concat([error,_df],axis=1)
        #Note Fix path for error Record
        error.to_csv(errorfile)

loading = instant_data()
# df_h = loading.hourly_instant()
df_d = loading.daily_instant()
TARGET,start_p,stop_p,save_host=station_sel('CPY012')
save_path ='/home/song/Public/Song/Work/Thesis/output_cpy012/Daily/DL/'
errorfile='/home/song/Public/Song/Work/Thesis/output_cpy012/Daily/eval.csv'

#load previous error rec
idx=['Modelname','Feature','n_in_time','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2']
try: error = pd.read_csv('/home/song/Public/Song/Work/Thesis/output_cpy012/Daily/eval.csv',index_col=0);print('LOAD SUCEESS')
except: error = pd.DataFrame(errorfile,index = idx);print("cannot find rec")

data = df_d[start_p:stop_p]
data = del_less_col(data,ratio=.85)
data['Day'] = data.index.dayofyear #add day
data = data.interpolate(limit=24).apply(lambda x: x.fillna(x.mean()),axis=0).astype('float32')#interpolate neighbor first, for rest NA fill with mean()

#####################
# MAR Feature selection
def mars_selection(data):
    global TARGET
    if TARGET=='CPY015_wl':
        MARfile='/home/song/Public/Song/Work/Thesis/MAR/featurelist_MAR_daily_7d.csv'
        # MARfile='/home/song/Public/Song/Work/Thesis/MAR/featurelist_MAR_hourly_7d.csv'
    elif TARGET=='CPY012_wl':
        MARfile='/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_daily_7d.csv'
        # MARfile='/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_hourly.csv'

    mar = pd.read_csv(MARfile)
    col = [i for i in data.columns]
    select_col = intersection(col,mar['feature'])
    select_col.append(TARGET) # add target
    data = data[select_col]
    return data

def build_lstm():
    global n_past,n_future,n_features,save_path
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
    global n_past,n_future,n_features,save_path
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_past, n_features)))
    model.add(RepeatVector(n_future))                                  # Decoder 
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    model.save(save_path+'endec.h5')
    
    return model

def build_cnn1d():
    global n_past,n_future,n_features,save_path
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_past, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_future))
    model.compile(optimizer='adam', loss='mse')    
    return model

def run_code(model,batch_size,syn,zoom=True):
    verbose, epochs = 1, 120
    # n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.savefig(save_path+'loss_{}.png'.format(syn), dpi=300, bbox_inches='tight') 
    plt.show()
    
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    record_result(y_train,y_test,trainPredict,testPredict,scaler_tar,syn)
    model.save(save_path+'{}.h5'.format(syn))
    plot_model(model, to_file='model_plot_hour_{}.png'.format(syn), show_shapes=True, show_layer_names=True)
    if zoom==True:
        plt.figure(figsize=(20,5))
        plt.plot(y_test[-500:,1],label='Actual_test')
        plt.plot(testPredict[-500:,1],label='Predict_test')
        plt.title('[{}] Hour1\n'.format(syn)+'Water Level CPY015 Forecast vs Actuals')
        plt.legend()
        plt.savefig(save_path+'Plot_{}_zoom500data.png'.format(syn), dpi=300, bbox_inches='tight') 
        plt.show()


data_mar = mars_selection(data)
# Move Y to first row
data_mar = move_column_inplace(data_mar,TARGET,0)

###### SETTING ################
n_past = 24*7
n_future = 24
n_features = len(data_mar.columns)


# SCALE
scaler_tar = MinMaxScaler()
scaler_tar.fit(data_mar[TARGET].to_numpy().reshape(-1,1))
scaler = MinMaxScaler()
data_mar[data_mar.columns] = scaler.fit_transform(data_mar[data_mar.columns])

# Train-Test split
split_pt = int(data_mar.shape[0]*.7)
train,test = data_mar.iloc[:split_pt,:],data_mar.iloc[split_pt:,:]

#Split XY
X_train, y_train = split_series(train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train[:,:,0]
X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test[:,:,0]

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
#######################################
callback_early_stopping = EarlyStopping(monitor='val_loss',patience=10, verbose=2)
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
callbacks = [callback_early_stopping,reduce_lr]

batch_size=128
run_code(build_cnn1d(),batch_size,'CNN_1D_MAR')
run_code(build_ende_lstm(),batch_size,'En_Dec_LSTM_MAR')
try:run_code(build_lstm(),batch_size,'LSTM_MAR')
except:pass


###### SETTING ################
n_past = 24*7
n_future = 24
n_features = len(data.columns)

# Move Y to first row
data = move_column_inplace(data,TARGET,0)
# SCALE
scaler_tar = MinMaxScaler()
scaler_tar.fit(data[TARGET].to_numpy().reshape(-1,1))
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Train-Test split
split_pt = int(data.shape[0]*.7)
train,test = data.iloc[:split_pt,:],data.iloc[split_pt:,:]

#Split XY
X_train, y_train = split_series(train.values,n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train[:,:,0]
X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test[:,:,0]

print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
#######################################
batch_size = 128
run_code(build_cnn1d(),batch_size,'CNN_1D')
run_code(build_ende_lstm(),batch_size,'En_Dec_LSTM')
run_code(build_lstm(),batch_size,'LSTM')