from DLtools.evaluation_rec import real_eva_error,error_rec,list_eva_error
from DLtools.Data import instant_data,intersection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector,TimeDistributed,Input
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(42)

def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df
def split_triantest(data,ratio=.7):
    split_pt = int(data.shape[0]*ratio)
    train,test = data.iloc[:split_pt,:],data.iloc[split_pt:,:]
    return train,test
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
#######################################
def preprate_data(data,n_past,n_future):
    train,test = split_triantest(data,ratio=0.7)
    X_train, y_train = split_series(train.values,n_past, n_future)
    X_train,y_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features)),y_train[:,:,0]
    
    X_test, y_test = split_series(test.values,n_past, n_future)
    X_test,y_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features)),y_test[:,:,0]
    return X_train,y_train,X_test,y_test
#######################################
def getPredict(model,X_train,X_test):
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    return trainPredict,testPredict
def getEvaluation(Y, Yhat,Y_t, Yhat_t,n_future):
    mse, nse,r2 = list_eva_error(Y, Yhat,n_future)
    Tmse, Tnse,Tr2 = list_eva_error(Y_t, Yhat_t,n_future)
    return mse, nse,r2,Tmse, Tnse,Tr2
#######################################
def MAR_sel(df,Target):
    mar = pd.read_csv('/home/song/Public/Song/Work/Thesis/featurelist_MAR_hourly_7d.csv')
    col = [i for i in df.columns]
    select_col = intersection(col,mar['feature'])                       ###Edit need
    select_col.append(Target) # add target
    return df[select_col]
###### SCALE#########
def scaler_return(df,target):
    scaler_target = MinMaxScaler()
    scaler_target.fit(df[target].to_numpy().reshape(-1,1))
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df,scaler_target


def build_seq(n_past,n_future,n_features):
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_past, n_features)))
    model.add(RepeatVector(n_future))                                              # Decoder 
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.summary()  
    return model

def build_lstm(n_past,n_future,n_features):
        model = Sequential()
        # model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(LSTM(200, activation='relu', input_shape=(n_past, n_features),return_sequences=True))
        model.add(LSTM(200, activation='relu', return_sequences=False))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_future))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model

def __init__(self,data,n_past,n_future,n_features,batchsize,syn_name,model_selection):
    self.data,self.n_past,self.n_future,self.n_features,self.batchsize = data,n_past,n_future,n_features,batchsize
    self.save_location = None
    self.n_epoch = 100
    self.syn_name,self.model_selection = syn_name,model_selection
    self.X_train,self.y_train,self.X_test,self.y_test = preprate_data(data,n_past,n_future)

def plot_loss(self,history_):    
    plt.plot(history_.history['loss'], label='train')
    plt.plot(history_.history['val_loss'], label='test')
    plt.title('{}\nin:{},out:{},n_fea:{},batch:{}'.format(self.syn_name,self.n_past,self.n_future,self.n_features,self.batchsize))
    plt.legend()
    plt.savefig(save_location+'loss_rec_{}_in{}_out{}_fea{}_bat{}.png'.format(self.syn_name,self.n_past,self.n_future,self.n_features,self.batchsize), dpi=300, bbox_inches='tight')         
    return 

def run(self):
    callback_early_stopping = EarlyStopping(monitor='val_loss',patience=10, verbose=2)
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
    callbacks = [callback_early_stopping,reduce_lr]
    verbose = 1
    
    model_fit = self.model_selection(self.n_past,self.n_future,self.n_features)
    history = model_fit.fit(self.X_train,self.y_train,epochs=self.n_epoch,validation_data=(self.X_test,self.y_test),batch_size=self.batchsize,verbose=verbose,callbacks=callbacks)
    self.plot_loss(history)

    trainPredict,testPredict = getPredict(model_fit,self.X_train,self.X_test)
    Y = scaler_tar.inverse_transform(self.y_train)
    Yhat = scaler_tar.inverse_transform(trainPredict.reshape(self.y_train.shape))
    Y_t = scaler_tar.inverse_transform(self.y_test)
    Yhat_t = scaler_tar.inverse_transform(testPredict.reshape(self.y_test.shape))
    for d in range(self.n_future):
        self.fig_save(self,d,Y,Yhat,Y_t,Yhat_t)

def fig_save(self,d,Y,Yhat,Y_t,Yhat_t):
    mse, nse,r2,Tmse, Tnse,Tr2 = getEvaluation(Y, Yhat,Y_t, Yhat_t,self.n_future)

    graph_index = np.arange(len(self.y_train)+len(self.y_test))
    g_Y= pd.Series(data=Y[:,d],index=graph_index[:len(self.y_train)])
    g_Yhat = pd.Series(data=(Yhat[:,d].ravel()),index=graph_index[:len(self.y_train)])
    g_Y_t= pd.Series(data=Y_t[:,d],index=graph_index[-len(self.y_test):])
    g_Yhat_t = pd.Series(data=(Yhat_t[:,d].ravel()),index=graph_index[-len(self.y_test):])
    
    if d ==0:
        plt.figure(figsize=(20,5))
        plt.plot(g_Y, label = "Actual")
        plt.plot(g_Yhat, label = "Predict")
        plt.plot(g_Y_t, label = "Actual_test")
        plt.plot(g_Yhat_t, label = "Predict_test")
        plt.title('[{}] Day{}\n'.format(self.syn_name,d+1)+'Water Level CPY015 Forecast vs Actuals\n'+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse[d],nse[d],r2[d])+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse[d],Tnse[d],Tr2[d]))
        plt.legend()
        plt.savefig(save_location+'result_{}_in{}_out{}_fea{}_bat{}_d{}.png'.format(self.syn_name,self.n_past,self.n_future,self.n_features,self.batchsize,d+1), dpi=300, bbox_inches='tight')


loaddata = instant_data()
# df = loaddata.daily()
df = loaddata.hourly_instant()
############ SETTTING ###########################
data = df["2013-01-01":"2013-07-31"].interpolate(method='time',limit=24)
TARGET = 'CPY015_wl'
# Feature selection
data_mar = MAR_sel(data,TARGET)
data_mar = move_column_inplace(data_mar,TARGET,0)
# Average NA, cofirm and drop if any
data_mar = data_mar.apply(lambda x: x.fillna(x.mean()),axis=0)
for col in data_mar.columns:
    if data_mar[col].isna().sum()>1:
        data_mar.drop(columns=col,inplace=True)
    else:pass

print(data_mar.isna().sum())
# move position of y
data_mar = move_column_inplace(data_mar,TARGET,0)
######### Record setting ###

save_location='/home/song/Public/Song/Work/Thesis/output/LSTM_EnDe/Hourly/'
idx=['Modelname','Feature','n_in_time','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2']
error = pd.DataFrame(index = idx)        
######## Parameter setting #########################
n_past = 24
n_future = 3
BATCH =128
n_features = data_mar.shape[1]

data_mar,scaler_tar = scaler_return(data_mar,TARGET)
# LSTM = MyML(data_mar,n_past,n_future,n_features,BATCH,'test',build_seq)
# # runing = LSTM.run()
print(n_features)
build_lstm(n_past,n_future.n_features)