from numpy.lib.npyio import load
from DLtools.evaluation_rec import real_eva_error,error_rec,list_eva_error
from DLtools.Data import load_data,del_less_col,check_specific_col
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, LSTM, RepeatVector,TimeDistributed,Input
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
def preprate_data(data):
    train,test = split_triantest(data,ratio=0.7)
    X_train, y_train = split_series(train.values,n_past, n_future)
    X_train,y_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features)),y_train[:,:,0]
    
    X_test, y_test = split_series(test.values,n_past, n_future)
    X_test,y_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features)),y_test[:,:,0]
    return X_train,y_train,X_test,y_test
#######################################
def getPredict(X_train,X_test):
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)
    return trainPredict,testPredict
def getEvaluation(Y, Yhat,Y_t, Yhat_t,n_future):
    mse, nse,r2 = list_eva_error(Y, Yhat,n_future)
    Tmse, Tnse,Tr2 = list_eva_error(Y_t, Yhat_t,n_future)
    return mse, nse,r2,Tmse, Tnse,Tr2
#######################################
loaddata = load_data(load_all=False)
# df_d = loaddata.daily()
df_h = loaddata.hourly()
def def_model_lstm(X_train,y_train,X_test,y_test,n_timesteps, n_features, n_outputs,batch_size):    
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))                                              # Decoder 
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    verbose, epochs = 0, 70
    callback_early_stopping = EarlyStopping(monitor='val_loss',patience=10, verbose=2)
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
    callbacks = [callback_early_stopping,reduce_lr]

    history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=batch_size,verbose=verbose,callbacks=callbacks)
    
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('Encode_Decode\nin:{},out:{},n_fea:{},batch:{}'.format(n_past,n_future,n_features,BATCH))
    plt.savefig(save_location+'loss_rec_in{}_out{}_fea{}_bat{}.png'.format(n_past,n_future,n_features,BATCH), dpi=300, bbox_inches='tight') 
    
    plt.legend()
    plt.show()
    return model

df = df_h["2013-01-01":"2017-12-31"].interpolate(limit=360).fillna(0)
TARGET = 'CPY015_wl'
df = move_column_inplace(df,TARGET,0)
######### Record setting #############################
save_location='output/LSTM_EnDe/temp/exp4_hourly/'

idx=['Modelname','Feature','n_in_time','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2']
error = pd.DataFrame(index = idx)        
error_d = pd.DataFrame(index = idx)    
######PARAMETER SETTING################
n_past = 24*7
n_future = 24 
BATCH = 64
n_features = df.shape[1]

# n_future_list = [3,7,14,21,30]
# n_past_list = [3,7,14,21,30]
# BATCH_LIST = [32,64,128]
data = df
###### SCALE###############
scaler_tar = MinMaxScaler()
scaler_tar.fit(df[TARGET].to_numpy().reshape(-1,1))
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

#####TRAIN TEST SPLIT###########
def run_loop(n_past_list,n_future_list,BATCH_LIST,sel_model):
    error = pd.DataFrame(index = idx)  
    for n_past in n_past_list:
        for n_future in n_future_list:
            if n_future<n_past:
                X_train,y_train,X_test,y_test = preprate_data(data)
                
                for BATCH in BATCH_LIST:
                    ######MODEL GEN################
                    model = sel_model(X_train,y_train,X_test,y_test,n_past, n_features, n_future,batch_size=BATCH)
                    #######PREDICT##################
                    trainPredict,testPredict = getPredict(X_train,X_test)
                    # Re scale
                    Y = scaler_tar.inverse_transform(y_train)
                    Yhat = scaler_tar.inverse_transform(trainPredict.reshape(y_train.shape))
                    Y_t = scaler_tar.inverse_transform(y_test)
                    Yhat_t = scaler_tar.inverse_transform(testPredict.reshape(y_test.shape))
                    ########EVALUATION#####################
                    mse, nse,r2,Tmse, Tnse,Tr2 = getEvaluation(Y, Yhat,Y_t, Yhat_t,n_future)          
                    graph_index = np.arange(len(y_train)+len(y_test))

                    for d in range(n_future):
                        g_Y= pd.Series(data=Y[:,d],index=graph_index[:len(y_train)])
                        g_Yhat = pd.Series(data=(Yhat[:,d].ravel()),index=graph_index[:len(y_train)])
                        g_Y_t= pd.Series(data=Y_t[:,d],index=graph_index[-len(y_test):])
                        g_Yhat_t = pd.Series(data=(Yhat_t[:,d].ravel()),index=graph_index[-len(y_test):])
                        
                        plt.figure(figsize=(20,5))
                        plt.plot(g_Y, label = "Actual")
                        plt.plot(g_Yhat, label = "Predict")
                        plt.plot(g_Y_t, label = "Actual_test")
                        plt.plot(g_Yhat_t, label = "Predict_test")
                        plt.title('[Encode-Decode LSTM] Day{}\n'.format(d+1)+'Water Level CPY015 Forecast vs Actuals\n'+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse[d],nse[d],r2[d])+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse[d],Tnse[d],Tr2[d]))
                        plt.legend()
                        plt.savefig(save_location+'result_in{}_out{}_fea{}_bat{}_d{}.png'.format(n_past,n_future,n_features,BATCH,d+1), dpi=300, bbox_inches='tight')


                    ####################################################
                        _df = pd.DataFrame(["EnDec_d{}".format(str(d+1)),n_features,(n_past,n_future),BATCH,mse[d], nse[d],r2[d],Tmse[d], Tnse[d],Tr2[d]],index=idx,columns=['LSTM'])
                        error = pd.concat([error,_df],axis=1)

                    error.to_csv(save_location+'evaluation.csv')

            else:
                pass


X_train,y_train,X_test,y_test = preprate_data(data)
model = def_model_lstm(X_train,y_train,X_test,y_test,n_past, n_features, n_future,batch_size=BATCH)
#######PREDICT##################
trainPredict,testPredict = getPredict(X_train,X_test)
# Re scale
Y = scaler_tar.inverse_transform(y_train)
Yhat = scaler_tar.inverse_transform(trainPredict.reshape(y_train.shape))
Y_t = scaler_tar.inverse_transform(y_test)
Yhat_t = scaler_tar.inverse_transform(testPredict.reshape(y_test.shape))
########EVALUATION#####################
mse, nse,r2,Tmse, Tnse,Tr2 = getEvaluation(Y, Yhat,Y_t, Yhat_t,n_future)          
graph_index = np.arange(len(y_train)+len(y_test))

for d in range(n_future):
    g_Y= pd.Series(data=Y[:,d],index=graph_index[:len(y_train)])
    g_Yhat = pd.Series(data=(Yhat[:,d].ravel()),index=graph_index[:len(y_train)])
    g_Y_t= pd.Series(data=Y_t[:,d],index=graph_index[-len(y_test):])
    g_Yhat_t = pd.Series(data=(Yhat_t[:,d].ravel()),index=graph_index[-len(y_test):])
    
    plt.figure(figsize=(20,5))
    plt.plot(g_Y, label = "Actual")
    plt.plot(g_Yhat, label = "Predict")
    plt.plot(g_Y_t, label = "Actual_test")
    plt.plot(g_Yhat_t, label = "Predict_test")
    plt.title('[Encode-Decode LSTM] Day{}\n'.format(d+1)+'Water Level CPY015 Forecast vs Actuals\n'+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse[d],nse[d],r2[d])+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse[d],Tnse[d],Tr2[d]))
    plt.legend()
    plt.savefig(save_location+'result_in{}_out{}_fea{}_bat{}_d{}.png'.format(n_past,n_future,n_features,BATCH,d+1), dpi=300, bbox_inches='tight')


####################################################
    _df = pd.DataFrame(["EnDec_d{}".format(str(d+1)),n_features,(n_past,n_future),BATCH,mse[d], nse[d],r2[d],Tmse[d], Tnse[d],Tr2[d]],index=idx,columns=['LSTM'])
    error = pd.concat([error,_df],axis=1)

error.to_csv(save_location+'evaluation.csv')