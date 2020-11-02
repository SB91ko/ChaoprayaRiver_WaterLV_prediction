from DLtools.Data_preprocess import load_data,preprocess
from DLtools.evaluation_rec import eva_error,error_rec
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D

np.random.seed(42)

def model_cnn(train_X, train_Y, validation):
    epochs = 100
    batch = 256
    lr = 0.0003
    adam = optimizers.Adam(lr)
    model_cnn = Sequential()
    model_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(50, activation='relu'))
    model_cnn.add(Dense(1))
    model_cnn.compile(loss='mse', optimizer=adam)
    model_cnn.summary()
    history = model_cnn.fit(train_X, train_Y, epochs=epochs, batch_size=batch, verbose=1,shuffle=False,validation_data=validation)
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()
    return model

def inputdata():        
    r='data/instant_data/rain_small.csv'
    w='data/instant_data/water_small.csv'
    rw = load_data(r,w)
    df =rw.df.resample('d').mean()

    X_in = df['2013-01-01':'2015-01-01'].interpolate(limit=15)
    X_in = X_in.astype('float32')
    return X_in

Rec = pd.DataFrame()
X_in = inputdata()
TARGET = 'CPY015_w'
FEATURE = X_in.shape[1]
###### TUNING PARAMETER: TIMELAGs,BATCHSIZE ######
for times in range(1,15):
    TIME = times
    prep = preprocess(X_in,Y_column=TARGET,n_timelag=TIME,n_feature=FEATURE,scale_ornot=True)
    train_X, train_Y = prep.train_X,prep.train_Y
    val_X,val_y = prep.val_X,prep.val_y
    test_X,test_y =prep.test_X,prep.test_Y
    tra_val_X,tra_val_Y = prep.train_val_X,prep.train_val_Y
    
    VALIDATION = (val_X,val_y) 
    model = model_cnn(train_X, train_Y, VALIDATION)
    trainPredict = model.predict(tra_val_X)
    testPredict = model.predict(test_X)
    mse_train,mse_test,nse_train,nse_test = eva_error(trainPredict,tra_val_Y,testPredict,test_y)

    scaler_target = MinMaxScaler()
    scaler_target.fit(X_in[TARGET].values.reshape(-1,1))
    scale_trainPredict = scaler_target.inverse_transform(trainPredict)
    scale_testPredict = scaler_target.inverse_transform(testPredict)
    
    gen_msetrain,gen_msetest,gen_nsetrain,gen_nsetest  = eva_error(scale_trainPredict,tra_val_Y,scale_testPredict,test_y)
    Rec = error_rec(Rec,'CNN1d_model',FEATURE,TIME,BATCH,mse_train,mse_test,nse_train,nse_test, gen_msetrain,gen_msetest,gen_nsetrain,gen_nsetest)
    Rec.to_csv('output/CNN_error_rec.csv')

