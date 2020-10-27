from DLtools import  prep_data,LSTMmodel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def LSTM_1(num_x_features, num_y_feature):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(None,num_x_features),
                activation='relu'))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dense(num_y_feature,activation='relu'))
    opt = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5)

    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    #model.summary()
    return model

if __name__ == "__main__":
    #test = LSTMmodel(water,water['BAKI'],timelag=7)
    #test.report()
    #PARAMETER SETTING
    rain = 'data/instant_data/rain.csv'
    water = 'data/instant_data/water.csv'    
    start_date = '2015-01-01'
    # Scale 1-0    
    rw = prep_data.instant_df(rain,water,start=start_date)
    df = rw.df
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    # Train LSTM
    TIMELAG = 1
    BATCH = 256
    input = LSTMmodel.prepLSTM(df,df['CPY008_w'],timelag=TIMELAG,batch_size = BATCH)
    X_FEATURE = input.num_x_feature
    Y_FEATURE = input.num_y_feature
    VALID = input.validation_data()
    XY_BATCH = input.generator
    #######################
    model = LSTM_1(X_FEATURE,Y_FEATURE)
    model.summary()
    history = model.fit(x=XY_BATCH, epochs=100, steps_per_epoch=100,validation_data=VALID)
