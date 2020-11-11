from DLtools.Data_preprocess import load_data
from DLtools.evaluation_rec import real_eva_error,error_rec

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten, LSTM, RepeatVector,TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

def train_test(data,n_out):    
    ratio = int(len(data)*.7)
    d_start = int(ratio-ratio%n_out)
    d_end = int(len(data)-len(data)%n_out)
    #train,test = X_in.iloc[:700,:].values,X_in.iloc[700:777,:].values
    train,test = data[:d_start,:],data[d_start:d_end,:]

    train = np.array(np.split(train, len(train)/n_out))
    test = np.array(np.split(test, len(test)/n_out))
    return train,test

def to_supervised(train, n_input, n_out=7):
    # flatten data
    try :
        data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    except:
        data = train.values
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out   
        # ensure we have enough data for this instance
        if out_end <= len(data):
            x_input = data[in_start:in_end, 0]
            x_input = x_input.reshape((len(x_input), 1))
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])   #Y is the first columns
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)
# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat
def build_model_seq2seq(train, n_input,validation=None):
    # define parameters
    train_x, train_y = to_supervised(train, n_input)
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    callback_early_stopping = EarlyStopping(monitor='val_loss',patience=5, verbose=2)
    #callbacks = [callback_early_stopping]
    callbacks = None

    # define model
    verbose, epochs, batch_size = 0, 50, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))                                  # Decoder 
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
    validation_data=validation,callbacks=callbacks)
    try:
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    except: pass
    return model

# evaluate a single model
def evaluate_model(train, test, n_input):
	# fit model
	model = build_model_seq2seq(train, n_input)
	# history is a list of weekly data
	history = [x for x in train]
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		# store the predictions
		predictions.append(yhat_sequence)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	# evaluate predictions days for each week
	predictions = np.array(predictions)
	score, mse_scores,nse_scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, mse_scores,nse_scores

def summarize_scores(name, score, scores,nse_score):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    n_scores = ', '.join(['%.1f' % s for s in nse_score])
    print('%s: [%.3f] %s' % (name, score, s_scores))
    print('%s: [%.3f] %s' % (name, score, n_scores))
def forecast(model, history, n_input):
	# flatten data
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat
def evaluate_forecasts(actual, predicted):
    mse_scores,nse_scores = list(),list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse,nse = real_eva_error(actual[:, i], predicted[:, i])
        # store
        mse_scores.append(mse)
        nse_scores.append(nse)
    return mse_scores,nse_scores

def move_column_inplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)
    return df

r='data/instant_data/rain_small.csv'
w='data/instant_data/water_small.csv'
rw = load_data(r,w)
df =rw.df.resample('d').mean().astype('float32')
data = df["2015-01-01":"2018-01-05"].interpolate(limit=360)
TARGET = 'CPY015_w'
data = move_column_inplace(data,TARGET,0)
data.head()
#SCALE
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])
n_input = 14
n_out = 7

def experiment(data,n_input,n_out):
    train, test = train_test(data.values,n_out=n_out)
    train_x,train_y = to_supervised(train,n_input)
    val_x, val_y = to_supervised(test, n_input)
    VALIDATION = (val_x,val_y)
    model = build_model_seq2seq(train,n_input,VALIDATION)
    # get prdict result as once
    trainPredict = model.predict(train_x)
    testPredict = model.predict(val_x)

    mse_scores,nse_scores = evaluate_forecasts(train_y, trainPredict)
    # Tscore, Tmse_scores,Tnse_scores = evaluate_forecasts(val_y, testPredict)
    print(mse_scores,nse_scores)
    # print(Tscore, Tmse_scores,Tnse_scores)

experiment(data,n_input,n_out)