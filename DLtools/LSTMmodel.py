import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import pandas as pd

class prepLSTM:
    train_split_ratio = 0.8 #8:2 train:test
    # don't forget to scale data first

    def __init__(self,input_df,y_series,timelag,batch_size=256):
        self.input_df = input_df        #df
        self.y_series = y_series        #df['target']
        self.timelag = timelag
        self.batch_size = batch_size

        self.X,self.y = self.value_point()
        self.num_x_feature = self.X.shape[1]
        self.num_y_feature = self.y.shape[1]
        self.num_data = len(self.X)
        self.num_train = int(self.train_split_ratio * self.num_data)
        self.x_train,self.y_train,self.x_test,self.y_test = self.train_test_split()
        
        self.x_batch_shape = (self.batch_size, self.timelag, self.num_x_feature)
        self.y_batch_shape = (self.batch_size, self.timelag, self.num_y_feature)


        self.generator = self.batch_gen()
        self.x_batch, self.y_batch = next(self.generator)

    def value_point(self):
        X = self.input_df.values[:-self.timelag]    # alinge x,y in same shape, del last NAN value (since y was shifted)

        y = self.y_series.shift(-self.timelag)      # shift y to future timeline
        y = y.values[:-self.timelag]                # alinge x,y in same shape, del last NAN value (since y was shifted)
        y = y.reshape(-1,1)
        return X,y

    def train_test_split(self):
        X,y = self.value_point()

        x_train = X[0:self.num_train]
        x_test = X[self.num_train:]
        y_train = y[0:self.num_train]
        y_test = y[self.num_train:]
        return x_train,y_train,x_test,y_test

    def batch_gen(self):
        while True:
            x_batch = np.zeros(shape = self.x_batch_shape, dtype=np.float16)
            y_batch = np.zeros(shape = self.y_batch_shape, dtype=np.float16)
            #fill batch with random sequences of data
            for i in range(self.batch_size):                
                #Get rand start index,
                idx = np.random.randint(self.num_train - self.timelag)
                
                #copy sequence of data start at this index.
                x_batch[i] = self.x_train[idx:idx+self.timelag]
                y_batch[i] = self.y_train[idx:idx+self.timelag]
                
            yield (x_batch, y_batch)      

    def report(self):
        print("\nFrom input X shape:{}.......target y shape:{}\n".format(self.X.shape,self.y.shape))
        print("Train X: {}, Train y:{}  |  Test X:{}, Test y:{}".format(self.x_train.shape,self.y_train.shape,self.x_test.shape,self.y_test.shape))
        print("Feature X:{}...............Feature y:{}".format(self.num_x_feature,self.num_y_feature))
        print("Batch gen to final shape .......X:{}......y:{}\n".format(self.x_batch_shape,self.y_batch_shape))
        
    def scale(self):
        self.water_scaler = MinMaxScaler()
        water_scaler_df =  self.water_scaler.fit_transform(self.water_df)
        self.rain_scaler = MinMaxScaler()
        rain_scaler_df =  self.rain_scaler.fit_transform(self.rain_df)
        #return self.rain_water_merge(water_scaler_df,rain_scaler_df)
        return water_scaler_df
    
    def validation_data(self):
        return (np.expand_dims(self.x_test, axis=0),np.expand_dims(self.y_test, axis=0))

"""rain=pd.read_csv("data/instant_data/rain.csv",index_col=['date'],parse_dates=['date'])
lstm = prepLSTM(rain,rain['WGTG'],7)
validation_data = lstm.validation_data()
print(validation_data[0].shape)
"""