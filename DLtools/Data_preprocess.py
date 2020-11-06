import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.ops.gen_math_ops import requantization_range
np.random.seed(42)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    #Source: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
    try:
        n_vars = data.shape[1]
        col_name = data.columns
    except IndexError:
        n_vars = 1
        col_name = ['var']
    except:
        n_vars = data.shape[1]
        col_name = ["var{}".format(i+1) for i in range(data.shape[1])] 

    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('{}(t-{})'.format(j, i)) for j in col_name]
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('{}(t)'.format(j)) for j in col_name]
        else:
            names += [('{}(t+{})'.format(j, i)) for j in col_name]
	# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

class load_data:
    def __init__(self,rain,water,start=None,stop=None):
        self.start, self.stop = start,stop
        self.water,self.rain = water,rain
        
        self.water_df = self.df_maker(self.water,'_w')
        self.rain_df = self.df_maker(self.rain,'_r').resample('H').pad()
        self.df = self.rain_water_merge(self.rain_df,self.water_df)
        print('DataFrame shape:',self.df.shape)

    def df_maker(self,csvfile,syn=''):
        df = pd.read_csv(csvfile,index_col=['date'],parse_dates=['date'])
        df.rename(columns=lambda x: x+syn, inplace=True)
        return df[self.start:self.stop]

    def rain_water_merge(self,rain,water):
        return pd.concat([rain,water])

def manual_trian_test_split(input_series,look_back):
    train_size = int(len(input_series) * 0.7)
    train, test = input_series[:train_size], input_series[train_size:len(input_series)]
    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    X_train, Y_train = create_dataset(train.reshape(-1,1), look_back)
    X_test, Y_test = create_dataset(test.reshape(-1,1), look_back)
    print('Shape of train:...',X_train.shape,Y_train.shape)
    print('Shape of test:...',X_test.shape,Y_test.shape)
    return X_train, Y_train, X_test, Y_test

class preprocess:
    def __init__(self,data, Y_column, n_timelag,n_feature,t_predict_ahead=1,scale_ornot=False):
        self.data = data
        self.Y_column = Y_column
        self.n_timelag = n_timelag
        self.n_feature = n_feature
        self.t_predict_ahead = t_predict_ahead
        self.scale_ornot =scale_ornot
        self.scaler = self.scaler_transform()

        self.ori_X,self.ori_Y =None,None
        self.test_X,self.test_Y = None,None
        self.train_X,self.train_Y, = None,None
        self.val_X,self.val_y = None,None
        self.prepare_data()

    def scaler_transform(self):
        if self.scale_ornot:
            scaler = MinMaxScaler()
            self.data[self.data.columns] = scaler.fit_transform(self.data[self.data.columns]) # scale each columns
        return 


    def prepare_data(self):
    # ensure all data is float
        values = self.data
        target = self.Y_column
        # frame as supervised learning
        reframed_data = series_to_supervised(values, n_in= self.n_timelag,n_out=self.t_predict_ahead)
        print("="*50,'preview Supervised',"="*50)
        print(reframed_data.head(5))
        print(reframed_data.tail(5))

        print(reframed_data.values.shape)
        print(len(reframed_data))
        print("="*50,'X-Y Shape',"="*50)
        # split into input and outputs
        X = reframed_data.iloc[:,:-self.n_feature]                # Eliminate last set of var(t)
        if self.Y_column is None:
            Y = reframed_data.iloc[:,-1].values                          # last columns as output   
        else:
            Y = reframed_data[str(target+'(t)')]
        self.ori_X,self.ori_Y =X,Y
        
        print(X.shape,"||******call ____ ori_x******")
        print(Y.shape,"||******call ____ ori_y******")
        try: 
            X,Y = X.values,Y.values.reshape(-1,1)
        except AttributeError:
            X= X.values
            Y = Y.reshape(-1,1)
        ###################EDIT2 #############################
        # X = reframed_data.iloc[:,:-self.n_feature].values                # Eliminate last set of var(t)
        # if self.Y_column is None:
        #     Y = reframed_data.iloc[:,-1].values                                   #last columns as output
            
        # else:
        #     Y = reframed_data[str(target+'(t)')].values
        # Y = Y.reshape(-1,1)
        # print(X.shape)
        # print(Y.shape)
        # ####################EDITED PARTED################
        #re_data = reframed_slide_windows.values
        #X = re_data[:,0:(self.n_timelag*self.n_feature)]   
        #Y = re_data[self.Y_column]
        #################################################
                
        len_train = int(len(reframed_data)*0.7)                   #Train test _7:3
        trainX,trainY=X[:len_train,:], Y[:len_train,:]
        testX,testY=X[len_train:,:], Y[len_train:,:]       
        train_X, val_X, train_y, val_y = train_test_split(trainX, trainY, test_size=0.3) #Val data 7:3
        
        print("="*50,'Train....Validation...Test',"="*50)
        print("Train:Test",trainX.shape, trainY.shape,testX.shape,testY.shape)
        print("Train....",train_X.shape, train_y.shape,"||")
        print("Val......",val_X.shape, val_y.shape,"||")
        
        # reshape input to be 3D for deeplearning [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], self.n_timelag, self.n_feature))
        val_X = val_X.reshape((val_X.shape[0], self.n_timelag, self.n_feature))
        test_X = testX.reshape((testX.shape[0], self.n_timelag, self.n_feature))
        
        ####Prep for report result#######
        trainX = trainX.reshape((trainX.shape[0], self.n_timelag, self.n_feature))
        self.train_val_X,self.train_val_Y =trainX,trainY
        ########

        self.train_X,self.train_Y = train_X,train_y 
        self.test_X,self.test_Y = test_X,testY
        self.val_X,self.val_y = val_X,val_y
        print("="*50,'[Reshape Train for 3D]Train....Validation...Test',"="*50)
        print("Train:Test",trainX.shape, trainY.shape,testX.shape,testY.shape,"|| ori_trainX,ori_trainY....test_X,test_Y")
        print("Train....",train_X.shape, train_y.shape,"|| train_X, train_Y")
        print("Val......",val_X.shape, val_y.shape,"|| val_X, val_Y")
        #return train_X,train_y,test_X,test_y,val_X,val_y
        return

def dummy_Example():
    #test = pd.read_csv('data/instant_data/rain_small.csv')
    # test = [x for x in range(10)]
    # print(series_to_supervised(test,n_in=5,n_out=2))

    # raw = pd.DataFrame()
    # raw['ob1'] = [x for x in range(10)]
    # raw['ob2'] = [x for x in range(50, 60)]
    # print(series_to_supervised(raw,n_in=7,n_out=1,dropnan=False))
    ######## reshape input###########

    # #Example: 
    # data = []
    # n=5000
    # for i in range(n):
    #     data.append([i+1,(i+1)*10])
    # data = np.array(data)
    # print(data.shape)
    # data[:,1]


    # samples =[]
    # timelag = 50
    # # for i in range(0,n,timelag):
    # #     #grab from i to i+200 batch
    # #     s = data[i:i+timelag]
    # #     samples.append(s)
    # data = data.reshape((len(samples),timelag,1))
    # print(data.shape) 
    pass
