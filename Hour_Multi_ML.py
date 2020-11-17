import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split
from DLtools.Data import instant_data,intersection,station_sel,del_less_col
from DLtools.evaluation_rec import record_list_result
from DLtools.feature_sel import call_mar

def high_corr(data,threshold=.95):
    """Eliminate first columns with high corr"""
    corr_matrix = data.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop
def corr_w_Y(data,target,threshold= 0.8):
    # correlation
    corr_test = data.corr(method='pearson')[target]
    corr_test = corr_test[(corr_test> threshold) | (corr_test< -threshold) ]
    corr_test = corr_test.sort_values(ascending=False)
    #corr_test =corr_test[1:] # eliminate Target it own
    return corr_test
def corr_select(data,target):
    col_feature = corr_w_Y(data,target,0.5).index
    data = data[col_feature]
    high_col = high_corr(data.iloc[:,1:]) #exclude target it own
    data.drop(columns=high_col,inplace=True)
    return data
def plot_corr(data,syn):
    global n_past
    ##Display / save
    corr = data.corr()
    plt.subplots(figsize=(10,10))
    mask = np.triu(data.corr())
    sns.heatmap(data.corr(), annot = True, vmin=-1, vmax=1, center= 0,mask=mask)
    plt.savefig(save_path+'Corr_{}lag{}.png'.format(syn,n_past), bbox_inches='tight')
    return

if __name__ == "__main__":
    loading = instant_data()
    ###########################################
    df = loading.hourly_instant()
    # df = loading.daily_instant()
    syn = ''
    mode = 'hour'
    st = 'CPY012'
    
    target,start_p,stop_p,host_path=station_sel(st,mode)
    save_path =host_path+'/DL/'
    ##########################################

    if mode =='hour': n_past,n_future = 24*7,24
    elif mode =='day': n_past,n_future = 30,14
    else: n_future=None; print('incorrect input')


    data = df[start_p:stop_p]
    data = del_less_col(data,ratio=.85)
    data['Day'] = data.index.dayofyear #add day
    data = data.interpolate(limit=30000000,limit_direction='both').astype('float32')
    
    for n_past in tqdm(range(1,n_future+1)):
        data = df.astype('float32')#interpolate neighbor first, for rest NA fill with mean()
        
        data[target]=data[target].shift(-n_past).dropna()
        
        #### Corr selection##
        # data = corr_select(data,TARGET)
        #### MAR selection ##
        data = call_mar(data,target,mode)

        #### plot ###
        plot_corr(data,syn)

        X = data.drop(columns=[target])
        Y = data[target]
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, shuffle=False)

        regr = linear_model.LinearRegression()
        regr.fit(trainX,trainY)
        trainPredict = regr.predict(trainX)
        testPredict = regr.predict(testX)
        # try:

        record_list_result(trainY,testY,trainPredict,testPredict,syn)
        # except:
        #     print('error on ',n_past)