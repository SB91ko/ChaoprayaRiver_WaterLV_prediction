import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split
from DLtools.Data import instant_data,intersection
from DLtools.evaluation_rec import real_eva_error, error_rec

def record_result(trainY,testY,trainPredict,testPredict,syn):
    global save_path,error,n_past,TARGET
    
    mse, nse,r2 = real_eva_error(trainY, trainPredict)
    Tmse, Tnse,Tr2 = real_eva_error(testY, testPredict)

    index = np.arange(len(trainY)+len(testY))
    Y= pd.Series(data=trainY.values,index=index[:len(trainY)])
    Yhat = pd.Series(data=(trainPredict),index=index[:len(trainY)])
    Y_t= pd.Series(data=testY.values,index=index[-len(testY):])
    Yhat_t = pd.Series(data=(testPredict),index=index[-len(testY):])
    ##### Plot
    plt.figure(figsize=(20,5))
    plt.plot(Y, label = "Actual")
    plt.plot(Yhat, label = "Predict")
    plt.plot(Y_t, label = "Actual_test")
    plt.plot(Yhat_t, label = "Predict_test")
    
    try:
        plt.title('[Multi Linear Regression] ahead {} day'.format(n_past)+'\nWater Level {} Forecast vs Actuals\n'.format(TARGET)+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse,nse,r2)+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse,Tnse,Tr2))
    except:
        plt.title('[Multi Linear Regression] ahead {} day'.format(n_past)+'\nWater Level {} Forecast vs Actuals\n'.format(TARGET)+'Train MSE: {} | NSE: {} | R2 score: {}'.format(mse,nse,r2)+'\nTest  MSE: {} | NSE: {} | R2 score: {}'.format(Tmse,Tnse,Tr2))
    plt.legend()
    plt.savefig(save_path+'result_Linear{}_{}day.png'.format(syn,n_past), dpi=300, bbox_inches='tight')
    ###### CSV output######
    idx=['Modelname','Feature','n_in_time','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2','Intercept','Coefficients']
    col = ['MultiLinearReg']
    _df = pd.DataFrame(["MultiLinearReg_{}".format(n_past),len([data.columns]),"None",'None',mse, nse,r2,Tmse, Tnse,Tr2,[regr.intercept_], [regr.coef_]],index=idx,columns=col)
    error = pd.concat([error,_df],axis=1)
    error.to_csv('/home/song/Public/Song/Work/Thesis/output_cpy012/Hourly/eval.csv')

def call_mar(data):
    mar = pd.read_csv('/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_hourly.csv')
    col = [i for i in data.columns]
    select_col = intersection(col,mar['feature'])

    select_col.append(TARGET) # add target
    return data[select_col]

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


save_path='/home/song/Public/Song/Work/Thesis/output_cpy012/Hourly/Linear/'
#load previous error rec
idx=['Modelname','Feature','n_in_time','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2']
try: error = pd.read_csv('/home/song/Public/Song/Work/Thesis/output_cpy012/Hourly/eval.csv',index_col=0)
except: error = pd.DataFrame(index = idx)

if __name__ == "__main__":
    loading = instant_data()
    df_h = loading.hourly_instant()
    df_d = loading.daily_instant()

    df = df_h["2014-02-01":"2018-03-31"].interpolate(limit=30000000)
    TARGET = 'CPY012_wl'
    
    for n_past in tqdm(range(1,30)):
        data = df.apply(lambda x: x.fillna(x.mean()),axis=0).astype('float32')#interpolate neighbor first, for rest NA fill with mean()
        ###### Setup #####
        syn = 'MAR'
        #######
        data[TARGET]=data[TARGET].shift(-n_past).dropna()
        #### Corr selection##
        # data = corr_select(data,TARGET)
        #### MAR selection ##
        data = call_mar(data)

        #### plot ###
        plot_corr(data,syn)

        X = data.drop(columns=[TARGET]).fillna(0)
        Y = data[TARGET].fillna(0)
        trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, shuffle=False)

        regr = linear_model.LinearRegression()
        regr.fit(trainX,trainY)
        trainPredict = regr.predict(trainX)
        testPredict = regr.predict(testX)
        # try:
        
        record_result(trainY,testY,trainPredict,testPredict,syn)
        # except:
        #     print('error on ',n_past)