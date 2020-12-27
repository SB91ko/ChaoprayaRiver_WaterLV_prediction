from sklearn.metrics import mean_squared_error,r2_score
from sklearn.metrics import mean_absolute_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

split_date = '2016-11-01'


def real_eva_error(Y,Y_hat):
    mse = np.mean((Y_hat - Y)**2)    #MSE
    nse = nashsutcliffe(Y,Y_hat)
    r2 = rsquared(Y,Y_hat)
    #---------added ------------#
    rmse = mse**.5  # RMSE
    mae = mean_absolute_error(Y,Y_hat)

    #except ValueError: mse,nse,r2,rmse,mae = np.NaN,np.NaN,np.NaN,np.NaN,np.NaN
    return mse,nse,r2,rmse,mae
def plot_rsquare(save_path,testY,testPredict,syn):
    _, _,Tr2,_,_ = real_eva_error(testY, testPredict,)
    m_test = monsoon_scope(testY)
    m_testPredict = monsoon_scope(testPredict)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(testY, testPredict,edgecolor='blue',facecolor='None',marker='.',label='other',alpha=0.3)       
    ax.scatter(m_test, m_testPredict,edgecolor='red',facecolor='None',marker='.',label='monsoon',alpha=0.3)       
    ax.plot([0, testY.max()+1], [0, testY.max()+1], 'b--', lw=1)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(syn+'\nR2: %.3f' % (Tr2))
    ax.legend(loc='upper left')
    fig.savefig(save_path+'/r2_{}.png'.format(syn), dpi=300, bbox_inches='tight') 
    fig.clear()
    plt.close(fig)
def monsoon_scope(df):
    monsoon=[8,9,10]
    # non_monsoon=[1,2,3,4,5,6,7,11,12]
    # df.iloc[(df.index.month.isin(non_monsoon))]=np.NaN
    # return df.dropna()
    return df.iloc[(df.index.month.isin(monsoon))]
def plot_moonson_l(save_path,trainY,testY,trainPredict,testPredict,syn):
    mse,nse,r2,rmse,mae = real_eva_error(monsoon_scope(trainY), monsoon_scope(trainPredict),)
    Tmse,Tnse,Tr2,Trmse,Tmae = real_eva_error(monsoon_scope(testY), monsoon_scope(testPredict),)

    def plot_out(monsoon,truemonsoon):
        fig,ax =  plt.subplots(2,1,figsize=(6.4, 4.8))
        fig.autofmt_xdate(rotation=45)
        if truemonsoon :
            fig.suptitle('\nMonsoon season:'+syn+'\nTrain MSE: %.3f | NSE: %.3f | MAE score: %.3f' % (mse,nse,mae)+'\nTest  MSE: %.3f | NSE: %.3f | MAE score: %.3f' % (Tmse,Tnse,Tmae))
        sel_m = trainY.iloc[(trainY.index.month.isin(monsoon))]
        Tsel_m = trainPredict.iloc[(trainPredict.index.month.isin(monsoon))]
        
        sel_2016 = sel_m.iloc[(sel_m.index.year.isin([2016]))]
        Tsel_2016 = Tsel_m.iloc[(Tsel_m.index.year.isin([2016]))]
        ax[0].plot(sel_2016,label='Y_2016')
        ax[0].plot(Tsel_2016,label='Yhat_2016')
        ax[0].set_title(' Year 2016')
        ax[0].legend(loc='upper left')

        sel_m = testY.iloc[(testY.index.month.isin(monsoon))]
        Tsel_m = testPredict.iloc[(testPredict.index.month.isin(monsoon))]
        sel_2017 = sel_m.iloc[(sel_m.index.year.isin([2017]))]
        Tsel_2017 = Tsel_m.iloc[(Tsel_m.index.year.isin([2017]))]
        ax[1].plot(sel_2017,label='Y_2017')
        ax[1].plot(Tsel_2017,label='Yhat_2017')
        ax[1].set_title('Year 2017')
        ax[1].legend(loc='upper left')

        plt.tight_layout()
        if truemonsoon: 
            fig.savefig(save_path+'/monsoon_line_{}.png'.format(syn), dpi=200, bbox_inches='tight') 
        else: 
            fig.savefig(save_path+'/zoom_line_{}.png'.format(syn), dpi=200, bbox_inches='tight') 
        fig.clear()
        plt.close(fig)
    monsoon = [8,9,10]
    plot_out(monsoon,True)
    general_zoom = [6,7]
    plot_out(general_zoom,False)
def monsoon_cal(trainY,testY,trainPredict,testPredict,syn):        
    m_trainPredict = monsoon_scope(trainPredict)
    m_testPredict = monsoon_scope(testPredict)
    m_train = monsoon_scope(trainY)
    m_test = monsoon_scope(testY)
    
    # plot_moonson_l(mode,save_path,trainY,testY,trainPredict,testPredict,syn)
    # plot_rsquare(mode,save_path,m_test,m_testPredict,syn+'monsoon')
    mse,nse,r2,rmse,mae = real_eva_error(m_train, m_trainPredict,)
    Tmse,Tnse,Tr2,Trmse,Tmae = real_eva_error(m_test, m_testPredict,)
    ###### CSV output######
    dict_data = {'Model_':syn,'MSE_trian*':mse,'Rmse_trian*':rmse,'NSE_train*':nse,'R2_train*':r2,'MAE_train*':mae,
    'MSE_test*':Tmse,'Rmse_test*':Trmse,'NSE_test*':Tnse,'R2_test*':Tr2,'MAE_test':Tmae}   
    _df = pd.DataFrame.from_dict(data=dict_data, orient ='index')
    return _df

##################################
def plotgraph(target,save_path,trainY,testY,trainPredict,testPredict,syn):
    mse,nse,r2,rmse,mae = real_eva_error(trainY, trainPredict,)
    Tmse,Tnse,Tr2,Trmse,mae = real_eva_error(testY, testPredict,)

    fig,ax = plt.subplots(1,figsize=(15,5))
    # plt.plot(trainPredict.index,trainPredict, label = "Predict_train",color='g')
    trainPredict.plot(label = "Predict_train",color='g',lw=1)
    testPredict.plot(label = "Predict_test",color='r',lw=1)
    ax.scatter(x=trainY.index,y=trainY,marker='.',label='Actual_train',alpha=0.3,edgecolor='lightskyblue',facecolor='None')
    ax.scatter(x=testY.index,y=testY,marker='.',label='Actual_test',alpha=0.3,edgecolor='orange',facecolor='None')

    ax.set_title('[{}]\n'.format(syn)+'Water Level {} Forecast vs Actuals\n'.format(target)+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse,nse,r2)+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse,Tnse,Tr2))
    ax.legend(loc='upper left', fontsize=8)
    fig.savefig(save_path+'/Plot_{}.png'.format(syn), dpi=300, bbox_inches='tight') 
    fig.clf()
    fig.clear()
    plt.close(fig)
def list_eva_error(Y,Y_hat,n_out):
    mse_l,nse_l,r2_l = list(),list(),list()
    rmse_l,mae_l = list(),list()
    if n_out==1: real_eva_error(Y,Y_hat)
    for i in range(n_out):
        try: mse,nse,r2,rmse,mae=real_eva_error(Y[:,i],Y_hat[:,i])
        except: mse,nse,r2,rmse,mae=real_eva_error(Y[:,i].reshape(-1,1),Y_hat[:,i])
        mse_l.append(mse)
        nse_l.append(nse)
        r2_l.append(r2)
        rmse_l.append(rmse)
        mae_l.append(mae)

    return mse_l,nse_l,r2_l,rmse_l,mae_l
def record_list_result(syn,df,mode,trainY,testY,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future=1):
    # mse_l, nse_l,r2_l = list_eva_error(trainY, trainPredict,n_future)
    # Tmse_l, Tnse_l,Tr2_l = list_eva_error(testY, testPredict,n_future)
    idx=None
    try:
        error = pd.read_csv(save_path+'/eval.csv',index_col=False)
        error = error.T
        error.reset_index(drop=True, inplace=True)
    except:
        error = pd.DataFrame()
        print('-- created new file --')

    for d in range(n_future):
        st_idx = n_past+d
        
        Y_tr= pd.Series(data=trainY[:,d],index=df[:split_date].index[st_idx:len(trainY)+st_idx])
        Yhat_tr = pd.Series(data=(trainPredict[:,d].ravel()),index=df[:split_date].index[st_idx:len(trainY)+st_idx])        
        Y_t= pd.Series(data=testY[:,d],index=df[split_date:].index[st_idx:len(testY)+st_idx])
        Yhat_t = pd.Series(data=(testPredict[:,d].ravel()),index=df[split_date:].index[st_idx:len(testY)+st_idx])

        mse,nse,r2,rmse,mae = real_eva_error(Y_tr, Yhat_tr,)
        Tmse, Tnse,Tr2,Trmse,Tmae = real_eva_error(Y_t, Yhat_t,)
        #------------ Plot trian-test ------------------#
        syn_new = syn+'_t'+str(d+1)
        if d in [0,11,23,47,71]: 
            plotgraph(target,save_path,Y_tr,Y_t,Yhat_tr,Yhat_t,syn_new) 
            plot_moonson_l(save_path,Y_tr,Y_t,Yhat_tr,Yhat_t,syn)
            plot_rsquare(save_path,Y_t,Yhat_t,syn_new)

        mon_df = monsoon_cal(Y_tr,Y_t,Yhat_tr,Yhat_t,syn_new) 
        
        
        #-----------------------
        print('{} saved !'.format(d+1))
        #---------------------
        dict_data = {'note':mode,'Model':syn,'timestep':d+1,'Feature':n_features,'Time_in':n_past,'Time_out':n_future,'Batch':batch_size,
                    'MSE_trian':mse,'Rmse_trian':rmse,'NSE_train':nse,'R2_train':r2,'MAE_train':mae,
                    'MSE_test':Tmse,'Rmse_test':Trmse,'NSE_test':Tnse,'R2_test':Tr2,'MAE_test':Tmae} 
        _df = pd.DataFrame.from_dict(data=dict_data, orient ='index')
        _df = pd.concat([_df,mon_df])
        
        idx=_df.index
        
        _df.reset_index(drop=True, inplace=True)
        error.reset_index(drop=True, inplace=True)
        error = pd.concat([error,_df],axis=1)
        try: 
            error.set_index(idx,inplace=True)
        except:
            idx=idx.insert(0,'0')
            error.set_index(idx,inplace=True)
    error.T.to_csv(save_path+'/eval.csv',index=False)
    return testY,testPredict
def record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1,rec_result=False):
    mse,nse,r2,rmse,mae = real_eva_error(trainY, trainPredict,)
    Tmse, Tnse,Tr2,Trmse,Tmae = real_eva_error(testY, testPredict,)
    try:
        trainPredict = pd.Series(data=(trainPredict),index=trainY.index)
        testPredict = pd.Series(data=(testPredict),index=testY.index)
    except:  pass
    ##################################
    plotgraph(target,save_path,trainY,testY,trainPredict,testPredict,syn)
    mon_df = monsoon_cal(trainY,testY,trainPredict,testPredict,syn)
    ########### R-square ################
    plot_rsquare(save_path,testY,testPredict,syn)
    plot_moonson_l(save_path,trainY,testY,trainPredict,testPredict,syn)
    ###### CSV output######
    dict_data = {'Model':syn,'Date':n_past,'Feature':n_features,'Time_in':n_past,'Time_out':n_future,'Use time':use_time,
                'MSE_trian':mse,'Rmse_trian':rmse,'NSE_train':nse,'R2_train':r2,'MAE_train':mae,
                'MSE_test':Tmse,'Rmse_test':Trmse,'NSE_test':Tnse,'R2_test':Tr2,'MAE_test':Tmae} 
    
    _df = pd.DataFrame.from_dict(data=dict_data, orient ='index')
    _df = pd.concat([_df,mon_df])
    
    try:
        error = pd.read_csv(save_path+'/eval.csv',index_col=False)
        error = error.T
        error.reset_index(drop=True, inplace=True)
    except:
        error = pd.DataFrame()
        print('-- created new file --')
    
    idx=_df.index
    _df.reset_index(drop=True, inplace=True)
    error.reset_index(drop=True, inplace=True)
    error = pd.concat([error,_df],axis=1)
    try: 
        error.set_index(idx,inplace=True)
    except:
        idx=idx.insert(0,'0')
        error.set_index(idx,inplace=True)
    error.T.to_csv(save_path+'/eval.csv',index=False)

    ##########################
    if rec_result: 
        res_train=pd.DataFrame({'model':syn,'Y':trainY,'Yhat':trainPredict,'type': 'train'})
        res_test=pd.DataFrame({'model':syn,'Y':testY,'Yhat':testPredict,'type': 'test'})

        result_ = pd.concat([res_train,res_test],axis=0)
        result_.to_csv(save_path+'/result{}.csv'.format(syn))
    ######################
    return testY,testPredict

#################################################################
import logging
logging.basicConfig(format='%(levelname)s: %(module)s.%(funcName)s(): %(message)s')
def rsquared(evaluation, simulation):
    """
    Coefficient of Determination
        .. math::
         r^2=(\\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}})^2
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Coefficient of Determination
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        return correlationcoefficient(evaluation, simulation)**2
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan
def correlationcoefficient(evaluation, simulation):
    """
    Correlation Coefficient
        .. math::
         r = \\frac{\\sum ^n _{i=1}(e_i - \\bar{e})(s_i - \\bar{s})}{\\sqrt{\\sum ^n _{i=1}(e_i - \\bar{e})^2} \\sqrt{\\sum ^n _{i=1}(s_i - \\bar{s})^2}}
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Corelation Coefficient
    :rtype: float
    """
    if len(evaluation) == len(simulation):
        correlation_coefficient = np.corrcoef(evaluation, simulation)[0, 1]
        return correlation_coefficient
    else:
        logging.warning("evaluation and simulation lists does not have the same length.")
        return np.nan
def nashsutcliffe(y, yhat):
    """
    Nash-Sutcliffe model efficinecy
        .. math::
         NSE = 1-\\frac{\\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\\sum_{i=1}^{N}(e_{i}-\\bar{e})^2} 
    :evaluation: Observed data to compared with simulation data.
    :type: list
    :simulation: simulation data to compared with evaluation data
    :type: list
    :return: Nash-Sutcliff model efficiency
    :rtype: float
    """
    if len(y) == len(yhat):
        simulation_yhat, evaluation_y = np.array(yhat), np.array(y)
        # s,e=simulation(Yhat),evaluation(Y)
        mean_observed = np.nanmean(evaluation_y)
        # compute numerator and denominator
        numerator = np.nansum((evaluation_y - simulation_yhat) ** 2)
        denominator = np.nansum((evaluation_y - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)
    else:
        print("evaluation and simulation lists does not have the same length.")
        return np.nan