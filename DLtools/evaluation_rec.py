from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def real_eva_error(Y,Y_hat):
    try: 
        mse = np.mean((Y_hat - Y)**2)       #MSE
        nse = nashsutcliffe(Y,Y_hat)
        r2 = rsquared(Y,Y_hat)
    except ValueError: mse,nse,r2 = np.NaN,np.NaN,np.NaN
    
    return mse,nse,r2
def plot_rsquare(mode,save_path,testY,testPredict,syn):
    _, _,Tr2 = real_eva_error(testY, testPredict,)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(testY, testPredict,edgecolor='red',facecolor='None',marker='.')       
    ax.plot([0, testY.max()+1], [0, testY.max()+1], 'b--', lw=1)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(syn+'\nR2: %.3f' % (Tr2))
    plt.savefig(save_path+'/r2_{}_{}.png'.format(syn,mode), dpi=300, bbox_inches='tight') 
def monsoon_scope(df):
    monsoon=[8,9,10]
    # non_monsoon=[1,2,3,4,5,6,7,11,12]
    # df.iloc[(df.index.month.isin(non_monsoon))]=np.NaN
    # return df.dropna()
    return df.iloc[(df.index.month.isin(monsoon))]

def plot_moonson(mode,save_path,trainY,testY,trainPredict,testPredict,syn):
    mse, nse,r2 = real_eva_error(trainY, trainPredict,)
    Tmse, Tnse,Tr2 = real_eva_error(testY, testPredict,)
    
    fig,ax =  plt.subplots(2,2,figsize=(15,7))
    monsoon = [8,9,10]
    fig.autofmt_xdate(rotation=45)
    fig.suptitle('\nMonsoon season:'+syn+'\nTrain MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse,nse,r2)+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse,Tnse,Tr2))
    sel_m = trainY.iloc[(trainY.index.month.isin(monsoon))]
    Tsel_m = trainPredict.iloc[(trainPredict.index.month.isin(monsoon))]

    sel_2014 = sel_m.iloc[(sel_m.index.year.isin([2014]))]
    Tsel_2014 = Tsel_m.iloc[(Tsel_m.index.year.isin([2014]))]
    ax[0][0].scatter(sel_2014.index,sel_2014,label='2014',alpha=0.2,edgecolor='blue',facecolor='None',marker='o')
    ax[0][0].plot(Tsel_2014,label='2014',color='g',lw=1)
    ax[0][0].set_title('(train) Year 2014')
    ax[0][0].legend()

    sel_2015 = sel_m.iloc[(sel_m.index.year.isin([2015]))]
    Tsel_2015 = Tsel_m.iloc[(Tsel_m.index.year.isin([2015]))]
    ax[0][1].scatter(sel_2015.index,sel_2015,label='2015',alpha=0.2,edgecolor='blue',facecolor='None',marker='o')
    ax[0][1].plot(Tsel_2015,label='2015',color='g',lw=1)
    ax[0][1].set_title('(train) Year 2015')
    ax[0][1].legend()
    
    sel_2016 = sel_m.iloc[(sel_m.index.year.isin([2016]))]
    Tsel_2016 = Tsel_m.iloc[(Tsel_m.index.year.isin([2016]))]
    ax[1][0].scatter(sel_2016.index,sel_2016,label='2016',alpha=0.2,edgecolor='blue',facecolor='None',marker='o')
    ax[1][0].plot(Tsel_2016,label='2016',color='g',lw=1)
    ax[1][0].set_title('(train) Year 2016')
    ax[1][0].legend()

    sel_m = testY.iloc[(testY.index.month.isin(monsoon))]
    Tsel_m = testPredict.iloc[(testPredict.index.month.isin(monsoon))]
    sel_2017 = sel_m.iloc[(sel_m.index.year.isin([2017]))]
    Tsel_2017 = Tsel_m.iloc[(Tsel_m.index.year.isin([2017]))]
    ax[1][1].scatter(sel_2017.index,sel_2017,label='2017',alpha=0.2,edgecolor='red',facecolor='None',marker='o')
    ax[1][1].plot(Tsel_2017,label='2017',color='orange',lw=1)
    ax[1][1].set_title('(test) Year 2017')
    ax[1][1].legend()

    plt.tight_layout()
    plt.savefig(save_path+'/monsoon_{}_{}.png'.format(syn,mode), dpi=300, bbox_inches='tight') 

def plot_moonson_l(mode,save_path,trainY,testY,trainPredict,testPredict,syn):
    mse, nse,r2 = real_eva_error(monsoon_scope(trainY), monsoon_scope(trainPredict),)
    Tmse, Tnse,Tr2 = real_eva_error(monsoon_scope(testY), monsoon_scope(testPredict),)

    fig,ax =  plt.subplots(2,2,figsize=(15,7))
    monsoon = [8,9,10]
    fig.autofmt_xdate(rotation=45)
    fig.suptitle('\nMonsoon season:'+syn+'\nTrain MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse,nse,r2)+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse,Tnse,Tr2))
    sel_m = trainY.iloc[(trainY.index.month.isin(monsoon))]
    Tsel_m = trainPredict.iloc[(trainPredict.index.month.isin(monsoon))]

    sel_2014 = sel_m.iloc[(sel_m.index.year.isin([2014]))]
    Tsel_2014 = Tsel_m.iloc[(Tsel_m.index.year.isin([2014]))]
    ax[0][0].plot(sel_2014,label='Y_2014')
    ax[0][0].plot(Tsel_2014,label='Yhat_22014')
    ax[0][0].set_title('(train) Year 2014')
    ax[0][0].legend()

    sel_2015 = sel_m.iloc[(sel_m.index.year.isin([2015]))]
    Tsel_2015 = Tsel_m.iloc[(Tsel_m.index.year.isin([2015]))]
    ax[0][1].plot(sel_2015,label='Y_2015')
    ax[0][1].plot(Tsel_2015,label='Yhat_22015')
    ax[0][1].set_title('(train) Year 2015')
    ax[0][1].legend()
    
    sel_2016 = sel_m.iloc[(sel_m.index.year.isin([2016]))]
    Tsel_2016 = Tsel_m.iloc[(Tsel_m.index.year.isin([2016]))]
    ax[1][0].plot(sel_2016,label='Y_2016')
    ax[1][0].plot(Tsel_2016,label='Yhat_2016')
    ax[1][0].set_title('(train) Year 2016')
    ax[1][0].legend()

    sel_m = testY.iloc[(testY.index.month.isin(monsoon))]
    Tsel_m = testPredict.iloc[(testPredict.index.month.isin(monsoon))]
    sel_2017 = sel_m.iloc[(sel_m.index.year.isin([2017]))]
    Tsel_2017 = Tsel_m.iloc[(Tsel_m.index.year.isin([2017]))]
    ax[1][1].plot(sel_2017,label='Y_2017',color='orange')
    ax[1][1].plot(Tsel_2017,label='Yhat_22017',color='red')
    ax[1][1].set_title('(test) Year 2017')
    ax[1][1].legend()

    plt.tight_layout()
    plt.savefig(save_path+'/monsoon_line_{}_{}.png'.format(syn,mode), dpi=300, bbox_inches='tight') 
  
def monsoon_cal(mode,target,save_path,trainY,testY,trainPredict,testPredict,syn):        
    m_trainPredict = monsoon_scope(trainPredict)
    m_testPredict = monsoon_scope(testPredict)
    m_train = monsoon_scope(trainY)
    m_test = monsoon_scope(testY)
    # plot_moonson(mode,save_path,trainY,testY,trainPredict,testPredict,syn)
    plot_moonson_l(mode,save_path,trainY,testY,trainPredict,testPredict,syn)
    plot_rsquare(mode,save_path,m_test,m_testPredict,syn+'monsoon')
    mse, nse,r2 = real_eva_error(m_train, m_trainPredict,)
    Tmse, Tnse,Tr2 = real_eva_error(m_test, m_testPredict,)
    ###### CSV output######
    idx=['Modelname','mse','nse','r2','Test_mse','Test_nse','Test_r2']
    _df = pd.DataFrame(["{}".format(syn),mse, nse,r2,Tmse, Tnse,Tr2],index=idx,columns=[syn+'monsoon'])
    try: error = pd.read_csv(save_path+'/eval_monsoon.csv',index_col=0);print('LOAD SUCEESS')
    except: error = pd.DataFrame();print("cannot find rec")
    error = pd.concat([error,_df],axis=1)
    error.to_csv(save_path+'/eval_monsoon.csv')

##################################
def plotgraph(mode,target,save_path,trainY,testY,trainPredict,testPredict,syn):
    mse, nse,r2 = real_eva_error(trainY, trainPredict,)
    Tmse, Tnse,Tr2 = real_eva_error(testY, testPredict,)

    plt.figure(figsize=(15,5))
    # plt.plot(trainPredict.index,trainPredict, label = "Predict_train",color='g')
    trainPredict.plot(label = "Predict_train",color='g',lw=1)
    testPredict.plot(label = "Predict_test",color='r',lw=1)
    plt.scatter(x=trainY.index,y=trainY,marker='.',label='Actual_train',alpha=0.3,edgecolor='b',facecolor='None')
    plt.scatter(x=testY.index,y=testY,marker='.',label='Actual_test',alpha=0.3,edgecolor='orange',facecolor='None')

    plt.title('[{}] {}\n'.format(syn,mode)+'Water Level {} Forecast vs Actuals\n'.format(target)+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse,nse,r2)+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse,Tnse,Tr2))
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig(save_path+'/Plot_{}_{}.png'.format(syn,mode), dpi=300, bbox_inches='tight') 
    plt.clf()


def list_eva_error(Y,Y_hat,n_out):
    mse_l,nse_l,r2_l = list(),list(),list()
    if n_out==1: real_eva_error(Y,Y_hat)
    for i in range(n_out):
        try: mse,nse,r2=real_eva_error(Y[:,i],Y_hat[:,i])
        except: mse,nse,r2=real_eva_error(Y[:,i].reshape(-1,1),Y_hat[:,i])
        mse_l.append(mse)
        nse_l.append(nse)
        r2_l.append(r2)
    return mse_l,nse_l,r2_l
def record_list_result(syn,df,mode,trainY,testY,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future=1,scaler_t=None):
    if scaler_t !=None:
        trainY = scaler_t.inverse_transform(trainY)
        trainPredict = scaler_t.inverse_transform(trainPredict.reshape(trainY.shape))
        testY = scaler_t.inverse_transform(testY)
        testPredict = scaler_t.inverse_transform(testPredict.reshape(testY.shape))
    # mse_l, nse_l,r2_l = list_eva_error(trainY, trainPredict,n_future)
    # Tmse_l, Tnse_l,Tr2_l = list_eva_error(testY, testPredict,n_future)
    for d in range(n_future):
        st_idx = n_past+d
        
        Y_tr= pd.Series(data=trainY[:,d],index=df[:'2017-01-01'].index[st_idx:len(trainY)+st_idx])
        Yhat_tr = pd.Series(data=(trainPredict[:,d].ravel()),index=df[:'2017-01-01'].index[st_idx:len(trainY)+st_idx])        
        Y_t= pd.Series(data=testY[:,d],index=df['2017-01-01':].index[st_idx:len(testY)+st_idx])
        Yhat_t = pd.Series(data=(testPredict[:,d].ravel()),index=df['2017-01-01':].index[st_idx:len(testY)+st_idx])
        
        mse, nse,r2 = real_eva_error(Y_tr, Yhat_tr,)
        Tmse, Tnse,Tr2 = real_eva_error(Y_t, Yhat_t,)
        ########### Plot trian-test ##################
        syn_new = syn+'_b'+str(batch_size)+'_t'+str(d+1)

        plotgraph(mode,target,save_path,Y_tr,Y_t,Yhat_tr,Yhat_t,syn_new)
        monsoon_cal(mode,target,save_path,Y_tr,Y_t,Yhat_tr,Yhat_t,syn_new)
        plot_rsquare(mode,save_path,testY,testPredict,syn_new)

        try: error = pd.read_csv(save_path+'/eval.csv',index_col=0);print('LOAD SUCEESS')
        except: error = pd.DataFrame();print("cannot find rec")

        dict_data = {'Model':syn,'timestep':d+1,'Feature':n_features,'Time_in':n_past,'Time_out':n_future,'Batch':batch_size,
                    'MSE_trian':mse,'NSE_train':nse,'R2_train':r2,'MSE_test':Tmse,'NSE_test':Tnse,'R2_test':Tr2}
        _df = pd.DataFrame.from_dict(data=dict_data, orient ='index')
        error = pd.concat([error,_df],axis=1)
        error.to_csv(save_path+'/eval.csv')
    return testY,testPredict
def record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1,rec_result=False):
    mse, nse,r2 = real_eva_error(trainY, trainPredict,)
    Tmse, Tnse,Tr2 = real_eva_error(testY, testPredict,)
    try:
        trainPredict = pd.Series(data=(trainPredict),index=trainY.index)
        testPredict = pd.Series(data=(testPredict),index=testY.index)
    except:  pass
    ##################################
    plotgraph(mode,target,save_path,trainY,testY,trainPredict,testPredict,syn)
    monsoon_cal(mode,target,save_path,trainY,testY,trainPredict,testPredict,syn)
    ########### R-square ################
    plot_rsquare(mode,save_path,testY,testPredict,syn)
    ###### CSV output######
    dict_data = {'Model':syn,'Date':n_past,'Feature':n_features,'Time_in':n_past,'Time_out':n_future,'Use time':use_time,
                'MSE_trian':mse,'NSE_train':nse,'R2_train':r2,'MSE_test':Tmse,'NSE_test':Tnse,'R2_test':Tr2}
    _df = pd.DataFrame.from_dict(data=dict_data, orient ='index')

    try: error = pd.read_csv(save_path+'/eval.csv',index_col=0);print('LOAD eva SUCEESS')
    except: error = pd.DataFrame();print("cannot find rec")
    error = pd.concat([error,_df],axis=1)
    error.to_csv(save_path+'/eval.csv')

    ##########################
    if rec_result==True: 
        try: result_csv = pd.read_csv(save_path+'/result.csv',index_col='date');print('LOAD result SUCEESS')
        except: result_csv = pd.DataFrame();print("cannot find result rec")
        res_train=pd.DataFrame({'model':syn,'Y':trainY,'Yhat':trainPredict,'type': 'train'})
        res_test=pd.DataFrame({'model':syn,'Y':testY,'Yhat':testPredict,'type': 'test'})

        result_ = pd.concat([res_train,res_test],axis=0)
        result_csv = pd.concat([result_csv,result_],axis=1)
        result_csv.to_csv(save_path+'/result.csv')
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