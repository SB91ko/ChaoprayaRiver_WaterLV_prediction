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

# def forecast_accuracy(forecast, actual,title):
#     mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    
#     mae = np.mean(np.abs(forecast - actual))    # MAE
#     mpe = np.mean((forecast - actual)/actual)   # MPE
#     rmse = np.mean((forecast - actual)**2)**.5  # RMSE
#     mse = np.mean((forecast - actual)**2)       #MSE
#     corr = np.corrcoef(forecast, actual)[0,1]   # corr
#     nse = nashsutcliffe(actual,forecast)
#     r2 = r2_score(actual, forecast)
#     result = {'MSE':mse,'rmse':rmse,'R2':r2,'NSE':nse,'mape':mape,  'mae': mae,
#             'mpe': mpe, 'corr':corr}
#     result =  pd.Series(result,name=title)
#     return result

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

def record_list_result(syn,mode,trainY,testY,trainPredict,testPredict,target,batch_size,save_path,n_past,n_features,n_future=1,scaler_t=None):
    """
    in : syn,trainY,testY,trainPredict,testPredict,target,n_past,n_feature
    """
    print(trainY.shape,testY.shape)
    print(trainPredict.shape,testPredict.shape)
    if scaler_t !=None:
        trainY = scaler_t.inverse_transform(trainY)
        trainPredict = scaler_t.inverse_transform(trainPredict.reshape(trainY.shape))
        testY = scaler_t.inverse_transform(testY)
        testPredict = scaler_t.inverse_transform(testPredict.reshape(testY.shape))
    if n_future==1:
        mse, nse,r2 = real_eva_error(trainY, trainPredict,)
        Tmse, Tnse,Tr2 = real_eva_error(testY, testPredict,)
    else:    
        mse, nse,r2 = list_eva_error(trainY, trainPredict,n_future)
        Tmse, Tnse,Tr2 = list_eva_error(testY, testPredict,n_future)
    index = np.arange(len(trainY)+len(testY))

    for d in range(n_future):
        index = np.arange(len(trainY)+len(testY))
        if n_future==1:
            Y_tr= pd.Series(data=trainY,index=index[:len(trainY)])
            Y_t= pd.Series(data=testY,index=index[-len(testY):])
            Yhat_tr = pd.Series(data=(trainPredict),index=index[:len(trainY)])
            Yhat_t = pd.Series(data=(testPredict),index=index[-len(testY):])
        else:
            Y_tr= pd.Series(data=trainY[:,d],index=index[:len(trainY)])
            Y_t= pd.Series(data=testY[:,d],index=index[-len(testY):])
            Yhat_tr = pd.Series(data=(trainPredict[:,d].ravel()),index=index[:len(trainY)])
            Yhat_t = pd.Series(data=(testPredict[:,d].ravel()),index=index[-len(testY):])
        plt.figure(figsize=(15,5))
        plt.plot(Y_tr, label = "Actual")
        plt.plot(Yhat_tr, label = "Predict")
        
        plt.plot(Y_t, label = "Actual_test")
        plt.plot(Yhat_t, label = "Predict_test")
        try: plt.title('[{}] {}{}\n'.format(syn,mode,d+1)+'Water Level {} Forecast vs Actuals\n'.format(target)+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse[d],nse[d],r2[d])+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse[d],Tnse[d],Tr2[d]))
        except: plt.title('[{}] {}{}\n'.format(syn,mode,d+1)+'Water Level {} Forecast vs Actuals\n'.format(target)+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse,nse,r2)+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse,Tnse,Tr2))
        plt.legend()
        plt.savefig(save_path+'/Plot_{}_{}{}_b{}.png'.format(syn,mode,d+1,batch_size), dpi=300, bbox_inches='tight') 
        plt.clf()

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.scatter(Y_t, Yhat_t,color='red',marker='.')       
        ax.plot([0, Y_t.max()+1], [0, Y_t.max()+1], 'b--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('R2: %.3f' % (Tr2[d]))
        plt.savefig(save_path+'/r2_{}_{}{}_b{}.png'.format(syn,mode,d+1,batch_size), dpi=300, bbox_inches='tight') 
        ###### CSV output######
        idx=['Modelname','Feature','t_in','t_out','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2']
        try:_df = pd.DataFrame(["{}_{}".format(syn,n_past),n_features,n_past,n_future,batch_size,mse[d], nse[d],r2[d],Tmse[d], Tnse[d],Tr2[d]],index=idx,columns=[syn])
        except: _df = pd.DataFrame(["{}_{}".format(syn,n_past),n_features,n_past,n_future,batch_size,mse, nse,r2,Tmse, Tnse,Tr2],index=idx,columns=[syn])
        try: error = pd.read_csv(save_path+'/eval.csv',index_col=0);print('LOAD SUCEESS')
        except: error = pd.DataFrame(index = idx);print("cannot find rec")
        error = pd.concat([error,_df],axis=1)
        error.to_csv(save_path+'/eval.csv')
    return testY,testPredict

    
def record_alone_result(syn,mode,trainY,testY,trainPredict,testPredict,target,use_time,save_path,n_past,n_features,n_future=1):
    """
    in : syn,trainY,testY,trainPredict,testPredict,target,n_past,n_feature
    """
    mse, nse,r2 = real_eva_error(trainY, trainPredict,)
    Tmse, Tnse,Tr2 = real_eva_error(testY, testPredict,)
    
    try:
        trainPredict = pd.Series(data=(trainPredict),index=trainY.index)
        testPredict = pd.Series(data=(testPredict),index=testY.index)
    except:
        pass

    plt.figure(figsize=(15,5))
    plt.plot(trainY, label = "Actual")
    plt.plot(trainPredict, label = "Predict")
    
    plt.plot(testY, label = "Actual_test")
    plt.plot(testPredict, label = "Predict_test")
    plt.title('[{}] {}\n'.format(syn,mode)+'Water Level {} Forecast vs Actuals\n'.format(target)+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse,nse,r2)+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse,Tnse,Tr2))
    plt.legend()
    plt.savefig(save_path+'/Plot_{}_{}.png'.format(syn,mode), dpi=300, bbox_inches='tight') 
    plt.clf()

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(testY, testPredict,color='red',marker='.')       
    ax.plot([0, testY.max()+1], [0, testY.max()+1], 'b--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('R2: %.3f' % (Tr2))
    plt.savefig(save_path+'/r2_{}_{}.png'.format(syn,mode), dpi=300, bbox_inches='tight') 
        
    ###### CSV output######
    idx=['Modelname','Feature','t_in','t_out','usetime','mse','nse','r2','Test_mse','Test_nse','Test_r2']
    _df = pd.DataFrame(["{}_{}".format(syn,n_past),n_features,n_past,n_future,use_time,mse, nse,r2,Tmse, Tnse,Tr2],index=idx,columns=[syn])
    try: error = pd.read_csv(save_path+'/eval.csv',index_col=0);print('LOAD SUCEESS')
    except: error = pd.DataFrame(index = idx);print("cannot find rec")
    error = pd.concat([error,_df],axis=1)
    error.to_csv(save_path+'/eval.csv')
    return testY,testPredict

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