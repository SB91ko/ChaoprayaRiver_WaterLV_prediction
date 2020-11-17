from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def real_eva_error(Y,Y_hat):
    mse = mean_squared_error(Y,Y_hat)
    nse = nashsutcliffe(Y_hat,Y)
    r2 = r2_score(Y,Y_hat)
    return mse,nse,r2

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

    mse, nse,r2 = list_eva_error(trainY, trainPredict,n_future)
    Tmse, Tnse,Tr2 = list_eva_error(testY, testPredict,n_future)
    index = np.arange(len(trainY)+len(testY))
    
    for d in range(n_future):
        index = np.arange(len(trainY)+len(testY))
        Y_tr= pd.Series(data=trainY[:,d],index=index[:len(trainY)])
        Yhat_tr = pd.Series(data=(trainPredict[:,d].ravel()),index=index[:len(trainY)])
        Y_t= pd.Series(data=testY[:,d],index=index[-len(testY):])
        Yhat_t = pd.Series(data=(testPredict[:,d].ravel()),index=index[-len(testY):])
        plt.figure(figsize=(15,5))
        plt.plot(Y_tr, label = "Actual")
        plt.plot(Yhat_tr, label = "Predict")
        
        plt.plot(Y_t, label = "Actual_test")
        plt.plot(Yhat_t, label = "Predict_test")
        plt.title('[{}] {}{}\n'.format(syn,mode,d+1)+'Water Level {} Forecast vs Actuals\n'.format(target)+'Train MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (mse[d],nse[d],r2[d])+'\nTest  MSE: %.3f | NSE: %.3f | R2 score: %.3f' % (Tmse[d],Tnse[d],Tr2[d]))
        plt.legend()
        plt.savefig(save_path+'/Plot_{}_{}{}.png'.format(syn,mode,d+1), dpi=300, bbox_inches='tight') 
        plt.clf()

        ###### CSV output######
        idx=['Modelname','Feature','t_in','t_out','batchsize','mse','nse','r2','Test_mse','Test_nse','Test_r2']
        try:_df = pd.DataFrame(["{}_{}".format(syn,n_past),n_features,n_past,n_future,batch_size,mse[d], nse[d],r2[d],Tmse[d], Tnse[d],Tr2[d]],index=idx,columns=[syn])
        except: _df = pd.DataFrame(["{}_{}".format(syn,n_past),n_features,n_past,n_future,batch_size,mse, nse,r2,Tmse, Tnse,Tr2],index=idx,columns=[syn])
        try: error = pd.read_csv(save_path+'/eval.csv',index_col=0);print('LOAD SUCEESS')
        except: error = pd.DataFrame(index = idx);print("cannot find rec")
        error = pd.concat([error,_df],axis=1)
        error.to_csv(save_path+'/eval.csv')
    return testY,testPredict

def nashsutcliffe(evaluation, simulation):
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
    if len(evaluation) == len(simulation):
        s, e = np.array(simulation), np.array(evaluation)
        # s,e=simulation(Yhat),evaluation(Y)
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)
    else:
        print("evaluation and simulation lists does not have the same length.")
        return np.nan