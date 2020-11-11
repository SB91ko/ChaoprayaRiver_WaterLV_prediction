from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import numpy as np
import math

def eva_error(Y_hat,Y,Y_hat_for_test,Y_for_test):
    mse_train,nse_train = real_eva_error(Y_hat,Y)
    mse_test,nse_test = real_eva_error(Y_hat_for_test,Y_for_test)
    return mse_train,mse_test,nse_train,nse_test

def real_eva_error(Y_hat,Y):
    mse = mean_squared_error(Y_hat,Y)
    nse = nashsutcliffe(Y_hat,Y)
    r2 = r2_score(Y,Y_hat)
    return mse,nse,r2

def list_eva_error(Y,Y_hat,n_out):
    mse_l,nse_l,r2_l = list(),list(),list()
    for i in range(n_out):
        mse = mean_squared_error(Y[:,i].reshape(-1,1),Y_hat[:,i],)
        nse = nashsutcliffe(Y[:,i].reshape(-1,1),Y_hat[:,i],)
        r2 = r2_score(Y[:,i].reshape(-1,1),Y_hat[:,i],)
        mse_l.append(mse)
        nse_l.append(nse)
        r2_l.append(r2)
    return mse_l,nse_l,r2_l

def error_rec(Base_df,model,n_feature,n_timelag,batch_size,mse_train,mse_test,nse_train,nse_test,gen_msetrain=None,gen_msetest=None,gen_nsetrain=None,gen_nsetest=None):
    df = pd.DataFrame({ 'model': model,
                        'n_feature': n_feature,
                        "n_timelag" : n_timelag,
                        "batch_size": batch_size,
                        'MSE_train':mse_train,
                        "MSE_test":mse_test,
                        'Gen_MSE_train': gen_msetrain,
                        "Gen_MSE_test": gen_msetest,
                        'RMSE_train':math.sqrt(mse_train),
                        "RMSE_test": math.sqrt(mse_test),
                        'Gen_RMSE_train': math.sqrt(gen_msetrain),
                        "Gen_RMSE_test": math.sqrt(gen_msetest),
                        'NSE_trian':nse_train,
                        "NSE_test":nse_test,
                        'Gen_NSE_trian': gen_nsetrain,
                        "Gen_NSE_test": gen_nsetest,
                        })
    return pd.concat([Base_df,df])

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
        # s,e=simulation,evaluation
        mean_observed = np.nanmean(e)
        # compute numerator and denominator
        numerator = np.nansum((e - s) ** 2)
        denominator = np.nansum((e - mean_observed)**2)
        # compute coefficient
        return 1 - (numerator / denominator)

    else:
        print("evaluation and simulation lists does not have the same length.")
        return np.nan