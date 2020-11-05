from sklearn.metrics import mean_squared_error
import hydroeval
import numpy as np
import pandas as pd
import math

def eva_error(Y_hat,Y,Y_hat_for_test,Y_for_test):
    mse_train = mean_squared_error(Y_hat,Y)
    mse_test = mean_squared_error(Y_hat_for_test,Y_for_test)
    nse_train = hydroeval.nse(Y_hat,Y)
    nse_test = hydroeval.nse(Y_hat_for_test,Y_for_test)

    return mse_train,mse_test,nse_train,nse_test


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