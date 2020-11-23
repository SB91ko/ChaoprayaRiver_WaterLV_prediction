from DLtools import *
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def test_stationarity(data):
    stats = ['Test Statistic','p-value','Lags','Observations']
    df_test = adfuller(data.dropna(), autolag='AIC')
    df_results = pd.Series(df_test[0:4], index=stats)
    for key,value in df_test[4].items():
        df_results['Critical Value (%s)'%key] = value        
    print(df_results)
    #extra
    if df_results[1] <= 0.05:
        print("strong evidence against the null hypothesis(H0), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a  unit root, indicating it is non-stationary")

def tsplot(y, title, lags=None, figsize=(10, 6)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    
    ############## Addup ######################

    ##################################
    ts_ax.set_title(title, fontsize=12, fontweight='bold')
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    sm.tsa.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    sm.tsa.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    sns.despine()
    plt.tight_layout()
    plt.savefig('/home/song/Public/Song/Work/Thesis/Data_viz/0_tsExp_{}.png'.format(title), bbox_inches='tight')
    # plt.show()
    return ts_ax, acf_ax, pacf_ax
