from DLtools import * 
from DLtools.Data import intersection


def call_mar(data,target,mode,sel_t=None,cutoff=0.1):
    if target=='CPY015_wl':
        if mode == 'hour': MARfile='./MAR/featurelist_MAR_hourly_7d.csv'
        elif mode == 'day': MARfile='./MAR/featurelist_MAR_daily_7d.csv'
    elif target=='CPY012_wl':
        if mode == 'hour': MARfile='./MAR_2/[CPY012]featurelist_MAR_hourly_Fullreport.csv'
        elif mode == 'day': MARfile='./MAR/[CPY012]featurelist_MAR_daily_Fullreport.csv'
    mar = pd.read_csv(MARfile)
    if sel_t!=None: mar = mar.loc[mar['timestep'] == sel_t]
    cutoff_mar = mar.loc[(mar['rss']>cutoff)|(mar['gcv']>cutoff)|(mar['nb_subset']>cutoff)]
    col = [i for i in data.columns]
    select_col = intersection(col,set(cutoff_mar['feature']))   #drop duplicate
    select_col.append(target)                                   # add target col
    return data[select_col]
    
def hi_corr_select(data,target):
    def corr_w_Y(data,target,threshold= 0.8):
        # correlation
        corr_test = data.corr(method='pearson')[target]
        corr_test = corr_test[(corr_test> threshold) | (corr_test< -threshold) ]
        corr_test = corr_test.sort_values(ascending=False)
        #corr_test =corr_test[1:] # eliminate Target it own
        print(corr_test)
        return corr_test
    def high_corr_RM(data,threshold=.95):
        """Eliminate first columns with high corr"""
        corr_matrix = data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return to_drop
    col_feature = corr_w_Y(data,target,0.8).index
    data = data[col_feature]
    high_col = high_corr_RM(data.iloc[:,1:]) #exclude target it own
    data = data.drop(columns=high_col)
    return data
