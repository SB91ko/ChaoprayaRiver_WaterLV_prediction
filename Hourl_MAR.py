from pyearth import Earth
from DLtools.Data import load_data,instant_data,check_specific_col,station_sel

import numpy as np
import pandas as pd
def to_supervise(data,target,n_out):
    data[target]=data[target].shift(-n_out)
    data = data.astype('float64').dropna()
    X = data.drop([target],axis=1)
    xlabels = list(X.columns)
    X = X.values
    y = data[target].values
    return X,y,xlabels
def toDF(rank):
    name,rss,gcv,nb_sub= list(),list(),list(),list()
    for i in range(len(rank)):
        if i%4==0:
            name.append(rank[i])
        elif i%4==1:
            rss.append(rank[i])
        elif i%4==2:
            gcv.append(rank[i])
        elif i%4==3:
            nb_sub.append(rank[i])
    data = {'feature':name,
    'rss':rss,
    'gcv':gcv,
    'nb_subset':nb_sub}
    score = pd.DataFrame(data)
    return score

loading = instant_data()
df,mode = loading.hourly_instant(),'hour'
# df,mode = loading.daily_instant(),'day'
st = 'CPY012'
target,start_p,stop_p,host_path=station_sel(st,mode)
if mode =='hour': n_past,n_future = 24*7,72
elif mode =='day': n_past,n_future = 60,30

data = df[start_p:stop_p]
data['Day'] = data.index.dayofyear #add day
data = data.interpolate(limit=300000000,limit_direction='both').astype('float32') #interpolate neighbor first, for rest NA fill with mean() 

conclude_df=pd.DataFrame()
for n_out in range(1,n_future+1):
    X,y,xlabels = to_supervise(data,target,n_out)
    criteria = ('rss', 'gcv', 'nb_subsets')
    model = Earth(enable_pruning = True,
                #   max_degree=3,
                #  max_terms=20,
                minspan_alpha=.5,
                feature_importance_type=criteria,
                verbose=True)
    model.fit(X,y,xlabels=xlabels)
    nbsub = model.summary_feature_importances(sort_by='nb_subsets')[:2000].split()[3:83]
    gcv = model.summary_feature_importances(sort_by='gcv')[:2000].split()[3:83]
    rss = model.summary_feature_importances(sort_by='rss')[:2000].split()[3:83]
    
    rss,gcv,nbsub = toDF(rss),toDF(gcv),toDF(nbsub)
    top20=pd.concat([rss,gcv,nbsub],ignore_index=True)
    top20 = pd.concat([rss,gcv,nbsub],ignore_index=True).drop_duplicates('feature')
    top20['timestep'] = n_out

    #ADDED combine all result
    conclude_df = pd.concat([conclude_df,top20],ignore_index=True)
    if mode=='day':
        conclude_df.drop_duplicates('feature').to_csv('/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_daily.csv',index=False)
        conclude_df.to_csv('/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_daily_Fullreport.csv',index=False)

    elif mode=='hour':
        conclude_df.drop_duplicates('feature').to_csv('/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_hourly.csv',index=False)
        conclude_df.to_csv('/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_hourly_Fullreport.csv',index=False)