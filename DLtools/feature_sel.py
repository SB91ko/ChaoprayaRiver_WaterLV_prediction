from DLtools import * 
from DLtools.Data import intersection


def call_mar(data,target,mode,sel_t=None,cutoff=0.1):
    if target=='CPY015_wl':
        if mode == 'hour': MARfile='/home/song/Public/Song/Work/Thesis/MAR/featurelist_MAR_hourly_7d.csv'
        elif mode == 'day': MARfile='/home/song/Public/Song/Work/Thesis/MAR/featurelist_MAR_daily_7d.csv'
    elif target=='CPY012_wl':
        if mode == 'hour': MARfile='/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_hourly_Fullreport.csv'
        elif mode == 'day': MARfile='/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_daily_Fullreport.csv'
    mar = pd.read_csv(MARfile)
    if sel_t!=None: mar = mar.loc[mar['timestep'] == sel_t]
    cutoff_mar = mar.loc[(mar['rss']>cutoff)|(mar['gcv']>cutoff)|(mar['nb_subset']>cutoff)]
    col = [i for i in data.columns]
    select_col = intersection(col,set(cutoff_mar['feature']))   #drop duplicate
    select_col.append(target)                                   # add target col
    return data[select_col]