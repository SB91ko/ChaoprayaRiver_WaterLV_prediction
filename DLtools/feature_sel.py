from DLtools import * 
from DLtools.Data import intersection

def call_mar(data,target,mode,sel_t=None):
    if target=='CPY015_wl':
        if mode == 'hour': MARfile='/home/song/Public/Song/Work/Thesis/MAR/featurelist_MAR_hourly_7d.csv'
        elif mode == 'day': MARfile='/home/song/Public/Song/Work/Thesis/MAR/featurelist_MAR_daily_7d.csv'
    elif target=='CPY012_wl':
        if mode == 'hour': MARfile='/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_hourly.csv'
        elif mode == 'day': MARfile='/home/song/Public/Song/Work/Thesis/MAR/[CPY012]featurelist_MAR_daily.csv'
    mar = pd.read_csv(MARfile)
    if sel_t!=None: mar = mar.loc[mar['timestep']<=sel_t]
    col = [i for i in data.columns]
    select_col = intersection(col,mar['feature'])
    select_col.append(target)               # add target col
    return data[select_col]

