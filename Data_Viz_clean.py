import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DLtools.Data import instant_data,check_specific_col,del_less_col,intersection,load_data,related_basins
loading = instant_data()

df_r = loading.df_r
df_w = loading.df_w
df_wet =loading.df_wet
df_dam = loading.df_d

# df_day=loading.daily_instant()
# df_hour =loading.hourly_instant()

df_solar = df_wet[check_specific_col(df_wet,'solar')]
df_rain1h = df_wet[check_specific_col(df_wet,'rain1h')]
df_temp = df_wet[check_specific_col(df_wet,'temp')]
df_press = df_wet[check_specific_col(df_wet,'press')]

def viz_data(save_name,save_path,data,n_graph,big_size=True,sample_size=7000):
    col = list(data.columns)
    print('working on ....{}'.format(save_name))
    for show in range(int(len(col)/n_graph)+1):
        col_show = col[:n_graph]
        values = data[col_show].values
        groups = range(1,len(col_show))
        if big_size==True: plt.figure(figsize=(11,7))
        for i,group in enumerate(groups):    
            plt.suptitle('Sample {} data points of {}'.format(sample_size,save_name),fontsize=20, y=.93)    
            plt.subplot(len(groups), 1, i+1)
            plt.plot(values[-sample_size:, group])
            box = dict(facecolor='white', pad=5, alpha=0.7)
            plt.title(col_show[group], y=0.2, loc='right',bbox=box)
            col = [x for x in col if x not in col_show]
        # plt.show()
        plt.savefig(save_path+'{}_{}.png'.format(save_name,show), bbox_inches='tight')

path = 'output/DataViz/'
# viz_data(save_name='water_lv',save_path=path,data=df_w,n_graph=9,)
viz_data(save_name='dam',save_path=path,data=df_dam,n_graph=7,)
# viz_data(save_name='rain_daily',save_path=path,data=df_r,n_graph=9,big_size=True,sample_size=1000)
# viz_data(save_name='solar',save_path=path,data=df_solar,n_graph=11,big_size=True,sample_size=1000)
# viz_data(save_name='rain1h',save_path=path,data=df_rain1h,n_graph=11,big_size=True,sample_size=500)
# viz_data(save_name='temp',save_path=path,data=df_temp,n_graph=11,big_size=True,sample_size=1000)
# viz_data(save_name='press',save_path=path,data=df_press,n_graph=11,big_size=True,sample_size=1000)

