import matplotlib.pyplot as plt
from tqdm import tqdm
from DLtools.Data import instant_data,check_specific_col,station_sel
from DLtools.MachineLearning import tsplot


def scope_data(data):
    global start_p,stop_p
    data = data[start_p:stop_p]
    # data = del_less_col(data)
    return data
##############################
st,mode = 'CPY012','day'
target,start_p,stop_p,_=station_sel(st,mode)
##############################
loading = instant_data()
df_r = scope_data(loading.df_r)
df_w = scope_data(loading.df_w)
df_wet = scope_data(loading.df_wet)
df_dam = scope_data(loading.df_d)

# df_day=loading.daily_instant()
# df_hour =loading.hourly_instant()

df_solar = df_wet[check_specific_col(df_wet,'solar')]
df_rain1h = df_wet[check_specific_col(df_wet,'rain1h')]
df_temp = df_wet[check_specific_col(df_wet,'temp')]
df_press = df_wet[check_specific_col(df_wet,'press')]
df_humid = df_wet[check_specific_col(df_wet,'humid')]

def viz_data(save_name,save_path,data,n_graph,big_size=True,sample_size=-7000):
    print('working on ....{}'.format(save_name))
    col = list(data.columns)
    for show in range(int(len(col)/n_graph)+1):
        col_show = col[:n_graph]
        values = data[col_show].values
        
        groups = range(1,len(col_show))
        if big_size==True: plt.figure(figsize=(11,7))
        for i,group in enumerate(groups):    
            plt.suptitle('Sample {} data points of {}'.format(sample_size,save_name),fontsize=20, y=.93)    
            plt.subplot(len(groups), 1, i+1)
            plt.plot(values[sample_size:, group])
            
            box = dict(facecolor='white', pad=5, alpha=0.7)
            plt.title(col_show[group], y=0.2, loc='right',bbox=box)
            col = [x for x in col if x not in col_show]
        plt.show()
        plt.savefig(save_path+'{}_{}.png'.format(save_name,show), bbox_inches='tight')

path = '/home/song/Public/Song/Work/Thesis/Data_viz/'
viz_data(save_name='over_water_lv',save_path=path,data=df_w,n_graph=9,sample_size=0)
viz_data(save_name='over_dam',save_path=path,data=df_dam,n_graph=7,sample_size=0)
viz_data(save_name='over_rain_daily',save_path=path,data=df_r,n_graph=9,big_size=True,sample_size=0)
viz_data(save_name='over_solar',save_path=path,data=df_solar,n_graph=11,big_size=True,sample_size=0)
viz_data(save_name='over_rain1h',save_path=path,data=df_rain1h,n_graph=11,big_size=True,sample_size=0)
viz_data(save_name='over_temp',save_path=path,data=df_temp,n_graph=11,big_size=True,sample_size=0)
viz_data(save_name='over_press',save_path=path,data=df_press,n_graph=11,big_size=True,sample_size=0)
viz_data(save_name='over_humid',save_path=path,data=df_humid,n_graph=11,big_size=True,sample_size=0)


viz_data(save_name='water_lv',save_path=path,data=df_w,n_graph=9,)
viz_data(save_name='dam',save_path=path,data=df_dam,n_graph=7,)
viz_data(save_name='rain_daily',save_path=path,data=df_r,n_graph=9,big_size=True,sample_size=-1000)
viz_data(save_name='solar',save_path=path,data=df_solar,n_graph=11,big_size=True,sample_size=-1000)
viz_data(save_name='rain1h',save_path=path,data=df_rain1h,n_graph=11,big_size=True,sample_size=-500)
viz_data(save_name='temp',save_path=path,data=df_temp,n_graph=11,big_size=True,sample_size=-1000)
viz_data(save_name='press',save_path=path,data=df_press,n_graph=11,big_size=True,sample_size=-1000)
viz_data(save_name='humid',save_path=path,data=df_humid,n_graph=11,big_size=True,sample_size=-700)
# ##########################################################
def auto_plot_ts_analyst(df):
    for col in tqdm(df.columns):
        
        tsplot(df[col],'hour_'+str(col))
print('work..rain')
auto_plot_ts_analyst(df_r.interpolate(limit=300000000,limit_direction='both'))
print('work..water')
auto_plot_ts_analyst(df_w.interpolate(limit=300000000,limit_direction='both'))
print('work..dam')
auto_plot_ts_analyst(df_dam.interpolate(limit=300000000,limit_direction='both'))
print('work..weather')
auto_plot_ts_analyst(df_wet.interpolate(limit=300000000,limit_direction='both'))
# ################################################################
