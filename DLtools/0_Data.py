import pandas as pd
import re,glob, os
import numpy as np
from tqdm import tqdm

def rename(path_name):
    path_name = re.search('[^\/]+$',path_name).group(0)
    return path_name[:-4]

def df_process_weather(file):
    try:
        df = pd.read_csv(file,parse_dates=[['year','month','day','time']]).rename(columns={"year_month_day_time":"date"})
    except:
        df = pd.read_csv(file,parse_dates=[['date', 'time']]).rename(columns={"date_time":"date","temp_out":"temp"})
    df.replace({-999:np.nan},inplace=True)
    df['station'] = rename(file)
    return df

def open_concat_csv(folder_path):
    all_filenames = [i for i in glob.glob(os.path.join(folder_path, '*.csv'))]
    df = pd.concat([df_process_weather(f) for f in all_filenames])
    return df

def convert_df(ori_df,col):
    df = pd.DataFrame()
    for name,group in ori_df.groupby('station'):
        if df.empty:
            df = group.set_index("date")[[col]].rename(columns={col:str(name+"_"+col)})
        else:
            df = df.join(group.set_index("date")[[col]].rename(columns={col:str(name+"_"+col)}))
    return df

class data_loading:
    basins = ['แม่น้ำปิง',"แม่น้ำวัง","แม่น้ำยม","แม่น้ำน่าน",'แม่น้ำป่าสัก',"แม่น้ำเจ้าพระยา"]
    
    path_file = '/home/song/Public/Song/Work/Thesis/data/'
    weather_st = pd.read_csv(path_file+'hii-telemetering-weather-data-master/station_metadata.csv')
    rain_st = pd.read_csv(path_file+"/hii-telemetering-batch-data-master/station_metadata-rain.csv")

    path_rain = path_file+"hii-telemetering-batch-data-master/rain2007-2020/"
    path_water = path_file+"hii-telemetering-batch-data-master/water-level2007-2020/"
    path_weather = path_file+"/home/song/Public/Song/Work/Thesis/data/hii-telemetering-weather-data-master"    

    def __init__(self):
        self.relate_stations_list = self.related_basins(self.weather_st,self.basins)
        

    def related_basins(station_df,basins_list):
        station_df = station_df.loc[station_df['basin'].isin(basins_list)]
        return list(station_df['code'])    
    
    def load_rain(self):
        rain_df = open_concat_csv(self.path_rain)

    def load_weather(self):
        all_filenames = [i for i in glob.glob(os.path.join(f'{self.path_weather}/*/*/','*.csv'))]
        related_file = list()
        for file in all_filenames:
            for x in related_basins:
                if x in file :
                    related_file.append(file)
        return pd.concat([df_process_weather(f) for f in tqdm(related_file)])

############################################################
print("Working on concat file.....")

print(we_df.head())
print(we_df.tail())
# # we_df.to_csv('/home/song/Public/Song/Work/Thesis/data/instant_data/weather_temp.csv')
############################################################

# df = ('/home/song/Public/Song/Work/Thesis/data/hii-telemetering-weather-data-master/2012/201201/BDAR.csv')
# df = '/home/song/Public/Song/Work/Thesis/data/hii-telemetering-weather-data-master/2019/201901/ABRT.csv'
# df = df_process_weather(df)

cols = list(we_df.columns[1:-1])
print("*"*50)
print(cols)


for col in tqdm(cols):
    com_df = pd.DataFrame()
    print(col)
    com_df = pd.concat([com_df,convert_df(we_df,col)])
    com_df.to_csv('/home/song/Public/Song/Work/Thesis/data/instant_data/weather/{}_pwyn.csv'.format(col))
    print(com_df.shape)
    print(com_df.tail())
# print(com_df.head())
print(com_df.shape)
# com_df.to_csv('/home/song/Public/Song/Work/Thesis/data/instant_data/weather.csv')
print("DONE")