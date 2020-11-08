import pandas as pd
import re,glob, os
import numpy as np
from tqdm import tqdm

########## misc tool ###########
def rename(path_name):
    path_name = re.search('[^\/]+$',path_name).group(0)
    return path_name[:-4]

def df_process(file,weather=False):
    if weather == True:
        try:
            df = pd.read_csv(file,parse_dates=[['year','month','day','time']]).rename(columns={"year_month_day_time":"date"})
        except:
            df = pd.read_csv(file,parse_dates=[['date', 'time']]).rename(columns={"date_time":"date","temp_out":"temp"})
    else:
        try:
            df = pd.read_csv(file,parse_dates=[['date', 'time']]).rename(columns={"date_time":"date"})
        except:
            df = pd.read_csv(file,parse_dates=['date'])
    
    df.replace({-999:np.nan},inplace=True)
    df['station'] = rename(file)
    df.set_index(['date','station'],inplace=True)
    return df

def open_concat_csv(folder_path,basins):
    all_filenames = [i for i in glob.glob(os.path.join(f'{folder_path}','*.csv'))]
    related_file = list()
    for file in all_filenames:
        for x in basins:
            if x in file :
                related_file.append(file)
    return related_file

def related_basins(station_df,basins_list):
    station_df = station_df.loc[station_df['basin'].isin(basins_list)]
    return list(station_df['code'])

def convert_df(ori_df,col):
    df = pd.DataFrame()
    ori_df = ori_df.reset_index()
    for name,group in ori_df.groupby('station'):
        if df.empty:
            df = group.set_index("date")[[col]].rename(columns={col:str(name+"_"+col)})
        else:
            df = df.join(group.set_index("date")[[col]].rename(columns={col:str(name+"_"+col)}))
    return df

def multi_convert_weather_df(df):
    cols = list(df.columns) #Exclude, date and Stations col
    com_df = pd.DataFrame()
    for col in cols:
        print(col)
        com_df = pd.concat([com_df,convert_df(df,col)])
        print(com_df.shape)
    return com_df
##########################################################################################################
class load_data:
    rain_st = pd.read_csv(f'/home/song/Public/Song/Work/Thesis/data/hii-telemetering-batch-data-master/station_metadata-rain.csv')
    water_st = pd.read_csv(f'/home/song/Public/Song/Work/Thesis/data/hii-telemetering-batch-data-master/station_metadata-water-level.csv')
    weather_st = pd.read_csv(f'/home/song/Public/Song/Work/Thesis/data/hii-telemetering-weather-data-master/station_metadata.csv')

    path_rain = "/home/song/Public/Song/Work/Thesis/data/hii-telemetering-batch-data-master/rain2007-2020/"
    path_water = "/home/song/Public/Song/Work/Thesis/data/hii-telemetering-batch-data-master/water-level2007-2020/"
    path_weather = "/home/song/Public/Song/Work/Thesis/data/hii-telemetering-weather-data-master/*/*/"
    

    def __init__(self):
        print("START")
        self.basins = ['แม่น้ำปิง',"แม่น้ำวัง","แม่น้ำยม","แม่น้ำน่าน",'แม่น้ำป่าสัก',"แม่น้ำเจ้าพระยา"]
        
        self.df_rain = self.rain_data()["2012-01-01":].resample('H').pad()      
        self.df_water = self.water_data()["2012-01-01":]
        self.df_weather = self.weather_data()["2012-01-01":]
        print("==========TOTAL FILE==========")
        print("Rain......Water.........Weather")
        print(self.df_rain.shape, self.df_water.shape, self.df_weather.shape)
        
    def rain_data(self):
        basin_list = related_basins(self.rain_st,self.basins)
        print(len(basin_list))
        related_file = open_concat_csv(self.path_rain,basin_list)
        print("Working on rain file.....")
        df = pd.concat([df_process(f) for f in tqdm(related_file)])
        return convert_df(df,'rain')

    def water_data(self):
        basin_list = related_basins(self.water_st,self.basins)
        print(len(basin_list))
        related_file = open_concat_csv(self.path_water,basin_list)
        print("Working on water file.....")
        df = pd.concat([df_process(f) for f in tqdm(related_file)])
        return convert_df(df,'wl')

    def weather_data(self):
        basin_list = related_basins(self.weather_st,self.basins)
        print(len(basin_list))
        yr_df = pd.DataFrame()
        for yr in range(2012,2021):
            path_weather = "/home/song/Public/Song/Work/Thesis/data/hii-telemetering-weather-data-master/{}/*/".format(yr)
            print("working on Weather year {}".format(yr))
            related_file = open_concat_csv(path_weather,basin_list)
            df = pd.concat([df_process(f,weather=True) for f in tqdm(related_file)])
            yr_df = pd.concat([yr_df,df])
        # related_file = open_concat_csv(self.path_weather,basin_list)
        # print("Working on Weather file.....")
        # df = pd.concat([df_process(f,weather=True) for f in tqdm(related_file)])
        return multi_convert_weather_df(yr_df)

if __name__ == "__main__":
    print("Test function load weather")
    data=load_data()
    tenki = data.df_weather
    print(tenki.head())
    print(tenki.tail())
    print(tenki.shape)