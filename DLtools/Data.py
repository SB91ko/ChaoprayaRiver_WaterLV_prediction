import re,glob, os
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def check_specific_col(df,column_name='rain1h'):
    """
    Detect columns name which has specific text, i.e. rain1h
    return columns list
    """
    _cols = [col for col in df.columns if column_name in col]
    return _cols
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

def del_less_col(df,ratio=.5):
    print("\nBefore del col are...",len(df.columns))
    for col in df.columns:
        # print(col,tenki[col].count(),"|",len(tenki[col]),"|",tenki[col].isnull().sum())
        if df[col].count()<len(df[col])*ratio:
            df.drop(col, axis=1, inplace=True)
    print("After...",len(df.columns))
    return df

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))
##########################################################################################################
class load_data:
    rain_st = pd.read_csv(f'./data/hii-telemetering-batch-data-master/station_metadata-rain.csv')
    water_st = pd.read_csv(f'./data/hii-telemetering-batch-data-master/station_metadata-water-level.csv')
    weather_st = pd.read_csv(f'./data/hii-telemetering-weather-data-master/station_metadata.csv')

    path_rain = "./data/hii-telemetering-batch-data-master/rain2007-2020/"
    path_water = "./data/hii-telemetering-batch-data-master/water-level2007-2020/"
    path_weather = "./data/hii-telemetering-weather-data-master/*/*/"
    path_dam = './data/Dam/clean.csv'

    def __init__(self,load_all=True):
        print("START LOADING DATA 2012-2020(July)")
        self.basins = ['แม่น้ำปิง',"แม่น้ำวัง","แม่น้ำยม","แม่น้ำน่าน",'แม่น้ำป่าสัก',"แม่น้ำเจ้าพระยา"]
        self.load_all =load_all
        self.df_rain =None
        self.df_water = None
        self.df_weather = None
        self.df_dam = None
        if load_all==True:
            print("\nrainfall (daily)....")
            self.df_rain = self.rain_data() # resample('H').pad()
            print("\nwater lv (hourly)....")
            self.df_water = self.water_data()
            print("\nweather lv (hourly)....")
            self.df_weather = self.weather_data()
            print("\nDam (daily)....")
            self.df_dam = self.dam_data()
            
            print("==========TOTAL FILE==========")
            print("Rain.............Water.........Weather...........Dam")
            print(self.df_rain.shape, self.df_water.shape, self.df_weather.shape,self.df_dam.shape)

    def daily(self):
        if self.df_rain ==None:
            self.df_rain = self.rain_data() # resample('H').pad()        
        elif  self.df_water==None:
            self.df_water = self.water_data()        
        elif  self.df_weather==None:
            self.df_weather = self.weather_data()
        elif  self.df_dam==None:
            self.df_dam = self.dam_data()

        rain1h = check_specific_col(self.df_weather,'rain1h')
        weather_d = self.df_weather.drop(rain1h, axis=1)
        daily = [self.df_rain,self.df_water.resample('d').mean(),weather_d.resample('d').mean(),self.df_dam.resample('d').mean()]
        df = pd.concat(daily,axis=1)
        return df

    def hourly(self):
        if (self.df_water ==None):
            self.df_water = self.water_data()
        elif  (self.df_weather==None):
            self.df_weather = self.weather_data()
        else: pass
        hourly = [self.df_water.resample('h').mean(),self.df_weather]
        df_h = pd.concat(hourly,axis=1)
        return df_h

    def rain_data(self):
        basin_list = related_basins(self.rain_st,self.basins)
        print(len(basin_list))
        related_file = open_concat_csv(self.path_rain,basin_list)
        print("Working on rain file.....")
        df = pd.concat([df_process(f) for f in tqdm(related_file)])
        col = list(df.columns)
        df.reset_index(inplace=True)
        # t_df = df.pivot(index='date', columns='station', values=col)
        # t_df.rename(columns=lambda x: x+"_r", inplace=True)
        # t_df = t_df["2013-01-01":"2020-07-31"]
        # return del_less_col(t_df)
        final_df=pd.DataFrame()
        for col in df.columns[2:]:
            df[col].dropna(inplace=True)
            t_df = df.pivot(index='date', columns='station', values=col)
            new_col = [(i+"_"+col) for i in t_df.columns]
            t_df.columns = list(new_col)
            final_df = pd.concat([final_df,t_df],axis=1)
        final_df = final_df["2013-01-01":"2020-07-31"]
        return del_less_col(final_df)

    def water_data(self):
        basin_list = related_basins(self.water_st,self.basins)
        print(len(basin_list))
        related_file = open_concat_csv(self.path_water,basin_list)
        print("Working on water file.....")
        df = pd.concat([df_process(f) for f in tqdm(related_file)])
        df.reset_index(inplace=True)

        final_df=pd.DataFrame()
        for col in df.columns[2:]:
            df[col].dropna(inplace=True)
            t_df = df.pivot(index='date', columns='station', values=col)
            new_col = [(i+"_"+col) for i in t_df.columns]
            t_df.columns = list(new_col)
            final_df = pd.concat([final_df,t_df],axis=1)
        # t_df = df.pivot(index='date', columns='station', values=col)
        # t_df.rename(columns=lambda x: x+"_w", inplace=True)
        final_df = del_less_col(final_df["2013-01-01":"2020-07-31"])
        ####clean noise####
        # for col in final_df.columns:
        #     station = self.water_st.loc[self.water_st['code']==col[:-3]]
        #     thr = station['ground_level']
        #     try: 
        #         thr = float(thr.values)
        #         final_df.loc[final_df[col]<(thr)] = np.NaN
        #     except: print('error',col,(thr.values))
        return final_df

    def weather_data(self):
        basin_list = related_basins(self.weather_st,self.basins)
        print(len(basin_list))
        yr_df = pd.DataFrame()
        for yr in range(2012,2021):
            path_weather = "/home/song/Public/Song/Work/Thesis/data/hii-telemetering-weather-data-master/{}/*/".format(yr)
            print("working on Weather year {}".format(yr))
            related_file = open_concat_csv(path_weather,basin_list)
            df = pd.concat([df_process(f,weather=True) for f in tqdm(related_file)])
            yr_df = pd.concat([yr_df,df]).sort_index()
        final_df = pd.DataFrame()
        cols = list(yr_df.columns) #Exclude, date and Stations col
        yr_df = yr_df.sort_index().reset_index()
        for col in cols:
            print(yr_df[col].shape,col)
            yr_df[col].dropna(inplace=True)
            t_df = yr_df.pivot(index='date', columns='station', values=col)
            new_col = [(i+"_"+col) for i in t_df.columns]
            t_df.columns = list(new_col)
            final_df = pd.concat([final_df,t_df],axis=1)
        final_df = final_df["2013-01-01":"2020-07-31"]
        return del_less_col(final_df)

    def dam_data(self):
        df = pd.read_csv(self.path_dam,index_col='date',parse_dates=['date'])
        df['Name'] = df['Name'].replace({'เขื่อนภูมิพล':'Dam_BH',
                            'เขื่อนสิริกิติ์' : 'Dam_SK',
                            'เขื่อนกิ่วลม' : 'Dam_KiewLom',
                            'เขื่อนกิ่วคอหมา': 'Dam_KiewKorMa',
                            'เขื่อนแควน้อย': 'Dam_KheawNoi',
                            'เขื่อนทับเสลา': 'Dam_ThapSalao',
                            'เขื่อนป่าสักฯ': 'Dam_Pasak',
                            'เขื่อนเจ้าพระยา': 'Dam_ChaoPY'})
        df.drop(columns='Unnamed: 0',inplace=True)
        cols = list(df.columns)
        cols = cols[1:-1]
        final_df = pd.DataFrame()
        df = df.sort_index().reset_index()
        
        for col in cols:
            print("working on column:",col)
            t_df = df.pivot(index='date', columns='Name', values=col)
            new_col = [(i+"_"+col) for i in t_df.columns]
            t_df.columns = list(new_col)
            final_df = pd.concat([final_df,t_df],axis=1)
            
        for col in final_df.columns:
            final_df[col] = final_df[col].astype(np.str)
            final_df[col] = final_df[col].str.replace(',', '').astype(np.float32)
        final_df = final_df["2013-01-01":"2020-07-31"]
        return del_less_col(final_df)


class instant_data:
    rain='./data/instant_data/all/rain.csv'
    water= './data/instant_data/all/water.csv'
    weather = './data/instant_data/all/weather.csv'
    dam = './data/instant_data/all/dam.csv'
    water_st = pd.read_csv(f'./data/hii-telemetering-batch-data-master/station_metadata-water-level.csv')
    def __init__(self):   
        self.df_w = self.df_maker(self.water)
        self.df_r = self.df_maker(self.rain)
        self.df_wet = self.df_maker(self.weather)
        self.df_d = self.df_maker(self.dam)

    def df_maker(self,csvfile):
        df = pd.read_csv(csvfile,index_col=['date'],parse_dates=['date'])
        # df.rename(columns=lambda x: x+syn, inplace=True)
        return df

    def daily_instant(self):
        rain1h = check_specific_col(self.df_wet,'rain1h')
        # close_bkk = check_specific_col(self.df_w,'BKK')
        solar = check_specific_col(self.df_wet,'solar')
        df_wl = self.df_w.resample('d').mean()
        clearnoise_wl(df_wl)
        daily = [self.df_r,df_wl,self.df_wet.resample('d').mean(),self.df_d.resample('d').mean()]
        df = pd.concat(daily,axis=1)
        df = df.drop(rain1h+solar, axis=1)
        # df = df.drop(rain1h+close_bkk+solar, axis=1)
        return df

    def hourly_instant(self):
        """
        lim_and_del is flag  ready to use data(as target defined)
        limit data to 2013-2017 || del col ratio less than 80%
        """
        df_wl = self.df_w.resample('h').mean()
        clearnoise_wl(df_wl)
        hourly = [df_wl,self.df_wet]
        df_h = pd.concat(hourly,axis=1)

        # close_bkk = check_specific_col(self.df_w,'BKK')
        solar = check_specific_col(self.df_wet,'solar')
        df_h = df_h.drop(solar, axis=1)
        # df_h = df_h.drop(close_bkk+solar, axis=1)
        return df_h

def station_sel(st,mode):
    """Select and return station status setting"""
    if st == 'CPY015':
        target='CPY015_wl'
        start_p = '2013-01-01'
        stop_p ='2017-12-31'
        if mode =='hour': host_path = './output/Hourly'
        elif mode =='day': host_path = '.output/Daily'
    elif st == 'CPY012':
        target='CPY012_wl'
        start_p ="2014-02-01"
        stop_p ="2018-03-31"
        if mode =='hour': host_path = './output_cpy012/Hourly'
        elif mode =='day': host_path = './output_cpy012/Daily'
    else: print('error nothing return from station sel') 
    return target,start_p,stop_p,host_path

def clearnoise_wl(df_wl):
    water_st = pd.read_csv(f'./data/hii-telemetering-batch-data-master/station_metadata-water-level.csv')
    for col in df_wl.columns:
        station = water_st.loc[water_st['code']==col[:-3]]
        thr = station['ground_level']
        try: 
            thr = float(thr.values)
            df_wl[col].loc[df_wl[col]<(thr)] = np.NaN
        except: pass
    return 

if __name__ == "__main__":

    # print("Test function load weather")
    path='./data/instant_data/all/'
    loaddata = load_data(load_all=False)
    
    water=loaddata.water_data()
    water.to_csv(path+'water.csv')
    
    weather=loaddata.weather_data()
    weather.to_csv(path+'weather.csv')
    
    rain = loaddata.rain_data()
    rain.to_csv(path+'rain.csv')

    dam = loaddata.dam_data()
    dam.to_csv(path+'dam.csv')


# def convert_df(ori_df,col):
#     df = pd.DataFrame()
#     ori_df = ori_df.dropna().reset_index()
#     for name,group in ori_df.groupby('station'):
#         if df.empty:
#             df = group.set_index("date")[[col]].rename(columns={col:str(name+"_"+col)})
#         else:
#             df = df.join(group.set_index("date")[[col]].rename(columns={col:str(name+"_"+col)}))
#     return df

# def multi_convert_weather_df(df):
#     cols = list(df.columns) #Exclude, date and Stations col
#     com_df = pd.DataFrame()
#     for col in cols:
#         print(col)
#         com_df = pd.concat([com_df,convert_df(df,col)])
#         print(com_df.shape)
#     return com_df