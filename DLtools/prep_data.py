from numpy.lib.shape_base import column_stack
import pandas as pd
import re, glob, os
import numpy as np


class merge_df:
    """
    Class method for merge all csv raw data to DataFrame.
    Can be skiped and use converted data in 'instat_data' instead.
    """
    def __init__(self,folder_path):
        self.folder_path = folder_path
        print('completed',self.folder_path)

    def rename(self,file):
        name = re.search('[^\/]+$',file).group(0)
        return name[:-4]

    def clean_df(self,file):
        try:
            df = pd.read_csv(file,parse_dates=[['date', 'time']])
            df.rename(columns={'date_time':'date'},inplace=True)
        except:
            df = pd.read_csv(file,parse_dates=['date'])
        df['station'] = self.rename(file)
        return df.replace({-999:np.nan})

    def open_concat_csv(self):
        all_filenames = [i for i in glob.glob(os.path.join(self.folder_path, '*.csv'))]
        df_dummy = pd.concat([self.clean_df(f) for f in all_filenames])
        return df_dummy

    def align_table(self):
        type = ('rain'or'wl')
        input_df = self.open_concat_csv()
        df_dummy = pd.DataFrame()
        for name,group in input_df.groupby('station'):
            if df_dummy.empty:
                df_dummy = group.set_index('date')[[type]].rename(columns={type:name})
            else:
                df_dummy = df_dummy.join(group.set_index('date')[[type]].rename(columns={type:name}))
        return df_dummy


class instant_df:
    def __init__(self,water_csv,rain_csv,start=None,stop=None):
        self.water_csv = water_csv
        self.rain_csv = rain_csv
        self.start = start
        self.stop = stop

        self.water_df = self.df_maker(water_csv,'_w')
        self.rain_df = self.df_maker(rain_csv,'_r')

        self.df = self.rain_water_merge(self.water_df,self.rain_df)
        self.rain_st_df = None
        self.water_st_df = None

        self.rain_scaler = None
        self.water_scaler = None

    def df_maker(self,csvfile,syn=''):
        df = pd.read_csv(csvfile,index_col=['date'],parse_dates=['date'])
        df.rename(columns=lambda x: x+syn, inplace=True)
        return df[self.start:self.stop]

    def rain_water_merge(self,water,rain):
        col_water,col_rain = self.only_related()
        return pd.concat([water[col_water],rain[col_rain]])

    def only_related(self):
        filepath = ("data/hii-telemetering-batch-data-master/")
        self.rain_st_df = pd.read_csv(filepath+"station_metadata-rain.csv")
        rain_st = self.rain_st_df.loc[(self.rain_st_df['basin']=="แม่น้ำปิง")|(self.rain_st_df['basin']=="แม่น้ำเจ้าพระยา")|(self.rain_st_df['basin']=="แม่น้ำน่าน")]
        self.water_st_df = pd.read_csv(filepath+'station_metadata-water-level.csv')
        water_st = self.water_st_df.loc[(self.water_st_df['basin']=="แม่น้ำปิง")|(self.water_st_df['basin']=="แม่น้ำเจ้าพระยา")|(self.water_st_df['basin']=="แม่น้ำน่าน")]

        def intersection(lst1, lst2): 
            return list(set(lst1) & set(lst2))
        water_station = [i for i in self.water_df.columns]
        rain_station = [i for i in self.rain_df.columns]

        col_water = [i+'_w' for i in water_st['code']]        
        col_water = intersection(col_water,water_station)   #เช็คว่า ชื่อสถานีมีอยู่ในDBจริงก่อนใช้
        col_rain = [i+'_r' for i in rain_st['code']]
        col_rain = intersection(col_rain,rain_station)
        return col_water,col_rain


    def report_df(self):
        print(".......Report........")
        print("rain shape:{}........water shape:......{}".format(self.rain_df.shape,self.water_df.shape))
        print("total",self.df.shape)
        print(self.df.head())

# rain = 'data/instant_data/rain.csv'
# water = 'data/instant_data/water.csv'

# rw = instant_df(water,rain,'2012-01-01','2014-01-01')
# print(rw.df.head())
# print(len(rw.only_related()[0]))
# print(len(rw.only_related()[1]))
# print(rain_water.rain_water_merge.head)