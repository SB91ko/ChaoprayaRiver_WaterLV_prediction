# from numpy.lib.shape_base import column_stack
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
        self.rain_st_df = None
        self.water_st_df = None

        self.water_df = self.df_maker(water_csv,'_w')
        self.rain_df = self.df_maker(rain_csv,'_r').resample('H').pad()
        
        self.col_water,self.col_rain = self.only_related()
        self.df = self.rain_water_merge(self.water_df,self.rain_df)
        #self.useful_col = self.report_missing_by_station()
        # self.rain_scaler = None
        # self.water_scaler = None

    def df_maker(self,csvfile,syn=''):
        df = pd.read_csv(csvfile,index_col=['date'],parse_dates=['date'])
        df.rename(columns=lambda x: x+syn, inplace=True)
        return df[self.start:self.stop]

    def rain_water_merge(self,water,rain):
        df = pd.concat([water[self.col_water],rain[self.col_rain]])
        df = df.reseample('d').mean()
        return df

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

    def report_missing_by_year(self):
        # Percent of missing data
        def count_miss(df):
            for from_yr in range(2007,2021):
                na_data = df.loc[str(from_yr)].count().sum()
                _total = df.loc[str(from_yr)].shape
                _total = _total[0]*_total[1]          
                print("avaliable data in *{}* is ........{:.2f} || missing....{:.2f}%".format(from_yr,_total,(100-na_data*100/_total)))
                #data,total_datapoint = df.loc[str(yr)].count()
        count_miss(self.df)

    def report_missing_by_station(self):
        av_ratio = 0.45
        useful_col = []
        print("ST.......nan|total")
        for col in self.df.columns:
            nan = self.df[col].isna().sum()
            avaliable =  self.df[col].count()
            total = len(self.df[col])
            print("{}.......{}||{},......{}".format(col,nan,avaliable,total))
            if avaliable/total >av_ratio:
                useful_col.append(col)
        return useful_col

    def report_df(self):
        print(".......Report........")
        print("rain shape:{}........water shape:......{}".format(self.rain_df.shape,self.water_df.shape))
        print("total",self.df.shape)
        print(self.df.head())

# rain = 'data/instant_data/rain_small.csv'
# water = 'data/instant_data/water_small.csv'
# rw = instant_df(water,rain,start='2013-01-01',stop='2020-01-01')

# df = rw.df
# df.interpolate(limit=3,inplace=True)
# col = rw.report_missing_by_station()

# print(df[col].head())