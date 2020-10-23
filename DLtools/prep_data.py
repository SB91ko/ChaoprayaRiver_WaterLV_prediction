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