from DLtools import  prep_data,LSTMmodel


if __name__ == "__main__":
    water_path = '/home/song/Public/Song/Work/Thesis/data/hii-telemetering-batch-data-master/rain2007-2020/'
    water = prep_data.merge_df(water_path).align_table()
    print(water.head())

    rain_path = '/home/song/Public/Song/Work/Thesis/data/hii-telemetering-batch-data-master/water-level2007-2020/'
    rain = prep_data.merge_df(rain_path).align_table()
    print(rain.head())
    #test = LSTMmodel(water,water['BAKI'],timelag=7)
    #test.report()