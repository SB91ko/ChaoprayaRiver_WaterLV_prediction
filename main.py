import pandas as pd
import re,glob
import numpy as np
import matplotlib.pyplot as plt

def batch_data(df,target,shift_day):
    """df : input dataframe
    target : predict target
    shift_day : time window, no. of lookahead data
    """
    x_data = df.values[:-shift_day]
    print(type(x_data))
    print("Shape:",x_data.shape)
    print("*"*20)
    y_data = df[target].values[:-shift_day]
    print(type(y_data))
    print("Shape:", y_data.shape)
    return x_data,y_data

water_lv = ['PIN001','NAN011','PIN005','CPY008','GLF001', 'DIV005']

if __name__ == "__main__":
    #cleaning input data

    # re-scale
    rain_min_max_scaler = MinMaxScaler()
    water_min_max_scaler = MinMaxScaler()

    #set lable data/ time window / batch data
    batch_data()
    
