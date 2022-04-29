#from matplotlib import style
#import datetime
import cv2
import matplotlib.ticker as ticker
import yfinance as yf
import pandas as pd
from pandas import DataFrame
from pandas_datareader import data as pdr

import matplotlib.pyplot as plt
import mplfinance as mpf

import numpy as np

yf.pdr_override()

# take stocks name
name = 'SAMSUNG' #input("Enter name of stocks: ")
# take ticker code
tkr = '005930.KS' #input("Enter ticker code: ")
# take start date
start_date = '2022-03-01' #input("Enter start date[format: (####)-(##)-(##)]: ")
# take end date
end_date = '2022-03-31' #input("Enter end date[format: (####)-(##)-(##)]: ")
# take periods of moving average
mav_list = []
while(False):
    tmp = int(input("Enter period of moving average(enter 0 for the end): "))
    if tmp == 0:
        break;
    mav_list.append(tmp)

mav_tuple = tuple(mav_list)

# take if the user wants to get volume data
show_volume = True #bool(int(input("Enter if you want volume data(enter 1 for true, 0 for false): ")))

# Three arguments - Ticker, start, end
data = pdr.get_data_yahoo(tkr, start=start_date, end=end_date)
# 휴일 제거
data = data[data['Volume'] > 0]
# Pandas DataFrame을 csv 파일로 내보내기 (DataFrame.to_csv())
data.to_csv('./samsung_data.csv')

filepath = './' + name + '_OHLCV.png'
mydpi = 10
seq_len = 20
dimension = 64

plt.style.use('dark_background')
data.reset_index(inplace=True)
figs = np.zeros((len(data)-1, dimension, dimension, 3))
labels = []

fig = plt.figure()
for i in range(0, len(data)-1):
    c = data.iloc[i:i + int(seq_len) - 1, :]
    c_ = data.iloc[i:i + int(seq_len), :]
    if len(c) == int(seq_len):
        mydpi = 64
        fig = plt.figure(figsize=(dimension / mydpi, dimension / mydpi), dpi=mydpi)
        ax1 = fig.add_subplot(1,1,1)
        mpf.candlestick2_ochl(ax1, c['Open'], c['Close'], c['High'], c['Low'], width=1, colorup='#77d879', colordown='#db3f3f')
        ax1.grid(False)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)
        ax1.axis('off')

    starting = c_['Close'].iloc[-2]
    ending = c_['Close'].iloc[-1]
    if ending > starting:
        label = 1
    else:
        label = 0
    labels.append(label)

    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer._renderer)
    figs[i] = fig_np[:,:,:3]

    plt.close(fig)

#savefig=filepath : 이미지 파일 저장
#fig, ax1 = mpf.plot(data, type='candle', mav=mav_tuple, volume=show_volume, axisoff=True, figratio=(10,10), 
#        tight_layout=True, style='yahoo', figsize=(32/mydpi,32/mydpi), returnfig=True, block=False)


#figsize=(32/mydpi,32/mydpi)
#image = cv2.imread(filepath, cv2.IMREAD_COLOR)
#image = cv2.resize(image, (64,64))

#plt.draw()
#fig = plt.gcf()
#plt.show()