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
plt.style.use('dark_background')

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

"""
<Format>
mpl_finance.plot(DataFrame, type='type of chart(default : Bar)',
mav=(moving average), volume='True/False(showing volume)', 
title='', figratio=(size of chart - x, y), tight_layout='remove the empty place',
style='')
"""
filepath = './' + name + '_OHLCV.png'
mydpi = 10

#savefig=filepath : 이미지 파일 저장
mpf.plot(data, type='candle', mav=mav_tuple, volume=show_volume, axisoff=True, figratio=(10,10), 
        tight_layout=True, style='yahoo', savefig=filepath)

image = cv2.imread(filepath, cv2.IMREAD_COLOR)
image_64X64 = cv2.resize(image, (64,64))
image_32X32 = cv2.resize(image, (32,32))
cv2.imwrite('./image_64X64.png', image_64X64)
cv2.imwrite('./image_32X32.png', image_32X32)


image_64X64_norm = cv2.normalize(image_64X64, None, 0, 255, cv2.NORM_MINMAX)

image_32X32_norm = cv2.normalize(image_32X32, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite('./image_64X64_Norm.png', image_64X64_norm)
cv2.imwrite('./image_32X32_Norm.png', image_32X32_norm)


#img_f = image_64X64.astype(np.float32)
#img_norm1 = ((img_f - img_f.min()) * (255) / (img_f.max() - img_f.min()))
#img_norm1 = img_norm1.astype(np.uint8)

#plt.draw()
#fig = plt.gcf()
#figsize=(32/mydpi,32/mydpi)
#plt.show()