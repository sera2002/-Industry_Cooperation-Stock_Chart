import yfinance as yf
import pandas as pd
from pandas import DataFrame
from pandas_datareader import data as pdr
yf.pdr_override()

# Three arguments - Ticker, start, end
data = pdr.get_data_yahoo('005930.KS', start='2021-01-01', end='2022-01-01')
print(data)
print(type(data))   # type : pandas.DataFrame

# Pandas DataFrame을 csv 파일로 내보내기 (DataFrame.to_csv())
data.to_csv('./samsung_data.csv')

