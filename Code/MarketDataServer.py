# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""

from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels_Futures, OrderBookSnapshot_FiveLevels_Stocks
import os
import pandas as pd
import time
import numpy as np
from datetime import datetime


from multiprocessing import Process, Queue

# import warnings
# warnings.filterwarnings('ignore')
# from concurrent.futures import ProcessPoolExecutor


class MarketDataService_Stock:

    def __init__(self, stock, marketData_2_exchSim_q, marketData_2_platform_q):
        print("[%d] <<<<< call MarketDataService.init" % (os.getpid(),))
        self.timegap = None
        self.done = False
        self.outfile_ls = []
        
        #Input the path of the specific stock
        self.stock_fp = './stocks/{}_md_202306_202306.csv.gz'.format(stock)
        self.produce_market_data(stock, marketData_2_exchSim_q, marketData_2_platform_q)

    def produce_market_data(self, stock, marketData_2_exchSim_q, marketData_2_platform_q):
        
        data = pd.read_csv(self.stock_fp)
        data = data[data.date == '2023-06-30'].reset_index(drop=True) #只提取最后一天的数据复刻
        del data['Unnamed: 0']
        data['lastPx'].fillna(method='ffill', inplace=True)
        data['size'].fillna(0, inplace=True)
        data['timeStamp'] = pd.to_datetime(data.date, format='%Y-%m-%d')\
                        - pd.to_datetime('1900-01-01',format='%Y-%m-%d')\
                        + pd.to_datetime(data.time, format='%H%M%S%f')
        self.timegap = np.diff(data.timeStamp)

        first_timegap = (data.at[0, 'time'] - 90000000)/1000

        time.sleep(15*60) #统一至9点
        time.sleep(first_timegap) #第一个数据delay至真实时间

        for i in range(data.shape[0]-1): # data.shape[0]
            self.produce_quote(marketData_2_exchSim_q, marketData_2_platform_q, i, stock, data)
            time.sleep(int(self.timegap[i])/1000000000) #Convert ns to s
        print("produce_quote has finished!")
    
    def produce_quote(self, marketData_2_exchSim_q, marketData_2_platform_q, i, stock, data):

        date = data.at[i, 'date']
        volume = data.at[i, 'volume']
        size = data.at[i, 'size']
        lastPx = data.at[i, 'lastPx']
        timeStamp = data.at[i, 'timeStamp']

        bidPrice, askPrice, bidSize, askSize = [], [], [], []
        for j in range(1,6):
            bidPrice.append(data.at[i,'BP{}'.format(j)])
            askPrice.append(data.at[i,'SP{}'.format(j)])
            bidSize.append(data.at[i,'BV{}'.format(j)])
            askSize.append(data.at[i,'SV{}'.format(j)])
        quoteSnapshot = OrderBookSnapshot_FiveLevels_Stocks(stock, date, timeStamp, 
                                                     volume, size, lastPx,
                                                     bidPrice, askPrice, bidSize, askSize)
        
        # print('[%d]MarketDataService_stock >>> produce_quote' % (os.getpid())) #输出实时数据
        # print(quoteSnapshot.outputAsDataFrame())
        self.outfile_ls.append(quoteSnapshot.outputAsDataFrame())
        marketData_2_exchSim_q.put(quoteSnapshot)
        marketData_2_platform_q.put(quoteSnapshot)
    
    def produce_df(self):
        stock_df = pd.concat(self.outfile_ls, axis = 0)
        return stock_df
    
class MarketDataService_Future:

    cols = ['timeStamp', 'ticker', 'date', 'time',\
                'askPrice5', 'askPrice4', 'askPrice3', 'askPrice2', 'askPrice1',
                'bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5', \
                'askSize5', 'askSize4', 'askSize3', 'askSize2', 'askSize1',\
                'bidSize1', 'bidSize2', 'bidSize3', 'bidSize4', 'bidSize5']
    
    def __init__(self, marketData_2_exchSim_q, marketData_2_platform_q, ticker:str):
        print("[%d] <<<<< call MarketDataService.init" % (os.getpid(),))
        self.tickerName = ticker
        self.outputAsDataFrame = pd.DataFrame()
        
        #Input the path of the specific future
        self.futures_fp = './futures/{}_md_202306_202306.csv.gz'.format(ticker)
                
        self.produce_market_data(marketData_2_exchSim_q, marketData_2_platform_q)

    def produce_market_data(self, marketData_2_exchSim_q, marketData_2_platform_q):
        dfFutures = pd.read_csv(self.futures_fp, compression='gzip')
        dfFutures['ticker'] = self.tickerName #Define the name of the futures
        
        #Output the timestamp in a format like '2000-01-01 00:00:00.000000'
        dfFutures['timeStamp'] = pd.to_datetime(dfFutures.date, format='%Y-%m-%d')\
                                - pd.to_datetime('1900-01-01',format='%Y-%m-%d')\
                                + pd.to_datetime(dfFutures.time, format='%H%M%S%f')
        
        df = dfFutures[self.cols]
        df.reset_index(drop=True,inplace=True)
        
        #Get the data of the last day which is '2023-06-30'
        lastDay = sorted(df.date.unique())[-1]
        lastDayIndex = df[df.date == lastDay].index.tolist()
        df = df.iloc[lastDayIndex,:]
        df.reset_index(drop=True,inplace=True)
        
        ask_price_columns = ['askPrice5', 'askPrice4', 'askPrice3', 'askPrice2', 'askPrice1']
        ask_size_columns = ['askSize5', 'askSize4', 'askSize3', 'askSize2', 'askSize1']
        bid_price_columns = ['bidPrice1', 'bidPrice2', 'bidPrice3', 'bidPrice4', 'bidPrice5']
        bid_size_columns = ['bidSize1','bidSize2','bidSize3','bidSize4','bidSize5']
        
        #Remove rows with all zeros in the specified columns
        df = df[~(df[ask_price_columns] == 0).all(axis=1)]
        df = df[~(df[ask_size_columns] == 0).all(axis=1)]
        df = df[~(df[bid_price_columns] == 0).all(axis=1)]
        df = df[~(df[bid_size_columns] == 0).all(axis=1)]
        
        for i in range(df.shape[0]):
            date_str = df.at[i,'date']
            time_str = str(df.at[i,'time'])
            combined_str = f"{date_str} {time_str}"
            datetime_obj = datetime.strptime(combined_str, "%Y-%m-%d %H%M%S%f")
            df.at[i,'time'] = datetime_obj.strftime('%H:%M:%S.%f')
        df.reset_index(drop=True, inplace=True)
        
        #Calculate the time intervals
        self.timeStamps = np.unique(np.array(df.timeStamp))
        self.timeIntervals = np.diff(self.timeStamps)
        self.allTickerData = [group.values[:, 1:] for _, group in df.groupby('ticker')]
        
        self.produce_quote(marketData_2_exchSim_q, marketData_2_platform_q)

    def produce_quote(self, marketData_2_exchSim_q, marketData_2_platform_q):
        df_futures = pd.DataFrame()
        
        for i in range(len(self.allTickerData[0])-1 ): #用于调整样本量 如果需要所有数据就改成len(self.allTickerData[0]) 
            #print(self.tickerName + ' future_quote')
            oneTicker = self.allTickerData[0][i]
            oneTicker = pd.DataFrame([oneTicker],columns=self.cols[1:])

            ticker = oneTicker.ticker.values[0]
            date = oneTicker.date.values[0]
            timeStamp = oneTicker.time.values[0]

            bidPrice, askPrice, bidSize, askSize = [], [], [], []

            for j in range(1,6):
                bidPrice.append(int(oneTicker['bidPrice{}'.format(j)].values))
                askPrice.append(int(oneTicker['askPrice{}'.format(j)].values))
                bidSize.append(int(oneTicker['bidSize{}'.format(j)].values))
                askSize.append(int(oneTicker['askSize{}'.format(j)].values))

            quoteSnapshot = OrderBookSnapshot_FiveLevels_Futures(ticker,date,timeStamp,bidPrice,askPrice,bidSize,askSize)
            temp = quoteSnapshot.outputAsDataFrame()
            # print(temp)
            df_futures = pd.concat([df_futures,temp])
            marketData_2_exchSim_q.put(quoteSnapshot)
            marketData_2_platform_q.put(quoteSnapshot)
            time.sleep(int(self.timeIntervals[i])/1000000000) #Convert ns to s

                    
        self.outputAsDataFrame = pd.concat([self.outputAsDataFrame,df_futures])
        self.outputAsDataFrame.reset_index(drop=True, inplace=True)
        print("produce_quote has finished!")


