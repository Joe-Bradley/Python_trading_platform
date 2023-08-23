# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:58:13 2023

@author: Guangyu
"""
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on Thu Jun 20 10:26:05 2020

@author: hongsong chou
"""

import os
import time
import pandas as pd

from common.Strategy import Strategy
from common.SingleStockOrder import SingleStockOrder
from common.SingleStockExecution import SingleStockExecution
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels_Futures, OrderBookSnapshot_FiveLevels_Stocks
#import matplotlib.dates as mdates
import joblib
import numpy as np
import threading
import matplotlib.pyplot as plt
import glob
#Performance of the strategy may have some difference due to the randomness of the GBRT model fitted in differenct local disk
class QuantStrategy(Strategy):
   
    def __init__(self, stratID, stratName, stratAuthor, ticker, day, fTicker_lis, sTicker_lis):

        #Create data structure to store data
        self.fTicker_lis =  fTicker_lis
        self.sTicker_lis =  sTicker_lis
        self.stock_future_dic = {}
        self.fmidQ_dic = {}
        self.smidQ_dic = {}
        self.features_dic = {}
        self.model_dic = {}
        self.position_dic = {}
        self.stock_pnl = {}
        self.buy_dic = {}
        self.sell_dic = {}
        self.pred_dic = {}
        featureCols = []
        for i in range(10, 110, 10):
            featureCols.extend(['fLaggingRtn_{}'.format(str(i))])
        super(QuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        for i in range(len(self.fTicker_lis)):
            self.fmidQ_dic[self.fTicker_lis[i]] = []
            self.smidQ_dic[self.sTicker_lis[i]] = []
            self.features_dic[self.fTicker_lis[i]] = pd.DataFrame(columns = featureCols)
            #May cause error due to the version difference of sklearn or joblib, run the 'RunThisFirst.py' to save model and avoid error
            self.model_dic[self.fTicker_lis[i]] = joblib.load('{}_model.joblib'.format(self.sTicker_lis[i])) 
            '''#Part for simulated trading of linear regression
            #mode = glob.glob('*{}*.joblib'.format(self.fTicker_lis[i][0:3] + '_linearRegression'))[0]
            #self.model_dic[self.fTicker_lis[i]] = joblib.load(mode)'''
            self.stock_future_dic[self.fTicker_lis[i]] = self.sTicker_lis[i]
            #position stands for the number of stocks
            self.stock_pnl[self.sTicker_lis[i]] = pd.DataFrame(data = [[0,0,0,0,0]], columns= ['average_price', 'position', 'pnl', 'invest', 'return'])
            
            self.buy_dic[self.sTicker_lis[i]] = 0
            self.sell_dic[self.sTicker_lis[i]] = 0
            self.pred_dic[self.sTicker_lis[i]] = 0
        self.strategy_metrics = pd.DataFrame(data = [[0,0,0,0,0,pd.to_datetime(90000000, format='%H%M%S%f').time()]], columns= ['return', 'sharpe_ratio', 'max_drawdown', 'total_pnl',  'std', 'time'])
        self.total_pnl = 0
        self.total_invest = 0
        self.initial_value = 1000000
        self.cash = self.initial_value
        self.asset = 0
        self.ticker = ticker #public field
        self.day = day #public field
        self.max_drawdown = 0
        
        
    def getStratDay(self):
        return self.day
    #Function to check the sign
    def check_sign(self, num):
        if num > 0:
            return 1
        elif num <0:
            return -1
        else:
            return 0 
    #Function to get the best window we selected to predict
    def get_best_window(self, stock_ticker):
        #change to 'linear Information.csv' for linear regression part
        df = pd.read_csv('information.csv', index_col = 0)
        if df['max_window'].loc[int(stock_ticker)][-3:] != '100':
            return int(df['max_window'].loc[int(stock_ticker)][-2:])
        else:
            return int(df['max_window'].loc[int(stock_ticker)][-3:])
        
    def get_max_drawdown(self, ser):
        cur_max = (ser.iloc[-1] - ser.max())/(ser.max()+1)
        if cur_max < self.max_drawdown:
            self.max_drawdown = cur_max
        else:
            cur_max = self.max_drawdown
        return cur_max
    
    def run(self, marketData, execution):
        if (marketData is None) and (execution is None):
            return None
        #execution part will calculate the position pnl of each stock (not the whole portfolio)
        elif (marketData is None) and ((execution is not None) and (isinstance(execution, SingleStockExecution))):
            #handle executions
            print('[%d] Strategy.handle_execution' % (os.getpid()))
            print(execution.outputAsArray())
            #To show the code clearly, assign the following variable with the following value
            date, ticker, timeStamp, execID, orderID, direction, price, size, common = execution.outputAsArray()
            #position stands for the number of stocks
            position = self.stock_pnl[ticker]['position'][0]
            average_price = self.stock_pnl[ticker]['average_price'][0] 
            pnl = self.stock_pnl[ticker]['pnl'][0] 
            invest_on_stock = self.stock_pnl[ticker]['invest'][0] 
            return_on_stock = self.stock_pnl[ticker]['return'][0] 
            if (execution.size is not None) and (size > 0):
                
                
                #when position = 0, the new execution price is the average price of the stock
                if position == 0:
                    average_price = price
                    #position stands for the number of stocks
                    position = direction*size
                    invest_on_stock = position*price
                else:
                    #if we have an opposite direction trade, we should realize the pnl
                    if self.check_sign(position) != self.check_sign(direction):
                        if abs(position) <= size:
                            #execution size is larger than what we have, then realize the PNL of the whole position
                            pnl += abs(position)*(price - average_price)*(-direction)
                            #the new average price is the execution price
                            average_price = price

                        else:
                            #execution size is smaller than what we have, realize the PNL of the execution size
                            pnl += size*(price - average_price)*(-direction)
             
                    #same direction, so no pnl is realized, just update the average
                    else:
                        average_price = (size*price + average_price*abs(position))/(size+abs(position))
                        #update the pnl of the stock we already have, since it price increase or decrease (although note realized)
                        pnl += position*(average_price - price)
                    #update position
                    position += direction*size
                    return_on_stock = pnl/abs(invest_on_stock)
                    invest_on_stock = position*price
                #long position will reduce cash and increase asset, and vice versa
                self.cash -= size*direction*price
                self.asset += size*direction*price

 
            #assign back to the class (position, average price and investment amout of single stock could only be updated at execution part)
            self.stock_pnl[ticker]['position'][0] = position
            self.stock_pnl[ticker]['average_price'][0] = average_price
            self.stock_pnl[ticker]['pnl'][0] = pnl
            self.stock_pnl[ticker]['invest'][0] = invest_on_stock
            self.stock_pnl[ticker]['return'][0] = return_on_stock        

            return None
        #market data part will calculate the book pnl
        elif ((marketData is not None) and (isinstance(marketData, OrderBookSnapshot_FiveLevels_Futures) or isinstance(marketData, OrderBookSnapshot_FiveLevels_Stocks))) and (execution is None):
            #handle new market data, then create a new order and send it via quantTradingPlatform.
            
            
            tick = marketData.ticker
            
            #if the ticker is future ticker
            if tick in self.fTicker_lis:
                future_ticker = tick
                
                df = marketData.outputAsDataFrame()
                #collect midQ data of futures which will used to predict
                if df['askPrice1'][0] == 0 or df['bidPrice1'][0] == 0:
                    if len(self.fmidQ_dic[future_ticker]) == 0: #the list should start with both bid and ask price are not 0
                        return None
                
                stock_Ticker = self.stock_future_dic[future_ticker]
                fMidQ = (df['askPrice1'][0]+df['bidPrice1'][0])/2
                self.fmidQ_dic[future_ticker].append(fMidQ)
                #To show the code clearly, assign the following variable with the following value
                cur_features = self.features_dic[future_ticker]
                
                #We can have lag100 return (which is a feature that will be used to predict) only if we have 101 or more future midQ data
                if len(self.fmidQ_dic[future_ticker]) < 101: 
                    return None
                else:
                    #collecting lagging return data
                    lagging_ret_lis = []
                    for i in range(10, 110, 10):
                        if self.fmidQ_dic[future_ticker][-i-1] == 0:
                            for j in range(-i-2, -1, -1): #make sure the zero will not be divided
                                if lagging_ret_lis[j] != 0:
                                    #replace the midQ with the last non-zero midQ
                                    lagging_ret_lis[-i-1] = lagging_ret_lis[j]
                                    break
                        #append lagging return from 10 to 1000
                        lagging_ret_lis.append(self.fmidQ_dic[future_ticker][-1]/self.fmidQ_dic[future_ticker][-i-1])
                    #collect one line data of future lagging return
                    self.features_dic[future_ticker] = pd.concat([cur_features, pd.DataFrame(data = [lagging_ret_lis], columns = cur_features.columns)])
                    #get  the corresponding model
                    model = self.model_dic[future_ticker]
                    #To show the code clearly, assign the following variable with the following value
                    cur_features = self.features_dic[future_ticker]
                    #tranform features data into the form that is suitable for prediction
                    d = pd.DataFrame(data = [cur_features.iloc[-1]], columns = cur_features.columns, index = [cur_features.index[-1]])
                    pred_ret = model.predict(d)
                    #collect the predicted return to the class
                    self.pred_dic[stock_Ticker] = pred_ret[0]
                return None   
            
            #if the ticker is stock ticker
            elif tick in self.sTicker_lis:
                timeStamp = marketData.timeStamp
                #Initialize th data
                direction = 0
                price = 0
                size = 0
                stock_df = marketData.outputAsDataFrame()
                stock_Ticker = tick
                #Don't want to outlier to influence the calculating of PNL
                if stock_df['askPrice1'][0] == 0 or stock_df['bidPrice1'][0] == 0:
                    if len(self.smidQ_dic[stock_Ticker]) == 0: #the list should start with both bid and ask price are not 0
                        return None
                    else:
                        #MidQ for calculating the PNL be assigned with normal value (both bid and ask price larger than 0)
                        sMidQ = self.smidQ_dic[stock_Ticker][-1]
                        self.smidQ_dic[stock_Ticker].append(sMidQ)
                        
                else:
                    sMidQ = (stock_df['askPrice1'][0]+stock_df['bidPrice1'][0])/2
                    self.smidQ_dic[stock_Ticker].append(sMidQ)
                #get the corresponding future ticker
                for fTick in self.stock_future_dic:
                    if self.stock_future_dic[fTick] == stock_Ticker:
                        future_ticker = fTick
                        break
                #If predicted return larger than 0 and there is available stocks at ask price 1
                if (self.pred_dic[stock_Ticker] > 0) and (stock_df['askSize1'][0] != 0):
                    
                    #Try to buy all stocks at ask price 1
                    size = stock_df['askSize1'][0]
                    price = stock_df['askPrice1'][0]
                    
                    #Turn the direction to 1 first, 
                    if self.cash > price*size:
                        
                        direction = 1
                        #give the trading time a mark, if the mark is 0, the order would be allowed here
                        if self.buy_dic[stock_Ticker] == 0:
                            
                            self.buy_dic[stock_Ticker] = len(self.smidQ_dic[stock_Ticker])
                        else:
                            #Only if the time after one trading longer than our predict window, we will do the next same direction order
                            if len(self.smidQ_dic[stock_Ticker]) - self.buy_dic[stock_Ticker] < self.get_best_window(stock_Ticker):
                                direction = 0
                                price = 0
                                size = 0
                            else:
                                #set it as 0 could be considered as a sign of allowing to trade again
                                self.buy_dic[stock_Ticker] = 0
                #Same logic as the case above
                if (self.pred_dic[stock_Ticker] < 0) and (stock_df['bidSize1'][0] != 0):
                    size = stock_df['bidSize1'][0]
                    price = stock_df['bidPrice1'][0]
                    direction = -1
                    if self.sell_dic[stock_Ticker] == 0:
                        self.sell_dic[stock_Ticker] = len(self.smidQ_dic[stock_Ticker])
                    else:
                        if len(self.smidQ_dic[stock_Ticker]) - self.sell_dic[stock_Ticker] < self.get_best_window(stock_Ticker):
                            direction = 0
                            price = 0
                            size = 0
                        else:
                            self.sell_dic[stock_Ticker] = 0
                
                #Put compulsory measure to make sure cash not drop a lot lower than 0
                if self.cash < 0:
                    size = stock_df['bidSize1'][0]
                    price = stock_df['bidPrice1'][0]
                    direction = -1
                #Put compulsory measure to make sure we not short too much (not more than our initial investment)
                if self.asset < -1000000:
                    size = stock_df['askSize1'][0]
                    price = stock_df['askPrice1'][0]
                    direction = 1
                
                
                #Calculate the book pnl
                
                if (len(self.fmidQ_dic[future_ticker]) >= 101) and (len(self.smidQ_dic[stock_Ticker]) >= 2): #Since if future's data less than 101, there will not be trade
                    if (stock_df['askPrice1'][0] != 0) and (stock_df['bidPrice1'][0] != 0):
                        #Calculating the PNL by midQ, 
                        pnl = (sMidQ - self.smidQ_dic[stock_Ticker][-2])*(self.stock_pnl[stock_Ticker]['position'][0])
                            
                        self.asset += pnl

                        #update strategy metrics after new information
                        last = self.strategy_metrics.iloc[-1]
                        last.total_pnl = self.cash + self.asset - self.initial_value

                        new_total_return = (self.cash + self.asset)/self.initial_value - 1
                        last['return'] = new_total_return
                        
                        new_return_series = pd.concat([self.strategy_metrics['return'],pd.Series([new_total_return])])
                        rtn_mean = new_return_series.mean()
                        rtn_std = new_return_series.std()
                        
                        #Assume risk-free rate is 0
                        if (rtn_std != 0) and (np.isnan(rtn_std) == False):
                            last['std'] = rtn_std
                            last.sharpe_ratio = rtn_mean/rtn_std
                        else:
                            last['std'] = rtn_std
                            last.sharpe_ratio = 0
                        #may have some fluctuation might due to multi-thread issue
                        last.max_drawdown = self.get_max_drawdown(new_return_series)
                        last['time'] = pd.to_datetime(timeStamp, format='%H%M%S%f').time()
                        new_metrics = list(last)
                        self.strategy_metrics = pd.concat([self.strategy_metrics, pd.DataFrame(data = [new_metrics], columns = self.strategy_metrics.columns)])
                       
                        print('[%d] Strategy.marketdata: ' % (os.getpid()) + 'position of ' + stock_Ticker + ': {}\n'.format(self.stock_pnl[stock_Ticker]['position'][0] ))
                        print('[{}] Strategy.marketdata: current cash amount is {}' .format(os.getpid(), str(self.cash)) )
                        print('[{}] Strategy.marketdata: current asset amount is {}' .format(os.getpid(), str(self.asset) ))
                        
                        print('[%d] Strategy.marketdata: ' % (os.getpid()) + 'current invest status of the portfolio: \n', self.strategy_metrics.iloc[-1] )
                    
                    
                        s = self.strategy_metrics.shape[0]
                        #Save return, sharpe ratio, max drawdown as picture at specific time
                        if  s%2000 == 0 :
                            self.strategy_metrics.sort_values('time', inplace = True)
                            self.strategy_metrics.to_csv( str(s)+'.csv')
                            #When we try single stock with linear regression, can collect return  in this way
                            #self.strategy_metrics.to_csv('./linear/' + stock_Ticker+'.csv')
                            #When we try single stock, can collect return  in this way
                            #self.strategy_metrics.to_csv('./GBRT_0629/' + stock_Ticker+'.csv')
        
                            self.strategy_metrics.plot(x = 'time', y = 'return', kind = 'line')
                            plt.tight_layout()
                            plt.savefig(str(s)+'return.png')
                            
                            self.strategy_metrics.plot(x = 'time', y = 'sharpe_ratio', kind = 'line')
                            plt.tight_layout()
                            plt.savefig(str(s)+'sharpe_ratio.png')
                            
                            self.strategy_metrics.plot(x = 'time', y = 'max_drawdown', kind = 'line')
                            plt.tight_layout()                        
                            plt.savefig(str(s)+'max_drawdown.png')

                date = marketData.date
                
                if (price != 0) and (size != 0):
                    
                    return SingleStockOrder(stock_Ticker,date,timeStamp, direction, price, size)
                else:
                    return None
            else:
                return None
            
        else:
            return None
                
        