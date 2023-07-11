# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""

import time
import random
import os
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels

class MarketDataService:

    def __init__(self, marketData_2_exchSim_q, marketData_2_platform_q):
        print("[%d]<<<<< call MarketDataService.init" % (os.getpid(),))
        time.sleep(3)
        self.produce_market_data(marketData_2_exchSim_q, marketData_2_platform_q)

    def produce_market_data(self, marketData_2_exchSim_q, marketData_2_platform_q):
        for i in range(10):
            self.produce_quote(marketData_2_exchSim_q, marketData_2_platform_q)
            time.sleep(5)

    def produce_quote(self, marketData_2_exchSim_q, marketData_2_platform_q):
        bidPrice, askPrice, bidSize, askSize = [], [], [], []
        bidPrice1 = 20+random.randint(0,100)/100
        askPrice1 = bidPrice1 + 0.01
        for i in range(5):
            bidPrice.append(bidPrice1-i*0.01)
            askPrice.append(askPrice1+i*0.01)
            bidSize.append(100+random.randint(0,100)*100)
            askSize.append(100+random.randint(0,100)*100)
        quoteSnapshot = OrderBookSnapshot_FiveLevels('testTicker', '20230706', time.asctime(time.localtime(time.time())), 
                                                     bidPrice, askPrice, bidSize, askSize)
        print('[%d]MarketDataService>>>produce_quote' % (os.getpid()))
        print(quoteSnapshot.outputAsDataFrame())
        marketData_2_exchSim_q.put(quoteSnapshot)
        marketData_2_platform_q.put(quoteSnapshot)