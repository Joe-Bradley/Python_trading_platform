



import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import threading
import os
import time
from common.SingleStockExecution import SingleStockExecution

class ExchangeSimulator:

    def __init__(self, marketData_2_exchSim_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        print("[%d]<<<<< call ExchSim.init" % (os.getpid(),))
        
        self.LastFutureDate = None
        self.current_orderbook_dict = {}
        self.res_queue = exchSim_2_platform_execution_q
        #execID
        self.execID = 0
        self.current_orderbook = None
        self.current_orderbook_rerange = None
        t_md = threading.Thread(name='exchsim.on_md', target=self.consume_md, args=(marketData_2_exchSim_q,))
        t_md.start()

        t_order = threading.Thread(name='exchsim.on_order', target=self.consume_order, args=(platform_2_exchSim_order_q, ))
        t_order.start()


    def consume_md(self,marketData_2_exchSim_q):
        while True:

            md = marketData_2_exchSim_q.get()
            self.current_orderbook_dict[md.ticker] = md

            # print('[%d]ExchSim.consume_md' % (os.getpid()))
            # print(res.outputAsDataFrame())


    def consume_order(self, platform_2_exchSim_order_q):
        while True:
            order = platform_2_exchSim_order_q.get()
            # print('[%d]ExchSim.on_order' % (os.getpid()))
            # print(res.outputAsArray())
            #order被传输到了produce_execution
            self.produce_execution(order)
    
    
    def produce_execution(self, order):
        if order.ticker in self.current_orderbook_dict:
            self.current_orderbook = self.current_orderbook_dict[order.ticker] 
            askPrice = [self.current_orderbook.askPrice1, self.current_orderbook.askPrice2,
                        self.current_orderbook.askPrice3, self.current_orderbook.askPrice4,
                        self.current_orderbook.askPrice5]
            bidPrice = [self.current_orderbook.bidPrice1, self.current_orderbook.bidPrice2,
                        self.current_orderbook.bidPrice3, self.current_orderbook.bidPrice4,
                        self.current_orderbook.bidPrice5]
            askSize = [self.current_orderbook.askSize1, self.current_orderbook.askSize2,
                       self.current_orderbook.askSize3, self.current_orderbook.askSize4,
                       self.current_orderbook.askSize5]
            bidSize = [self.current_orderbook.bidSize1, self.current_orderbook.bidSize2,
                       self.current_orderbook.bidSize3, self.current_orderbook.bidSize4,
                       self.current_orderbook.bidSize5]
            self.current_orderbook_rerange = pd.DataFrame(
                {'askPrice': askPrice, 'bidPrice': bidPrice, 'askSize': askSize, 'bidSize': bidSize})


            if order.direction == 1:
                match_order = self.current_orderbook_rerange[self.current_orderbook_rerange["askPrice"] <= order.price]
                fill_size = min(match_order["askSize"].sum(), order.size)
            
                if fill_size == order.size:
                    match_order['size_cumsum'] = match_order["askSize"].cumsum()
                    match_order['not_excute_size'] = match_order['size_cumsum'].apply(lambda x: max(x - order.size, 0))
                    match_order['excute_size'] = match_order["askSize"] - match_order['not_excute_size']
                    match_order['excute_size'] = match_order['excute_size'].apply(lambda x: max(x, 0))
                    price = (match_order['excute_size'] * match_order["askPrice"]).sum() / match_order['excute_size'].sum()
                else:
                    price = (match_order["askPrice"] * match_order["askSize"]).sum() / match_order[
                        "askSize"].sum()
        
            elif order.direction == -1:
                match_order = self.current_orderbook_rerange[self.current_orderbook_rerange["bidPrice"] >= order.price]
                fill_size = min(match_order["bidSize"].sum(), order.size)

                if fill_size == order.size:
                    match_order['size_cumsum'] = match_order["bidSize"].cumsum()
                    match_order['not_excute_size'] = match_order['size_cumsum'].apply(lambda x: max(x - order.size, 0))
                    match_order['excute_size'] = match_order["bidSize"] - match_order['not_excute_size']
                    match_order['excute_size'] = match_order['excute_size'].apply(lambda x: max(x, 0))
                    price = (match_order['excute_size'] * match_order["bidPrice"]).sum() / match_order['excute_size'].sum()
                else:
                    price = (match_order["bidPrice"] * match_order["bidSize"]).sum() / match_order[
                    "bidSize"].sum()
            else:
                price = None
                fill_size = None

        else:
            price = None
            fill_size = None
        
        if (price is not None) and (np.isnan(price)):
            price = None
            fill_size = None


        execution = SingleStockExecution(
            execID = self.execID,
            orderID = order.orderID,
            ticker = order.ticker,
            date = order.date,
            timeStamp = order.submissionTime,
            direction = order.direction,
            price = price,
            size = fill_size,
        )
        self.res_queue.put(execution)
        if price is not None:
            print('[%d]ExchSim.produce_execution' % (os.getpid()))
            print(execution.outputAsArray())
     
