# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""

import threading
import os
import time
from common.SingleStockExecution import SingleStockExecution

class ExchangeSimulator:
    
    def __init__(self, marketData_2_exchSim_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        print("[%d]<<<<< call ExchSim.init" % (os.getpid(),))

        # listening market data
        t_md = threading.Thread(name='exchsim.on_md', target=self.consume_md, args=(marketData_2_exchSim_q,))
        t_md.start()

        # listening trading platform
        t_order = threading.Thread(name='exchsim.on_order', target=self.consume_order, args=(platform_2_exchSim_order_q, exchSim_2_platform_execution_q, ))
        t_order.start()

    # consume market data
    def consume_md(self, marketData_2_exchSim_q):
        while True:
            res = marketData_2_exchSim_q.get()
            print('[%d]ExchSim.consume_md' % (os.getpid()))
            print(res.outputAsDataFrame())

    # consume order
    def consume_order(self, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        while True:
            res = platform_2_exchSim_order_q.get()
            print('[%d]ExchSim.on_order' % (os.getpid()))
            print(res.outputAsArray())
            self.produce_execution(res, exchSim_2_platform_execution_q)
    
    def produce_execution(self, order, exchSim_2_platform_execution_q):
        execution = SingleStockExecution(order.ticker, order.date, time.asctime(time.localtime(time.time())))
        exchSim_2_platform_execution_q.put(execution)
        print('[%d]ExchSim.produce_execution' % (os.getpid()))
        print(execution.outputAsArray())
