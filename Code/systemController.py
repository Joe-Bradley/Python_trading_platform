# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 16:27:03 2023

@author: 84752
"""



from multiprocessing import Process, Queue
#from marketDataService import MarketDataService
from exchangeSimulator import ExchangeSimulator
import threading
from quantTradingPlatform import TradingPlatform
from MarketDataServer import MarketDataService_Stock, MarketDataService_Future
from QuantStrategy import QuantStrategy

if __name__ == '__main__':
    ###########################################################################
    # Define all components
    ###########################################################################
    '''#for calculating information of all 10 stocks portfolio
    fTicker_lis = ['JBF1', 'QWF1', 'HCF1', 'DBF1', 'EHF1', 'IPF1', 'IIF1', 'QXF1', 'PEF1', 'NAF1']
    sTicker_lis =  ['3443','2388', '2498', '2610', '1319', '3035', '3006', '2615', '5425', '3105']'''
    '''#for calculating return for single stock
    fTicker_lis = ['DBF1']
    sTicker_lis = ['2610']'''
    #Our final selection of 3-stock portfolio
    fTicker_lis = ['EHF1', 'IPF1', 'IIF1']
    sTicker_lis =  [ '1319', '3035', '3006']
    queue_lis = []
    strategy = QuantStrategy("1", "leading_lagging_effect", "Team 4", "Future_stock", "20230801", fTicker_lis, sTicker_lis)
    for i in range(len(fTicker_lis)):
        mktData_string = '''marketData_2_platform_q_{} = Queue(); marketData_2_platform_q_{} = Queue()'''.format(sTicker_lis[i], fTicker_lis[i])
        exec(mktData_string)
    marketData_2_exchSim_q = Queue()
    
    
    platform_2_exchSim_order_q = Queue()
    exchSim_2_platform_execution_q = Queue()
    # stocks = ['1319','2388']#,'2498','2610','2615','3006','3035','3105','3443','5425']
    # futures = ['EHF1', 'QWF1']
    ### 方法1 
    s_md_thread_list = []
    f_md_thread_list = []
    for i in range(len(sTicker_lis)):  #开启5个子进程执行fun1函数
        string = '''ts = threading.Thread(target=MarketDataService_Stock,args=(sTicker_lis[{}],marketData_2_exchSim_q, marketData_2_platform_q_{}))
tf = threading.Thread(target=MarketDataService_Future,args=(marketData_2_exchSim_q, marketData_2_platform_q_{},fTicker_lis[{}],))
ts.start()
tf.start()
s_md_thread_list.append(ts)
f_md_thread_list.append(tf)
        '''.format(i, sTicker_lis[i], fTicker_lis[i], i)
        exec(string)
        
    
    #Process(name='md', target=MarketDataService, args=(marketData_2_exchSim_q, marketData_2_platform_q, )).start()
    t2 = threading.Thread(name='sim', target=ExchangeSimulator, args=(marketData_2_exchSim_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q, ))
    t2.start()
    s_plat_thread_list = []
    f_plat_thread_list = []
    for i in range(len(sTicker_lis)):  #开启5个子进程执行fun1函数
        string = '''ts_plat = threading.Thread(target=TradingPlatform,args=(marketData_2_platform_q_{},platform_2_exchSim_order_q,
        exchSim_2_platform_execution_q, strategy))
tf_plat = threading.Thread(target=TradingPlatform,args=(marketData_2_platform_q_{},platform_2_exchSim_order_q,
exchSim_2_platform_execution_q, strategy))
ts_plat.start()
tf_plat.start()
s_plat_thread_list.append(ts_plat)
f_plat_thread_list.append(tf_plat)
        '''.format(sTicker_lis[i], fTicker_lis[i])
        exec(string)
        
    for i in range(len(s_md_thread_list)):
        s_md_thread_list[i].join()
        f_md_thread_list[i].join()
    t2.join()
    for i in range(len(s_plat_thread_list)):
        s_plat_thread_list[i].join()
        f_plat_thread_list[i].join()

    
    