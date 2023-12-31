

import threading
import os
from QuantStrategy import QuantStrategy


class TradingPlatform:
    # quantStart是tradingPlatform下的一个属性
    quantStrat = None

    def __init__(self, marketData_2_platform_q,
                    platform_2_exchSim_order_q,
                    exchSim_2_platform_execution_q, strategy):
        print("[%d]<<<<< call Platform.init" % (os.getpid(),))

        # Instantiate individual strategies
        self.quantStrat = strategy

        t_md = threading.Thread(name='platform.on_marketData', target=self.consume_marketData,
                                args=(platform_2_exchSim_order_q,
                                      marketData_2_platform_q))
        t_md.start()

        t_exec = threading.Thread(name='platform.on_exec', target=self.handle_execution,
                                  args=(exchSim_2_platform_execution_q,))
        t_exec.start()

    def consume_marketData(self, platform_2_exchSim_order_q,
                           marketData_2_platform_q):
        print('[%d]Platform.consume_marketData' % (os.getpid(),))
        while True:
            
            res = marketData_2_platform_q.get()
            

                #print('[%d] Platform.on_md of ' % (os.getpid()) + sTicker_lis[i] + ' and ' + fTicker_lis[i] )
            print('[%d] Platform.on_md' % (os.getpid()))
            print(res.outputAsDataFrame())
            # 根据策略来得到新的order
            result = self.quantStrat.run(res, None)
            if result is None:
                pass
            else:
                #do something with the new order
                platform_2_exchSim_order_q.put(result)
    # 将execution传递给量化策略
    def handle_execution(self, exchSim_2_platform_execution_q):
        print('[%d]Platform.handle_execution' % (os.getpid(),))
        while True:
            execution = exchSim_2_platform_execution_q.get()
            if execution.price is not None:
                print('[%d] Platform.handle_execution' % (os.getpid()))
                print(execution.outputAsArray())
                self.quantStrat.run(None, execution)