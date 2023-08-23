# -*- coding: utf-8 -*-
"""
Created on Sat Jul 3 07:11:28 2019

@author: hongs
"""

class SingleStockExecution():
    
    def __init__(self, execID, orderID, ticker, date, timeStamp, direction, price, size):
        self.execID = execID
        self.orderID = orderID
        self.ticker = ticker
        self.date = date
        self.timeStamp = timeStamp
        self.direction = direction
        self.price = price
        self.size = size
        self.comm = 0 #commission for this transaction

    def outputAsArray(self):
        output = []
        output.append(self.date)
        output.append(self.ticker)
        output.append(self.timeStamp)
        output.append(self.execID)
        output.append(self.orderID)
        output.append(self.direction)
        output.append(self.price)
        output.append(self.size)
        output.append(self.comm)

        return output