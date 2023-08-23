#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 21:54:28 2023

@author: tracy
"""

import pandas as pd

import modelnew as model
fTicker_lis = ['JBF', 'QWF', 'HCF', 'DBF', 'EHF', 'IPF', 'IIF', 'QXF', 'PEF', 'NAF']
sTicker_lis =  ['3443','2388', '2498', '2610', '1319', '3035', '3006', '2615', '5425', '3105']
if __name__ == '__main__':
    df = pd.DataFrame(index = sTicker_lis, columns = ['max_window', 'past_5d_mean', 'past_5d_std'])
    for i in range(len(fTicker_lis)):
        max_window, aver, sd = model.save_model(sTicker_lis[i], fTicker_lis[i])
        df.loc[sTicker_lis[i]] = [max_window, aver, sd]
    df.to_csv('Information.csv')