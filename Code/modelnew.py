import pandas as pd
import numpy as np
from sklearn import ensemble
import joblib
from multiprocessing import Pool
np.__version__ = '1.23'


fTicker_lis = ['JBF', 'QWF', 'HCF', 'DBF', 'EHF', 'IPF', 'IIF', 'QXF', 'PEF', 'NAF']
sTicker_lis =  ['3443','2388', '2498', '2610', '1319', '3035', '3006', '2615', '5425', '3105']
#Process the data in the same way as the lecture
def read_data(stockTicker, futuresTicker):
    stockDataFileName = './stocks/'+ stockTicker +'_md_202306_202306.csv.gz'
    indexFuturesFileName = './futures/'+futuresTicker +'1_md_202306_202306.csv.gz'
    stockData = pd.read_csv(stockDataFileName, compression='gzip', index_col=0)
    futuresData = pd.read_csv(indexFuturesFileName, compression='gzip')
    return stockData, futuresData

def preprocess_data(stockData, futuresData):
    stockData = stockData.copy()
    futuresData = futuresData.copy()
    #get the data used to train and building model
    #Select about five days data to train since we consider recent data would be more helpful for building model
    #And the data of the last trading day has being removed here, so the whole training process will not involve the data on the last trading day
    sd = np.unique(stockData.date)[-7:-1]#Change number in the [], to select the model for which day we want, do back-testing in this way.
    fd = np.unique(futuresData.date)[-7:-1]#[-7:-1] just to create the model we used to simulated trading
    stockData = stockData[stockData.date.isin(sd)]
    futuresData = futuresData[futuresData.date.isin(fd)]
    stockData['lastPx']=stockData.groupby('date')['lastPx'].fillna(method='ffill')
    stockData['size']=stockData.groupby('date')['size'].fillna(0)
    futuresData['midQ'] = (futuresData['bidPrice1'] + futuresData['askPrice1'])/2
    stockData_dates = np.unique(stockData.date)
    stoD=pd.to_datetime(stockData_dates, format="%Y-%m-%d")
    qqqq=stoD.year*10000+stoD.month*100+stoD.day
    indexData_dates = np.unique(futuresData.date)
    indD=pd.to_datetime(indexData_dates, format="%Y-%m-%d")
    pppp=indD.year*10000+indD.month*100+indD.day
    commonDays=pd.to_datetime(pppp.intersection(qqqq),format="%Y%m%d")
    d_futures=pd.to_datetime(futuresData.date,format="%Y-%m-%d")
    futuresData.date=d_futures
    futuresData=futuresData[futuresData.date.isin(commonDays)]
    d_stock=pd.to_datetime(stockData.date,format="%Y-%m-%d")
    stockData.date=d_stock
    stockData=stockData[stockData.date.isin(commonDays)]
    stockData_DateTime = pd.to_datetime(stockData.date.astype(str) + ' ' + stockData.time.astype(str), format="%Y-%m-%d %H%M%S%f")
    futuresData_DateTime = pd.to_datetime(futuresData.date.astype(str) + ' ' + futuresData.time.astype(str), format="%Y-%m-%d %H%M%S%f")
    stockData.index = stockData_DateTime
    stockData = stockData[~stockData.index.duplicated(keep='last')]
    futuresData.index = futuresData_DateTime
    futuresData = futuresData[~futuresData.index.duplicated(keep='last')]
    new_index1=stockData.index.union(futuresData.index)
    new_index=np.unique(new_index1)
    resampledFuturesData = futuresData.reindex(new_index)
    resampledFuturesData.fillna(method='ffill',inplace=True)
    futuresData_downsampled=resampledFuturesData.loc[stockData.index]
    return stockData, futuresData_downsampled

def feature_engineering(stockData, futuresData_downsampled):
    basicCols = ['date', 'time', 'sAskPrice1','sBidPrice1','sMidQ', 'fAskPrice1','fBidPrice1', 'fMidQ']
    featureCols = []
    labelCols = []
    for i in range(10, 110, 10):
        basicCols.extend(['fLaggingRtn_{}'.format(str(i))])
        featureCols.extend(['fLaggingRtn_{}'.format(str(i))])
    for i in range(10, 110, 10):
        basicCols.extend(['sForwardRtn_{}'.format(str(i))])
        labelCols.extend(['sForwardRtn_{}'.format(str(i))])
    df = pd.DataFrame(columns=basicCols)
    df['date']=stockData['date']
    df['time']=stockData['time']
    df['sAskPrice1']=stockData['SP1']
    df['sBidPrice1']=stockData['BP1']
    df['sMidQ']=(stockData['SP1'] + stockData['BP1'])/2
    df['fAskPrice1']=futuresData_downsampled['askPrice1']
    df['fBidPrice1']=futuresData_downsampled['bidPrice1']
    df['fMidQ']=futuresData_downsampled['midQ']
    for i in range(10, 110, 10):
        df['sForwardRtn_{}'.format(str(i))]=df.groupby('date')['sMidQ'].shift(-i) / df['sMidQ'].shift(0) - 1
        df['fLaggingRtn_{}'.format(str(i))]=df['fMidQ'].shift(0) / df.groupby('date')['fMidQ'].shift(i) - 1
        df['sForwardRtn_{}'.format(str(i))].fillna(0, inplace=True)
        df['fLaggingRtn_{}'.format(str(i))].fillna(0, inplace=True)
    return df, featureCols, labelCols

def train_models(df, featureCols, labelCols,stockData):
    numOfProcesses = 16
    gbrtModels = {}
    GBRTModels = {}
    for j in range(10,110,10):
        gbrtModels['Y_M_{}'.format(str(j))] = ensemble.GradientBoostingRegressor(n_estimators=300,learning_rate=0.01)
    r2Cols = ['date']
    r2Cols.extend(labelCols)
    r2InSample = pd.DataFrame(columns=r2Cols)
    or2Cols = ['date']
    or2Cols.extend(labelCols)
    or2 = pd.DataFrame(columns=or2Cols)
    allDays = stockData.date
    date_index=np.unique(allDays)
    for i in range(len(date_index)-1):
        df_forThisDay = df[df.date == date_index[i]]
        df_forNextDay = df[df.date == date_index[i+1]]
        features = pd.DataFrame(columns=featureCols)
        features_next = pd.DataFrame(columns = featureCols)
        for j in range(10, 110, 10):
            features['fLaggingRtn_{}'.format(str(j))] = df_forThisDay['fLaggingRtn_{}'.format(str(j))]
            features_next['fLaggingRtn_{}'.format(str(j))] = df_forNextDay['fLaggingRtn_{}'.format(str(j))]
        labels = pd.DataFrame(columns=['label'])
        labels_next = pd.DataFrame(columns=['label'])
        oneLineData = [np.datetime_as_string(date_index[i]).split('T')[0]]
        oneLineData_next = [np.datetime_as_string(date_index[i+1]).split('T')[0]]
        pool = Pool(processes=numOfProcesses)
        for k in range(10, 110, 10):
            labels['label'] = df_forThisDay['sForwardRtn_{}'.format(str(k))]
            labels_next['label'] = df_forNextDay['sForwardRtn_{}'.format(str(k))]
            labels = np.array(labels).reshape(-1,)
            GBRTModels['Y_M_{}'.format(str(k))] = pool.apply_async(gbrtModels['Y_M_{}'.format(str(k))].fit, args=(features, labels))
            labels = pd.DataFrame(columns=['label'])
        pool.close()
        pool.join()
        for l in range(10, 110,10):
            labels['label'] = df_forThisDay['sForwardRtn_{}'.format(str(k))]
            labels_next['label'] = df_forNextDay['sForwardRtn_{}'.format(str(k))]
            inSampleR2 = GBRTModels['Y_M_{}'.format(str(l))].get().score(features, labels)
            #get the score of the model fitted by the data of the previous day as the out of sample R2
            oor = GBRTModels['Y_M_{}'.format(str(l))].get().score(features_next, labels_next)
            oneLineData.extend([inSampleR2])
            oneLineData_next.extend([oor])
            print(str(date_index[i]) + ',in sample model_{}'.format(str(l)) + "," + str(inSampleR2))
            print(str(date_index[i]) + ',out of sample model_{}'.format(str(l)) + "," + str(oor))
        r2InSample = pd.concat([r2InSample, pd.DataFrame(data = [oneLineData], columns=r2Cols)])
        #Collect the data of the out of sample R2
        or2 = pd.concat([or2, pd.DataFrame(data = [oneLineData_next], columns = or2Cols)])
    #After looping, here the df_forNextDay would be the data with the date just before the last day of the earliest input data
    return GBRTModels, r2InSample, or2,features_next,df_forNextDay


def save_model(stockTicker, futuresTicker):
    #the whole training process will not involve the data on the last trading day
    stockData, futuresData = read_data(stockTicker, futuresTicker)
    stockData, futuresData_downsampled = preprocess_data(stockData, futuresData)
    df, featureCols, labelCols = feature_engineering(stockData, futuresData_downsampled)
    GBRTModels, r2InSample, or2,features_next,df_forNextDay = train_models(df, featureCols, labelCols,stockData)
    or2.set_index('date', inplace = True)
    #get the forward window with the highest out of sample R2
    max_window = or2.mean().idxmax()
    #here the df_forNextDay would be the data with the date just before the last day of the earliest input data
    maxOOR2_label = df_forNextDay[max_window]
    #build the model with the data of the day before the last trading day
    model = ensemble.GradientBoostingRegressor().fit(features_next, maxOOR2_label)
    #save the model into joblib files
    joblib.dump(model,  stockTicker + '_model.joblib')
    print('Model for '+ stockTicker + ' successfully saved')
    #return the information that maybe used
    return max_window, df[max_window].mean(), df[max_window].std()