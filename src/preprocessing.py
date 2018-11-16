'''
    Preprocessing
'''

import math
import logging
import pandas as pd

Logger = logging.getLogger('preprocessing.stdout')

def roundfn(var):
    return float(format(var, '.6f'))

def preprocessing(dataFrame):
    '''
    Preprocessing of data frame adding some useful columns
    :param dataFrame: data frame after interpolation
    :return: preprocessed data frame
    '''

    prev = 0.0
    diff = 0.0
    avg = 0.0
    sum = 0.0
    num_moving_avg = 50
    volatile_sum = 0.0
    volatile_avg = 0.0
    num_volatile = 10
    curr_volatility = 0.0

    dataFrame['prev_day_diff'] = pd.Series(0,index=dataFrame.index)
    dataFrame['50_day_moving_avg'] = pd.Series(0,index=dataFrame.index)
    dataFrame['10_day_volatility'] = pd.Series(0,index=dataFrame.index)

    indexList = dataFrame.index

    for rowNo,index in enumerate(indexList):

        if not rowNo:
            try:
                dataFrame.loc[index, 'prev_day_diff'] = diff
            except Exception:
                Logger.debug("Error in appending diff to 0th row")
        else:
            diff = roundfn(float(dataFrame.loc[index , 'adj_close']) - float(prev))
            try:
                dataFrame.loc[index, 'prev_day_diff'] = diff
            except Exception:
                Logger.debug("Error in updating diff of date: {} and rowNo: {}".format(index, rowNo))

        prev = dataFrame.loc[index, 'adj_close']

        if rowNo<num_moving_avg:
            sum += float(dataFrame.loc[index, 'adj_close'])
            avg = roundfn(sum / (rowNo+1))
        else:
            sum = sum + float(dataFrame.loc[index, 'adj_close']) - float(dataFrame.loc[indexList[rowNo - num_moving_avg
                                                                                       ], 'adj_close'])
            avg = roundfn(sum / num_moving_avg)

        try:
            dataFrame.loc[index,'50_day_moving_avg'] = avg
        except Exception:
            Logger.debug("Error in updating 50 day moving avg. of date: {} and rowNo: {}".format(index, rowNo))

        if rowNo < num_volatile:
            volatile_sum += float(dataFrame.loc[index,'adj_close'])
            volatile_avg = roundfn(volatile_sum/ (rowNo + 1))
        else:
            volatile_sum = volatile_sum + float(dataFrame.loc[index,'adj_close']) - float(dataFrame.loc[indexList[rowNo - num_volatile
                                                                                                        ],'adj_close'])
            volatile_avg = roundfn(volatile_sum / num_volatile)

        if rowNo:
            count = min(rowNo,num_volatile)

            for i in range(count):
                curr_volatility += math.pow((dataFrame.loc[index,'adj_close'] - volatile_avg),2)

            curr_volatility = roundfn(math.sqrt(curr_volatility/count))

        try:
            dataFrame.loc[index,'10_day_volatility'] = curr_volatility
        except Exception:
            Logger.debug("Error in updating 10 day volatility of date: {} and row: {}".format(index, rowNo))

        curr_volatility = 0.0

    dataFrame.to_csv("../AugmentedStock.csv", encoding='utf-8')

    return dataFrame
