'''
    Data Normalization
'''

import pandas as pd
from sklearn import preprocessing

def normalize(attributes, target):
    '''
        Data Normalization.
        :param attributes: the data frame of attributes(candidates for features) received from main function
        :param target: the data frame of target variable(close price)
        :returns dataFrame: normalized data frame
    '''

    normDataFrame = preprocessing.normalize(attributes, axis=0, norm="l2")
    normDataFrame = pd.DataFrame(normDataFrame, index=attributes.index, columns=list(attributes.columns.values))

    normalizedDataSet = normDataFrame.copy()
    normalizedDataSet['close'] = target
    #normalizedDataSet.to_csv("../NormalizedStock.csv", encoding='utf-8')

    return normDataFrame, target