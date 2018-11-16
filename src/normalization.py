'''
    Data Normalization
'''

import pandas as pd
from sklearn import preprocessing

def normalize(dataFrame):
    '''
        Data Normalization.
        :param dataFrame: the data frame received from main function
        :param colsToNormalize: the names of columns to be normalized
        :returns dataFrame: normalized data frame
    '''

    normDataFrame = preprocessing.normalize(dataFrame, axis=0, norm="l2")

    dataFrame = pd.DataFrame(normDataFrame, index=dataFrame.index, columns=list(dataFrame.columns.values))

    dataFrame.to_csv("../NormalizedStock.csv", encoding='utf-8')

    return dataFrame