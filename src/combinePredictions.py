'''
File to combine the predictions from both Time Series and News
'''

import pandas as pd
from makeGraph import makeGraph
from sklearn.metrics import mean_absolute_error
from scipy.stats import mannwhitneyu

def combinePredictions(errorTimeSeries, errorNews, predictionTimeSeries, predictionNews, target):
    '''

    :param errorTimeSeries: dictionary having error values of different algorithms run for time series
    :param errorNews: dictionary having error values of different algorithms run for news
    :param predictionTimeSeries: dictionary having predicted values of different algorithms run for time series
    :param predictionNews: dictionary having predicted values of different algorithms run for news
    :param target: the actual values of stock for same days as predictions
    '''

    combinedErrors = {}
    pValues = {}

    for key,value in predictionTimeSeries.items():
        value1 = predictionTimeSeries[key]*errorNews[key]
        value2 = predictionNews[key]*errorTimeSeries[key]
        value =  value1.add(value2)
        value /= (errorTimeSeries[key] + errorNews[key])
        #print(value)
        makeGraph(target,valueCombined=value,name="Combined "+ str(key))
        combinedErrors[key] = mean_absolute_error(target,value)
        statistic,pvalue = mannwhitneyu(target,pd.Series(value[0]))
        pValues[key] = pvalue

    minKeyTimeSeries = min(errorTimeSeries, key = errorTimeSeries.get)
    minKeyNews = min(errorNews, key = errorNews.get)

    value1 = predictionTimeSeries[minKeyTimeSeries]*errorNews[minKeyNews]
    value2 = predictionNews[minKeyNews]*errorTimeSeries[minKeyTimeSeries]
    value = value1.add(value2)
    value /= (errorTimeSeries[minKeyTimeSeries] + errorNews[minKeyNews])
    #print(value)
    makeGraph(target,valueCombined=value,name=str(minKeyTimeSeries) + " - " + str(minKeyNews))
    combinedErrors["time series:" + str(minKeyTimeSeries) + " and news: " + str(minKeyNews)] = mean_absolute_error(target,value)
    statistic,pvalue = mannwhitneyu(target,pd.Series(value[0]))
    pValues["time series:" + str(minKeyTimeSeries) + " and news: " + str(minKeyNews)] = pvalue

    print("Error scores in combined prediction = {}".format(combinedErrors))
    print("P-Values in combined prediction = {}".format(pValues))

    return