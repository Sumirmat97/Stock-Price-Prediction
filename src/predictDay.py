'''
    predicting the stock price of a certain set of days
'''

import pandas as pd

from sklearn.externals import joblib
from interpolation import interpolate
from preprocessing import preprocessing
from cleaning import cleanForPrediction
from embeddings import embeddings
import json, re
import logging

Logger = logging.getLogger('predictToday.stdout')

def predictDay(timeSeriesDataFrame,newsDataFrame,modelTimeSeries,modelNews):
    '''
        Predicts the stock price of a certain day as per the given inputs
    :param timeSeriesDataFrame: data frame containing data of the time series inputs of the days for which prediction is desired
    :param newsDataFrame: data frame containing the news inputs of the days for which prediction is desired
    :param modelTimeSeries: trained model for prediction from time series if provided
    :param modelNews: trained model for prediction from news if provided
    :return: the final predicted value
    '''

    #loading the dictionaries of Mean Absolute Errors
    with open('..\errors.json','r') as errorFile:
        errors = json.load(errorFile)

    errorTimeSeries = errors['TimeSeries']
    errorNews = errors['News']

    with open(str('..\\features.json'),'r') as featureFile:
        features = json.load(featureFile)

    #Selecting best time series model if not given by user
    if not modelTimeSeries:
        minKeyTimeSeries = min(errorTimeSeries, key = errorTimeSeries.get)
        modelName = getModelFromKey(minKeyTimeSeries)
        timeSeriesModel = joblib.load(str("..\\Models\\"+modelName+"TimeSeries.sav"))
        timeSeriesError = errorTimeSeries[minKeyTimeSeries]
        timeSeriesFeatures = features[minKeyTimeSeries]
    else:
        timeSeriesModel = joblib.load(modelTimeSeries)
        key = getKeyFromModel(modelTimeSeries)
        timeSeriesError = errorTimeSeries[key]
        timeSeriesFeatures = features[key]

    #Selecting best model predicting from news if not given by user
    if not modelNews:
        minKeyNews = min(errorNews, key = errorNews.get)
        modelName = getModelFromKey(minKeyNews)
        newsModel = joblib.load(str("..\\Models\\"+modelName+"News.sav"))
        newsError = errorNews[minKeyNews]
    else:
        newsModel = joblib.load(modelNews)
        newsError = errorNews[getKeyFromModel(modelNews)]

    #processing dataframe
    colsToInterpolate = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    timeSeriesDataFrame = interpolate(timeSeriesDataFrame, colsToInterpolate)
    timeSeriesDataFrame = preprocessing(timeSeriesDataFrame)

    timeSeriesDataFrame = timeSeriesDataFrame.drop('close',axis=1)
    timeSeriesDataFrame['volume'] /= 100000

    #prediction of stock price using time series model
    timeSeriesPrediction = timeSeriesModel.predict(timeSeriesDataFrame[timeSeriesFeatures])
    timeSeriesPrediction = pd.Series(timeSeriesPrediction,index=timeSeriesDataFrame.index)

    #newsDataFrame = newsDataFrame.drop('close',axis=1)

    #cleaning of textual data
    newsDataFrame = cleanForPrediction(newsDataFrame)

    #finding the word embeddings of textual data
    newsData = embeddings(newsDataFrame)
    newsData = newsData[0]
    newsDataFrame = pd.DataFrame(newsData,index=newsDataFrame.index)

    #prediction of stock price using news model
    newsPrediction = newsModel.predict(newsDataFrame)
    newsPrediction = pd.Series(newsPrediction,index=newsDataFrame.index)

    #combining the predictions
    value1 = timeSeriesPrediction*newsError
    value2 = newsPrediction*timeSeriesError
    value = value1.add(value2)
    value /= (timeSeriesError + newsError)

    print(value)

    return

def getModelFromKey(keyName):
    '''
        Gives the name of model from keys of dictionary
    :param keyName: one key of the dictionary
    :return: the name of a model (without the space in between its name)
    '''
    list = keyName.split(" ")
    name = "".join(list)
    return name

def getKeyFromModel(modelName):
    '''
        Gives key of the dictionaries from model name
    :param modelName: the name of a model with extension(without the space in between its name)
    :return: the corresponding key of the dictionary
    '''

    pos = modelName.find("TimeSeries.sav")
    if pos != -1:
        modelName = modelName[0:pos]
    else:
        pos = modelName.find("News.sav")
        modelName = modelName[0:pos]

    if modelName == "SVR":
        return modelName

    list = re.findall('([A-Z][a-z]*)', modelName)
    modelName = " ".join(list)

    return modelName