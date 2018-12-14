import pandas as pd
import psutil, argparse, logging
import json

from interpolation import interpolate
from preprocessing import preprocessing
from predictFromTimeSeries import predictFromTimeSeries
from cleaning import clean
from embeddings import embeddings
from predictFromNews import predictFromNews
from combinePredictions import combinePredictions
from predictDay import predictDay

logging.basicConfig(filename='StockLogger.log', filemode='w', level=logging.DEBUG)
Logger = logging.getLogger('main.stdout')

def main(Args):
    '''
    Main function for stock price prediction
    :param args: arguments acquired from command lines(refer to ParseArgs() for list of args)
    '''

    stockFilePath = Args.stockfilepath
    newsFilePath = Args.newsfilepath
    nCpuCores = Args.ncpucores
    testStockFilePath = Args.teststockfilepath
    testNewsFilePath = Args.testnewsfilepath
    modelTimeSeries = Args.modeltimeseries
    modelNews = Args.modelnews

    Logger.debug("StockDataPath: {}, NewsDataPath: {}, NCpuCores: {}, TestStockFilePath: {}, TestNewsFilePath: {}, modelTimeSeries: {}, modelNews: {}"
                     .format(stockFilePath, newsFilePath, nCpuCores,testStockFilePath,testNewsFilePath,modelTimeSeries,modelNews))
    if Args.predict == 1:
        timeSeriesDataFrame = pd.read_csv(stockFilePath,parse_dates=True,index_col="date")
        newsDataFrame = pd.read_csv(newsFilePath,parse_dates=True,index_col="date")
        predictDay(timeSeriesDataFrame,newsDataFrame,modelTimeSeries,modelNews)
    else:
        #Time Series Analysis
        dataFrame = pd.read_csv(stockFilePath, parse_dates=True,index_col="date")
        testDataFrame = pd.read_csv(testStockFilePath,parse_dates=True,index_col="date")

        colsToInterpolate = ['open', 'high', 'low', 'close', 'adj_close', 'volume']

        #Interpolate
        dataFrame = interpolate(dataFrame, colsToInterpolate)
        testDataFrame = interpolate(testDataFrame, colsToInterpolate)

        #Preprocessing
        dataFrame = preprocessing(dataFrame)
        attributes = dataFrame.drop('close', axis=1)
        target = dataFrame.loc[:,'close']
        testDataFrame = preprocessing(testDataFrame)
        testAttributes = testDataFrame.drop('close',axis=1)
        testTarget = testDataFrame.loc[:,'close']

        #Normalization - converting values to comparable range
        attributes['volume'] /= 100000
        testAttributes['volume'] /= 100000

        #Predictions and errors of different algorithms for news
        errorTimeSeries,predictionTimeSeries = predictFromTimeSeries(attributes, target, testAttributes, testTarget, nCpuCores)

        #News Analysis
        newsDataFrame = pd.read_csv(newsFilePath,parse_dates=True,index_col='date')
        testNewsDataFrame = pd.read_csv(testNewsFilePath,parse_dates=True,index_col='date')

        #Cleaning of textual data
        newsAttributes,newsTarget = clean(newsDataFrame)
        newsTestAttributes,newsTestTarget = clean(testNewsDataFrame)

        #Embeddings
        newsAttributes,newsTestAttributes = embeddings(newsAttributes,newsTestAttributes)

        #Predictions and errors of different algorithms for news
        errorNews,predictionNews = predictFromNews(newsAttributes,newsTarget,newsTestAttributes,newsTestTarget,nCpuCores)

        #writing the errors in models in file for future use
        errors = dict()
        errors['TimeSeries'] = errorTimeSeries
        errors['News'] = errorNews
        with open('..\errors.json','w') as errorFile:
            json.dump(errors,errorFile)

        combinePredictions(errorTimeSeries,errorNews,predictionTimeSeries,predictionNews,testTarget)

def ParseArgs():
    Args = argparse.ArgumentParser(description="Prediction of Stock Prices")
    Args.add_argument("--predict", type=int, default=0,
                      help="Whether to predict or train models (0 to train 1 to predict for certain days)")
    Args.add_argument("--stockfilepath",
                      help="Absolute path of csv file having history data of stock prices for training")
    Args.add_argument("--newsfilepath",
                      help="Absolute path of csv file having news for training")
    Args.add_argument("--teststockfilepath",
                      help="Absolute path of csv file having data of stock prices for testing")
    Args.add_argument("--testnewsfilepath",
                      help="Absolute path of csv file having news for testing")
    Args.add_argument("--ncpucores", type=int, default=psutil.cpu_count(),
                      help="Number of CPUs that will be used for processing")
    Args.add_argument("--modeltimeseries",
                      help="Absolute path to the saved model file for time series prediction(.sav extension)")
    Args.add_argument("--modelnews",
                      help="Absolute path to the saved model file for news prediction(.sav extension)")
    return Args.parse_args()

if __name__ == "__main__":
    main(ParseArgs())
