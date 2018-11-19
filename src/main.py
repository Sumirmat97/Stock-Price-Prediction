import pandas as pd
import psutil, argparse, logging

from interpolation import interpolate
from preprocessing import preprocessing
from normalization import normalize
from predictFromHistory import predictFromHistory
import matplotlib.pyplot as plt

logging.basicConfig(filename='StockLogger.log', filemode='w', level=logging.DEBUG)
Logger = logging.getLogger('main.stdout')


def main(Args):
    '''
    Main function for stock price prediction
    :param args: arguments acquired from command lines(refer to ParseArgs() for list of args)
    '''

    stockFilePath = Args.stockfilepath
    newsFilePath = Args.newsfilepath
    nCpuCores= Args.ncpucores
    testStockFilePath = Args.teststockfilepath
    testNewsFilePath = Args.testnewsfilepath
    model= Args.model

    Logger.debug("StockDataPath: {}, NewsDataPath: {}, NCpuCores: {}, TestStockFilePath: {}, TestNewsFilePath: {}"
                     .format(stockFilePath, newsFilePath, nCpuCores,testStockFilePath,testNewsFilePath))
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

    #Normalization
    attributes, target = normalize(attributes, target)
    testAttributes,testTarget = normalize(testAttributes, testTarget)

    #Finding best model to predict Time Series of stock prices
    predictFromHistory(attributes, target, testAttributes, testTarget, nCpuCores)

def ParseArgs():
    Args = argparse.ArgumentParser(description="Prediction of Stock Prices")
    Args.add_argument("--stockfilepath",
                      help="Absolute path of csv file having history data of stock prices for training")
    Args.add_argument("--newsfilepath",
                      help="Absolute path of csv file having news for training")
    Args.add_argument("--teststockfilepath",
                      help="Absolute path of csv file having data of stock prices for testing")
    Args.add_argument("--testnewsfilepath",
                      help="Absolute path of csv file having news for testing")
    Args.add_argument("--ncpucores", type= int, default= psutil.cpu_count(),
                      help= "Number of CPUs that will be used for processing")
    Args.add_argument("--model",
                      help="Absolute path to the saved model file(.pkl extension)")
    return Args.parse_args()

if __name__ == "__main__":
    main(ParseArgs())
