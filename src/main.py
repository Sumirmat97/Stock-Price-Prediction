import numpy as np
import pandas as pd
import psutil, argparse, logging

from interpolation import interpolate
from preprocessing import preprocessing
from normalization import normalize

logging.basicConfig(filename='StockLogger.log',filemode='w',level=logging.DEBUG)
Logger = logging.getLogger('main.stdout')

def main(Args):
    '''
    Main function for stock price prediction
    :param args: arguments acquired from command lines(refer to ParseArgs() for list of args)
    '''

    historyFilePath = Args.historyfilepath
    newsFilePath = Args.newsfilepath
    nCpuCores= Args.ncpucores
    model= Args.model

    testSize= Args.testsize

    Logger.debug("HistoryDataPath: {}, NewsDataPath: {}, NCpuCores: {}, TestSize: {}"
                     .format(historyFilePath, newsFilePath, nCpuCores, testSize))
    dataFrame = pd.read_csv(historyFilePath, parse_dates=True,index_col="date")

    colsToInterpolate = ['open', 'high', 'low', 'close', 'adj_close', 'volume']

    dataFrame = interpolate(dataFrame, colsToInterpolate)
    dataFrame = preprocessing(dataFrame)
    dataFrame = normalize(dataFrame)

    print(dataFrame.head())

def ParseArgs():
    Args =  argparse.ArgumentParser(description="Prediction of Stock Prices")
    Args.add_argument("--historyfilepath",
                      help="Absolute path of csv file having history data of stock")
    Args.add_argument("--newsfilepath",
                      help="Absolute path of csv file having news")
    Args.add_argument("--ncpucores", type= int, default= psutil.cpu_count(),
                      help= "Number of CPUs that will be used for processing")
    Args.add_argument("--testsize", type= float, default= 0.3,
                      help= "Size of the test set when split by Scikit Learn's Train Test Split module")
    Args.add_argument("--model",
                      help= "Absolute path to the saved model file(.pkl extension)")
    return Args.parse_args()

if __name__ == "__main__":
    main(ParseArgs())
