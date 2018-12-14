'''
Shows and saves graphs of predictions
'''
import matplotlib.pyplot as plt

def makeGraph(valueOriginal, valueFromTimeSeries = None, valueFromNews = None, valueCombined = None, name = "AlgorithmName"):
    '''
        Shows and saves graphs of the data in input
    :param valueOriginal: the original closing prices of stock
    :param valueFromTimeSeries: the predicted values of closing prices of stock predicted from time series analysis
    :param valueFromNews: the predicted values of closing prices of stock predicted from news analysis
    :param valueCombined: the predicted values of closing prices of stock predicted from combination of both
    :param name: name of the graph
    '''

    if valueFromTimeSeries is not None and valueFromNews is not None:
        plt.plot(valueOriginal.index,valueOriginal,'r',label='Original Values')
        plt.plot(valueFromTimeSeries.index,valueFromTimeSeries,'b',label='Time Series Prediction')
        plt.plot(valueFromNews.index,valueFromNews,'g',label='Prediction from news')

    elif valueFromTimeSeries is not None:
        plt.plot(valueOriginal.index,valueOriginal,'r',label='Original Values')
        plt.plot(valueFromTimeSeries.index,valueFromTimeSeries,'b',label='Time Series Prediction')

    elif valueFromNews is not None :
        plt.plot(valueOriginal.index,valueOriginal,'r',label='Original Value')
        plt.plot(valueFromNews.index,valueFromNews,'b',label='Prediction from news')

    elif valueCombined is not None:
        plt.plot(valueOriginal.index,valueOriginal,'r',label='Original Value')
        plt.plot(valueCombined.index,valueCombined,'b',label='Combined Prediction')

    plt.legend()
    plt.xlabel("Dates")
    plt.ylabel("Stock closing values")
    plt.title(name)
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    graph = plt.gcf()
    plt.show()
    graph.savefig('../PredictedGraphs/' + name + ".png" )
