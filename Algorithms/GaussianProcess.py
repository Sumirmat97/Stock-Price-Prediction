import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import joblib
from sklearn.metrics import mean_absolute_error
from makeGraph import makeGraph
from scipy.stats import mannwhitneyu

import logging
Logger = logging.getLogger('GaussianProcess.stdout')

def gaussianProcess(X_train, y_train, X_test, y_test, Identifier):

    '''
        Fits Gaussian Process Regression model on the data provided after feature selection.
    :param X_train: the data frame containing the selected features for training
    :param y_train: the data frame of target variable used for training
    :param X_test: the data frame of containing the selected features for testing
    :param y_test: the data frame of target variable for testing
    :param Identifier: whether called for time series prediction or news prediction
    :return: returns the error score and predicted values
    '''

    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train,y_train)

    Logger.info("Gradient Boost Model: {}".format(gpr))

    # save the model to disk
    if Identifier != "News":
        filename = '../Models/GaussianProcess'+'TimeSeries'+'.sav'
    else:
        filename = '../Models/GaussianProcess'+'News'+'.sav'
    joblib.dump(gpr, filename)

    prediction = gpr.predict(X_test)
    prediction = pd.DataFrame(prediction,index=y_test.index)
    error = mean_absolute_error(y_test,prediction)

    if Identifier != "News":
        makeGraph(y_test,valueFromTimeSeries=prediction,name="Time Series - Gaussian Process")
    else:
        makeGraph(y_test,valueFromNews=prediction,name="News - Gaussian Process")

    #print(prediction)
    statistic,pvalue = mannwhitneyu(y_test,pd.Series(prediction[0]))

    return error,prediction,pvalue
