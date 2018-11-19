import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

import logging
Logger = logging.getLogger('GaussianProcess.stdout')

def gaussianProcess(X_train,y_train,X_test,y_test):

    '''
        Fits Gaussian Process Regression model on the data provided after feature selection.
    :param X_train: the data frame containing the selected features for training
    :param y_train: the data frame of target variable used for training
    :param X_test: the data frame of containing the selected features for testing
    :param y_test: the data frame of target variable for testing
    :param cpus: no of cores to be used (refer to ParseArgs() in main.py)
    :return: returns the error score and predicted values
    '''

    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train,y_train)

    Logger.info("Gradient Boost Model: {}".format(gpr))


    # save the model to disk
    filename = '..\Models\GaussianProcess.sav'
    joblib.dump(gpr, filename)

    prediction = gpr.predict(X_test)
    prediction = pd.DataFrame(prediction,index=y_test.index)
    error = mean_absolute_error(y_test,prediction)

    plt.plot(y_test.index,y_test,'r',prediction.index,prediction,'b')
    plt.xlabel("Dates")
    plt.ylabel("Stock closing values")
    plt.title("Gaussian Process")
    plt.show()

    print(prediction)

    return error,prediction
