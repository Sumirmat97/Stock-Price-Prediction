import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, make_scorer
from makeGraph import makeGraph
from scipy.stats import mannwhitneyu

def bayesianRidge(X_train,y_train,X_test,y_test,Identifier):
    '''
        Fits Bayesian Ridge model on the data provided after feature selection.
    :param X_train: the data frame containing the selected features for training
    :param y_train: the data frame of target variable used for training
    :param X_test: the data frame of containing the selected features for testing
    :param y_test: the data frame of target variable for testing
    :param Identifier: whether called for time series prediction or news prediction
    :return: returns the error score and predicted values
    '''

    bayesianRidge = BayesianRidge()

    bayesianRidge.fit(X_train,y_train)

    # save the model to disk
    if Identifier != "News":
        filename = '../Models/BayesianRidge'+'TimeSeries'+'.sav'
    else:
        filename = '../Models/BayesianRidge'+'News'+'.sav'
    joblib.dump(bayesianRidge, filename)

    prediction = bayesianRidge.predict(X_test)

    prediction = pd.DataFrame(prediction,index=y_test.index)
    error = mean_absolute_error(y_test,prediction)

    if Identifier != "News":
        makeGraph(y_test,valueFromTimeSeries=prediction,name="Time Series - Bayesian Ridge")
    else:
        makeGraph(y_test,valueFromNews=prediction,name="News - Bayesian Ridge")

    print(prediction)
    statitic,pvalue = mannwhitneyu(y_test,pd.Series(prediction[0]))

    return error,prediction,pvalue



