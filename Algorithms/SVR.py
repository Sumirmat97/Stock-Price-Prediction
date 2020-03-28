import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
from sklearn.metrics import mean_absolute_error, make_scorer
from makeGraph import makeGraph
from scipy.stats import mannwhitneyu

import logging
Logger = logging.getLogger('SVR.stdout')

def svr(X_train, y_train, X_test, y_test, cpus, Identifier):
    '''
        Fits Support Vector Regression model on the data provided after feature selection.
    :param X_train: the data frame containing the selected features for training
    :param y_train: the data frame of target variable used for training
    :param X_test: the data frame of containing the selected features for testing
    :param y_test: the data frame of target variable for testing
    :param cpus: no of cores to be used (refer to ParseArgs() in main.py)
    :param Identifier: whether called for time series prediction or news prediction
    :return: returns the error score and predicted values
    '''

    svr = SVR()
    params = {'kernel':['linear'],'C':[1e-2,1e-1,1,1e1,1e2],'gamma':[0.1,0.01]}

    scorer = make_scorer(mean_absolute_error)
    timeSeriesSplit = TimeSeriesSplit(n_splits=3).split(X_train)
    gridSearch = GridSearchCV(estimator = svr, cv = timeSeriesSplit, param_grid = params, n_jobs = cpus, scoring = scorer)

    gridSearch.fit(X_train,y_train)

    model = gridSearch.best_estimator_

    Logger.info("Best SVR score: {}".format(gridSearch.best_score_))
    Logger.info("Best SVR Model: {}".format(model))

    # save the model to disk
    if Identifier != "News":
        filename = '../Models/SVR'+'TimeSeries'+'.sav'
    else:
        filename = '../Models/SVR'+'News'+'.sav'
    joblib.dump(model, filename)

    prediction = model.predict(X_test)

    prediction = pd.DataFrame(prediction,index=y_test.index)
    error = mean_absolute_error(y_test,prediction)

    if Identifier != "News":
        makeGraph(y_test,valueFromTimeSeries=prediction,name="Time Series - SVR")
    else:
        makeGraph(y_test,valueFromNews=prediction,name="News - SVR")

    #print(prediction)
    statistic,pvalue = mannwhitneyu(y_test,pd.Series(prediction[0]))

    return error,prediction,pvalue


