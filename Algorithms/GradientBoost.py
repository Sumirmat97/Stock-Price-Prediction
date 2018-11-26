import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, make_scorer
from makeGraph import makeGraph
from scipy.stats import mannwhitneyu

import logging
Logger = logging.getLogger('GradientBoost.stdout')

def gradientBoost(X_train, y_train, X_test, y_test, cpus, Identifier):
    '''
        Fits Gradient Boosting Regression model on the data provided after feature selection.
    :param X_train: the data frame containing the selected features for training
    :param y_train: the data frame of target variable used for training
    :param X_test: the data frame of containing the selected features for testing
    :param y_test: the data frame of target variable for testing
    :param cpus: no of cores to be used (refer to ParseArgs() in main.py)
    :param Identifier: whether called for time series prediction or news prediction
    :return: returns the error score and predicted values
    '''

    gradientBoost = GradientBoostingRegressor()
    params = {'n_estimators': [1,2,5,10,50,100], 'max_depth': [1,2,3,4,5], 'min_samples_split': [2,3],
          'learning_rate': [0.01,0.1,1], 'loss': ['ls'], 'random_state':[1]}

    scorer = make_scorer(mean_absolute_error)
    timeSeriesSplit = TimeSeriesSplit(n_splits=3).split(X_train)
    gridSearch = GridSearchCV(estimator = gradientBoost, cv = timeSeriesSplit, param_grid = params, n_jobs = cpus, scoring=scorer)

    gridSearch.fit(X_train,y_train)

    model = gridSearch.best_estimator_

    Logger.info("Best Gradient Boost score: {}".format(gridSearch.best_score_))
    Logger.info("Best Gradient Boost Model: {}".format(model))

    # save the model to disk
    if Identifier != "News":
        filename = '../Models/GradientBoost'+'TimeSeries'+'.sav'
    else:
        filename = '../Models/GradientBoost'+'News'+'.sav'
    joblib.dump(model, filename)

    prediction = model.predict(X_test)
    prediction = pd.DataFrame(prediction,index=y_test.index)
    error = mean_absolute_error(y_test,prediction)

    if Identifier != "News":
        makeGraph(y_test,valueFromTimeSeries=prediction,name="Time Series - Gradient Boosting")
    else:
        makeGraph(y_test,valueFromNews=prediction,name="News - Gradient Boosting")

    print(prediction)
    statitic,pvalue = mannwhitneyu(y_test,pd.Series(prediction[0]))

    return error,prediction,pvalue
