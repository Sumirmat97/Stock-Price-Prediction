import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
import matplotlib.pyplot as plt

import logging
Logger = logging.getLogger('DecisionTree.stdout')

def decisionTree(X_train,y_train,X_test,y_test,cpus):

    '''
        Fits Decision Tree Regression model on the data provided after feature selection.
    :param X_train: the data frame containing the selected features for training
    :param y_train: the data frame of target variable used for training
    :param X_test: the data frame of containing the selected features for testing
    :param y_test: the data frame of target variable for testing
    :param cpus: no of cores to be used (refer to ParseArgs() in main.py)
    :return: returns the error score and predicted values
    '''

    decisionTree = DecisionTreeRegressor()

    params = {'max_depth':[1,5,10,50,100,500],'random_state':[1]}

    scorer = make_scorer(mean_absolute_error)
    timeSeriesSplit = TimeSeriesSplit(n_splits=3).split(X_train)
    gridSearch = GridSearchCV(estimator = decisionTree, cv = timeSeriesSplit, param_grid = params, n_jobs = cpus, scoring=scorer)

    gridSearch.fit(X_train,y_train)

    model = gridSearch.best_estimator_

    Logger.info("Best Random Forest score: {}".format(gridSearch.best_score_))
    Logger.info("Best Random Forest Model: {}".format(model))

    # save the model to disk
    filename = '..\Models\DecisionTree.sav'
    joblib.dump(model, filename)

    prediction = model.predict(X_test)
    prediction = pd.DataFrame(prediction,index=y_test.index)
    error = mean_absolute_error(y_test,prediction)

    plt.plot(y_test.index,y_test,'r',prediction.index,prediction,'b')
    plt.xlabel("Dates")
    plt.ylabel("Stock closing values")
    plt.title("Decision Tree")
    plt.show()

    print(prediction)

    return error,prediction
