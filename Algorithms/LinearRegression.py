import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error
from makeGraph import makeGraph
from scipy.stats import mannwhitneyu

def linearRegression(X_train,y_train,X_test,y_test,cpus,Identifier):
    '''
        Fits Linear Regression model on the data provided after feature selection.
    :param X_train: the data frame containing the selected features for training
    :param y_train: the data frame of target variable used for training
    :param X_test: the data frame of containing the selected features for testing
    :param y_test: the data frame of target variable for testing
    :param cpus: no of cores to be used (refer to ParseArgs() in main.py)
    :return: returns the error score and predicted values
    '''

    linear = LinearRegression(n_jobs=cpus)

    linear.fit(X_train,y_train)

    # save the model to disk
    if Identifier != "News":
        filename = '../Models/LinearRegression'+'TimeSeries'+'.sav'
    else:
        filename = '../Models/LinearRegression'+'News'+'.sav'
    joblib.dump(linear, filename)

    prediction = linear.predict(X_test)

    prediction = pd.DataFrame(prediction,index=y_test.index)
    error = mean_absolute_error(y_test,prediction)

    if Identifier != "News":
        makeGraph(y_test,valueFromTimeSeries=prediction,name="Time Series - Linear Regression")
    else:
        makeGraph(y_test,valueFromNews=prediction,name="News - Linear Regression")

    print(prediction)
    statitic,pvalue = mannwhitneyu(y_test,pd.Series(prediction[0]))

    return error,prediction,pvalue



