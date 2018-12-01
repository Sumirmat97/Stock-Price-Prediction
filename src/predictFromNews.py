'''
Predicting stock close prices using news
'''

import sys
sys.path.insert(0, '..\Algorithms')

from SVR import svr
from GradientBoost import gradientBoost
from RandomForest import randomForest
from KNeighbor import kNeighbor
from GaussianProcess import gaussianProcess
from DecisionTree import decisionTree
from BayesianRidge import bayesianRidge
from LinearRegression import linearRegression

import logging
Logger = logging.getLogger('predictFromNews.stdout')

def predictFromNews(attributes, target, testAttributes, testTarget, nCpuCores):
    '''
        Predict stock close prices using the text analysis on news
    :param attributes: the data frame of attributes(candidates for features) for training
    :param target: the data frame of target variable(close price) for training
    :param testAttributes: the data frame of attributes for testing
    :param testTarget: the data frame of target variable for testing
    :param nCpuCores: no of cores to be used (refer to ParseArgs() in main.py)
    :return: dictionaries of errors and predicted values of stock closing price as predicted by all models
    '''

    scores = {}
    predictions = {}
    pValues = {}
    Identifier = "News"

    #Training and testing Bayesian Ridge model

    error,predictedValues,pvalue = bayesianRidge(attributes,target,testAttributes,testTarget,Identifier)
    scores['Bayesian Ridge'] = error
    predictions['Bayesian Ridge'] = predictedValues
    pValues['Bayesian Ridge'] = pvalue

    #Training and testing Decision Tree Regression model
    error,predictedValues,pvalue = decisionTree(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['Decision Tree'] = error
    predictions['Decision Tree'] = predictedValues
    pValues['Decision Tree'] = pvalue

    #Training and testing Gaussian Process model
    error,predictedValues,pvalue = gaussianProcess(attributes,target,testAttributes,testTarget,Identifier)
    scores['Gaussian Process'] = error
    predictions['Gaussian Process'] = predictedValues
    pValues['Gaussian Process'] = pvalue

    #Training and testing Gradient Boosting Regression model
    error,predictedValues,pvalue = gradientBoost(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['Gradient Boost'] = error
    predictions['Gradient Boost'] = predictedValues
    pValues['Gradient Boost'] = pvalue

    #Training and testing K Neighbor model
    error,predictedValues,pvalue = kNeighbor(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['K Neighbor'] = error
    predictions['K Neighbor'] = predictedValues
    pValues['K Neighbor'] = pvalue

    #Training and testing Linear Regression model
    error,predictedValues,pvalue = linearRegression(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['Linear Regression'] = error
    predictions['Linear Regression'] = predictedValues
    pValues['Linear Regression'] = pvalue

    #Training and testing Random Forests Regression model
    error,predictedValues,pvalue = randomForest(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['Random Forest'] = error
    predictions['Random Forest'] = predictedValues
    pValues['Random Forest'] = pvalue

    #Training and testing SVR model
    error,predictedValues,pvalue = svr(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['SVR'] = error
    predictions['SVR'] = predictedValues
    pValues['SVR'] = pvalue

    print("Error scores in news prediction = {}".format(scores))
    print("P-Values in news prediction = {}".format(pValues))
    return scores,predictions