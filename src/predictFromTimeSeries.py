'''
Predicting stock close prices using past data of stock with suitable features
'''

import sys,json
sys.path.insert(0, '..\Algorithms')

from featureSelection import FeatureSelector
from SVR import svr
from GradientBoost import gradientBoost
from RandomForest import randomForest
from KNeighbor import kNeighbor
from GaussianProcess import gaussianProcess
from DecisionTree import decisionTree
from allowedModels import AllowedModels
from BayesianRidge import bayesianRidge
from LinearRegression import linearRegression

import logging
Logger = logging.getLogger('predictFromTimeSeries.stdout')

def predictFromTimeSeries(attributes, target, testAttributes, testTarget,nCpuCores):

    '''
        Predict stock close prices using the past data of stock after doing feature extraction
    :param attributes: the data frame of attributes(candidates for features) for training
    :param target: the data frame of target variable(close price) for training
    :param testAttributes: the data frame of attributes for testing
    :param testTarget: the data frame of target variable for testing
    :param nCpuCores: no of cores to be used (refer to ParseArgs() in main.py)
    :return: dictionaries of errors and predicted values of stock closing price as predicted by all models
    '''

    #dictionary have error values of all the models
    scores = {}
    predictions = {}
    pValues = {}
    features = {}
    Identifier = "Time Series"

    #Feature Selection
    featureSelector = FeatureSelector()

    #Feature Selection for Bayesian Ridge
    attributesBR, targetBR = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.BAYESIAN_RIDGE)
    testAttributesBR = testAttributes[attributesBR.columns]
    features['Bayesian Ridge'] = list(attributesBR.columns)

    #Training and testing Bayesian Ridge model
    error,predictedValues,pvalue = bayesianRidge(attributesBR,targetBR,testAttributesBR,testTarget,Identifier)
    scores['Bayesian Ridge'] = error
    predictions['Bayesian Ridge'] = predictedValues
    pValues['Bayesian Ridge'] = pvalue

    #Feature Selection for Decision Trees Regression
    attributesDT, targetDT = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.DECISION_TREE)
    testAttributesDT = testAttributes[attributesDT.columns]
    features['Decision Tree'] = list(attributesDT.columns)

    #Training and testing Decision Tree Regression model
    error,predictedValues,pvalue = decisionTree(attributesDT,targetDT,testAttributesDT,testTarget,nCpuCores,Identifier)
    scores['Decision Tree'] = error
    predictions['Decision Tree'] = predictedValues
    pValues['Decision Tree'] = pvalue

    #Training and testing Gaussian Process model (Feature selection not possible for Gaussian Process using RFE)
    error,predictedValues,pvalue = gaussianProcess(attributes,target,testAttributes,testTarget,Identifier)
    scores['Gaussian Process'] = error
    predictions['Gaussian Process'] = predictedValues
    pValues['Gaussian Process'] = pvalue

    #Feature Selection for Gradient Boosting Regression
    attributesGB, targetGB = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.GRADIENT_BOOST)
    testAttributesGB = testAttributes[attributesGB.columns]
    features['Gradient Boost'] = list(attributesGB.columns)

    #Training and testing Gradient Boosting Regression model
    error,predictedValues,pvalue = gradientBoost(attributesGB,targetGB,testAttributesGB,testTarget,nCpuCores,Identifier)
    scores['Gradient Boost'] = error
    predictions['Gradient Boost'] = predictedValues
    pValues['Gradient Boost'] = pvalue

    #Training and testing K Neighbor model (Feature selection not possible for K Neighbors using RFE)
    error,predictedValues,pvalue = kNeighbor(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['K Neighbor'] = error
    predictions['K Neighbor'] = predictedValues
    pValues['K Neighbor'] = pvalue

    #Feature Selection for Linear Regression
    attributesLR, targetLR = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.LINEAR_REGRESSION)
    testAttributesLR = testAttributes[attributesLR.columns]
    features['Linear Regression'] = list(attributesLR.columns)

    #Training and testing Linear Regression model
    error,predictedValues,pvalue = linearRegression(attributesLR,targetLR,testAttributesLR,testTarget,nCpuCores,Identifier)
    scores['Linear Regression'] = error
    predictions['Linear Regression'] = predictedValues
    pValues['Linear Regression'] = pvalue

    #Feature Selection for Random Forests Regression
    attributesRF, targetRF = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.RANDOM_FOREST)
    testAttributesRF = testAttributes[attributesRF.columns]
    features['Random Forest'] = list(attributesRF.columns)

    #Training and testing Random Forests Regression model
    error,predictedValues,pvalue = randomForest(attributesRF,targetRF,testAttributesRF,testTarget,nCpuCores,Identifier)
    scores['Random Forest'] = error
    predictions['Random Forest'] = predictedValues
    pValues['Random Forest'] = pvalue

    #Feature Selection for SVR
    attributesSVR, targetSVR = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.SVR)
    testAttributesSVR = testAttributes[attributesSVR.columns]
    features['SVR'] = list(attributesSVR.columns)

    #Training and testing SVR model
    error,predictedValues,pvalue = svr(attributesSVR,targetSVR,testAttributesSVR,testTarget,nCpuCores,Identifier)
    scores['SVR'] = error
    predictions['SVR'] = predictedValues
    pValues['SVR'] = pvalue

    print("Error scores in time series prediction = {}".format(scores))
    print("P-Values in time series prediction = {}".format(pValues))

    with open(str('..\\features.json'),'w') as featureFile:
        json.dump(features,featureFile)

    return scores,predictions