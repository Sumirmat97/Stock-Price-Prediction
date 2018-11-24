'''
Predicting stock close prices using past data of stock with suitable features
'''

import sys
sys.path.insert(0, '..\Algorithms')

from featureSelection import FeatureSelector
from SVR import svr
from GradientBoost import gradientBoost
from RandomForest import randomForest
from KNeighbor import kNeighbor
from GaussianProcess import gaussianProcess
from DecisionTree import decisionTree
from allowedModels import AllowedModels
from other import other
from other2 import other2

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
    :return: predicted values of stock closing price as predicted by best model
    '''

    #dictionary have error values of all the models
    scores = {}
    predictions = {}
    Identifier = "Time Series"

    #Feature Selection
    featureSelector = FeatureSelector()

    #Feature Selection for Decision Trees Regression
    attributesDT, targetDT = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.DECISION_TREE)
    testAttributesDT = testAttributes[attributesDT.columns]

    #Training and testing Decision Tree Regression Algo
    error,predictedValues = decisionTree(attributesDT,targetDT,testAttributesDT,testTarget,nCpuCores,Identifier)
    scores['Decision Tree'] = error
    predictions['Decision Tree'] = predictedValues

    #Training and testing Gaussian Process Algo (Feature selection not possible for Gaussian Process using RFE)
    error,predictedValues = gaussianProcess(attributes,target,testAttributes,testTarget,Identifier)
    scores['Gaussian Process'] = error
    predictions['Gaussian Process'] = predictedValues

    #Feature Selection for Gradient Boosting Regression
    attributesGB, targetGB = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.GRADIENT_BOOST)
    testAttributesGB = testAttributes[attributesGB.columns]

    #Training and testing Gradient Boosting Regression Algo
    error,predictedValues = gradientBoost(attributesGB,targetGB,testAttributesGB,testTarget,nCpuCores,Identifier)
    scores['Gradient Boost'] = error
    predictions['Gradient Boost'] = predictedValues

    #Training and testing K Neighbor Algo (Feature selection not possible for K Neighbors using RFE)
    error,predictedValues = kNeighbor(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['K Neighbor'] = error
    predictions['K Neighbor'] = predictedValues

    #Feature Selection for Random Forests Regression
    attributesRF, targetRF = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.RANDOM_FOREST)
    testAttributesRF = testAttributes[attributesRF.columns]

    #Training and testing Random Forests Regression Algo
    error,predictedValues = randomForest(attributesRF,targetRF,testAttributesRF,testTarget,nCpuCores,Identifier)
    scores['Random Forest'] = error
    predictions['Random Forest'] = predictedValues

    #Feature Selection for SVR
    attributesSVR, targetSVR = featureSelector.featureSelection(attributes, target, 6)
    testAttributesSVR = testAttributes[attributesSVR.columns]

    #Training and testing SVR Algo
    error,predictedValues = svr(attributesSVR,targetSVR,testAttributesSVR,testTarget,nCpuCores,Identifier)
    scores['SVR'] = error
    predictions['SVR'] = predictedValues

    error,prediction = other(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['other'] = error
    predictions['other'] = predictedValues

    error,prediction = other2(attributes,target,testAttributes,testTarget,nCpuCores,Identifier)
    scores['other2'] = error
    predictions['other2'] = predictedValues

    print(scores)
    return scores,predictions