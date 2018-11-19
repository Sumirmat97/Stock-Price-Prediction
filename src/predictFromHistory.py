import sys
sys.path.insert(0, '..\Algorithms')

from featureSelection import FeatureSelector
from SVR import svr
from GradientBoost import gradientBoost
from RandomForest import randomForest
from KNeighbor import kNeighbor
from GaussianProcess import gaussianProcess
from DecisionTree import decisionTree
from AllowedModels import AllowedModels

def predictFromHistory(attributes, target, testAttributes, testTarget,nCpuCores):

    '''
        Predict stock prices using the past data of stock
    :param attributes: the data frame of attributes(candidates for features) for training
    :param target: the data frame of target variable(close price) for training
    :param testAttributes: the data frame of attributes for testing
    :param testTarget: the data frame of target variable for testing
    :param nCpuCores: no of cores to be used (refer to ParseArgs() in main.py)
    :return: predicted values of stock closing price as predicted by best model
    '''

    #dictionary have error values of all the models
    scores = {}

    #Feature Selection
    featureSelector = FeatureSelector()

    #Feature Selection for SVR
    attributesSVR, targetSVR = featureSelector.featureSelection(attributes, target, 6)
    testAttributesSVR = testAttributes[attributesSVR.columns]

    #Training and testing SVR Algo
    error,predictedValues = svr(attributesSVR,targetSVR,testAttributesSVR,testTarget,nCpuCores)
    scores['SVR'] = error

    #Training and testing K Neighbor Algo (Feature selection not possible for K Neighbors using RFE)
    error,predictedValues = kNeighbor(attributes,target,testAttributes,testTarget,nCpuCores)
    scores['KNeighbor'] = error

    #Training and testing Gaussian Process Algo (Feature selection not possible for Gaussian Process using RFE)
    error,predictedValues = gaussianProcess(attributes,target,testAttributes,testTarget)
    scores['GaussianProcess'] = error

    #Feature Selection for Gradient Boosting Regression
    attributesGB, targetGB = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.GRADIENT_BOOST)
    testAttributesGB = testAttributes[attributesGB.columns]

    #Training and testing Gradient Boosting Regression Algo
    error,predictedValues = gradientBoost(attributesGB,targetGB,testAttributesGB,testTarget,nCpuCores)
    scores['GradientBoost'] = error

    #Feature Selection for Random Forests Regression
    attributesRF, targetRF = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.RANDOM_FOREST)
    testAttributesRF = testAttributes[attributesRF.columns]

    #Training and testing Random Forests Regression Algo
    error,predictedValues = randomForest(attributesRF,targetRF,testAttributesRF,testTarget,nCpuCores)
    scores['RandomForest'] = error


    #Feature Selection for Decision Trees Regression
    attributesDT, targetDT = featureSelector.featureSelection(attributes,target,6,modelName = AllowedModels.DECISION_TREE)
    testAttributesDT = testAttributes[attributesDT.columns]

    #Training and testing Decision Tree Regression Algo
    error,predictedValues = decisionTree(attributesDT,targetDT,testAttributesDT,testTarget,nCpuCores)
    scores['Decision Tree'] = error

    print(scores)
    return