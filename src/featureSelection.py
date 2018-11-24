'''
Feature Selection
'''

from allowedModels import AllowedModels
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import logging


class FeatureSelector:

    def __init__(self):
        self.data = []
        self.Logger = logging.getLogger('featureSelection.stdout')

    def featureSelection(self, attributes, target, numberOfRequiredFeatures = 1, modelName = AllowedModels.SVR):
        '''
            Feature Selection
                Select the most important features
            :param attributes: the data frame of attributes(candidates for features) received from main function
            :param target: the data frame of target variable(close price)
            :param numberOfRequiredFeatures: The number of features required, by default 1
            :param modelName: The minimum number of features required, by default SVR
            :returns selected_attributes, target
        '''

        if modelName == AllowedModels.SVR:
            return self.__svr(attributes, target, numberOfRequiredFeatures)
        elif modelName == AllowedModels.RANDOM_FOREST:
            return self.__randomForest(attributes,target,numberOfRequiredFeatures)
        elif modelName == AllowedModels.GRADIENT_BOOST:
            return self.__gradientBoost(attributes,target,numberOfRequiredFeatures)
        elif modelName == AllowedModels.DECISION_TREE:
            return self.__decisionTree(attributes,target,numberOfRequiredFeatures)
        else:
            return attributes, target

    def __svr(self, attributes, target, numberOfRequiredFeatures):
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, numberOfRequiredFeatures, step=1)
        selector = selector.fit(attributes, target)
        self.Logger.info(selector.support_)
        self.Logger.info(selector.ranking_)
        selected_attributes = attributes.copy()
        index = 0
        indexNames = list(attributes.columns.values)
        for name in indexNames:
            if not selector.support_[index]:
                selected_attributes = selected_attributes.drop(name, axis=1)
            index = index + 1
        featuredSelectedDataSet = selected_attributes.copy()
        featuredSelectedDataSet['close'] = target
        featuredSelectedDataSet.to_csv("..\FeatureSelectedStockSVR" + str(numberOfRequiredFeatures) + ".csv", encoding='utf-8')
        return selected_attributes, target

    def __gradientBoost(self,attributes,target,numberOfRequiredFeatures):
        estimator = GradientBoostingRegressor(random_state=1)
        selector = RFE(estimator, numberOfRequiredFeatures, step=1)
        selector = selector.fit(attributes, target)
        self.Logger.info(selector.support_)
        self.Logger.info(selector.ranking_)
        selected_attributes = attributes.copy()
        index = 0
        indexNames = list(attributes.columns.values)
        for name in indexNames:
            if not selector.support_[index]:
                selected_attributes = selected_attributes.drop(name, axis=1)
            index = index + 1
        featuredSelectedDataSet = selected_attributes.copy()
        featuredSelectedDataSet['close'] = target
        featuredSelectedDataSet.to_csv("..\FeatureSelectedStockGradientBoost" + str(numberOfRequiredFeatures) + ".csv", encoding='utf-8')
        return selected_attributes, target

    def __randomForest(self,attributes,target,numberOfRequiredFeatures):
        estimator = RandomForestRegressor(random_state=1)
        selector = RFE(estimator, numberOfRequiredFeatures, step=1)
        selector = selector.fit(attributes, target)
        self.Logger.info(selector.support_)
        self.Logger.info(selector.ranking_)
        selected_attributes = attributes.copy()
        index = 0
        indexNames = list(attributes.columns.values)
        for name in indexNames:
            if not selector.support_[index]:
                selected_attributes = selected_attributes.drop(name, axis=1)
            index = index + 1
        featuredSelectedDataSet = selected_attributes.copy()
        featuredSelectedDataSet['close'] = target
        featuredSelectedDataSet.to_csv("..\FeatureSelectedStockRandomForest" + str(numberOfRequiredFeatures) + ".csv", encoding='utf-8')
        return selected_attributes, target

    def __decisionTree(self,attributes,target,numberOfRequiredFeatures):
        estimator = DecisionTreeRegressor(random_state=1)
        selector = RFE(estimator, numberOfRequiredFeatures, step=1)
        selector = selector.fit(attributes, target)
        self.Logger.info(selector.support_)
        self.Logger.info(selector.ranking_)
        selected_attributes = attributes.copy()
        index = 0
        indexNames = list(attributes.columns.values)
        for name in indexNames:
            if not selector.support_[index]:
                selected_attributes = selected_attributes.drop(name, axis=1)
            index = index + 1
        featuredSelectedDataSet = selected_attributes.copy()
        featuredSelectedDataSet['close'] = target
        featuredSelectedDataSet.to_csv("..\FeatureSelectedStockDecisionTree" + str(numberOfRequiredFeatures) + ".csv", encoding='utf-8')
        return selected_attributes, target