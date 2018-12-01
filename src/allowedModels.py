'''
Allowed Models
'''

from enum import Enum


class AllowedModels(Enum):
    SVR = 1
    RANDOM_FOREST = 2
    GRADIENT_BOOST = 3
    DECISION_TREE = 4
    BAYESIAN_RIDGE = 5
    LINEAR_REGRESSION = 6
