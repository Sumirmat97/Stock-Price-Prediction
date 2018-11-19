'''
Allowed Models
'''

from enum import Enum


class AllowedModels(Enum):
    SVR = 1
    RANDOM_FOREST = 2
    GRADIENT_BOOST = 3
    DECISION_TREE = 4
    NO_MODEL = 5