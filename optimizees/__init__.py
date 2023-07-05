from .base import BaseOptimizee
from .lasso import LASSO
from .logistic_l1 import LogisticL1
from .logistic_l1_cifar10 import LogisticL1CIFAR10

OPTIMIZEE_DICT = {
    'LASSO': LASSO,
    'LogisticL1': LogisticL1,
    'LogisticL1CIFAR10': LogisticL1CIFAR10,
}

