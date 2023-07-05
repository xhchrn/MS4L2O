import copy
import math
import numpy as np

from optimizees.base import BaseOptimizee

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ProximalGradientDescent():
    def __init__(self, *args, **kwargs):
        """
        TBA
        """
        self.current_optimizees = None
        self.step_size = None

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'ProximalGradientDescent'
        
    def detach_state(self):
        pass

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        """
        TBA
        """
        self.current_optimizees = optimizees
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())

    def __call__(self, optimizees, *args, **kwargs):
        """
        TBA
        """
        step_size = kwargs.get('step_size', self.step_size)
        if self.current_optimizees is None or step_size is None:
            raise RuntimeError('The optimizer is not properly initialized yet. '
                               'Please run `reset_state` method first.')
        else:
            if optimizees is not self.current_optimizees:
                print('WARNING: a new set of optimizees are fed to the optimzer.'
                      'Please run `optimizer.reset_state(optimizees, step_size)`'
                      ' properly before calling the optimzers.')

        smooth_grad = optimizees.smooth_grad(compute_grad=False)
        Z = optimizees.X - step_size * smooth_grad
        Xnew = optimizees.prox(dict(X=Z, P=step_size))
        optimizees.X = Xnew

        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()

