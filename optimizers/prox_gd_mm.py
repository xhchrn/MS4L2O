import copy
import math
import numpy as np

from optimizees.base import BaseOptimizee

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ProximalGradientDescentMomentum(object):
    def __init__(self, *args, **kwargs):
        """
        An implement of the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) 
        """
        self.current_optimizees = None
        self.step_size = None
        self.tau = None
        self.Z = None

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'ProximalGradientDescentMomentum'

    def detach_state(self):
        pass

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        """
        TBA
        """
        self.current_optimizees = optimizees
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())
        self.tau = 1.0
        self.Z = copy.deepcopy(optimizees.X)

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

        smooth_grad = optimizees.smooth_grad(dict(X=self.Z), compute_grad=False)
        temp = self.Z - step_size * smooth_grad
        Xnew = optimizees.prox(dict(X=temp, P=step_size))

        prev_tau = self.tau
        self.tau = (1.0 + math.sqrt(1.0 + 4.0 * prev_tau**2)) / 2.0
        self.Z = Xnew + ((prev_tau - 1.0) / self.tau) * (Xnew - optimizees.X)

        optimizees.X = Xnew

        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()

