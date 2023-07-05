import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F
from optimizees.base import BaseOptimizee


class AdamHD(object):
    def __init__(self, *args, **kwargs):
        """
        An implement of :
        https://github.com/gbaydin/hypergradient-descent
        """
        self._optimizees = None
        self.step_size = None
        self.momentum1 = None 
        self.momentum2 = None 
        self.eps = None
        self.t = None
        self.step_size_hyper = None
        self.hyperGrad = None
        self.aveGra1 = None
        self.aveGra2 = None

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'AdamHD'

    def detach_state(self):
        pass

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        """
        TBA
        """
        self._optimizees = optimizees
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())
        self.momentum1 = kwargs.get('momentum1', None)
        self.momentum2 = kwargs.get('momentum2', None) 
        self.eps = kwargs.get('eps', 1e-8) 
        self.aveGra1 = torch.zeros_like(optimizees.X)
        self.aveGra2 = torch.zeros_like(optimizees.X)
        self.t = 1
        self.step_size_hyper = kwargs.get('hyper_step', None)
        self.hyperGrad = torch.zeros_like(optimizees.X)

    def clean_state(self, optimizees):
        """
        TBA
        """
        self._optimizees = None
        self.step_size = None
        self.momentum1 = None 
        self.momentum2 = None 
        self.aveGra1 = None
        self.aveGra2 = None
        self.t = None
        self.eps = None
        self.step_size_hyper = None
        self.hyperGrad = None

    def __call__(self, optimizees, step_size=None, *args, **kwargs):
        """
        TBA
        """
        if self._optimizees is None or self.step_size is None:
            self.reset_state(optimizees)
        else:
            if optimizees is not self._optimizees:
                print('WARNING: a new set of optimizees are fed to the optimzer.'
                      'Please run `optimizer.reset_state(optimizees)` properly '
                      'before calling the optimzers.')

        subgrad = optimizees.subgrad(compute_grad=False)
        
        if self.t > 1:
            self.step_size = self.step_size - self.step_size_hyper * torch.dot(subgrad.reshape(-1), self.hyperGrad.reshape(-1))
            self.step_size = torch.maximum(self.step_size, torch.ones_like(self.step_size) * 1e-8)

        self.aveGra1 = self.aveGra1 * self.momentum1 + (1-self.momentum1) * subgrad
        self.aveGra2 = self.aveGra2 * self.momentum2 + (1-self.momentum2) * (subgrad ** 2)
        
        bias_correction1 = 1 - self.momentum1 ** self.t
        bias_correction2 = 1 - self.momentum2 ** self.t
        M1 = self.aveGra1 / bias_correction1
        M2 = self.aveGra2 / bias_correction2
        
        Xnew = optimizees.X - self.step_size * M1 / (M2.sqrt() + self.eps)
        optimizees.X = Xnew
        self.t += 1
        self.hyperGrad = - M1 / (M2.sqrt() + self.eps)
        
        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()