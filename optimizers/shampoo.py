import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F
from optimizees.base import BaseOptimizee

'''
modified from:
https://pytorch-optimizer.readthedocs.io/en/latest/_modules/torch_optimizer/shampoo.html
'''

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)

class Shampoo(object):
    def __init__(self, *args, **kwargs):
        """
        TBA
        """
        self._optimizees = None
        self.step_size = None
        self.L = None 
        self.R = None 

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'Shampoo'

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        """
        TBA
        """
        self._optimizees = optimizees
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())
        num_batch = self._optimizees.X.shape[0]
        size1 = self._optimizees.X.shape[1]
        size2 = self._optimizees.X.shape[2]
        device = self._optimizees.X.device
        self.L = 1e-8 * torch.eye(size1, device=device).reshape((1,size1,size1)).repeat(num_batch,1,1)
        self.R = 1e-8 * torch.eye(size2, device=device).reshape((1,size2,size2)).repeat(num_batch,1,1)

    def clean_state(self, optimizees):
        """
        TBA
        """
        self._optimizees = None
        self.step_size = None
        self.L = None 
        self.R = None 

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

        subgrad = optimizees.subgrad(compute_grad=False) # the same shape with X
        
        self.L = self.L + torch.matmul(subgrad, subgrad.transpose(1,2))
        self.R = self.R + torch.matmul(subgrad.transpose(1,2), subgrad)
        
        L_precond_inv = torch.clone(self.L)
        for i in range(self.L.shape[0]): # for all batches
            L_precond_inv[i] = _matrix_power(self.L[i], -1/4)
        R_precond_inv = torch.clone(self.R)
        for i in range(self.R.shape[0]): # for all batches
            R_precond_inv[i] = _matrix_power(self.R[i], -1/4)
            
        update_dir = torch.matmul(subgrad, R_precond_inv)
        update_dir = torch.matmul(L_precond_inv, update_dir)
        
        Xnew = optimizees.X - self.step_size * update_dir
        optimizees.X = Xnew

        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()