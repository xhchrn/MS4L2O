import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F


class SubGradientDescent(object):
    def __init__(self, *args, **kwargs):
        """
        TBA
        """
        self._optimizees = None
        self._step_size = None

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'SubGradientDescent'

    def reset_state(self, optimizees):
        """
        TBA
        """
        self._optimizees = optimizees
        self._step_size = 0.9999 / optimizees.grad_lipschitz()

    def clean_state(self, optimizees):
        """
        TBA
        """
        self._currnt_optimizees = None
        self._step_size = None

    @property
    def step_size(self):
        """TODO: Docstring for step_size.
        :returns: TODO

        """
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        """TODO: Docstring for step_size.setter
        """
        self._step_size = value

    def get_step_size(self, step_size):
        """TODO: Docstring for set_step_size.
        :returns: TODO

        """
        if step_size is None:
            assert self._step_size is not None
            return self._step_size
        else:
            return step_size

    def __call__(self, optimizees, step_size=None, *args, **kwargs):
        """
        TBA
        """
        if self._optimizees is None or self._step_size is None:
            self.reset_state(optimizees)
        else:
            if optimizees is not self._optimizees:
                print('WARNING: a new set of optimizees are fed to the optimzer.'
                      'Please run `optimizer.reset_state(optimizees)` properly '
                      'before calling the optimzers.')

        subgrad = optimizees.subgrad(compute_grad=False)
        Xnew = optimizees.X - self.step_size * subgrad
        optimizees.X = Xnew

        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()

