import torch
import torch.nn as nn


class BaseOptimizee(object):

    def __init__(self) -> None:
        raise NotImplementedError('Tried to call the \`__init__\` function in '
                                  'the BaseOptimizee class.')
    
    def generate(self, batch_size:int):
        raise NotImplementedError('Tried to call the \`generate\` function in '
                                  'the BaseOptimizee class.')

    def get_grad(
        self,
        grad_method: str,
        inputs: dict,
        compute_grad: bool,
        **kwargs,
    ):
        raise NotImplementedError('Tried to call the \`get_grad\` function in '
                                  'the BaseOptimizee class.')

    def cuda(self):
        raise NotImplementedError('Tried to call the \`cuda\` function in '
                                  'the BaseOptimizee class.')

