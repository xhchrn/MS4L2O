import math
import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

from optimizees.base import BaseOptimizee

def shrink(x, theta):
    return x.sign() * F.relu(x.abs() - theta)


class AdaLISTA(nn.Module):
    def __init__(self, layers: int, input_dim: int, output_dim: int):
        """
        Adaptive LISTA model for LASSO.
        See https://github.com/aaberdam/AdaLISTA
        """
        super().__init__()

        self.input_dim  = input_dim
        self.output_dim = output_dim

        print("---", self.input_dim, self.output_dim)

        self.layers = layers # Number of layers in the network

        #-------------------------------------
        # Define the parameters for the model
        #-------------------------------------
        self.W1 = nn.Parameter(torch.eye(self.output_dim).unsqueeze(0))
        self.W2 = nn.Parameter(torch.eye(self.output_dim).unsqueeze(0))
        self.step_sizes = nn.ParameterList()
        self.thresholds = nn.ParameterList()
        self.P = nn.ParameterList()
        for i in range(self.layers):
            L = 5.0
            self.step_sizes.append(nn.Parameter(torch.tensor(1/L)))
            self.thresholds.append(nn.Parameter(torch.tensor(1/L)))

        # print('---non smooth ---')

    """
    Function: get_optimizer
    Purpose : Return the desired optimizer for the model.
    """
    def get_meta_optimizer(
        self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3
    ):
        param_groups = []

        # W1, W2 group
        param_groups.append(
            {
                'params': [self.W1, self.W2],
                'lr' : init_lr * (lr_decay_layer ** (layer - 1))
            }
        )
        # Current layer
        param_groups.append(
            {
                'params': [self.step_sizes[layer-1], self.thresholds[layer-1]],
                'lr': init_lr
            }
        )

        # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': [self.step_sizes[i], self.thresholds[i]],
                        'lr'    : init_lr * (lr_decay_layer ** (layer-i-1))
                    }
                )

            # Stage decay
            stage_decay = lr_decay_stage2 if stage == 2 else lr_decay_stage3
            for group in param_groups:
                group['lr'] *= stage_decay

        return optim.Adam(param_groups)

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'AdaLISTA'

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        """
        TBA
        """
        self.current_step = 0
        return

    def forward(self, optimizees, *args, **kwargs):
        """
        TBA
        """
        # Optional to input the desired number of layers. The default value is
        #   the number of layers.
        K = kwargs.get('K', self.layers)
        A = optimizees.W
        X = optimizees.X
        Y = optimizees.Y

        if self.current_step < self.layers:
            # Normal iteration
            ss = self.step_sizes[self.current_step]
            th = self.thresholds[self.current_step]
        else:
            ss = self.step_sizes[-1]
            th = self.thresholds[-1]

        w1a = self.W1 @ A
        # mat1 = torch.bmm(A.T, self.W1.T, self.W1, A)
        mat1 = w1a.transpose(1,2) @ w1a
        mat2 = A.transpose(1,2) @ self.W2

        hidden = X - ss * mat1 @ X + ss * mat2 @ Y
        optimizees.X = shrink(hidden, th)

        self.current_step += 1

        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()

