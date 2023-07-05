import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseOptimizee

from collections import defaultdict


class LogisticL1(BaseOptimizee):

    def __init__(
        self,
        batch_size : int,
        W = None,
        Y = None,
        rho = 0.1,
        s = 5,
        device = 'cpu',
        **options
    ) -> None:
        """
        The unconstrained logistic regression with l1 regularization is
        formulated as

            minimize logistic(X, \lambda) + rho * ||X||_1
               X

        Params:
            batch_size: number of regression problems to be solved.
            input_dim: dimension of the coefficient vector.
            output_dim: size of the dataset for each logistic regression problem.
        """
        self.device = device
        self.vars = dict()

        self.batch_size = batch_size
        self.input_dim  = options.get('input_dim')
        self.output_dim = options.get('output_dim')
        self.rho = rho
        self.s = s

        seed = options.get('seed', None)
        if seed:
            rng_state = torch.get_rng_state()
            torch.manual_seed(seed)

        if W is None:
            self.W = torch.randn(self.batch_size,
                                 self.output_dim,
                                 self.input_dim).to(self.device)
        else:
            raise NotImplementedError('Not implemented')

        # Set output Y
        if Y is None:
            X_gt = torch.randn(batch_size, self.input_dim).to(self.device)
            non_zero_idx = torch.multinomial(
                torch.ones_like(X_gt), num_samples=self.s, replacement=False
            )
            self.X_gt = torch.zeros_like(X_gt).scatter(
                dim=1, index=non_zero_idx, src=X_gt
            ).unsqueeze(-1)
            self.Y = torch.where(torch.bmm(self.W, self.X_gt) >= 0.0, 1.0, 0.0)
        else:
            raise NotImplementedError('Not implemented')

        # Initialize the first iterate X
        self.X = torch.zeros(batch_size, self.input_dim, 1).to(self.device)
        prox_out = torch.zeros(batch_size, self.input_dim, 1).to(self.device)
        self.set_var('Z', prox_out)

        if seed:
            rng_state = torch.set_rng_state(rng_state)

    def get_var(self, var_name):
        return self.vars[var_name]

    def set_var(self, var_name, var_value):
        self.vars[var_name] = var_value

    def detach_vars(self):
        for var in self.vars.values():
            var.detach_()

    @property
    def X(self):
        return self.get_var('X')

    @X.setter
    def X(self, value):
        self.set_var('X', value)

    def grad_lipschitz(self):
        lip = torch.linalg.norm(self.W, dim=(-2,-1), ord=2)
        return lip.reshape(self.batch_size, 1, 1)

    def objective(self, inputs: dict = None, compute_grad: bool = False):
        """Calculate the objective value.
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            inner_prod = torch.bmm(W, X)
            logistic = torch.sigmoid(inner_prod)
            reg_loss = F.binary_cross_entropy(logistic, Y)
            l1 = self.rho * torch.abs(X).sum(dim=(1,2)).mean()
            if reg_loss.isnan().any():
                import ipdb
                ipdb.set_trace(context=10)
            return reg_loss + l1

    def objective_batch(self, inputs: dict = None, compute_grad: bool = False):
        """Calculate the objective value.
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            inner_prod = torch.bmm(W, X)
            logistic = torch.sigmoid(inner_prod)
            reg_loss = F.binary_cross_entropy(logistic, Y, reduction='none').mean(dim=(1,2))
            l1 = self.rho * torch.abs(X).sum(dim=(1,2))
            return reg_loss + l1

    def objective_batch_shift(self, inputs: dict = None, compute_grad: bool = False):
        if inputs is None:
            inputs = {}
        obj = self.objective_batch(inputs, compute_grad)
        fstar = self.fstar.reshape_as(obj)
        return (obj - fstar) / fstar

    def objective_shift(self, inputs: dict = None, compute_grad: bool = False):
        if inputs is None:
            inputs = {}
        return self.objective_batch_shift(inputs, compute_grad).mean()

    def get_grad(
        self,
        grad_method: str,
        inputs: dict = None,
        compute_grad: bool = False,
        **kwargs
    ):
        """
        Calculate a specific type of gradient, specified by `grad_method`.
        `grad_method` can be chosen from [`smooth_grad`, `subgrad`, `bp_grad`].
        """
        grad_func = getattr(self, grad_method, None)
        if grad_func is None:
            raise RuntimeError(f'Invalid grad method specified: {grad_method}')
        return grad_func(inputs, compute_grad, **kwargs)

    def bp_grad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """
        Calculate the gradient of the iterate w.r.t. the objective using back-
        propagation mechanism built in PyTorch.
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X).detach()
        X.requires_grad_(True)
        X.retain_grad()
        inputs.update({'X': X})

        # Calculate the objective value
        objective = self.objective(inputs=inputs, compute_grad=True)

        # Run backpropagation with computation graph retained, if necessary
        objective.backward(retain_graph=compute_grad)

        return X.grad

    def smooth_grad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """Calculate the gradient of the smooth part of the objective function.
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            inner_prod = torch.bmm(W, X)
            # exponetial = torch.exp(- inner_prod)
            # logistic = 1 / (1 + exponetial)
            logistic = torch.sigmoid(inner_prod)

            return ((logistic - Y) * W).mean(dim=1).unsqueeze(-1)

    def subgrad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """Calculate the subgrad
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            smooth = self.smooth_grad(dict(X=X,Y=Y,W=W), compute_grad, **kwargs)
            return smooth + self.rho * torch.sign(X)

    def prox(self, inputs: dict, compute_grad: bool = False):
        P = inputs['P']
        X = inputs['X']
        with torch.set_grad_enabled(compute_grad):
            mag = nn.functional.relu(torch.abs(X) - self.rho * P)
            return torch.sign(X) * mag

    def print4debug(self, args):
        return

    def save_to_file(self, path):
        Wcpu = self.W.cpu().numpy()
        Ycpu = self.Y.cpu().numpy()
        sio.savemat(path, {'W':Wcpu, 'Y':Ycpu, 'rho':self.rho})

    def load_from_file(self, path):
        mats = sio.loadmat(path)
        self.W = torch.from_numpy(mats['W']).to(self.device)
        self.Y = torch.from_numpy(mats['Y']).to(self.device)
        rho = mats['rho']
        while isinstance(rho, np.ndarray):
            rho = rho[0]
        self.rho = rho
        self.batch_size = self.W.shape[0]
        self.output_dim = self.W.shape[1]
        self.input_dim = self.W.shape[2]
        self.Y = torch.reshape(self.Y, (self.batch_size, self.output_dim, 1))

    def save_sol(self, sol, path):
        sio.savemat(path, {'fstar':sol})

    def load_sol(self, path):
        fstar = sio.loadmat(path)['fstar']
        self.fstar = torch.from_numpy(fstar).to(self.device)
