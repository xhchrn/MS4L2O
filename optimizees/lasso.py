import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from .base import BaseOptimizee


class LASSO(BaseOptimizee):

    def __init__(
        self,
        batch_size: int,
        W = None,
        Y = None,
        rho = 0.1,
        s = 5,
        device = 'cpu',
        **options
    ) -> None:
        """
        The LASSO optimization problem is formulated as

            minimize (1/2) * ||Y - W @ X||_2^2  + rho * ||X||_1
               X
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
            W = torch.randn(self.batch_size, self.output_dim, self.input_dim).to(self.device)
            self.W = W / torch.sum(W**2, dim=1, keepdim=True).sqrt()
        else:
            if isinstance(W, np.ndarray):
                W = torch.from_numpy(W).to(self.device)
            elif isinstance(W, torch.Tensor):
                W = W.to(self.device)
            else:
                raise ValueError(f'Invalid type {type(W)} for W')
            assert W.dim() == 2
            self.W = W.unsqueeze(0).repeat(self.batch_size, 1, 1)

        # Set output Y
        if Y is None:
            X_gt = torch.randn(batch_size, self.input_dim).to(self.device)
            non_zero_idx = torch.multinomial(
                torch.ones_like(X_gt), num_samples=self.s, replacement=False
            )
            self.X_gt = torch.zeros_like(X_gt).scatter(
                dim=1, index=non_zero_idx, src=X_gt
            ).unsqueeze(-1)
            self.Y = torch.bmm(self.W, self.X_gt)
        else:
            if isinstance(Y, np.ndarray):
                Y = torch.from_numpy(Y).to(self.device)
            elif isinstance(Y, torch.Tensor):
                Y = Y.to(self.device)
            else:
                raise ValueError(f'Invalid type {type(Y)} for Y')
            assert Y.dim() == 2
            self.Y = Y.unsqueeze(0).repeat(self.batch_size, 1, 1)
            self.X_gt = None

        # Initialize the first iterate X
        ## X here is Y in the paper, prox_out here is X in paper
        self.initialize()

        # Restore RNG state after generation
        if seed:
            rng_state = torch.set_rng_state(rng_state)

    def initialize(self):
        self.X = torch.zeros(self.batch_size, self.input_dim, 1).to(self.device)
        prox_out = torch.zeros(self.batch_size, self.input_dim, 1).to(self.device)
        self.set_var('Z', prox_out)

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
        lip = torch.linalg.norm(self.W, dim=(-2,-1), ord=2) ** 2
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
            residual = torch.bmm(self.W, X) - self.Y
            l2 = 0.5 * (residual**2.0).sum(dim=(1,2)).mean()
            l1 = self.rho * torch.abs(X).sum(dim=(1,2)).mean()
            return l1 + l2

    def objective_batch(self, inputs: dict = {}, compute_grad: bool = False):
        """Calculate the objective value.
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            residual = torch.bmm(self.W, X) - self.Y
            l2 = 0.5 * (residual**2.0).sum(dim=(1,2))
            l1 = self.rho * torch.abs(X).sum(dim=(1,2))
            return l1 + l2

    def objective_batch_shift(self, inputs: dict = {}, compute_grad: bool = False):
        if inputs is None:
            inputs = {}
        obj = self.objective_batch(inputs, compute_grad)
        fstar = self.fstar.reshape_as(obj)
        valid_ind = (fstar != 0)
        ret = obj - fstar 
        ret[valid_ind] /= fstar[valid_ind]
        return ret
        
    def objective_shift(self, inputs: dict = {}, compute_grad: bool = False):
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
        assert grad_func, f'Invalid grad method specified: {grad_method}'
        return grad_func(inputs, compute_grad, **kwargs)

    def bp_grad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """
        Calculate the gradient of the iterate w.r.t. the objective using back-
        propagation mechanism built in PyTorch.
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        X = X.detach()
        X.requires_grad_(True)
        X.retain_grad()
        inputs['X'] = X

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
            residual = torch.bmm(W, X) - Y
            return torch.bmm(W.permute(0,2,1), residual)

    def subgrad(self, inputs: dict = None, compute_grad: bool = False, **kwargs):
        """Calculate the subgrad
        """
        if inputs is None:
            inputs = {}
        X = inputs.get('X', self.X)
        Y = inputs.get('Y', self.Y)
        W = inputs.get('W', self.W)

        with torch.set_grad_enabled(compute_grad):
            residual = torch.bmm(W, X) - Y
            return (torch.bmm(W.permute(0,2,1), residual) +
                    self.rho * torch.sign(X))

    def prox(self, inputs: dict, compute_grad: bool = False, **kwargs):
        P = inputs['P']
        X = inputs['X']
        with torch.set_grad_enabled(compute_grad):
            mag = nn.functional.relu(torch.abs(X) - self.rho * P)
            return torch.sign(X) * mag
        
    def save_to_file(self, path):
        Wcpu = self.W.cpu().numpy()
        Ycpu = self.Y.cpu().numpy()
        sio.savemat(path, {'W':Wcpu, 'Y':Ycpu, 'rho':self.rho})

    def load_from_file(self, path):
        mats = sio.loadmat(path)
        self.W = torch.from_numpy(mats['W']).type(torch.float32).to(self.device)
        self.Y = torch.from_numpy(mats['Y']).type(torch.float32).to(self.device)
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

