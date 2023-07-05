## Modified from lstm2. new models: p,a,b,b1,b2

import torch
import numpy               as np
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F

from collections import defaultdict
from optimizees.base import BaseOptimizee

NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'softplus': nn.Softplus(),
}

class CoordMathLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layers,
                 p_use=True, p_scale=1.0, p_scale_learned=True, p_norm='eye',
                 b_use=True, b_scale=1.0, b_scale_learned=True, b_norm='eye',
                 a_use=True, a_scale=1.0, a_scale_learned=True, a_norm='eye',
                 b1_use=True, b1_scale=1.0, b1_scale_learned=True, b1_norm='eye',
                 b2_use=True, b2_scale=1.0, b2_scale_learned=True, b2_norm='eye',
                 **kwargs):
        """
        Coordinate-wise non-smooth version of our proposed model.
		Please check (18) and (19) in the following paper:
		Liu et al. (2023) "Towards Constituting Mathematical Structures for Learning to Optimize."
        """
        super().__init__()

        self.input_size  = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        use_bias = True

        self.hist = defaultdict(list)

        self.layers = layers  # Number of layers for LSTM

        self.lstm = nn.LSTM(input_size, hidden_size, layers, bias=use_bias)
        # one more hidden laer before the output layer.
        # borrowed from NA-ALISTA: https://github.com/feeds/na-alista
        self.linear = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        # pre-conditioner
        self.linear_p = nn.Linear(hidden_size, output_size, bias=use_bias)
        # bias
        self.linear_b = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.linear_b1 = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.linear_b2 = nn.Linear(hidden_size, output_size, bias=use_bias)
        # momentum
        self.linear_a = nn.Linear(hidden_size, output_size, bias=use_bias)

        self.state = None
        self.step_size = kwargs.get('step_size', None)

        self.p_use = p_use
        if p_scale_learned:
            self.p_scale = nn.Parameter(torch.tensor(1.) * p_scale)
        else:
            self.p_scale = p_scale
        self.p_norm = NORM_FUNC[p_norm]

        self.b_use = b_use
        if b_scale_learned:
            self.b_scale = nn.Parameter(torch.tensor(1.) * b_scale)
        else:
            self.b_scale = b_scale
        self.b_norm = NORM_FUNC[b_norm]

        self.b1_use = b1_use
        if b1_scale_learned:
            self.b1_scale = nn.Parameter(torch.tensor(1.) * b1_scale)
        else:
            self.b1_scale = b1_scale
        self.b1_norm = NORM_FUNC[b1_norm]

        self.b2_use = b2_use
        if b2_scale_learned:
            self.b2_scale = nn.Parameter(torch.tensor(1.) * b2_scale)
        else:
            self.b2_scale = b2_scale
        self.b2_norm = NORM_FUNC[b2_norm]

        self.a_use = a_use
        if a_scale_learned:
            self.a_scale = nn.Parameter(torch.tensor(1.) * a_scale)
        else:
            self.a_scale = a_scale
        self.a_norm = NORM_FUNC[a_norm]

    @property
    def device(self):
        return self.linear_p.weight.device

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        batch_size = optimizees.X.numel()
        self.state = (
            # hidden_state
            torch.randn(
                self.layers, batch_size, self.hidden_size
            ).to(self.device),
            # cell_state
            torch.randn(
                self.layers, batch_size, self.hidden_size
            ).to(self.device),
        )
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())

    def detach_state(self):
        if self.state is not None:
            self.state = (self.state[0].detach(), self.state[1].detach())

    # def clear_hist(self):
    #     for l in self.hist.values():
    #         l.clear()

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model.
        """
        return 'CoordMathLSTM'

    def forward(
        self,
        optimizees: BaseOptimizee,
        grad_method: str,
        reset_state: bool = False,
        detach_grad: bool = True,
    ):
        """docstrings
        TBA
        """
        batch_size = optimizees.batch_size

        if self.state is None or reset_state:
            self.reset_state(optimizees)

        lstm_input = optimizees.get_grad(
            grad_method=grad_method,
            compute_grad=self.training,
            retain_graph=self.training,
        )
        lstm_input2 = optimizees.X
        if detach_grad:
            lstm_input = lstm_input.detach()
            lstm_input2 = lstm_input2.detach()

        # Here the `grad` is of dimension (batch_size, input_size, 1). Need
        # to reshape it into (1, batch_size, input_size) to be used by LSTM.
        # lstm_input = lstm_input.squeeze().unsqueeze(0)
        lstm_input = lstm_input.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_input2 = lstm_input2.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_in = torch.cat((lstm_input,lstm_input2), dim = 2)

        # Core update by LSTM.
        output, self.state = self.lstm(lstm_in, self.state)
        output = F.relu(self.linear(output))
        P = self.linear_p(output).reshape_as(optimizees.X)
        B = self.linear_b(output).reshape_as(optimizees.X)
        A = self.linear_a(output).reshape_as(optimizees.X)
        B1 = self.linear_b1(output).reshape_as(optimizees.X)
        B2 = self.linear_b2(output).reshape_as(optimizees.X)

        P = self.p_norm(P) * self.p_scale if self.p_use else 1.0
        B = self.b_norm(B) * self.b_scale if self.b_use else 1.0
        A = self.a_norm(A) * self.a_scale if self.a_use else 0.0
        B1 = self.b_norm(B1) * self.b1_scale if self.b1_use else 0.0
        B2 = self.b_norm(B2) * self.b2_scale if self.b2_use else 0.0

        # Calculate the update and reshape it back to the shape of the iterate
        smooth_grad = optimizees.get_grad(
            grad_method='smooth_grad',
            compute_grad=self.training,
            retain_graph=False
        )
        updateX = - P * self.step_size * smooth_grad
        smooth_grad2 = optimizees.get_grad(
            grad_method='smooth_grad',
            inputs = {'X':optimizees.get_var('Z')},
            compute_grad=self.training,
            retain_graph=False
        )
        updateZ = - P * self.step_size * smooth_grad2

        # Apply the update to the iterate
        prox_in = B * (optimizees.X + updateX) + (1 - B) * (optimizees.get_var('Z') + updateZ) + B1
        prox_out = optimizees.prox({'P':P * self.step_size, 'X':prox_in}, compute_grad=self.training)
        prox_diff = prox_out - optimizees.get_var('Z')
        optimizees.X = prox_out + A * prox_diff + B2

        # Clean up after the current iteration
        # optimizees.Z = prox_out
        optimizees.set_var('Z', prox_out)
        # optimizees.hist['prox_in'].append(prox_in)
        # optimizees.hist['prox_out'].append(prox_out)
        # optimizees.hist['rnn_out'].append(output)
        # optimizees.hist['rnn_in'].append(lstm_input)
        # optimizees.hist['P'] += [P * self.step_size] if self.p_use else []
        # optimizees.hist['B'] += [B] if self.b_use else []
        # optimizees.hist['A'] += [A] if self.a_use else []
        # optimizees.hist['B1'] += [B1] if self.b1_use else []
        # optimizees.hist['B2'] += [B2] if self.b2_use else []

        return optimizees


def test():
    return True


if __name__ == "__main__":
    test()
