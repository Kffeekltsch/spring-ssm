import torch
import torch.nn.functional as F
import numpy as np
from typing import overload, Callable, Iterable, List, TypeVar, Any, Literal, Union, Sequence, Tuple, Optional
from .jax_compat import associative_scan
from .init import *
import random
import math
#from custom_layers import GatedAF
from copy import deepcopy


@torch.jit.script
def binary_operator(q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    # return A_j * A_i, A_j * b_i + b_j
    return A_j * A_i, torch.addcmul(b_j, A_j, b_i)

@torch.jit.script
def binary_operator_2(q_i: torch.Tensor, q_j: torch.Tensor):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    # q_i[...,0], q_i[...,1] = q_i
    # q_j[...,0], q_j[...,1] = q_j
    # return q_j[...,0] * q_i[...,0], q_j[...,0] * q_i[...,1] + q_j[...,1]
    return torch.stack([q_j[...,0] * q_i[...,0],  q_j[...,0] * q_i[...,1] + q_j[...,1]],dim=-1)

def store_results(fun):
    precomputed = {}
    def wrapper(*args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in precomputed:
            precomputed[key] = fun(*args, **kwargs)
        return deepcopy(precomputed[key])
    return wrapper

@store_results
def get_indices(length, step, device):
    base = np.arange(int(2**(step-1))-1, length, int(2**step)).reshape(-1, 1)
    a_indices = base + np.zeros((2**(step-1),1)).reshape(1, -1)
    b_indices = base + np.arange(2**(step-1)).reshape(1, -1)+1
    return torch.tensor(a_indices.reshape(-1),device=device).long(), torch.tensor(b_indices.reshape(-1),device=device).long()

# @torch.jit.script
def associative_scan_2(operator, x):
    print('starting', x.shape)
    #assume shape = (bs, length, whatever, ...)
    length = x.shape[1]

    pow2 = np.log2(length)
    # print(pow2)
    perfect_power_of_2 = pow2 % 1 == 0
    upcast = pow2 % 1 > 0.5

    if perfect_power_of_2:
        pow2 = int(pow2)
        pass
    elif upcast : # upcast to next power of 2
        zeros = torch.zeros((x.shape[0],2**(np.ceil(pow2).astype(int))-length,*x.shape[2:]), device=x.device, dtype=x.dtype)
        x = torch.cat([x, zeros], dim=1)
        pow2 = int(np.ceil(pow2))
    else:   # downcast and handle the rest later
        rest = x[:,2**np.floor(np.log2(length)).astype(int)-1:,...]
        x = x[:,:2**np.floor(np.log2(length)).astype(int),...]
        pow2 = int(np.floor(pow2))

    length_clipped = x.shape[1]

    for step in range(1, pow2+1):
        read_a, read_b = get_indices(length_clipped, step, x.device)
        # print(read_a.shape)
        # print(read_a, read_b, write)
        x[:,read_b,...] = operator(x[:,read_a,...], x[:,read_b,...])


    if perfect_power_of_2:
        out = x
    elif upcast:
        out = x[:,:length,...]
    else:
        rest[:,0,...] = x[:,-1,...]
        out = torch.cat([x,associative_scan_2(operator, rest)[:,1:,...]], dim=1)

    return out

# def apply_ssm(Lambda_bars: torch.Tensor, B, B_bias, C, C_bias, input_sequence, complex_output):
#     cinput_sequence = input_sequence.type(
#         Lambda_bars.dtype)  # Cast to correct complex type
#     # Static timesteps
#     Bu_elements = torch.vmap(lambda u: B @ u)(cinput_sequence) + B_bias

#     # print('B_bias', B_bias.shape)
#     # print('C_bias', C_bias.shape)

#     if Lambda_bars.ndim == 1:  # Repeat for associative_scan
#         Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

#     _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))
#     if complex_output:
#         out = torch.vmap(lambda x: (C @ x))(xs) + C_bias
#     else:
#         out = (torch.vmap(lambda x: (C @ x))(xs) + C_bias).real
#     return out

def apply_ssm(Lambda_bars: torch.Tensor, B, B_bias, C, C_bias, input_sequence, complex_output):
    cinput_sequence = input_sequence.type(
        Lambda_bars.dtype)  # Cast to correct complex type
    # Static 

    Bu_elements = cinput_sequence@B.T + B_bias.view(1,1,-1)
    # Bu_elements = torch.vmap(lambda u: B @ u)(cinput_sequence) + B_bias

    #print('B_bias', B_bias.shape)
    #print('C_bias', C_bias.shape)
    #print('Bu_elements', Bu_elements.shape)
    #print('cinput_sequence', cinput_sequence.shape)

    if Lambda_bars.ndim == 1:  # Repeat for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[1], 1)
    

    # _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))
    _, xs = torch.vmap(lambda Bu : associative_scan(binary_operator, (Lambda_bars, Bu)))(Bu_elements)

    if complex_output:
        # out = torch.vmap(lambda x: (C @ x))(xs) + C_bias
        out = xs@C.T + C_bias.view(1,1,-1)
    else:
        # out = (torch.vmap(lambda x: (C @ x))(xs) + C_bias).real
        out = (xs@C.T + C_bias.view(1,1,-1)).real
    return out


def discretize_zoh(Lambda, B, B_bias, Delta, bias):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B      (complex64): input matrix + bias                (P, H + 1)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H + 1)
    """
    if bias:
        B_concat = torch.cat((B, B_bias.unsqueeze(1)), dim=-1)
    else:
        B_concat = B
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = ((Lambda_bar - 1)/Lambda)[..., None] * B_concat
    return Lambda_bar, B_bar


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt

class SSM(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 dt_min: float,
                 dt_max: float,
                 step_scale: float = 1.0,
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 complex_output=False,
                 B_C_init='orthogonal',
                 ensure_stability='abs',
                 what2train='eigenvalues_delta'):  # ['phase_norm', 'real_eigenvalues', 'eigenvalues_delta']
        """The Modified S5 SSM
        Args:
            d_in        (int32):     Number of features of input
            d_state     (int32):     state size
            d_out       (int32):     Number of output features
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_scale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()
        self.symmetric = False

        # lambdaInit  (float32): Initial diagonal state matrix       (P,2)
        self.Lambda = torch.nn.Parameter(make_linear_eigenvalues(d_state, symmetric=self.symmetric))
        self.log_step = torch.nn.Parameter(init_log_steps(d_state, dt_min, dt_max))
        self.discretize = discretize_zoh

        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.step_scale = step_scale
        self.ensure_stability = ensure_stability
        self.what2train = what2train

        if self.input_bias:
            if bias_init == 'zero':
                self.B_bias = torch.nn.Parameter(
                    torch.zeros(d_state, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.B_bias = torch.nn.Parameter(
                    torch.rand(d_state, 2, dtype=torch.float))
        else:
            self.B_bias = torch.nn.Parameter(torch.zeros(
                d_state, 2, dtype=torch.float), requires_grad=False)

        if self.output_bias:
            if bias_init == 'zero':
                self.C_bias = torch.nn.Parameter(
                    torch.zeros(d_out, 2, dtype=torch.float))
            elif bias_init == 'uniform':
                self.C_bias = torch.nn.Parameter(
                    torch.rand(d_out, 2, dtype=torch.float))
        else:
            self.C_bias = torch.nn.Parameter(torch.zeros(
                d_out, 2, dtype=torch.float), requires_grad=False)
        if B_C_init == 'S5':
            lamb, B, C = S5_init(d_in, d_out, d_state)
            self.Lambda.data = lamb
            self.B = torch.nn.Parameter(B,requires_grad=True)
            self.C = torch.nn.Parameter(2*C,requires_grad=True)
            self.B_bias = torch.nn.Parameter(self.B_bias.data[:self.Lambda.shape[0],...],requires_grad=True)
            self.log_step = torch.nn.Parameter(init_log_steps(self.Lambda.shape[0], dt_min, dt_max))

            print('S5 init')
            print('A', self.Lambda.shape)
            print('B', self.B.shape)
            print('C', self.C.shape)
            print('B_bias', self.B_bias.shape)
            print('C_bias', self.C_bias.shape)


        elif B_C_init == 'orthogonal':
            gain = np.sqrt(4/12)
            B_r = torch.empty(d_state, d_in)
            B_i = torch.empty(d_state, d_in)
            if d_in == 1:
                B_r = torch.nn.init.normal_(B_r, std=gain)
                B_i = torch.nn.init.normal_(B_i, std=gain)
            else:
                B_r = torch.nn.init.orthogonal_(B_r.T, gain=gain).T
                B_i = torch.nn.init.orthogonal_(B_i.T, gain=gain).T

            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))

            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.orthogonal_(C_r.T, gain=gain).T
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.orthogonal_(C_i.T, gain=gain).T
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_uniform_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_uniform_(B_i,  nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_uniform_(C_r,  nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_uniform_(C_i,  nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'kaiming_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.kaiming_normal_(B_r, nonlinearity='relu')
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.kaiming_normal_(B_i,  nonlinearity='relu')
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.kaiming_normal_(C_r,  nonlinearity='relu')
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.kaiming_normal_(C_i,  nonlinearity='relu')
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_uniform':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_uniform_(
                B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_uniform_(
                B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_uniform_(
                C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_uniform_(
                C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        elif B_C_init == 'xavier_normal':
            B_r = torch.empty(d_state, d_in)
            B_r = torch.nn.init.xavier_normal_(
                B_r, gain=torch.nn.init.calculate_gain('relu'))
            B_i = torch.empty(d_state, d_in)
            B_i = torch.nn.init.xavier_normal_(
                B_i, gain=torch.nn.init.calculate_gain('relu'))
            self.B = torch.nn.Parameter(torch.stack((B_r, B_i), dim=-1))
            C_r = torch.empty(d_out, d_state)
            C_r = torch.nn.init.xavier_normal_(
                C_r, gain=torch.nn.init.calculate_gain('relu'))
            C_i = torch.empty(d_out, d_state)
            C_i = torch.nn.init.xavier_normal_(
                C_i, gain=torch.nn.init.calculate_gain('relu'))
            self.C = torch.nn.Parameter(torch.stack((C_r, C_i), dim=-1))

        # print('A', self.Lambda.shape)
        # print('B', self.B.shape)
        # print('C', self.C.shape)
        # print('B_bias', self.B_bias.shape)
        # print('C_bias', self.C_bias.shape)



    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return torch.zeros((*batch_shape, self.C.shape[-2]))

    def forward_rnn(self, signal, prev_state):
        Lambda_c = as_complex(self.Lambda)
        if self.ensure_stability == 'relu':
            Lambda_c = torch.complex(-F.relu(-Lambda_c.real), Lambda_c.imag)
            # Lambda_c.real = -F.relu(-Lambda_c.real) # Ensure stability
        elif self.ensure_stability == 'abs':
            Lambda_c = torch.complex(-torch.abs(Lambda_c.real), Lambda_c.imag)
        else:
            # raise not implemented error
            raise NotImplementedError(
                'Only relu and abs stability are implemented')

        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        cinput_sequence = signal.type(C_c.dtype)

        step = self.step_scale * torch.exp(self.log_step)
        # print('step', step)
        Lambda_bar, B_bars = self.discretize(
            Lambda_c, B_c, B_bias_c, step, self.input_bias)

        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bar[:, 0])

        # print('Lambda_bar', Lambda_bar.shape)
        # print('input', cinput_sequence)
        # print('B_bar_forward_rnn', B_bar)
        # print('B_bias_bar_forward_rnn', B_bias_bar)
        # print('C_c', C_c)

        Bu = B_bar @ cinput_sequence + B_bias_bar
        x = Lambda_bar * prev_state + Bu
        y = C_c @ x + C_bias_c

        if self.complex_output:
            y_out = y
        else:
            y_out = y.real
        return y_out, x

    def forward(self, signal):
        with torch.no_grad():
            if self.ensure_stability == 'relu':
                self.Lambda.data[:, 0] = -F.relu(-self.Lambda.data[:, 0])
                # Lambda_c.real = -F.relu(-Lambda_c.real) # Ensure stability
            elif self.ensure_stability == 'abs':
                self.Lambda.data[:, 0] = -torch.abs(self.Lambda.data[:, 0])
                # Lambda = torch.complex(-torch.abs(Lambda.real), Lambda.imag)

            if not self.symmetric:
                self.Lambda.data[:, 1] = torch.abs(self.Lambda.data[:, 1])

        Lambda = as_complex(self.Lambda)

        step = self.step_scale * torch.exp(self.log_step)
        # print('Lambda', Lambda.shape)

        # self.B.data[...,1] = 0  # set imaginary part to zero

        B_c = as_complex(self.B)
        B_bias_c = as_complex(self.B_bias)
        C_c = as_complex(self.C)
        C_bias_c = as_complex(self.C_bias)

        Lambda_bars, B_bars = self.discretize(
            Lambda, B_c, B_bias_c, step, self.input_bias)
        if self.input_bias:
            B_bar = B_bars[:, 0:-1]
            B_bias_bar = B_bars[:, -1]
        else:
            B_bar = B_bars
            B_bias_bar = torch.zeros_like(B_bars[:, 0])
        # forward = apply_ssm
        return apply_ssm(Lambda_bars, B_bar, B_bias_bar, C_c, C_bias_c, signal, self.complex_output)


class MIMOSSM(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 step_scale: float = 1.0,
                 dt_min: float = 0.001, #0.001
                 dt_max: float = 0.10, #0.1
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 complex_output=True,
                 B_C_init='orthogonal',
                 stability='abs'):
        super().__init__()
        self.d_in = d_in
        self.d_state = d_state
        self.d_out = d_out
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.step_scale = step_scale
        self.previous_step_scale = step_scale
        self.complex_output = complex_output

        self.seq = SSM(
            d_in,
            d_state,
            d_out,
            dt_min,
            dt_max,
            step_scale,
            input_bias=input_bias,
            bias_init=bias_init,
            output_bias=output_bias,
            complex_output=complex_output,
            B_C_init=B_C_init,
            ensure_stability=stability
        )

    def initial_state(self, batch_size: Optional[int] = None):
        return self.seq.initial_state(batch_size)

    def forward(self, signal):
        # return torch.vmap(lambda s: self.seq(s))(signal)
        return self.seq(signal)

    def set_step_scale(self, step_scale):
        self.step_scale = step_scale
        self.seq.step_scale = step_scale


class ChannelDropout(torch.nn.Dropout1d):
    def __init__(self, p = 0.5, inplace = False):
        super().__init__(p, inplace)
    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        return super().forward(x.permute(0,2,1)).permute(0,2,1)


class SequenceLayer(torch.nn.Module):
    def __init__(self,
                 d_in: int,
                 d_state: int,
                 d_out: int,
                 step_scale: float = 1.0, #1.0
                 input_bias=False,
                 bias_init='zero',
                 output_bias=False,
                 norm=True, #switched
                 norm_type='ln',
                 complex_input=False,
                 complex_output=False,
                 B_C_init='orthogonal',#'orthogonal'
                 stability='abs',
                 trainable_SkipLayer=False,
                 dropout=0.0,
                 act='RELu',
                 dt_min=0.001,            # Use potentially different dt ranges
                 dt_max=0.1,
                 kernel_size = 4,
                 pre_conv = True
                 ):
        super().__init__()
        
        self.s5 = MIMOSSM(d_in, d_state, d_out, step_scale=step_scale,dt_min=dt_min,dt_max=dt_max,
                          input_bias=input_bias, bias_init=bias_init,
                          output_bias=output_bias,
                          complex_output=complex_output, B_C_init=B_C_init, stability=stability)
        self.d_in=d_in
        self.d_state=d_state
        self.d_out=d_out
        self.step_scale=step_scale
        self.input_bias=input_bias
        self.bias_init=bias_init
        self.output_bias=output_bias
        self.norm=norm
        self.norm_type=norm_type
        self.complex_input=complex_input #false
        self.complex_output=complex_output #true #norm bilden
        self.B_C_init=B_C_init
        self.stability=stability
        self.trainable_SkipLayer=trainable_SkipLayer
        self.dropout=dropout
        self.act=act
        self.pre_conv = pre_conv
        
        if self.pre_conv:
            
            self.local_conv = torch.nn.Conv1d(d_in, d_in, kernel_size, padding=kernel_size-1, groups=d_in) # Causal, group conv
        self.trainable_SkipLayer = trainable_SkipLayer
        if self.trainable_SkipLayer:
            self.skipLayer = torch.nn.Linear(d_in, d_out, bias=False)
            #self.skipLayer_real = torch.nn.Linear(d_in, d_out, bias=False)
            #self.skipLayer_imag = torch.nn.Linear(d_in, d_out, bias=False)
            #self.skipLayer_real.weight.data = np.sqrt(4/12) * self.skipLayer_real.weight.data / torch.norm(self.skipLayer_real.weight.data, dim=1, keepdim=True)
            #self.skipLayer_imag.weight.data = np.sqrt(4/12) * self.skipLayer_imag.weight.data / torch.norm(self.skipLayer_imag.weight.data, dim=1, keepdim=True)
            self.skipLayer.weight.data = np.sqrt(4/12)*self.skipLayer.weight.data/torch.norm(self.skipLayer.weight.data, dim=1, keepdim=True)
            ##### self.skipLayer.weight.data = torch.nn.init.orthogonal_(self.skipLayer.weight.data, gain=np.sqrt(4/12)) 

        
        self.dropout = ChannelDropout(p=self.dropout, inplace=False) if dropout > 0 else torch.nn.Identity()


        if norm and (norm_type == 'ln'):
            if complex_input:
                class LayerNormComplex(torch.nn.Module):
                    # init function
                    def __init__(self, d_in):
                        super().__init__()
                        self.ln_real = torch.nn.LayerNorm(d_in)
                        self.ln_imag = torch.nn.LayerNorm(d_in)

                    def forward(self, x):
                        return self.ln_real.forward(x.real) + 1j*self.ln_imag.forward(x.imag)

                self.attn_norm = LayerNormComplex(d_in)
            else:
                self.attn_norm = torch.nn.LayerNorm(d_in)

        elif norm and (norm_type == 'bn'):
            # batch norm 1d needs NxCxL input shape
            if complex_input:
                class BatchNorm1dT(torch.nn.Module):
                    # init function
                    def __init__(self, d_in):
                        super().__init__()
                        self.bn_real = torch.nn.BatchNorm1d(d_in)
                        self.bn_imag = torch.nn.BatchNorm1d(d_in)

                    def forward(self, x):
                        return self.bn_real.forward(x.real.permute(0, 2, 1)).permute(0, 2, 1) + 1j*self.bn_imag.forward(x.imag.permute(0, 2, 1)).permute(0, 2, 1)

            else:
                class BatchNorm1dT(torch.nn.BatchNorm1d):
                    def forward(self, x):
                        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)

            self.attn_norm = BatchNorm1dT(d_in)

        else:
            self.attn_norm = torch.nn.Identity()

        if act == 'RELu':
            self.act = torch.nn.ReLU() #norm bilden bei complex
        elif act == 'LeakyRELu':
            self.act = torch.nn.LeakyReLU()
        elif act == 'Identity':
            self.act = torch.nn.Identity()
        elif act == 'Hardtanh':
            self.act = torch.nn.Hardtanh()
        elif act == 'Sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'prelu' :
            self.act = torch.nn.PReLU()
        elif act == 'Softplus':
            self.act = torch.nn.Softplus()
        elif act == 'Gated':
            self.act = GatedAF()
        elif act == 'SiLU':
            self.act = torch.nn.SiLU()
        elif act == 'gelu':
            self.act = torch.nn.GELU()
        elif act == 'Softsign':
            self.act = torch.nn.Softsign()
        elif act == 'identity':
            self.act =torch.nn.Identity()
        else:
            print(f'act: {act}')
            raise NotImplementedError('Activation function not implemented')

    def forward(self, x):

        step_scale = self.s5.step_scale
        previous_step_scale = self.s5.previous_step_scale

        if step_scale != previous_step_scale:
            x = x[:, ::int(step_scale/previous_step_scale), :]

        if self.trainable_SkipLayer:
            res = self.skipLayer(x.clone())
            
            #res = self.skipLayer_real(x.real.clone())
            #res = self.skipLayer_imag(x.imag.clone())
        else:
            res = x.clone()
        if self.pre_conv:
            local_fx = self.local_conv(x.permute(0, 2, 1)).permute(0, 2, 1)[:, :x.shape[1], :] # Apply short conv
            fx = self.attn_norm(local_fx) #x input #localfx
        else:
            fx = self.attn_norm(x)
        out = self.s5(fx)

        if self.s5.complex_output:
            x = (self.act(out.real) + 1j*self.act(out.imag)) + res
        else:
            x = self.dropout(self.act(out.real)) + res
            #x = F.leaky_relu(out).real + res #was a comment
            #x = x/np.sqrt(2)
        return x

    def set_step_scale(self, step_scale, previous_step_scale=None):
        self.step_scale = step_scale
        self.previous_step_scale = previous_step_scale

        if previous_step_scale is None:
            self.s5.previous_step_scale = step_scale
            self.s5.set_step_scale(step_scale)
        else:
            self.s5.previous_step_scale = previous_step_scale
            self.s5.set_step_scale(step_scale)

    def get_number_of_parameters(self):
        # analytical calculation of the number of parameters
        # B/ in Bias Matrix element C
        B = self.d_in*self.d_state*2
        B_bias = self.d_state*2

        # A matrix element C
        A = self.d_state*2

        # C/ out Bias matrix element C
        C = self.d_state*self.d_out*2
        C_bias = self.d_out*2

        # Skip Layer element R
        skip = 0
        if self.trainable_SkipLayer:
            skip = self.d_in*self.d_out
        
        return A + B + B_bias + C + C_bias + skip





    # function the get the number of MUltipliy-Add operations of the model
    def get_number_of_MACs(self):
        # analytical calculation of the number of parameters
        # B Matrix element C @ u element R , bias can be ignored
        B = self.d_in*self.d_state*2
        # A matrix element C @ x element C
        A = self.d_state*4
        # C Matrix element C @ x element C, bias can be ignored, but output is elemet R
        C = self.d_state*self.d_out*2

        # Skip Layer element R
        skip = self.d_out
        if self.trainable_SkipLayer:
            skip = self.d_in*self.d_out

        # now combine based on step scale
        update_macs = A+B
        output_macs = C+skip

        return (update_macs+output_macs)/self.s5.step_scale



class RandomZeroOrderHold(torch.nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if self.training:
            random_val = random.uniform(0, 1)
            random_val /= self.prob
            if random_val < 1e-3:   # avoid division by zero, also sets upperbound for downscale
                return x
            downscale = 2**(math.floor(-math.log2(random_val)))
            if downscale < 1:
                return x
            if downscale > 128:
                downscale = 128
            for offset in range(1, int(downscale)):
                x[:, offset::int(downscale), :] = x[:, ::int(downscale), :]
            return x
        else:
            return x

    def set_step_scale(self, step_scale, previous_step_scale=None):
        pass

    def get_number_of_MACs(self):
        return 0

    def get_number_of_parameters(self):
        return 0


class EWMAReduction(torch.nn.Module):
    def __init__(self, third_point=0.1, samples=16000):
        super().__init__()
        alpha = third_point**(3/(2*samples))
        precomputed = torch.arange(samples-1,-1,-1).float()
        
        self.alpha = torch.nn.Parameter(torch.tensor(alpha),requires_grad=False)
        self.precomputed = torch.nn.Parameter(self.alpha**precomputed,requires_grad=False)

        self.step_scale = 1

    def set_step_scale(self, step_scale, previous_step_scale=None):
        self.step_scale = step_scale

    def forward(self, x):
        scale = 1-self.alpha**self.step_scale
        precomputed_slice = self.precomputed[::self.step_scale]

        x = x*scale*precomputed_slice.view(1,-1,1)
        x = torch.sum(x,dim=1)
        return x
        

class SC_Model_classifier(torch.nn.Module):
    def __init__(self, *, input_size=1, classes=35, hidden_sizes=[], output_sizes=[], ZeroOrderHoldRegularization=[],
                 input_bias=False, bias_init='zero',
                 output_bias=False, complex_output=False,
                 norm=False, norm_type='bn', B_C_init='orthogonal', stability='relu', trainable_SkipLayer=False,Reduction='mean', act='RELu',dropout=0.0, **kwargs):
        super(SC_Model_classifier, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.hidden_sizes = hidden_sizes
        self.output_sizes = output_sizes
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.complex_output = complex_output
        self.trainable_SkipLayer = trainable_SkipLayer
        self.reduction=Reduction
        self.act=act
        self.dropout = dropout

        if 'n_layer' in kwargs:
            raise ValueError(
                'n_layer is deprecated in SC_Model_classifier, use lists of hidden_sizes and output_sizes instead')

        if len(kwargs) > 0:
            raise ValueError('Unknown keyword arguments: ' + str(kwargs))

        if not isinstance(hidden_sizes, list):
            raise ValueError('hidden_sizes must be a list')

        if not isinstance(output_sizes, list):
            raise ValueError('output_sizes must be a list')

        if not isinstance(ZeroOrderHoldRegularization, list):
            if isinstance(ZeroOrderHoldRegularization, float):
                ZeroOrderHoldRegularization = [ZeroOrderHoldRegularization]*len(hidden_sizes)
            elif ZeroOrderHoldRegularization is None:
                ZeroOrderHoldRegularization = []
            else:
                raise ValueError(
                    'ZeroOrderHoldRegularization must be a list of floats, or a float or None or empty list')

        if len(hidden_sizes) != len(output_sizes):
            raise ValueError(
                'hidden_sizes and output_sizes must have the same length')

        if len(ZeroOrderHoldRegularization) != 0 and len(ZeroOrderHoldRegularization) != len(hidden_sizes):
            print(ZeroOrderHoldRegularization)
            print(hidden_sizes)
            raise ValueError('ZeroOrderHoldRegularization must have the same length as hidden_sizes')

        sequence_layers = []
        sequence_layers.append(SequenceLayer(d_in=input_size,
                                             d_state=hidden_sizes[0],
                                             d_out=output_sizes[0],
                                             input_bias=self.input_bias,
                                             bias_init=bias_init,
                                             output_bias=self.output_bias, norm=norm, norm_type=norm_type,
                                             complex_input=False,
                                             complex_output=self.complex_output,
                                             B_C_init=B_C_init, stability=stability,
                                             trainable_SkipLayer=self.trainable_SkipLayer,
                                             act=self.act,
                                             dropout=dropout,
                                             ))
        for i in range(1, len(hidden_sizes)):
            sequence_layers.append(SequenceLayer(d_in=output_sizes[i-1],
                                                 d_state=hidden_sizes[i],
                                                 d_out=output_sizes[i],
                                                 input_bias=self.input_bias,
                                                 bias_init=bias_init,
                                                 output_bias=self.output_bias, norm=norm, norm_type=norm_type,
                                                 complex_input=False,
                                                 complex_output=self.complex_output,
                                                 B_C_init=B_C_init, stability=stability,
                                                 trainable_SkipLayer=self.trainable_SkipLayer,
                                                 act=self.act,
                                                 dropout=dropout,
                                                 ))

        Reg_layers = []
        for i,_ in enumerate(ZeroOrderHoldRegularization):
            Reg_layers.append(RandomZeroOrderHold(
                ZeroOrderHoldRegularization[i]))

        self.seq = torch.nn.Sequential(*sequence_layers)

        if len(ZeroOrderHoldRegularization) > 0:
            self.reg = torch.nn.Sequential(*Reg_layers)
        else:
            self.reg = torch.nn.Sequential(*([torch.nn.Identity()]*len(self.seq)))

        self.decoder = torch.nn.Linear(output_sizes[-1], self.classes)
        # print('decoder norm', torch.norm(self.decoder.weight.data, dim=1, keepdim=True))
        # print(np.sqrt(4/12))
        self.decoder.weight.data = np.sqrt(4/12)*self.decoder.weight.data/torch.norm(self.decoder.weight.data, dim=1, keepdim=True)

        self.input_norm = torch.nn.BatchNorm1d(input_size, affine=True, momentum=1e-2)  # works


        if self.reduction == 'EWMA':
            self.time_Reduction = EWMAReduction()
        elif self.reduction == 'mean':
            pass
        else:
            raise ValueError('Unknown Reduction type: ' + str(self.reduction))

    def forward(self, inputs):
        # inputs = self.input_norm(inputs.transpose(1, 2)).transpose(1, 2)

        for i,_ in enumerate(self.seq):
            inputs = self.seq[i](inputs)
            inputs = self.reg[i](inputs)

        if self.reduction == 'EWMA':
            output_mean = self.time_Reduction(inputs)
        elif self.reduction == 'mean':
            output_mean = torch.mean(inputs, dim=1)
        else:
            raise ValueError('Unknown Reduction type: ' + str(self.reduction))

        logits = self.decoder(output_mean)
        return logits

    def set_step_scale(self, step_scales, previous_step_scales=None):
        # self.step_scales = step_scales
        if previous_step_scales is None:
            previous_step_scales = step_scales
        for previous_step_scale, step_scale, layer in zip(previous_step_scales, step_scales, self.seq):
            layer.set_step_scale(step_scale, previous_step_scale)
        
        if self.reduction == 'EWMA':
            self.time_Reduction.set_step_scale(step_scales[-1], previous_step_scales[-1])


    def get_number_of_parameters(self):
        params = 0
        for layer in self.seq:
            params += layer.get_number_of_parameters()

        params += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)

        params += 2 # Input BN

        return params

    def get_number_of_MACs(self):
        n_MACs = 0
        for layer in self.seq:
            n_MACs += layer.get_number_of_MACs()

        n_MACs += self.decoder.in_features/self.seq[-1].s5.step_scale       ## reduction layer, only additions
        n_MACs += self.decoder.in_features * self.decoder.out_features/16000 ## decoder layer (16000 samples)

        n_MACs += 1 # Input BN

        return n_MACs
