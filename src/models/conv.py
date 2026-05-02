
import abc as _abc
import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory
from typing import Optional as Optional, Tuple as _Tuple
from tqdm import tqdm
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.custom_layers import (
    Conv1dCausal,
    FiLM,
    GatedAF,
    TanhAF,
)

class Conv1D(nn.Module):
    """
    Causal helper convolution permuting shapes internally.
    """
    def __init__(self, in_ch, out_ch, kernel_size, 
                 padding_mode='causal', dilation=1,  bias=False):
        
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.conv = Conv1dCausal(in_ch, out_ch, kernel_size,
                             dilation=dilation, stride =1, bias=bias)

    def forward(self,x):
        x = self.conv(x)
        return x
        
class ShallowFusion(nn.Module):
    def __init__(self, inter_size, out_size):
        super(ShallowFusion, self).__init__()
        self.norm = nn.LayerNorm(inter_size)
        self.hidden = nn.Linear(inter_size, inter_size)
        self.act2 = nn.GELU()
        self.act = nn.Tanh()
        self.out = nn.Linear(inter_size, out_size)
    
    def forward(self, x):
        # x: [B, T, inter_size]
        x = self.norm(x)
        x = self.hidden(x)
        x = self.act2(x)
        x = self.out(x)
        x = self.act(x)
        return x


class CONV(nn.Module):
    """
    A simple baseline model using a stack of dilated convolutions (FIR-like).
    This model is fully convolutional and can handle variable-length inputs.
    """
    def __init__(self, in_ch=1, out_ch=1,
                 n_channels=8, kernel_size=12, num_layers=15):
        super().__init__()

        self.input_proj = Conv1dCausal(in_ch, n_channels, kernel_size,
                             dilation=2, stride =1, bias=False)

        self.gatedaf = GatedAF()
        conv_layers = []
        for i in range(num_layers): 
            dilation = 2 ** i
            conv_layers.append(
                Conv1dCausal(n_channels, n_channels, kernel_size, dilation=dilation,stride=1, bias=True)
            )
        self.conv_stack = nn.Sequential(*conv_layers)
        self.activation = nn.Tanh()
        self.output_head = ShallowFusion(n_channels, out_ch)

    def forward(self, x):
        # Input: [B, T, C]
        x = x.permute(0, 2, 1) 
        x = self.input_proj(x)
        x = self.conv_stack(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1) 
        output = self.output_head(x)
        
        return output

    def calc_receptive_field(self):
        """Calculates the total receptive field of the model in samples."""
        rf = 1
        for i in range(self.num_layers):
            dilation = 2 ** i
            rf += (self.kernel_size - 1) * dilation
        return rf

  

