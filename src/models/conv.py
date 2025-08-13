from ssm.model import SequenceLayer, MIMOSSM
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

# --- 1. Conv1d_permute with Dilation-Aware Padding ---

class Conv1d_permute(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 padding_mode='causal', dilation=1,  bias=False):
        super(Conv1d_permute, self).__init__()
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.dilation = dilation  # store dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             dilation=dilation,  bias=bias)
    
    def forward(self, x):
        # x: [batch, seq, channels] → [batch, channels, seq]
        x = x.permute(0, 2, 1)
        if self.padding_mode == 'causal':
            pad_left = self.dilation * (self.kernel_size - 1)
            x = F.pad(x, (pad_left, 0))
        elif self.padding_mode == 'symmetric':
            left_pad = (self.dilation * (self.kernel_size - 1)) // 2
            right_pad = self.dilation * (self.kernel_size - 1) - left_pad
            x = F.pad(x, (left_pad, right_pad))
        elif self.padding_mode == 'reflect':
            left_pad = (self.dilation * (self.kernel_size - 1)) // 2
            right_pad = self.dilation * (self.kernel_size - 1) - left_pad
            x = F.pad(x, (left_pad, right_pad), mode='reflect')
        x = self.conv(x)
        return x.permute(0, 2, 1)

# --- 2. DilatedConvStack (optionally using conv1d_permute) ---
class DilatedConvStack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, dilation_base=2, 
                 causal=True, use_conv1d_permute=True, activation='relu'):
        super(DilatedConvStack, self).__init__()
        self.use_conv1d_permute = use_conv1d_permute
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(num_layers-1):
            dilation = dilation_base ** i
            if causal:
                if use_conv1d_permute:
                    conv_layer = Conv1d_permute(in_channels, out_channels, kernel_size, 
                                                padding_mode='causal', dilation=dilation)
                    #layer = nn.Sequential(conv_layer, nn.BatchNorm1d(out_channels))
                else:
                    pad = dilation * (kernel_size - 1)
                    conv_layer = nn.Sequential(
                        nn.ConstantPad1d((pad, 0), 0),
                        nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
                    )
            else:
                pad = dilation * (kernel_size - 1) // 2
                if use_conv1d_permute:
                    conv_layer = Conv1d_permute(in_channels, out_channels, kernel_size, 
                                                padding_mode='symmetric', dilation=dilation)
                else:
                    conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=pad)
            self.layers.append(conv_layer)

    
    def forward(self, x):
        if not self.use_conv1d_permute:
            x = x.permute(0, 2, 1)
            #res = x
        for layer in self.layers:
            #residual = x
            x = layer(x)
        if not self.use_conv1d_permute:
            x = x.permute(0, 2, 1)
        return x

# --- Shallow Fusion Module for Channel Mixing ---
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


# --- 6. Tail Branch (using dilated convs and sequence layers) ---
class TailBranch(nn.Module):
    def __init__(self, input_size, tail_hidden_size, tail_num_layers, inter_size, activation, 
                 sample_rate, filter_duration=5.0, use_dilated_conv=True, conv_kernel=7, 
                 num_dilated_layers=16, padding_mode='causal', use_conv1d_permute_in_dilated=True):
        super(TailBranch, self).__init__()
        self.inter_size = inter_size
        self.input_size = input_size
        filter_length = int(filter_duration * sample_rate)
        
        if use_dilated_conv:
            self.conv_block = nn.Sequential(
                DilatedConvStack(inter_size, inter_size, conv_kernel, num_dilated_layers, 
                                 causal=(padding_mode=='causal'),
                                 use_conv1d_permute=use_conv1d_permute_in_dilated,
                                 activation=activation),
                nn.PReLU() if activation=='prelu' else nn.GELU()
            ) #now inter size instead of in_channels
        else:
            self.conv_block = nn.Sequential(
                Conv1d_permute(input_size, input_size, filter_length, padding_mode=padding_mode),
                #nn.PReLU() if activation=='prelu' else nn.GELU()
            )
        

        self.in_mod = nn.Linear(self.input_size, self.inter_size)
        
        #self.shallow_fusion = ShallowFusion(inter_size*6, input_size)
        self.shallow_fusion = ShallowFusion(inter_size, input_size)


    def forward(self, x):
        #res1 = x
        
        x = self.in_mod(x)
        x = self.conv_block(x)
        x1 = self.shallow_fusion(x)
        

        return x1


# --- 7. Overall Model: SpringReverbNet with Fusion Gating Conditioned on Cutoff ---
class CONV(nn.Module):
    def __init__(self, 
                 input_size, output_size,
                 er_filters, tail_inter_size,
                 tail_hidden_size, tail_num_layers,
                 activation, sample_rate, er_kernel_size, fusion_method,
                 use_dilated_conv=True, conv_kernel=1, num_dilated_layers=4,
                 padding_mode='causal',
                ):
        super(CONV, self).__init__()

        self.tail_branch = TailBranch(input_size, tail_hidden_size, tail_num_layers, tail_inter_size, activation, 
                                      sample_rate, filter_duration=5.0, use_dilated_conv=use_dilated_conv, 
                                      conv_kernel=conv_kernel, num_dilated_layers=num_dilated_layers, 
                                      padding_mode=padding_mode, use_conv1d_permute_in_dilated=True)

        self.macs, self.params = self._compute_macs_params()

        print(f"MACs: {self.macs}")
        print(f"Parameters: {self.params}")
    
    def _compute_macs_params(self):
        # Hybrid parameter counting: add custom layers using get_number_of_parameters if available,
        # then add remaining parameters using p.numel().
        counted = set()
        macs = 0
        params = 0
        for m in self.modules():
            if hasattr(m, "get_number_of_parameters"):
                try:
                    p_val = m.get_number_of_parameters()
                    params += p_val
                    for p in m.parameters():
                        counted.add(id(p))
                    if hasattr(m, "get_number_of_MACs"):
                        macs += m.get_number_of_MACs()
                except:
                    pass
        # Add remaining parameters not counted:
        for p in self.parameters():
            if id(p) not in counted:
                params += p.numel()
        return macs, params
        

    def forward(self, x, gain_factor = None):
        
        # x: [batch, seq, input_size]
        out = self.tail_branch(x)

        return  out


def log_model_report(config, model, modelname):
    """
    Generate a detailed log report including configuration, training progress,
    and the full model architecture.
    """
    # Build configuration report string
    config_report = ""
    for key, value in config.items():
        config_report += f"{key}: {value}\n"
    
    # Get a text version of the model architecture
    architecture_report = repr(model)
    
    report = (
        f"|{'-'*90}|\n"
        f"Processing model: {modelname}\n"
        f"Configuration:\n{config_report}"
        f"Macs: {model.macs}\n"
        f"Parameters: {model.params}\n"
        f"Model architecture:\n"
        f"{'='*90}\n"
        f"{architecture_report}\n"
        f"{'='*90}\n"
        f"|{'-'*90}|\n"
    )
    
    return report