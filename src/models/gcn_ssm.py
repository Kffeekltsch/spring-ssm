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
import torch.profiler
from ssm.src.model.sequence_layer import SequenceLayer 

from src.models.custom_layers import (
    Conv1dCausal,
    FiLM,
    GatedAF,
    TanhAF,
)


class GCNBlock(nn.Module):
    """Single block of a Gated Convolutional Network (GCN) with conditional modulation.

    Parameters:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel.
        dilation (int, optional): Dilation rate for dilated convolutions.
        stride (int, optional): Stride for the convolution.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        cond_dim: int = 0,
        tail_hidden_size = 4
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.cond_dim = cond_dim
        self.tail_hidden_size = tail_hidden_size
        
        self.conv = Conv1dCausal(
            in_channels=in_ch,
            out_channels=out_ch*2 , #*2 # adapt for the Gated Activation Function
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

        #self.film = FiLM(cond_dim=cond_dim, n_features=out_ch * 2)

        self.gated_activation = GatedAF()
        self.act = nn.PReLU()

        self.res = nn.Conv1d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=(1,), bias=False
        )
        
    def forward(self, x): #cond
        x_in = x
        x = self.conv(x)  
        x = self.gated_activation(x) 
        x_res = self.res(x_in)  
        x = x + x_res

        return x


class GCN_SSM(nn.Module):
    """Gated Convolutional Network (GCN) model, often used in sequence modeling tasks.

    Parameters:
        in_ch (int, optional): Number of input channels.
        out_ch (int, optional): Number of output channels.
        n_blocks (int, optional): Number of GCN blocks.
        n_channels (int, optional): Number of channels in the GCN blocks.
        dilation_growth (int, optional): Growth rate for dilation in the GCN blocks.
        kernel_size (int, optional): Size of the convolution kernel.
        cond_dim (int, optional): Dimensionality of the conditional input for FiLM.

    Returns:
        Tensor: The output of the GCN model.
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        n_blocks: int = 2,
        n_channels: int = 32,
        dilation_growth: int = 8,
        kernel_size: int = 3,
        d_state: int = 32,
        cond_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch  # input channels
        self.out_ch = out_ch  # output channels
        self.kernel_size = kernel_size
        self.cond_dim = cond_dim

        # Compute convolution channels and dilations
        self.channels = [n_channels] * n_blocks
        self.dilations = [dilation_growth**idx for idx in range(n_blocks)]
        print(f"Dilations: {self.dilations}")

        # Blocks number is given by the number of elements in the channels list
        self.n_blocks = len(self.channels)
        assert len(self.dilations) == self.n_blocks

        # Create a list of strides
        self.strides = [1] * self.n_blocks


        # Create a list of GCN blocks
        self.blocks = nn.ModuleList()
        block_out_ch = None
        

        for idx, (curr_out_ch, dil, stride) in enumerate(
            zip(self.channels, self.dilations, self.strides)
        ):
            if idx == 0:
                block_in_ch = in_ch
            else:
                block_in_ch = block_out_ch
            block_out_ch = curr_out_ch

            self.blocks.append(
                GCNBlock(
                    block_in_ch,
                    block_out_ch,
                    kernel_size,
                    dil,
                    stride,
                    cond_dim
                )
            )
            self.blocks.append(
                SequenceLayer(
                    d_in=block_out_ch,
                    d_out=block_out_ch,
                    d_state=d_state,
                    input_bias=False,
                    output_bias=False,
                    complex_input = False,
                    complex_output=False,
                    trainable_SkipLayer=True, #new
                    act="Hardtanh"
         
                )
            )


        # Output layer
        self.out_net = nn.Conv1d(
            self.channels[-1], out_ch, kernel_size=(1,), stride=(1,), bias=False
        )

        # Activation function
        self.af = TanhAF()

    def forward(self, x): #cond
        
        # x.shape = (batch_size, in_ch, samples)
        # cond.shape = (batch_size, cond_dim)
        x = x.permute(0,2,1)
        for block_idx, block in enumerate(self.blocks):
            if isinstance(block, GCNBlock):
                # GCNBlock expects [B, C, T] and outputs [B, C_out, T]
                x = block(x)
            elif isinstance(block, SequenceLayer):
                # SequenceLayer expects [B, T, C]
                x_for_ssm = x.permute(0, 2, 1) # Permute to [B, T, C]
                x_ssm_out = block(x_for_ssm)
                x = x_ssm_out.permute(0, 2, 1) # Permute back to [B, C, T] if next block is GCN
            else:
                x = block(x) # Fallback for other layer types
        x = self.out_net(x)  # Apply output layer
        x = self.af(x)  # Apply tanh activation function
        x = x.permute(0,2,1)
        return x

    def get_trainable_parameters(self):
        """Calculates the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_manual_macs(self, sequence_length):
        """
        Calculates the MACs for the SSM part of the model by calling the
        analytical MACs function of each SequenceLayer.
        """
        total_macs = 0
        

        for module in self.blocks:

            if "SequenceLayer" in module.__class__.__name__:
                macs_per_step = module.get_number_of_MACs()
                total_macs += macs_per_step * sequence_length

        return total_macs
    def calc_receptive_field(self) -> int:
        """Compute the receptive field in samples.

        Returns:
            int: The receptive field of the model.
        """
        assert all(_ == 1 for _ in self.strides) 
        assert self.dilations[0] == 1 
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf = rf + ((self.kernel_size - 1) * dil)
        return rf
