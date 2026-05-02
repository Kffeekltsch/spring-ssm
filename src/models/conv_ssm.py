import torch
import torch.nn as nn
from ssm.src.model.sequence_layer import SequenceLayer 
from src.models.conv import ShallowFusion 
from src.models.custom_layers import (
    Conv1dCausal,
    FiLM,
    GatedAF,
    TanhAF,
)


class CONV_SSM_L(nn.Module):
    """
    A sequential hybrid model: FIR-like convolutional conv followed by
    an IIR-like SSM stack for refinement.
    """
    def __init__(self, in_ch=1, out_ch=1, n_channels=8,
                 kernel_size=12, num_layers=15, 
                 d_state=32, num_ssm_layers=6, activation = "prelu"): 
        super().__init__()

        self.num_layers = num_layers
        self.kernel_size = kernel_size
        conv_layers = [nn.Conv1d(in_ch, n_channels, 1)]
        for i in range(num_layers):
            dilation = 2 ** i
            conv_layers.append(
                Conv1dCausal(n_channels, n_channels, kernel_size, dilation=dilation, stride=1, bias=False)
            )
        self.conv = nn.Sequential(*conv_layers)
        self.conv_activation = nn.Tanh()

        # stack of SSM layers
        ssm_stack = []
        for _ in range(num_ssm_layers):
            ssm_stack.append(
                SequenceLayer(
                    d_in=n_channels, 
                    d_out=n_channels, 
                    d_state=d_state, 
                    act=activation
                )
            )
        self.ssm_layers = nn.Sequential(*ssm_stack)
        self.output_head = ShallowFusion(n_channels, out_ch)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        conv_features = self.conv(x)
        ssm_input = conv_features.permute(0, 2, 1)
        ssm_features = self.ssm_layers(ssm_input)
        output = self.output_head(ssm_features)
        
        return output

    def calc_receptive_field(self):
        """Calculates the receptive field."""
        rf = 1
        for i in range(self.num_layers):
            dilation = 2 ** i
            rf += (self.kernel_size - 1) * dilation
        return rf

    def get_manual_macs(self, sequence_length):
        total_macs = 0
        for layer in self.ssm_layers:
            macs_per_step = layer.get_number_of_MACs()
            total_macs += macs_per_step * sequence_length


        return total_macs