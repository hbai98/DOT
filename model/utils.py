from timm.models.layers import create_act_layer
import torch.nn as nn


class TreeConv(nn.Module):
    def __init__(self, in_channels, out_channels, degree, act='gelu', inplace=True):
        """
        Collect local features. 

        :param in_channels: int hidden dimension, include hidden_dim/postion/
        :param out_channels: int output dimension
        :param degree: int degree of the tree
        :param act: activation layer name

        :return: density or color(r, g, b)

        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=degree, bias=True)
        self.norm  = nn.LayerNorm(out_channels)
        self.act = create_act_layer(act)
    def forward(self, x):
        """
        x is expected to have shape (N, C, L) or (C, L), where L is the number of children
        
        : return: the encoded leaf nodes' feature (out_channels)
        """
        x = self.conv(x).squeeze(-1)
        x = self.norm(x)
        x = self.act(x)
        return x  