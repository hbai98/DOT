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
        self.conv = nn.Conv1d(in_channels, out_channels, degree, bias=True)
        self.act  = create_act_layer(act, inplace=inplace)

    def forward(self, x):
        """
        x is expected to have shape (B, N, C), where N is the number of children
        """
        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.act(x)
        return x  # (B, C, H//2, W//2)