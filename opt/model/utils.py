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
    
def setup_render_opts(opt, args):
    """
    Pass render arguments to the Adtree renderer options
    """
    # opt.step_size = args.step_size
    opt.sigma_thresh = args.sigma_thresh
    opt.stop_thresh = args.stop_thresh
    # opt.background_brightness = args.background_brightness
    # opt.backend = args.renderer_backend
    opt.random_sigma_std = args.random_sigma_std
    opt.random_sigma_std_background = args.random_sigma_std_background
    opt.last_sample_opaque = args.last_sample_opaque
    opt.near_clip = args.near_clip
    # opt.use_spheric_clip = args.use_spheric_clip