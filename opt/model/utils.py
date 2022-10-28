from timm.models.layers import create_act_layer
import torch.nn as nn
import torch
from typing import Optional
import numpy as np
from skimage.filters._gaussian import gaussian
from skimage.filters.thresholding import threshold_li, threshold_otsu, threshold_yen, threshold_minimum, threshold_triangle

def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

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
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=degree, bias=True)
        self.norm = nn.LayerNorm(out_channels)
        self.act = create_act_layer(act, inplace=inplace)

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

# borrow from https://github.com/Ragheb2464/preto-front/blob/master/2d.py


def pareto_2d(data):
    from operator import itemgetter
    sorted_data = sorted(data, key=itemgetter(0, 1), reverse=True)
    pareto_idx = list()
    pareto_idx.append(0)
    cutt_off = sorted_data[0][1]
    for i in range(1, len(sorted_data)):
        if sorted_data[i][1] > cutt_off:
            pareto_idx.append(i)
            cutt_off = sorted_data[i][1]
    return pareto_idx


def threshold(data, method, sigma=3):
    device = data.device
    data = gaussian(data.cpu().detach().numpy(), sigma=sigma)
    if method == 'li':
        return torch.tensor(threshold_li(data), device=device)
    elif method == 'otsu':
        return torch.tensor(threshold_otsu(data), device=device)
    elif method == 'yen':
        return torch.tensor(threshold_yen(data), device=device)
    elif method == 'minimum':
        return torch.tensor(threshold_minimum(data), device=device)
    elif method == 'triangle':
        return torch.tensor(threshold_triangle(data), device=device)
    else:
        assert False, f'the method {method} is not implemented.'


def posenc(
    x: torch.Tensor,
    cov_diag: Optional[torch.Tensor],
    min_deg: int,
    max_deg: int,
    include_identity: bool = True,
    enable_ipe: bool = True,
    cutoff: float = 1.0,
):
    """
    Positional encoding function. Adapted from jaxNeRF
    (https://github.com/google-research/google-research/tree/master/jaxnerf).
    With support for mip-NeFF IPE (by passing cov_diag != 0, keeping enable_ipe=True).
    And BARF-nerfies frequency attenuation (setting cutoff)

    Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1],
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    :param x: torch.Tensor (..., D), variables to be encoded. Note that x should be in [-pi, pi].
    :param cov_diag: torch.Tensor (..., D), diagonal cov for each variable to be encoded (IPE)
    :param min_deg: int, the minimum (inclusive) degree of the encoding.
    :param max_deg: int, the maximum (exclusive) degree of the encoding. if min_deg >= max_deg,
                         positional encoding is disabled.
    :param include_identity: bool, if true then concatenates the identity
    :param enable_ipe: bool, if true then uses cov_diag to compute IPE, if available.
                             Note cov_diag = 0 will give the same effect.
    :param cutoff: float, in [0, 1], a relative frequency cutoff as in BARF/nerfies. 1 = all frequencies,
                          0 = no frequencies

    :return: encoded torch.Tensor (..., D * (max_deg - min_deg) * 2 [+ D if include_identity]),
                     encoded variables.
    """
    if min_deg >= max_deg:
        return x
    scales = torch.tensor(
        [2 ** i for i in range(min_deg, max_deg)], device=x.device)
    half_enc_dim = x.shape[-1] * scales.shape[0]
    # (..., D * (max_deg - min_deg))
    shapeb = list(x.shape[:-1]) + [half_enc_dim]
    xb = torch.reshape((x[..., None, :] * scales[:, None]), shapeb)
    four_feat = torch.sin(
        torch.cat([xb, xb + 0.5 * np.pi], dim=-1)
    )  # (..., D * (max_deg - min_deg) * 2)
    if enable_ipe and cov_diag is not None:
        # Apply integrated positional encoding (IPE)
        xb_var = torch.reshape(
            (cov_diag[..., None, :] * scales[:, None] ** 2), shapeb)
        xb_var = torch.tile(xb_var, (2,))  # (..., D * (max_deg - min_deg) * 2)
        four_feat = four_feat * torch.exp(-0.5 * xb_var)
    # if cutoff < 1.0:
    #     # BARF/nerfies, could be made cleaner
    #     cutoff_mask = _cutoff_mask(
    #         scales, cutoff * (max_deg - min_deg)
    #     )  # (max_deg - min_deg,)
    #     four_feat = four_feat.view(shapeb[:-1] + [2, scales.shape[0], x.shape[-1]])
    #     four_feat = four_feat * cutoff_mask[..., None]
    #     four_feat = four_feat.view(shapeb[:-1] + [2 * scales.shape[0] * x.shape[-1]])
    if include_identity:
        four_feat = torch.cat([x] + [four_feat], dim=-1)
    return four_feat


def _SOFTPLUS_M1(x):
    return torch.log(torch.exp(x-1)+1)


def asoftplus(x):
    return x + torch.log(-torch.expm1(-x))
