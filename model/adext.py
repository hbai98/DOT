from turtle import forward
from torch import nn as nn
from svox import N3Tree, VolumeRenderer as VR, NDCConfig, Rays
import torch
import numpy as np
from .utils import TreeConv
from timm.models.layers import Mlp
from timm.models.layers import lecun_normal_, trunc_normal_
from einops import rearrange
from svox.helpers import DataFormat, _get_c_extension, LocalIndex
from torch import autograd
from warnings import warn

_C = _get_c_extension()

def _rays_spec_from_rays(rays):
    spec = _C.RaysSpec()
    spec.origins = rays.origins
    spec.dirs = rays.dirs
    spec.vdirs = rays.viewdirs
    return spec

# class _AdRenderAutugradFunction(autograd.Function):
#     @staticmethod
#     def forward(ctx, tree, rays, opt):
#         out = _C.volume_render(tree, rays, opt)
#         ctx.tree = tree
#         ctx.rays = rays
#         ctx.opt = opt
#         return out
    
#     @staticmethod
#     def backward(ctx, grad_out_data_to_render):   
#         if ctx.needs_input_grad[0]:
#             return _C.volume_render_backward(
#                 ctx.tree, ctx.rays, ctx.opt, grad_out_data_to_render.contiguous()
#             ), None, None, None
#         return None, None, None, None         

        
class AdExternal_N3Tree(nn.Module):
    def __init__(self,
                 data_dim=32,
                 depth_limit=10,
                 drop=0.2,
                 mlp_ratio=2,
                 init_reserve=1,
                 init_refine=0,
                 data_format='RGBA',
                 device="cuda",
                 ):
        """
        :param data_format: a string to indicate the data format. :code:`RGBA | SH# | SG# | ASG#`
        :param init_reserve: int amount of nodes to reserve initially
        :param init_refine: int number of times to refine entire tree initially
                            inital resolution will be :code:`[N^(init_refine + 1)]^3`.
        """
        super().__init__()
        self.depth_limit = depth_limit
        self.data_dim = data_dim
        self.tree = N3Tree(data_dim=data_dim, depth_limit=depth_limit, 
                        init_refine=init_refine,
                        init_reserve=init_reserve,
                        device=device)
        self.N = self.tree.N
        self.device = device
        
        # output RGB by default
        self.data_format = DataFormat(data_format)
        if self.data_format.data_dim == None:
            self.data_format.data_dim = 3 + 1
                      
        self.data_format_txt = data_format
        self.depth_weight = nn.Parameter(torch.FloatTensor(np.arange(1, depth_limit+1))/(depth_limit+1))
        self.mlp_ratio = mlp_ratio
        self.dict_convs = {}
        for d in range(depth_limit):
            self.dict_convs[d] = TreeConv(data_dim, data_dim, self.N**3).to(device=device)
        self.head_sigma = Mlp(
            self.data_dim, self.data_dim*self.mlp_ratio, 1, drop=drop).to(device=device)
        self.head_f = Mlp(self.data_dim, self.data_dim *
                              self.mlp_ratio, self.data_format.data_dim-1, drop=drop).to(device=device)
        self.new_data_dim = self.data_format.data_dim
        self.init_weights()
        
    # def forward(self, rays : Rays, cuda=True, fast=False):

    #     with torch.enable_grad():
    #         return _AdRenderAutugradFunction.apply(
    #             self.tree._spec(self.data_to_render),
    #             _rays_spec_from_rays(rays),
    #             self._get_options(fast)            
    #         )        
    def init_weights(self):
        nn.init.normal_(self.tree.data)
        self.apply(_init_Adtree_weights)
        
    def encode_at(self, intnode_idx):
        """
        Advanced: Encode features at leaves of the internal node by the Treeconv operation. 
        :param intnode_idx: index of internal node for identifying leaves
        """
        assert intnode_idx<self.tree.n_internal, f"The intnode_idx is the index of the node at the internal nodes array of length ({self.n_internal}), while intnode_idx={intnode_idx}."
        assert self.tree.parent_depth[intnode_idx, 1] < self.tree.depth_limit, 'the operation is legal for depths less than the depth limit.'
        
        data = rearrange(self.tree.data[intnode_idx], 'N1 N2 N3 D -> D (N1 N2 N3)')
        depth = self.tree.parent_depth[intnode_idx]
        conv = self.dict_convs[depth[1].item()]
        
        return conv(data)
    
    def encode(self):
        """
        Advanced: Encode features of the entire tree by the Treeconv operation.
        The convolution is a recursive process that starts from the deepest layer to the top.
        """   
        B = self.tree.data.size(0)
        N = self.N
        device = self.tree.data.device
        new_data_dim = self.new_data_dim
        dtype = self.tree.data.dtype
        
        # encode
        depth, indexes = torch.sort(self.tree.parent_depth, dim=0, descending=True)
        features = torch.zeros(self.tree.data_dim, device=self.device)
        
        for d in depth:
            idx = d[0]
            depth = d[1]
            xyzi = self.tree._unpack_index(idx)
            intnode_idx, xi, yi, zi = xyzi
            feature = self.encode_at(intnode_idx)
            # revise the internal node's data
            self.tree.data.data[intnode_idx, xi, yi, zi] = feature
            features += self.depth_weight[depth]*feature
        
        # revise the tree's leaf nodes for rendering
        self.tree += features
        # the internal nodes will be skipped by the rendering algorithm. 
        # Note: leaf nodes are accessed by t[:]
        features = self.tree[:]
        _f = self.head_f(features)
        f_ = self.head_sigma(features)
        leaf_data = torch.cat((_f, f_), dim=1)
        leaf_idx = self.tree._all_leaves()
        leaf_idx = self.tree._pack_index(leaf_idx)
        # prepare OUT tree 
        # t_out = N3Tree(N=self.N, data_dim=new_data_dim,
        #         data_format=str(self.data_format),
        #         depth_limit=self.depth_limit,
        #         geom_resize_fact=self.tree.geom_resize_fact,
        #         dtype=dtype,
        #         device=device
        # )
        
        # def copy_to_device(x):
        #     return torch.empty(x.shape, dtype=x.dtype, device=device).copy_(x)
        # t_out.invradius = copy_to_device(self.tree.invradius)
        # t_out.offset = copy_to_device(self.tree.offset)
        # t_out.child = copy_to_device(self.tree.child)
        # t_out.parent_depth = copy_to_device(self.tree.parent_depth)
        # t_out._n_internal = copy_to_device(self.tree._n_internal)
        # t_out._n_free = copy_to_device(self.tree._n_free)
        
        # if self.tree.extra_data is not None:
        #     t_out.extra_data = copy_to_device(self.tree.extra_data)
        # else:
        #     t_out.extra_data = None   
        t_out = self.tree.partial()
        t_out.expand(self.data_format_txt, data_dim=self.new_data_dim)
        # render = VR(t_out)
        # target =  torch.tensor([[0.0, 1.0, 0.5]]).cuda()
        # ray_ori = torch.tensor([[0.1, 0.1, -0.1]]).cuda()
        # ray_dir = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
        # ray = Rays(origins=ray_ori, dirs=ray_dir, viewdirs=ray_dir)
        # print(render(ray, cuda=True))

        size_ = t_out.data.shape
        data = torch.zeros(size_, device=device)
        B, N1, N2, N3, D = data.size()
        data = rearrange(data, 'B N1 N2 N3 D -> (B N1 N2 N3) D')
        data[leaf_idx] = leaf_data
        del t_out.data
        t_out.data = rearrange(data, '(B N1 N2 N3) D ->B N1 N2 N3 D', B=B, N1=N1, N2=N2, N3=N3)
        return t_out
                   
    def get_parent_intnode(self, idx):
        """Get the internode's parent node idx.

        Args:
            idx: the internode's idx. 
        Return:
            the parent's node idx if exist, else return -1.
        """
        inter_nodes_xyzi = self.tree.child[:self.tree.n_internal].nonzero()
        xyzi = self.tree._unpack_index(idx)
        int_idx = ((xyzi - inter_nodes_xyzi)==0).all(1)
        res = inter_nodes_xyzi[int_idx]
        
        if res.size(0) >0:
            return res[0]
        else:
            return -1

def _init_Adtree_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)