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
from warnings import warn
from torch.nn.functional import gumbel_softmax

_C = _get_c_extension()

def _rays_spec_from_rays(rays):
    spec = _C.RaysSpec()
    spec.origins = rays.origins
    spec.dirs = rays.dirs
    spec.vdirs = rays.viewdirs
    return spec
       

        
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
            self.dict_convs[str(d)] = TreeConv(data_dim, data_dim, self.N**3).to(device=device)
        self.dict_convs = nn.ModuleDict(self.dict_convs)
        self.head_sigma = Mlp(
            self.data_dim, self.data_dim*self.mlp_ratio, 1, drop=drop).to(device=device)
        self.head_f = Mlp(self.data_dim, self.data_dim *
                              self.mlp_ratio, self.data_format.data_dim-1, drop=drop).to(device=device)
        self.new_data_dim = self.data_format.data_dim
        self.init_weights()
        
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
        conv = self.dict_convs[str(depth[1].item())]
        
        return conv(data)
    
    def encode(self):
        """
        Advanced: Encode features of the entire tree by the Treeconv operation.
        The convolution is a recursive process that starts from the deepest layer to the top.
        
        Return:

        """   
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
        # the internal nodes will be skipped by the rendering algorithm. 
        # Note: leaf nodes are accessed by t[:]
        features = self.tree[:] + features
        _f = self.head_f(features)
        f_ = self.head_sigma(features)
        leaf_data = torch.cat((_f, f_), dim=1)
        return leaf_data
    
    
    def out_tree(self, leaf_data):
        B = self.tree.data.size(0)
        device = self.tree.data.device
        leaf_idx = self.tree._all_leaves()
        leaf_idx = self.tree._pack_index(leaf_idx)
        
        # prepare OUT tree 
        t_out = self.tree.partial()
        t_out.expand(self.data_format_txt, data_dim=self.new_data_dim)
        
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
    
    def expand_grad(self, grad, k=5):
        """The adaptive sampling based on greedy gradient based selection. 
        We expand the tree based on max(|dMSE/dc|). 
        This operation is only feasible whene data.grad is availabel and the current depth <= depth_limit. return signal True.
        If depth > depth_limit, the operation stops, and return signal False.
        """
        assert self.tree.data.grad is not None, 'The grad is needed for adaptive selection. '
        
        # grad = self.tree.data.grad.clone()
        leaf_idx = self.tree._all_leaves()
        leaf_idx = self.tree._pack_index(leaf_idx) 
        grad = rearrange(grad, 'B N1 N2 N3 D -> (B N1 N2 N3) D')
        grad = grad[leaf_idx] 
        grad = grad.sum(dim=-1)
        _, node_idx = torch.topk(grad, k=k)
        idxes = leaf_idx[node_idx]
        
        for idx in idxes:
            intnode_idx, xi, yi, zi = self.tree._unpack_index(idx)
            if self.tree.parent_depth[intnode_idx, 1] >= self.depth_limit:
                continue
            # expand
            flag = self.tree._refine_at(intnode_idx, (xi, yi, zi))
            break
        return flag
        

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