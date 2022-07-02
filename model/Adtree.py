#  Copyright 2022 AdNerf Authors.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

from einops import rearrange
from matplotlib.pyplot import axis
from svox import N3Tree
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from svox.helpers import _get_c_extension
from .utils import TreeConv
from timm.models.layers import lecun_normal_, trunc_normal_

class AdTree(N3Tree):
    """
    PyTorch :math:`N^3`-tree library with CUDA acceleration.
    By :math:`N^3`-tree we mean a 3D tree with branching factor N at each interior node,
    where :math:`N=2` is the familiar octree.

    .. warning::
        `nn.Parameters` can change size, which
        makes current optimizers invalid. If any :code:`refine(): or
        :code:`shrink_to_fit()` call returns True,
        or :code:`expand(), shrink()` is used,
        please re-make any optimizers
    """
    def __init__(self, N=2, data_dim=32, mlp_ratio=2, depth_limit=10,
            init_reserve=1, init_refine=0, geom_resize_fact=1.0,
            radius=0.5, center=[0.5, 0.5, 0.5],
            data_format="RGBA",
            extra_data=None,
            device="cpu",
            dtype=torch.float32,
            map_location=None,
            drop=0.2,):
        """
        Construct N^3 Tree

        :param N: int branching factor N
        :param data_dim: int size of data stored at each leaf (NEW in 0.2.28: optional if data_format other than RGBA is given).
                        If data_format = "RGBA" or empty, this defaults to 32.
        :param hidden_dim: int size of hidden representation of data, defaults to None. If it is not None, the data has deep representation. 
        :param depth_limit: int maximum depth  of tree to stop branching/refining
                            Note that the root is at depth -1.
                            Size :code:`N^[-10]` leaves (1/1024 for octree) for example
                            are depth 9. :code:`max_depth` applies to the same
                            depth values.
        :param init_reserve: int amount of nodes to reserve initially
        :param init_refine: int number of times to refine entire tree initially
                            inital resolution will be :code:`[N^(init_refine + 1)]^3`.
                            initial max_depth will be init_refine.
        :param geom_resize_fact: float geometric resizing factor
        :param radius: float or list, 1/2 side length of cube (possibly in each dim)
        :param center: list center of space
        :param data_format: a string to indicate the data format. :code:`RGBA | SH# | SG# | ASG#`
        :param extra_data: extra data to include with tree
        :param device: str device to put data
        :param dtype: str tree data type, torch.float32 (default) | torch.float64
        :param map_location: str DEPRECATED old name for device (will override device and warn)
        :param drop: float, the dropout rate for MLP. Both color and density. 
        """
        super().__init__(N=N,
                         data_dim=data_dim,
                         depth_limit=depth_limit,
                         init_reserve=init_reserve,
                         init_refine=init_refine,
                         geom_resize_fact=geom_resize_fact,
                         radius=radius,
                         center=center,
                         data_format=data_format,
                         extra_data=extra_data,
                         device=device,
                         dtype=dtype,
                         map_location=map_location)
                
        self.mlp_ratio = mlp_ratio
        self.dict_convs = {}
        for d in range(depth_limit):
            self.dict_convs[d] = TreeConv(data_dim, data_dim, N**3)
        self.head_density = Mlp(self.data_dim, self.data_dim*self.mlp_ratio, 1, drop=drop)
        self.head_color =  Mlp(self.data_dim, self.data_dim*self.mlp_ratio, 3, drop=drop)
    
        # initialize high dimensional data with data_dim
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.data)
        self.apply(_init_Adtree_weights)
    
    def expand_at(self, intnode_idx, xyzi):
        """
        Advanced: insert children at a specific leaf node. Concatenating the torch.array instead of 
        creating the new data. 

        :param intnode_idx: index of internal node for identifying leaf
        :param xyzi: tuple of size 3 with each element in :code:`{0, ... N-1}`
                    in xyz orde rto identify leaf within internal node
        """       
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        assert min(xyzi) >= 0 and max(xyzi) < self.N
        if self.parent_depth[intnode_idx, 1] >= self.depth_limit:
            return

        xi, yi, zi = xyzi
        if self.child[intnode_idx, xi, yi, zi] != 0:
            # Already has child
            return

        resized = False
        filled = self.n_internal
        if filled >= self.capacity:
            self._resize_add_cap(1)
            resized = True

        self.child[filled] = 0
        self.child[intnode_idx, xi, yi, zi] = filled - intnode_idx
        depth = self.parent_depth[intnode_idx, 1] + 1
        self.parent_depth[filled, 0] = self._pack_index(torch.tensor(
            [[intnode_idx, xi, yi, zi]], dtype=torch.int32))[0]
        self.parent_depth[filled, 1] = depth
        ## different from original _refine_at()
        self.data.data[filled, :, :, :] = self.data.data[intnode_idx, xi, yi, zi] # -> children copy data from their parents
        # self.data.data[intnode_idx, xi, yi, zi] = 0 -> paraent to zero [is unecessary and may destroy gradients]
        
        self._n_internal += 1
        self._invalidate()
        return resized      
      
    def encode_at(self, intnode_idx):
        """
        Advanced: Encode features at leaves of the internal node by the Treeconv operation. 
        :param intnode_idx: index of internal node for identifying leaves
        """
        assert intnode_idx<self.n_internal, f"The intnode_idx is the index of the node at the internal nodes array of length ({self.n_internal}), while intnode_idx={intnode_idx}."
        assert self.parent_depth[intnode_idx, 1] < self.depth_limit, 'the operation is legal for depths less than the depth limit.'
        
        data = rearrange(self.data[intnode_idx], 'N1 N2 N3 D -> D (N1 N2 N3)')
        depth = self.parent_depth[intnode_idx]
        conv = self.dict_convs[depth[1].item()]
        
        return conv(data)
    
    def encode(self):
        """
        Advanced: Encode features of the entire tree by the Treeconv operation.
        The convolution is a recursive process that starts from the deepest layer to the top.
        
        :return: muti-scale features after mean()[data_dim]
        """
        depth, indexes = torch.sort(self.parent_depth, dim=0, descending=True)
        features = torch.zeros(1, self.data_dim)
        for d in depth:
            idx = d[0]
            xyzi = self._unpack_index(idx)
            intnode_idx, xi, yi, zi = xyzi
            feature = self.encode_at(intnode_idx)
            # revise the internal node's data
            self.data.data[intnode_idx, xi, yi, zi] = feature
            features = torch.cat((features, feature.unsqueeze(0)), axis=0)
        return features[1:].mean(dim=0)
            
    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat = torch.div(flat, self.N, rounding_mode='floor')
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)
    
def _init_Adtree_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
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
       