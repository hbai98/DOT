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

from svox import N3Tree
import torch
import torch.nn as nn
from timm.models import Mlp
from svox.helpers import N3TreeView, DataFormat, LocalIndex, _get_c_extension
from utils import TreeConv
from warnings import warn

_C = _get_c_extension()

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
    def __init__(self, N=2, data_dim=None, hidden_dim=None, mlp_ratio=2, depth_limit=10,
            init_reserve=1, init_refine=0, geom_resize_fact=1.0,
            radius=0.5, center=[0.5, 0.5, 0.5],
            data_format="RGBA",
            extra_data=None,
            device="cpu",
            dtype=torch.float32,
            map_location=None):
        """
        Construct N^3 Tree

        :param N: int branching factor N
        :param data_dim: int size of data stored at each leaf (NEW in 0.2.28: optional if data_format other than RGBA is given).
                        If data_format = "RGBA" or empty, this defaults to 4.
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
        
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        if self.hidden_dim is not None:
            self.conv = TreeConv(self.hidden_dim, self.hidden_dim, N^3)
            self.head_density = Mlp(self.hidden_dim, self.hidden_dim*self.mlp_ratio, 1)
            self.head_color =  Mlp(self.hidden_dim, self.hidden_dim*self.mlp_ratio, 3)
    
    
        
    def forward(self, indices, cuda=True, want_node_ids=False, world=True):
        """
        Get tree values. Differentiable.

        :param indices: :code:`(Q, 3)` the points
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.
        :param want_node_ids: if true, returns node ID for each query.
        :param world: use world space instead of :code:`[0,1]^3`, default True

        :return: :code:`(Q, data_dim), [(Q)]`

        """
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert len(indices.shape) == 2

        if not cuda or _C is None or not self.data.is_cuda:
            if not want_node_ids:
                warn("Using slow query")
            if world:
                indices = self.world2tree(indices)

            indices.clamp_(0.0, 1.0 - 1e-10)

            n_queries, _ = indices.shape
            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            result = torch.empty((n_queries, self.data_dim), dtype=self.data.dtype,
                                  device=indices.device)
            remain_indices = torch.arange(n_queries, dtype=torch.long, device=indices.device)
            ind = indices.clone()

            if want_node_ids:
                subidx = torch.zeros((n_queries, 3), dtype=torch.long, device=indices.device)

            while remain_indices.numel():
                ind *= self.N
                ind_floor = torch.floor(ind)
                ind_floor.clamp_max_(self.N - 1)
                ind -= ind_floor

                sel = (node_ids[remain_indices], *(ind_floor.long().T),)

                deltas = self.child[sel]

                term_mask = deltas == 0
                term_indices = remain_indices[term_mask]

                vals = self.data[sel]
                result[term_indices] = vals[term_mask]
                if want_node_ids:
                    subidx[term_indices] = ind_floor.to(torch.long)[term_mask]

                node_ids[remain_indices] += deltas
                remain_indices = remain_indices[~term_mask]
                ind = ind[~term_mask]

            if want_node_ids:
                txyz = torch.cat([node_ids[:, None], subidx], axis=-1)
                return result, self._pack_index(txyz)

            return result
        else:
            result, node_ids = _QueryVerticalFunction.apply(
                                self.data, self._spec(world), indices);
            return (result, node_ids) if want_node_ids else result