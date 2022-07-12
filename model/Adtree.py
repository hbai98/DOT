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

import torch
import torch.nn as nn
from timm.models.layers import Mlp

from torch import float64, nn, autograd
from .utils import TreeConv, posenc
from timm.models.layers import lecun_normal_, trunc_normal_
from typing import Union, List
from warnings import warn
from adsvox.utils import _get_c_extension, N3TreeView, LocalIndex
from adsvox import Rays
from einops import rearrange
from dataclasses import dataclass
import numpy as np
import math

from typing import Optional
_C = _get_c_extension()
class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, tree, rays, opt):
        out = _C.volume_render(tree, rays, opt)
        ctx.tree = tree
        ctx.rays = rays
        ctx.opt = opt
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_input_grad[0]:
            return _C.volume_render_backward(
                ctx.tree, ctx.rays, ctx.opt, grad_out.contiguous()
            ), None, None, None
        return None, None, None, None
    
@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    available:
    :param backend: str, renderer backend
    :param background_brightness: float
    :param step_size: float, step size for rendering
    :param sigma_thresh: float
    :param stop_thresh: float
    """

    backend: str = "cuvol"  # One of cuvol, svox1, nvol

    background_brightness: float = 1.0  # [0, 1], the background color black-white

    step_size: float = 0.5  # Step size, in normalized voxels (not used for svox1)
    #  (i.e. 1 = 1 voxel width, different from svox where 1 = grid width!)

    sigma_thresh: float = 1e-10  # Voxels with sigmas < this are ignored, in [0, 1]
    #  make this higher for fast rendering

    stop_thresh: float = (
        1e-7  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    )
    #  probability is <= this much (forward only)
    #  make this higher for fast rendering

    last_sample_opaque: bool = False   # Make the last sample opaque (for forward-facing)

    near_clip: float = 0.0
    use_spheric_clip: bool = False

    random_sigma_std: float = 1.0                   # Noise to add to sigma (only if randomize=True)
    random_sigma_std_background: float = 1.0        # Noise to add to sigma
                                                    # (for the BG model; only if randomize=True)

    def _to_cpp(self, randomize: bool = False):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.background_brightness = self.background_brightness
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip

        opt.last_sample_opaque = self.last_sample_opaque
        #  opt.randomize = randomize
        #  opt.random_sigma_std = self.random_sigma_std
        #  opt.random_sigma_std_background = self.random_sigma_std_background

        #  if randomize:
        #      # For our RNG
        #      UINT32_MAX = 2**32-1
        #      opt._m1 = np.random.randint(0, UINT32_MAX)
        #      opt._m2 = np.random.randint(0, UINT32_MAX)
        #      opt._m3 = np.random.randint(0, UINT32_MAX)
        #      if opt._m2 == opt._m3:
        #          opt._m3 += 1  # Prevent all equal case
        # Note that the backend option is handled in Python
        return opt


class _QueryVerticalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, data, tree_spec, indices):
        out, node_ids = _C.query_vertical(tree_spec, indices)

        ctx.mark_non_differentiable(node_ids)
        ctx.tree_spec = tree_spec
        ctx.save_for_backward(indices)
        return out, node_ids

    @staticmethod
    def backward(ctx, grad_out, dummy):
        if ctx.needs_input_grad[0]:
            return _C.query_vertical_backward(ctx.tree_spec,
                         ctx.saved_tensors[0],
                         grad_out.contiguous()), None, None
        return None, None, None
    
class AdTree(nn.Module):
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

    def __init__(
            self,
            N=2,
            data_dim=32, mlp_ratio=2, depth_limit=10,
            init_reserve=1, init_refine=0, geom_resize_fact=1.0,
            radius: Union[float, List[float]] = .5,
            center: Union[float, List[float]] = [.5, .5, .5],
            extra_data=None,
            dtype=torch.float32,
            map_location=None,
            drop=0.2,
            mlp_posenc_size : int = 0,
            background_nlayers: int = 0,
            background_reso: int = 256,
            device: Union[torch.device, str] = "cpu",
    ):
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
        :param mlp_posenc_size: int, if using BASIS_TYPE_MLP, then enables standard axis-aligned positional encoding of
                                    given size on MLP; if 0 then does not use positional encoding
        """
        super().__init__()
        assert N >= 2
        assert depth_limit >= 0
        self.N : int = N

        self.opt = RenderOptions()
        if map_location is not None:
            warn('map_location has been renamed to device and may be removed')
            device = map_location
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'
        
        self.data_dim : int = data_dim
        if init_refine > 0:
            for i in range(1, init_refine + 1):
                init_reserve += (N ** i) ** 3

        self.register_parameter("data", nn.Parameter(
            torch.zeros(init_reserve, N, N, N, self.data_dim, dtype=dtype, device=device)))
        self.register_buffer("child", torch.zeros(
            init_reserve, N, N, N, dtype=torch.int32, device=device))
        self.register_buffer("parent_depth", torch.zeros(
            init_reserve, 2, dtype=torch.int32, device=device))

        self.register_buffer("_n_internal", torch.tensor(1, device=device))
        self.register_buffer("_n_free", torch.tensor(0, device=device))

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        radius = torch.tensor(radius, dtype=dtype, device=device)
        center = torch.tensor(center, dtype=dtype, device=device)

        self.register_buffer("invradius", 0.5 / radius)
        self.register_buffer("offset", 0.5 * (1.0 - center / radius))

        self.depth_limit = depth_limit
        self.geom_resize_fact = geom_resize_fact

        if extra_data is not None:
            assert isinstance(extra_data, torch.Tensor)
            self.register_buffer("extra_data", extra_data.to(dtype=dtype, device=device))
        else:
            self.extra_data = None

        self._ver = 0
        self._invalidate()
        self._lock_tree_structure = False
        self._weight_accum = None
        self._weight_accum_op = None

        self.refine(repeats=init_refine)
        
        self.depth_weight = nn.Parameter(torch.FloatTensor(np.arange(1, depth_limit+1))/(depth_limit+1))
        self.mlp_ratio = mlp_ratio
        self.dict_convs = {}
        for d in range(depth_limit):
            self.dict_convs[d] = TreeConv(data_dim, data_dim, N**3).to(device=device)
        self.head_sigma = Mlp(
            self.data_dim, self.data_dim*self.mlp_ratio, 1, drop=drop).to(device=device)
        self.head_sh = Mlp(self.data_dim, self.data_dim *
                              self.mlp_ratio, 3, drop=drop).to(device=device)

        # self.background_nlayers = background_nlayers
        # assert background_nlayers == 0 or background_nlayers > 1, "Please use at least 2 MSI layers (trilerp limitation)"
        # self.background_reso = background_reso

        # # TODO: background data for 360
        # self.background_links: Optional[torch.Tensor]
        # self.background_data: Optional[torch.Tensor]
        # if self.use_background:
        #     background_capacity = (self.background_reso ** 2) * 2
        #     background_links = torch.arange(
        #         background_capacity,
        #         dtype=torch.int32, device=device
        #     ).reshape(self.background_reso * 2, self.background_reso)
        #     self.register_buffer('background_links', background_links)
        #     self.background_data = nn.Parameter(
        #         torch.zeros(
        #             background_capacity,
        #             self.background_nlayers,
        #             4,
        #             dtype=torch.float32, device=device
        #         )
        #     )
        # else:
        #     self.background_data = nn.Parameter(
        #         torch.empty(
        #             0, 0, 0,
        #             dtype=torch.float32, device=device
        #         ),
        #         requires_grad=False
        #     )
        self.init_weights()
        
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
                
    # Main accesors
    def set(self, indices, values, cuda=True):
        """
        Set tree values,

        :param indices: torch.Tensor :code:`(Q, 3)`
        :param values: torch.Tensor :code:`(Q, K)`
        :param cuda: whether to use CUDA kernel if available. If false,
                     uses only PyTorch version.

        """
        assert len(indices.shape) == 2
        assert not indices.requires_grad  # Grad wrt indices not supported
        assert not values.requires_grad  # Grad wrt values not supported
        indices = indices.to(device=self.data.device)
        values = values.to(device=self.data.device)

        if not cuda or _C is None or not self.data.is_cuda:
            warn("Using slow assignment")
            indices = self.world2tree(indices)

            n_queries, _ = indices.shape
            indices.clamp_(0.0, 1.0 - 1e-10)
            ind = indices.clone()

            node_ids = torch.zeros(n_queries, dtype=torch.long, device=indices.device)
            remain_mask = torch.ones(n_queries, dtype=torch.bool, device=indices.device)
            while remain_mask.any():
                ind_floor = torch.floor(ind[remain_mask] * self.N)
                ind_floor.clamp_max_(self.N - 1)
                sel = (node_ids[remain_mask], *(ind_floor.long().T),)

                deltas = self.child[sel]
                vals = self.data.data[sel]

                nonterm_partial_mask = deltas != 0
                nonterm_mask = torch.zeros(n_queries, dtype=torch.bool, device=indices.device)
                nonterm_mask[remain_mask] = nonterm_partial_mask

                node_ids[remain_mask] += deltas
                ind[remain_mask] = ind[remain_mask] * self.N - ind_floor

                term_mask = remain_mask & ~nonterm_mask
                vals[~nonterm_partial_mask] = values[term_mask]
                self.data.data[sel] = vals

                remain_mask &= nonterm_mask
        else:
            _C.assign_vertical(self._spec(), indices, values)
            
    def init_weights(self):
        nn.init.normal_(self.data)
        self.apply(_init_Adtree_weights)
        
    def _invalidate(self):
        self._ver += 1
        self._last_all_leaves = None
        self._last_frontier = None
        
    def world2tree(self, indices):
        """
        Scale world points to tree (:math:`[0,1]^3`)
        """
        return torch.addcmul(self.offset, indices, self.invradius)        
    
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
        # different from original _refine_at()
        # -> children copy data from their parents
        self.data.data[filled, :, :,
                       :] = self.data.data[intnode_idx, xi, yi, zi]
        # self.data.data[intnode_idx, xi, yi, zi] = 0 -> paraent to zero [is unecessary and may destroy gradients]

        self._n_internal += 1
        self._invalidate()
        return resized
    def _pack_index(self, txyz):
        return txyz[:, 0] * (self.N ** 3) + txyz[:, 1] * (self.N ** 2) + \
               txyz[:, 2] * self.N + txyz[:, 3]
               
    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing capacity
        """
        cap_needed = max(cap_needed, int(self.capacity * (self.geom_resize_fact - 1.0)))
        may_oom = self.capacity + cap_needed > 1e7  # My CPU Memory is limited
        if may_oom:
            # Potential OOM prevention hack
            self.data = nn.Parameter(self.data.cpu())
        self.data = nn.Parameter(torch.cat((self.data.data,
                        torch.zeros((cap_needed, *self.data.data.shape[1:]),
                                dtype=self.data.dtype,
                                device=self.data.device)), dim=0))
        if may_oom:
            self.data = nn.Parameter(self.data.to(device=self.child.device))
        self.child = torch.cat((self.child,
                                torch.zeros((cap_needed, *self.child.shape[1:]),
                                   dtype=self.child.dtype,
                                   device=self.data.device)))
        self.parent_depth = torch.cat((self.parent_depth,
                                torch.zeros((cap_needed, *self.parent_depth.shape[1:]),
                                   dtype=self.parent_depth.dtype,
                                   device=self.data.device)))    
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

        """
        depth, indexes = torch.sort(self.parent_depth, dim=0, descending=True)
        features = torch.zeros(1, self.data_dim)
        for d in depth:
            idx = d[0]
            depth = d[1]
            xyzi = self._unpack_index(idx)
            intnode_idx, xi, yi, zi = xyzi
            feature = self.encode_at(intnode_idx)
            # revise the internal node's data
            self.data.data[intnode_idx, xi, yi, zi] = feature
            features += self.depth_weight[depth]*feature
        
        return features

    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat = torch.div(flat, self.N, rounding_mode='floor')
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)
    
    # Leaf refinement & memory management methods
    def refine(self, repeats=1, sel=None):
        """
        Refine each selected leaf node, respecting depth_limit.

        :param repeats: int number of times to repeat refinement
        :param sel: :code:`(N, 4)` node selector. Default selects all leaves.

        :return: True iff N3Tree.data parameter was resized, requiring
                 optimizer reinitialization if you're using an optimizer

        .. warning::
            The parameter :code:`tree.data` can change due to refinement. If any refine() call returns True, please re-make any optimizers
            using :code:`tree.params()`.

        .. warning::
            The selector :code:`sel` is assumed to contain unique leaf indices. If there are duplicates
            memory will be wasted. We do not dedup here for efficiency reasons.

        """
        if self._lock_tree_structure:
            raise RuntimeError("Tree locked")
        with torch.no_grad():
            resized = False
            for repeat_id in range(repeats):
                filled = self.n_internal
                if sel is None:
                    # Default all leaves
                    sel = (*self._all_leaves().T,)
                depths = self.parent_depth[sel[0], 1]
                # Filter by depth & leaves
                good_mask = (depths < self.depth_limit) & (self.child[sel] == 0)
                sel = [t[good_mask] for t in sel]
                leaf_node =  torch.stack(sel, dim=-1).to(device=self.data.device)
                num_nc = len(sel[0])
                if num_nc == 0:
                    # Nothing to do
                    return False
                new_filled = filled + num_nc

                cap_needed = new_filled - self.capacity
                if cap_needed > 0:
                    self._resize_add_cap(cap_needed)
                    resized = True

                new_idxs = torch.arange(filled, filled + num_nc,
                        device=leaf_node.device, dtype=self.child.dtype) # NNC

                self.child[filled:new_filled] = 0
                self.child[sel] = new_idxs - leaf_node[:, 0].to(torch.int32)
                self.data.data[filled:new_filled] = self.data.data[
                        sel][:, None, None, None]
                self.parent_depth[filled:new_filled, 0] = self._pack_index(leaf_node)  # parent
                self.parent_depth[filled:new_filled, 1] = self.parent_depth[
                        leaf_node[:, 0], 1] + 1  # depth

                if repeat_id < repeats - 1:
                    # Infer new selector
                    t1 = torch.arange(filled, new_filled,
                            device=self.data.device).repeat_interleave(self.N ** 3)
                    rangen = torch.arange(self.N, device=self.data.device)
                    t2 = rangen.repeat_interleave(self.N ** 2).repeat(
                            new_filled - filled)
                    t3 = rangen.repeat_interleave(self.N).repeat(
                            (new_filled - filled) * self.N)
                    t4 = rangen.repeat((new_filled - filled) * self.N ** 2)
                    sel = (t1, t2, t3, t4)
                self._n_internal += num_nc
        if repeats > 0:
            self._invalidate()
        return resized

    def _eval(self, rays: Rays):
        dirs = rays.dirs
        if self.mlp_posenc_size > 0:
            dirs_enc = posenc(
                dirs,
                None,
                0,
                self.mlp_posenc_size,
                include_identity=True,
            )
        else:
            dirs_enc = dirs
            
        features = self.encode()
        
        return features, dirs_enc
        
    def volume_render(
        self,
        rays: Rays,
        fast=False
    ):
        """
        Standard volume rendering with fused MSE gradient generation,
            given a ground truth color for each pixel.
        Will update the *.grad tensors for each parameter
        You can then subtract the grad manually or use the optim_*_step methods

        See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param rgb_gt: (N, 3), GT pixel colors, each channel in [0, 1]
        :param randomize: bool, whether to enable randomness
        :param beta_loss: float, weighting for beta loss to add to the gradient.
                                 (fused into the backward pass).
                                 This is average voer the rays in the batch.
                                 Beta loss also from neural volumes:
                                 [Lombardi et al., ToG 2019]
        :return: (N, 3), predicted RGB
        """
        assert (
            _C is not None and self.data.is_cuda
        ), "CUDA extension is currently required for fused"
        assert rays.is_cuda
        grad_data, grad_bg = self._get_data_grads()
        with torch.enable_grad():
            dir_enc, multi_scale_data = self._eval(rays) # basis_data [D] -> multi-scale features 
            multi_scale_data = self.data + multi_scale_data
            # inference sh and sigma.
            sh = self.head_sh(multi_scale_data)
            sigma = self.head_sigma(multi_scale_data)
        
        rgb_out = _VolumeRenderFunction.apply(
            self.data,
            self._spec(replace_data=torch.cat()),
            rays._to_cpp(),
            self._get_options(fast))
        # grad_sh = torch.empty_like(sh)
        # grad_sigma = torch.empty_like(sigma)
        
        # grad_holder = _C.TreeOutputGrads()
        # grad_holder.grad_sh = grad_sh
        # grad_holder.grad_sigma = grad_sigma
        
        # if self.use_background:
        #     grad_holder.grad_background_out = grad_bg
        #     self.sparse_background_indexer = torch.zeros(list(self.background_data.shape[:-1]),
        #             dtype=torch.bool, device=self.background_data.device)
        #     grad_holder.mask_background_out = self.sparse_background_indexer
        
        # cu_fn = _C.__dict__[f"volume_render_{self.opt.backend}_fused"]
        # #  with utils.Timing("actual_render"):
        # cu_fn(
        #     self._spec(replace_data=torch.cat()),
        #     rays._to_cpp(),
        #     self.opt._to_cpp(randomize=randomize),
        #     rgb_gt,
        #     beta_loss,
        #     sparsity_loss,
        #     rgb_out,
        #     grad_holder
        # )
        # # Manually trigger conv+MLP backward!
        # sh.backward(grad_sh)
        # sigma.backward(grad_sigma)

        return rgb_out    
    
    def check_integrity(self):
            """
            Do some checks to verify the tree's structural integrity,
            mostly for debugging. Errors with message if check fails;
            does nothing else.
            """
            n_int = self.n_internal
            n_free = self._n_free.item()
            assert n_int - n_free > 0, "Tree has no root"
            assert self.data.shape[0] == self.capacity, "Data capacity mismatch"
            assert self.child.shape[0] == self.capacity, "Child capacity mismatch"
            assert (self.parent_depth[0] == 0).all(), "Node at index 0 must be root"

            free = self.parent_depth[:n_int, 0] == -1
            remain_ids = torch.arange(n_int, dtype=torch.long, device=self.child.device)[~free]
            remain_child = self.child[remain_ids]
            assert (remain_child >= 0).all(), "Nodes not topologically sorted"
            link_next = remain_child + remain_ids[..., None, None, None]

            assert link_next.max() < n_int, "Tree has an out-of-bounds child link"
            assert (self.parent_depth[link_next.reshape(-1), 0] != -1).all(), \
                    "Tree has a child link to a deleted node"

            remain_ids = remain_ids[remain_ids != 0]
            if remain_ids.numel() == 0:
                return True
            remain_parents = (*self._unpack_index(
                self.parent_depth[remain_ids, 0]).long().T,)
            assert remain_parents[0].max() < n_int, "Parent link out-of-bounds (>=n_int)"
            assert remain_parents[0].min() >= 0, "Parent link out-of-bounds (<0)"
            for i in range(1, 4):
                assert remain_parents[i].max() < self.N, "Parent sublink out-of-bounds (>=N)"
                assert remain_parents[i].min() >= 0, "Parent sublink out-of-bounds (<0)"
            assert (remain_parents[0] + self.child[remain_parents] == remain_ids).all(), \
                    "parent->child cycle consistency failed"
            return True    
        
    def _all_leaves(self):
        if self._last_all_leaves is None:
            self._last_all_leaves = (self.child[
                :self.n_internal] == 0).nonzero(as_tuple=False).cpu()
        return self._last_all_leaves
    
    def _get_data_grads(self):
        ret = []
        for subitem in ["data", "background_data"]:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (
                    not hasattr(param, "grad")
                    or param.grad is None
                    or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret
    
    def _spec(self, world=True, replace_data=None):
        """
        Pack tree into a TreeSpec (for passing data to C++ extension)
        """
        tree_spec = _C.TreeSpec()
        if replace_data:
            tree_spec.data = replace_data
        else:
            tree_spec.data = self.data
        tree_spec.child = self.child
        tree_spec.parent_depth = self.parent_depth
        tree_spec.extra_data = self.extra_data if self.extra_data is not None else \
                torch.empty((0, 0), dtype=self.data.dtype, device=self.data.device)
        tree_spec.offset = self.offset if world else torch.tensor(
                  [0.0, 0.0, 0.0], dtype=self.data.dtype, device=self.data.device)
        tree_spec.scaling = self.invradius if world else torch.tensor(
                  [1.0, 1.0, 1.0], dtype=self.data.dtype, device=self.data.device)
        if hasattr(self, '_weight_accum'):
            tree_spec._weight_accum = self._weight_accum if \
                    self._weight_accum is not None else torch.empty(
                            0, dtype=self.data.dtype, device=self.data.device)
            tree_spec._weight_accum_max = (self._weight_accum_op == 'max')
        return tree_spec    
    def shrink_to_fit(self):
            """
            Shrink data & buffers to tightly needed fit tree data,
            possibly dealing with fragmentation caused by merging.
            This is called by the :code:`save()` function by default, unless
            :code:`shrink=False` is specified there.

            .. warning::
                    Will change the nn.Parameter size (data), breaking optimizer!
            """
            if self._lock_tree_structure:
                raise RuntimeError("Tree locked")
            n_int = self.n_internal
            n_free = self._n_free.item()
            new_cap = n_int - n_free
            if new_cap >= self.capacity:
                return False
            if n_free > 0:
                # Defragment
                free = self.parent_depth[:n_int, 0] == -1
                csum = torch.cumsum(free, dim=0)

                remain_ids = torch.arange(n_int, dtype=torch.long)[~free]
                remain_parents = (*self._unpack_index(
                    self.parent_depth[remain_ids, 0]).long().T,)

                # Shift data over
                par_shift = csum[remain_parents[0]]
                self.child[remain_parents] -= csum[remain_ids] - par_shift
                self.parent_depth[remain_ids, 0] -= par_shift * (self.N ** 3)

                # Remake the data now
                self.data = nn.Parameter(self.data.data[remain_ids])
                self.child = self.child[remain_ids]
                self.parent_depth = self.parent_depth[remain_ids]
                self._n_internal.fill_(new_cap)
                self._n_free.zero_()
            else:
                # Direct resize
                self.data = nn.Parameter(self.data.data[:new_cap])
                self.child = self.child[:new_cap]
                self.parent_depth = self.parent_depth[:new_cap]
            self._invalidate()
            return True

    # Misc
    @property
    def n_leaves(self):
        return self._all_leaves().shape[0]

    @property
    def n_internal(self):
        return self._n_internal.item()

    @property
    def capacity(self):
        return self.parent_depth.shape[0]

    @property
    def max_depth(self):
        """
        Maximum tree depth - 1
        """
        return torch.max(self.depths).item()

    def accumulate_weights(self, op : str='sum'):
        """
        Begin weight accumulation.

        :param op: reduction to apply weight in each voxel,
                   sum | max

        .. warning::

            Weight accumulator has not been validated
            and may have bugs

        :Example:

        .. code-block:: python

            with tree.accumulate_weights() as accum:
                ...

            # (n_leaves) in same order as values etc.
            accum = accum()
        """
        return WeightAccumulator(self, op)

    # Persistence
    def save(self, path, shrink=True, compress=True):
        """
        Save to from npz file

        :param path: npz path
        :param shrink: if True (default), applies shrink_to_fit before saving
        :param compress: whether to compress the npz; may be slow

        """
        if shrink:
            self.shrink_to_fit()
        data = {
            "data_dim" : self.data_dim,
            "child" : self.child.cpu(),
            "parent_depth" : self.parent_depth.cpu(),
            "n_internal" : self._n_internal.cpu().item(),
            "n_free" : self._n_free.cpu().item(),
            "invradius3" : self.invradius.cpu(),
            "offset" : self.offset.cpu(),
            "depth_limit": self.depth_limit,
            "geom_resize_fact": self.geom_resize_fact,
            "data": self.data.data.half().cpu().numpy()  # save CPU Memory
        }
        if self.data_format is not None:
            data["data_format"] = repr(self.data_format)
        if self.extra_data is not None:
            data["extra_data"] = self.extra_data.cpu()
        if compress:
            np.savez_compressed(path, **data)
        else:
            np.savez(path, **data)

    @classmethod
    def from_grid(cls, grid, *args, **kwargs):
        """
        Construct from a grid

        :param grid: (D, D, D, data_dim)

        """
        D = grid.shape[0]
        assert grid.ndim == 4 and grid.shape[1] == D and grid.shape[2] == D, \
               "Grid must be a 4D array with first 3 dims equal"
        logD = int(math.log2(D))
        assert 2**logD == D, "Grid size must be power of 2"
        kwargs['init_refine'] = logD - 1
        tree = cls(*args, **kwargs)
        tree.set_grid(grid)
        return tree

    def set_grid(self, grid):
        """
        Set current tree to grid.
        Assumes the tree's resolution is less than the grid's reslution

        :param grid: (D, D, D, data_dim)

        """
        D = grid.shape[0]
        assert grid.ndim == 4 and grid.shape[1] == D and \
               grid.shape[2] == D and grid.shape[-1] == self.data_dim
        idx = gen_grid(D).reshape(-1, 3)
        self[LocalIndex(idx)] = grid.reshape(-1, self.data_dim)

    @classmethod
    def load(cls, path, device='cpu', dtype=torch.float32, map_location=None):
        """
        Load from npz file

        :param path: npz path
        :param device: str device to put data
        :param dtype: str torch.float32 (default) | torch.float64
        :param map_location: str DEPRECATED old name for device

        """
        if map_location is not None:
            warn('map_location has been renamed to device and may be removed')
            device = map_location
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'
        tree = cls(dtype=dtype, device=device)
        z = np.load(path)
        tree.data_dim = int(z["data_dim"])
        tree.child = torch.from_numpy(z["child"]).to(device)
        tree.N = tree.child.shape[-1]
        tree.parent_depth = torch.from_numpy(z["parent_depth"]).to(device)
        tree._n_internal.fill_(z["n_internal"].item())
        if "invradius3" in z.files:
            tree.invradius = torch.from_numpy(z["invradius3"].astype(
                                np.float32)).to(device)
        else:
            tree.invradius.fill_(z["invradius"].item())
        tree.offset = torch.from_numpy(z["offset"].astype(np.float32)).to(device)
        tree.depth_limit = int(z["depth_limit"])
        tree.geom_resize_fact = float(z["geom_resize_fact"])
        tree.data.data = torch.from_numpy(z["data"].astype(np.float32)).to(device)
        if 'n_free' in z.files:
            tree._n_free.fill_(z["n_free"].item())
        else:
            tree._n_free.zero_()
        # tree.data_format = DataFormat(z['data_format'].item()) if \
        #         'data_format' in z.files else None
        tree.extra_data = torch.from_numpy(z['extra_data']).to(device) if \
                          'extra_data' in z.files else None
        return tree

    # Magic
    def __repr__(self):
        return (f"svox.N3Tree(N={self.N}, data_dim={self.data_dim}, " +
                f"depth_limit={self.depth_limit}, " +
                f"capacity:{self.n_internal - self._n_free.item()}/{self.capacity}, " +
                f"data_format:{self.data_format or 'RGBA'})");

    def __getitem__(self, key):
        """
        Get N3TreeView
        """
        return N3TreeView(self, key)

    def __setitem__(self, key, val):
        N3TreeView(self, key).set(val)

    def __iadd__(self, val):
        self[:] += val
        return self

    def __isub__(self, val):
        self[:] -= val
        return self

    def __imul__(self, val):
        self[:] *= val
        return self

    def __idiv__(self, val):
        self[:] /= val
        return self

        
    # Misc
    @property
    def n_leaves(self):
        return self._all_leaves().shape[0]

    @property
    def n_internal(self):
        return self._n_internal.item()

    @property
    def capacity(self):
        return self.parent_depth.shape[0]

    @property
    def max_depth(self):
        """
        Maximum tree depth - 1
        """
        return torch.max(self.depths).item()
    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return torch.Size((self.n_leaves, self.data_dim))

    def size(self, dim):
        return self.data_dim if dim == 1 else self.n_leaves

    def numel(self):
        return self.data_dim * self.n_leaves

    def __len__(self):
        return self.n_leaves
    
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

# Redirect functions to N3TreeView so you can do tree.depths instead of tree[:].depths
def _redirect_to_n3view():
    redir_props = ['depths', 'lengths', 'lengths_local', 'corners', 'corners_local',
                   'values', 'values_local']
    redir_funcs = ['sample', 'sample_local', 'aux',
            'normal_', 'clamp_', 'uniform_', 'relu_', 'sigmoid_', 'nan_to_num_']
    def redirect_func(redir_func):
        def redir_impl(self, *args, **kwargs):
            return getattr(self[:], redir_func)(*args, **kwargs)
        setattr(AdTree, redir_func, redir_impl)
    for redir_func in redir_funcs:
        redirect_func(redir_func)
    def redirect_prop(redir_prop):
        def redir_impl(self, *args, **kwargs):
            return getattr(self[:], redir_prop)
        setattr(AdTree, redir_prop, property(redir_impl))
    for redir_prop in redir_props:
        redirect_prop(redir_prop)
_redirect_to_n3view()

class WeightAccumulator():
    def __init__(self, tree, op):
        assert op in ['sum', 'max'], 'Unsupported accumulation'
        self.tree = tree
        self.op = op

    def __enter__(self):
        self.tree._lock_tree_structure = True
        self.tree._weight_accum = torch.zeros(
                self.tree.child.shape, dtype=self.tree.data.dtype,
                device=self.tree.data.device)
        self.tree._weight_accum_op = self.op
        self.weight_accum = self.tree._weight_accum
        return self

    def __exit__(self, type, value, traceback):
        self.tree._weight_accum = None
        self.tree._weight_accum_op = None
        self.tree._lock_tree_structure = False

    @property
    def value(self):
        return self.weight_accum

    def __call__(self):
        return self.tree.aux(self.weight_accum)
    
def gen_grid(D):
    """
    Generate D^3 grid centers (coordinates in [0, 1]^3, xyz)

    :param D: resolution
    :return: (D, D, D, 3)
    """
    arr=(torch.arange(D) + 0.5) / D
    X, Y, Z = torch.meshgrid(arr, arr, arr)
    XYZ = torch.stack([X, Y, Z], -1)
    return XYZ