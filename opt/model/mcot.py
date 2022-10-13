import torch.nn as nn
from svox import N3Tree, VolumeRenderer, LocalIndex
from svox.renderer import NDCConfig, _rays_spec_from_rays
from torch.nn.functional import softplus
import torch
from svox.helpers import DataFormat
from warnings import warn
from svox.svox import _get_c_extension
import numpy as np
from .utils import pareto_2d, _SOFTPLUS_M1, asoftplus
from einops import rearrange
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import sys
_C = _get_c_extension()



class SMCT(N3Tree):
    def __init__(self, N=2, data_dim=None, depth_limit=10,
                 init_reserve=1, init_refine=0, geom_resize_fact=1.0,
                 radius=0.5, center=[0.5, 0.5, 0.5],
                 data_format="SH9",
                 extra_data=None,
                 device="cuda",
                 dtype=torch.float32,
                 map_location=None
                 ):
        """
        Construct N^3 Tree: spatial mento carlo tree
        :param pre_data: torch.Tensor, the previous record of data. if None, the data is registered 
        as the buffer for recording purpose. 

        :param N: int branching factor N
        :param data_dim: int size of data stored at each leaf (NEW in 0.2.28: optional if data_format other than RGBA is given).
                        If data_format = "RGBA" or empty, this defaults to 4.
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
        super(N3Tree, self).__init__()
        assert N >= 2
        assert depth_limit >= 0
        self.N: int = N

        if map_location is not None:
            warn('map_location has been renamed to device and may be removed')
            device = map_location
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'

        self.data_format = DataFormat(
            data_format) if data_format is not None else None
        self.data_dim: int = data_dim
        self._maybe_auto_data_dim()
        del data_dim

        if init_refine > 0:
            for i in range(1, init_refine + 1):
                init_reserve += (N ** i) ** 3

        self.register_parameter("data",
                                nn.Parameter(torch.empty(init_reserve, N, N, N, self.data_dim, dtype=dtype, device=device)))
        nn.init.constant_(self.data, 0.01)

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
            self.register_buffer("extra_data", extra_data.to(
                dtype=dtype, device=device))
        else:
            self.extra_data = None

        self._ver = 0
        self._invalidate()
        self._lock_tree_structure = False
        self._weight_accum = None
        self._weight_accum_op = None

        self.refine(repeats=init_refine)

    def get_depth(self):
        return torch.max(self.parent_depth[:, 1])

    def _unpack_index(self, flat):
        t = []
        for i in range(3):
            t.append(flat % self.N)
            flat = torch.div(flat, self.N, rounding_mode='trunc')
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)

    def _invalidate(self):
        self._ver += 1
        self._last_all_leaves = None
        self._last_frontier = None
        self._last_all_inter = None

    def _all_internals(self):
        if self._last_all_inter is None:
            self._last_all_inter = (self.child[
                :self.n_internal] != 0).nonzero(as_tuple=False).cpu()
        return self._last_all_inter

    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing capacity
        """
        cap_needed = max(cap_needed, int(
            self.capacity * (self.geom_resize_fact - 1.0)))
        # may_oom = self.capacity + cap_needed > 2.8e7  # My CPU Memory is limited
        # if may_oom:
        #     # print('Potential OOM: shift to cpu. It will be extremely slow.')
        #     print('Out of memory. Stop sampling.')
        #     # Potential OOM prevention hack
        #     # self.data = nn.Parameter(self.data.cpu())
        #     return False
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(h)
        need = sys.getsizeof(torch.zeros((cap_needed, *self.data.data.shape[1:]),
                                                        dtype=self.data.dtype,).storage())
        if info.free < need:
            print('Memory runs out, and stop sampling.')
            return False
                
        self.data = nn.Parameter(torch.cat((self.data.data,
                                            torch.zeros((cap_needed, *self.data.data.shape[1:]),
                                                        dtype=self.data.dtype,
                                                        device=self.data.device)), dim=0))
        self.child = torch.cat((self.child,
                                torch.zeros((cap_needed, *self.child.shape[1:]),
                                            dtype=self.child.dtype,
                                            device=self.data.device)))
        self.parent_depth = torch.cat((self.parent_depth,
                                       torch.zeros((cap_needed, *self.parent_depth.shape[1:]),
                                                   dtype=self.parent_depth.dtype,
                                                   device=self.data.device)))
        return True
    
    def _spec(self, world=True):
        """
        Pack tree into a TreeSpec (for passing data to C++ extension)
        """
        tree_spec = _C.TreeSpec()
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
                    resized = self._resize_add_cap(cap_needed)
                    # resized = True
                    if not resized:
                        return resized

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
    
    # def _refine_at(self, intnode_idx, xyzi):
    #     """
    #     Advanced: refine specific leaf node. Mostly for testing purposes.

    #     :param intnode_idx: index of internal node for identifying leaf
    #     :param xyzi: tuple of size 3 with each element in :code:`{0, ... N-1}`
    #                 in xyz orde rto identify leaf within internal node

    #     """
    #     if self._lock_tree_structure:
    #         raise RuntimeError("Tree locked")
    #     assert min(xyzi) >= 0 and max(xyzi) < self.N
    #     if self.parent_depth[intnode_idx, 1] >= self.depth_limit:
    #         return

    #     xi, yi, zi = xyzi
    #     if self.child[intnode_idx, xi, yi, zi] != 0:
    #         # Already has child
    #         return

    #     resized = False
    #     filled = self.n_internal
    #     if filled >= self.capacity:
    #         self._resize_add_cap(1)
    #         resized = True

    #     self.child[filled] = 0
    #     self.child[intnode_idx, xi, yi, zi] = filled - intnode_idx
    #     depth = self.parent_depth[intnode_idx, 1] + 1
    #     self.parent_depth[filled, 0] = self._pack_index(torch.tensor(
    #         [[intnode_idx, xi, yi, zi]], dtype=torch.int32))[0]
    #     self.parent_depth[filled, 1] = depth
    #     # the data is initalized as zero for furhter processing(expansion)
    #     # self.data.data[filled, :, :, :] = self.data.data[intnode_idx, xi, yi, zi]
    #     # the data is kept as original for further inferencing
    #     # self.data.data[intnode_idx, xi, yi, zi] = 0
    #     self._n_internal += 1
    #     self._invalidate()
    #     return resized


class MCOT(nn.Module):
    def __init__(self,
                 radius,
                 center,
                 step_size,
                 init_refine=0,
                 sigma_thresh=0.0,
                 stop_thresh=0.0,
                 data_dim=None,
                 depth_limit=12,
                 device="cpu",
                 data_format="SH9",
                 dtype=torch.float32,
                 policy='greedy',
                 p_sel=['ctb'],
                 density_softplus=True,
                 record=True,
                 init_weight_sparsity_loss=0.01,
                 init_tv_sigma_loss=0.01,
                 init_tv_color_loss=0.01,
                 ):
        """Main mcts based octree data structure

        Args:
            radius: float or List[float, float, float], 1/2 side length of cube (possibly in each dim).
            center:  float or List[float, float, float], the center of cube.
            step_size: float, render step size (in voxel size units)
            data_format (str, optional): _description_. Defaults to "SH9".
            num_round: int, the number of round to play. 
            depth_limit: int maximum depth  of tree to stop branching/refining
                         Note that the root is at depth -1.
                         Size :code:`N^[-10]` leaves (1/1024 for octree) for example
                         are depth 9. :code:`max_depth` applies to the same
                         depth values.
        """
        super(MCOT, self).__init__()
        # the octree is always kept to record the overview
        # the sh, density value are updated constantly
        self.radius = radius
        self.center = center
        self.data_format = data_format
        self.tree = SMCT(radius=radius,
                         center=center,
                         data_format=data_format,
                         init_refine=init_refine,
                         data_dim=data_dim,
                         depth_limit=depth_limit,
                         dtype=dtype)
        self.step_size = step_size
        self.dtype = dtype
        n_nodes = self.tree.n_internal
        N = self.tree.N

        self.round = 0
        self.gstep_id = 0
        self.gstep_id_base = 0
        self.basis_rms = None
        self.sigma_thresh = sigma_thresh
        self.stop_thresh = stop_thresh
        self.policy = policy
        self.p_sel = p_sel
        self.density_softplus = density_softplus
        self.record = record

        if init_weight_sparsity_loss is not None:
            _ = asoftplus(torch.tensor(init_weight_sparsity_loss))
            self.w_sparsity = torch.nn.Parameter(torch.tensor([_], device=device))
        if init_tv_sigma_loss is not None:
            _ = asoftplus(torch.tensor(init_tv_sigma_loss))
            self.w_sigma_tv = torch.nn.Parameter(torch.tensor([_], device=device)) 
        if init_tv_color_loss is not None:
            _ = asoftplus(torch.tensor(init_tv_color_loss))
            self.w_color_tv = torch.nn.Parameter(torch.tensor([_], device=device))        
        if record:
            self.register_buffer("num_visits", torch.zeros(
                n_nodes, N, N, N, dtype=torch.int32, device=device))
            self.register_buffer("reward", torch.zeros(
                n_nodes, N, N, N, len(p_sel), dtype=dtype, device=device))

    def merge(self, nids):
        device = self.tree.data.device
        nids = nids.to(device)
        idxs = [f in nids for f in self.tree._frontier]
        self.tree.merge(idxs)
    @property
    def _w_sparsity(self):
        if self.w_sparsity is not None:
            return softplus(self.w_sparsity)
    @property
    def _w_color_tv(self):
        if self.w_sparsity is not None:
            return softplus(self.w_color_tv)
    @property
    def _w_sigma_tv(self):
        if self.w_sparsity is not None:
            return softplus(self.w_sigma_tv)       
            
    def reweight_rays(self, rays, error, opt):
        assert error.size(0) == rays.origins.size(0)
        assert error.is_cuda 
        self.tree._weight_accum = None
        with self.tree.accumulate_weights(op="sum") as accum:
            _C.reweight_rays(self.tree._spec(), _rays_spec_from_rays(rays), opt, error)
        return accum.value      
        
    def select(self, max_sel, reward, rw_idxs):
        """Deep first search based on policy value: from root to the tail.
        Select the top k nodes.

        reward:  leaves, objectives
        the instant reward is conputed as: weight*exp(-mse)
        """

        if self.policy == 'greedy':
            p_val = reward  # only MSE
            sel = min(p_val.size(0), max_sel)
            vals, idxs = torch.topk(p_val, sel)
            idxs = rw_idxs[idxs]
            return idxs
        elif self.policy == 'pareto':
            device = self.tree.data.device
            D = torch.Tensor([reward.size(-1)])
            res = []
            reward = reward[:, :len(self.p_sel)]

            def DFS(todo, idxs_pq):
                # print(f'todo:{todo}')
                while True:
                    if len(todo) == 0 or len(res) >= max_sel:
                        # print(f'num_leaves:{self.player.n_leaves}')
                        print(f'Select {len(res)}/{max_sel}')
                        # print(len(todo))
                        break

                    root = todo.pop(0)
                    idxs_pq.update(
                        {k: idxs_pq[k]-1 for k in idxs_pq if idxs_pq[k] != 0})

                    nid = self.tree._pack_index(
                        torch.tensor(root).unsqueeze(0))
                    if self.tree.child[(*root,)] == 0:
                        res.append(root)
                    else:
                        # internal nodes
                        sel = (self.tree.parent_depth[1:, 0] == nid.to(
                            device)).nonzero(as_tuple=True)[0].item()
                        # root as parent
                        n_visits = self.num_visits[sel]+1
                        n_visits = n_visits.unsqueeze(-1).to(device)

                        if len(self.p_sel) == 1:
                            p_val = self.reward[sel].to(device) +\
                                torch.sqrt(torch.log((n_visits.sum())).to(
                                    device)/n_visits)

                        else:
                            p_val = self.reward[sel].to(device) +\
                                torch.sqrt((4*torch.log((n_visits.sum()).to(device))+torch.log(
                                    D).to(device))/(2*n_visits))

                        # pareto optimal set (equals thenbvm intersection)
                        # p_sel  [2,2,2,p_sel]
                        if len(self.p_sel) == 1:
                            p_val = p_val.squeeze(-1)
                            idxs = (p_val == p_val.max()).nonzero().to(device)
                        else:
                            idxs = pareto_2d(
                                rearrange(p_val.cpu().detach().numpy(), 'X Y Z N -> (X Y Z) N'))
                            idxs = self.tree._unpack_index(
                                torch.tensor(idxs).to(device))
                        _add = (torch.ones(idxs.size(0), 1)*sel).to(device)
                        idxs = torch.cat([_add, idxs], dim=-1).int().tolist()
                        # pick one at random
                        # insert into a priority queue ranked by the depth of tree
                        depth = self.tree.parent_depth[sel, 1].item()
                        _sel = np.random.randint(len(idxs))

                        # print(todo)
                        # print(depth)
                        # print(idxs_pq)
                        # insert
                        todo.insert(0, idxs.pop(_sel))  # deep-first
                        # children on the next layer
                        todo[idxs_pq[depth+1]:idxs_pq[depth+1]-1] = idxs
                        # update idxs after insertion
                        if (depth+2) not in idxs_pq:
                            idxs_pq[depth+2] = idxs_pq[depth+1]
                        idxs_pq.update({k: idxs_pq[k]+len(idxs)
                                       for k in idxs_pq if k > depth+1})
                        # print(todo)
                        # print(depth)
                        # print(idxs_pq)

            idxs_pq = dict()
            for k in range(1, self.tree.get_depth()+2):
                idxs_pq[k] = 0
            # select all leaves of the root of the tree
            sel = torch.nonzero(torch.ones(1, self.tree.N, self.tree.N, self.tree.N))[
                1:].tolist()
            idxs_pq[2] = self.tree.N**3
            # create the priority queue by a dict
            DFS(sel, idxs_pq)
            sel = torch.tensor(res)
            return sel

    def backtrace(self, reward, idxs):
        idxs_ = idxs.clone()
        # reward
        reward = reward[:, :len(self.p_sel)]
        # initalize rewards on selected leaves
        idxs_sel = self.tree._pack_index(idxs_)
        idxs_leaves = self.tree._pack_index(self.tree._all_leaves())
        sel_ = [l in idxs_sel for l in idxs_leaves]
        pre = reward[sel_]

        while True:
            sel = (*idxs_.long().T,)
            self.num_visits[sel] += 1
            # print(pre.shape)
            # print(self.reward[sel].shape)
            # back-propagate rewards
            self.reward[sel] += (pre-self.reward[sel]) / \
                (1+self.num_visits[sel].unsqueeze(-1))
            pre = self.reward[sel]

            nid = idxs_[:, 0]
            sel_ = nid > 0
            nid = nid[sel_]
            pre = pre[sel_]

            if nid.size(0) == 0:
                break
            else:
                idxs_ = self.tree._unpack_index(
                    self.tree.parent_depth[nid.long(), 0])


    def _volumeRenderer(self):
        return VolumeRenderer(self.tree, step_size=self.step_size,
                              sigma_thresh=self.sigma_thresh, stop_thresh=self.stop_thresh, density_softplus=self.density_softplus)

    def expand(self, idxs, repeats):
        # group expansion
        i = idxs.size(0)
        idxs = (*idxs.long().T,)
        res = self.tree.refine(sel=idxs, repeats=repeats)

        # # clean up memory on internal nodes' data
        # sel = [*self.tree._all_internals().T.long(),]
        # self.tree.data[sel] = None

        if self.record:
            self.num_visits = torch.cat((self.num_visits,
                                        torch.zeros((i, *self.num_visits.shape[1:]),
                                                    dtype=self.num_visits.dtype,
                                                    device=self.num_visits.device)
                                         ))
            self.reward = torch.cat((self.reward,
                                    torch.zeros((i, *self.reward.shape[1:]),
                                                dtype=self.reward.dtype,
                                                device=self.reward.device)
                                     ))
        return res

    def optim_basis_step(self, lr: float, beta: float = 0.9, epsilon: float = 1e-8,
                             optim: str = 'rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        data = self.tree.data
        
        assert (
            _C is not None and data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        if optim == 'rmsprop':
            if self.basis_rms is None or self.basis_rms.shape != data.shape:
                del self.basis_rms
                self.basis_rms = torch.zeros_like(data.data)
            self.basis_rms.mul_(beta).addcmul_(
                data.grad, data.grad, value=1.0 - beta)
            denom = self.basis_rms.sqrt().add_(epsilon)
            data.data.addcdiv_(
                data.grad, denom, value=-lr)
        elif optim == 'sgd':
            data.grad.mul_(lr)
            data.data -= data.grad
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

        data.grad.zero_()

    def optim_basis_all_step(self, lr_sigma: float, lr_sh: float, beta: float = 0.9, epsilon: float = 1e-8,
                             optim: str = 'rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """

        data = self.tree.data

        assert (
            _C is not None and data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        if optim == 'rmsprop':
            if self.basis_rms is None or self.basis_rms.shape != data.shape:
                del self.basis_rms
                self.basis_rms = torch.zeros_like(data.data)
            self.basis_rms.mul_(beta).addcmul_(
                data.grad, data.grad, value=1.0 - beta)
            denom = self.basis_rms.sqrt().add_(epsilon)
            data.data[..., -1].addcdiv_(data.grad[..., -1],
                                        denom[..., -1], value=-lr_sigma)
            data.data[..., :-1].addcdiv_(data.grad[..., :-1],
                                         denom[..., :-1], value=-lr_sh)

        elif optim == 'sgd':
            data.grad[..., -1].mul_(lr_sigma)
            data.grad[..., :-1].mul_(lr_sh)
            data.data -= data.grad
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')

    def _sigma(self):
        sel = (*self.tree._all_leaves().long().T, )
        val = self.tree.data[sel][..., -1]
        if self.density_softplus:
            val = _SOFTPLUS_M1(val)
        return val
    def _color(self):
        sel = (*self.tree._all_leaves().long().T, )
        val = self.tree.data[sel][..., :-1]
        return val
def get_expon_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class VolumeRenderer(VolumeRenderer):
    def __init__(self, tree, step_size: float = 0.001,
                 background_brightness: float = 1,
                 ndc: NDCConfig = None,
                 min_comp: int = 0,
                 max_comp: int = -1,
                 density_softplus: bool = False,
                 rgb_padding: float = 0,
                 stop_thresh=0.0,
                 sigma_thresh=1e-3):
        super().__init__(tree, step_size, background_brightness,
                         ndc, min_comp, max_comp, density_softplus, rgb_padding)
        self.sigma_thresh = sigma_thresh
        self.stop_thresh = stop_thresh
