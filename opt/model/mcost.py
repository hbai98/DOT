import torch.nn as nn
from svox import N3Tree, VolumeRenderer
from svox.renderer import NDCConfig

import torch
from svox.helpers import DataFormat
from warnings import warn
from svox.svox import _get_c_extension
import numpy as np
from collections import defaultdict

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

    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing capacity
        """
        cap_needed = max(cap_needed, int(
            self.capacity * (self.geom_resize_fact - 1.0)))
        may_oom = self.capacity + cap_needed > 1e7  # My CPU Memory is limited
        if may_oom:
            print('Potential OOM: shift to cpu. It will be extremely slow.')
            # Potential OOM prevention hack
            self.data = nn.Parameter(self.data.cpu())

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

    def _refine_at(self, intnode_idx, xyzi):
        """
        Advanced: refine specific leaf node. Mostly for testing purposes.

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
        # the data is initalized as zero for furhter processing(expansion)
        # self.data.data[filled, :, :, :] = self.data.data[intnode_idx, xi, yi, zi]
        # the data is kept as original for further inferencing
        # self.data.data[intnode_idx, xi, yi, zi] = 0
        self._n_internal += 1
        self._invalidate()
        return resized
    


class Mcost(nn.Module):
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
        super(Mcost, self).__init__()
        # the octree is always kept to record the overview
        # the sh, density value are updated constantly
        self.radius = radius
        self.center = center
        self.data_format = data_format
        self.player = SMCT(radius=radius,
                           center=center,
                           data_format=data_format,
                           init_refine=init_refine,
                           data_dim=data_dim,
                           depth_limit=depth_limit,
                           dtype=dtype)
        self.step_size = step_size
        self.dtype = dtype
        n_nodes = self.player.n_internal
        N = self.player.N

        self.round = 0
        self.gstep_id = 0
        self.gstep_id_base = 0
        self.basis_rms = None
        self.sigma_thresh = sigma_thresh
        self.stop_thresh = stop_thresh
        self.policy = policy
        self.p_sel = p_sel

        if policy == 'pareto':
            # the n_visits for mcst
            self.register_buffer("num_visits", torch.zeros(
                n_nodes, N, N, N, dtype=torch.int32, device=device))
            self.register_buffer("reward", torch.zeros(
                n_nodes, N, N, N, len(p_sel), dtype=dtype, device=device))

    def select(self, k, reward):
        """Deep first search based on policy value: from root to the tail.
        Select the top k nodes.

        reward:  leaves, objectives
        the instant reward is conputed as: weight*exp(-mse)
        """
        
        if self.policy == 'greedy':
            sel = [*self.player._all_leaves().long().T,]
            p_val = reward[sel, 0] # only MSE
            vals, idxs = torch.topk(p_val, k)
            idxs = self.player._all_leaves()[idxs]
            return idxs.long()
        elif self.policy == 'pareto':
            device = self.player.data.device
            D = torch.Tensor([reward.size(-1)])
            res = []
            reward = reward[:,:len(self.p_sel)]  
            
            def DFS(todo, idxs_pq):
                # print(f'todo:{todo}')
                while True:
                    if len(todo)==0 or len(res) >= k:
                        break
                    
                    root = todo.pop(0)
                    idxs_pq.update({k:idxs_pq[k]-1 for k in idxs_pq if idxs_pq[k]!=0})
                    
                    nid = self.player._pack_index(torch.tensor(root).unsqueeze(0))
                    if self.player.child[(*root,)]==0:
                        res.append(root)
                    else:
                        # internal nodes 
                        sel = (self.player.parent_depth[1:,0]==nid.to(device)).nonzero(as_tuple=True)[0].item()
                        n_visits = self.num_visits[sel]+1
                        if len(self.p_sel) == 1:
                            p_val = self.reward[sel].to(device)+\
                                torch.sqrt(torch.log((n_visits.sum())).to(device)/(self.num_visits[sel]+1).to(device))
                        else:
                            p_val = self.reward[sel].to(device) +\
                                torch.sqrt((4*torch.log((n_visits.sum()).to(device))+torch.log(D).to(device))/(2*(self.num_visits[sel]+1).to(device)))
                        p_val = p_val.squeeze(0) #[2,2,2,p_sel]
                        # pareto optimal set (equals the intersection)
                        idxs = [torch.nonzero(p_val[...,d]==p_val[...,d].max()) for d in range(p_val.size(-1))]
                        idxs = torch.cat(idxs).unique(dim=0).to(device)
                        _add = (torch.ones(idxs.size(0), 1)*sel).to(device)
                        idxs = torch.cat([_add, idxs], dim=-1).int().tolist()
                        # pick one at random 
                        # insert into a priority queue ranked by the depth of tree
                        depth = self.player.parent_depth[sel,1].item()
                        _sel = np.random.randint(len(idxs))
                        
                        # print(todo)
                        # print(depth)
                        # print(idxs_pq)
                        # insert 
                        todo.insert(0, idxs.pop(_sel)) # deep-first
                        todo[idxs_pq[depth+1]:idxs_pq[depth+1]-1] = idxs # children on the next layer
                        # update idxs after insertion
                        if (depth+2) not in idxs_pq:
                            idxs_pq[depth+2]=idxs_pq[depth+1] 
                        idxs_pq.update({k:idxs_pq[k]+len(idxs) for k in idxs_pq if k > depth+1})
                        # print(todo)
                        # print(depth)
                        # print(idxs_pq)
                                          
            idxs_pq = dict()
            for k in range(1, self.player.get_depth()+2):
                idxs_pq[k] = 0         
            # select all leaves of the root of the tree 
            sel = torch.nonzero(torch.ones(1, self.player.N, self.player.N, self.player.N))[1:].tolist()
            idxs_pq[2] = self.player.N**3
            # create the priority queue by a dict
            DFS(sel, idxs_pq)
            sel = torch.tensor(res)
            self.backtrace(reward, sel)
            return sel
    
    def backtrace(self, reward, idxs):
        idxs_ = idxs.clone()
        # initalize rewards on selected leaves 
        idxs_sel = self.player._pack_index(idxs_)
        idxs_leaves = self.player._pack_index(self.player._all_leaves())
        sel_ = [l in idxs_sel for l in idxs_leaves]
        pre = reward[sel_]

        while True:
            sel = (*idxs_.long().T,)
            self.num_visits[sel] += 1
            # print(pre.shape)
            # print(self.reward[sel].shape)
            # back-propagate rewards
            self.reward[sel] += (pre-self.reward[sel])/(1+self.num_visits[sel].unsqueeze(-1))
            pre = self.reward[sel]

            nid = idxs_[:, 0]
            sel_ = nid > 0
            nid = nid[sel_]
            pre = pre[sel_]

            if nid.size(0) == 0:
                break
            else:
                idxs_ = self.player._unpack_index(
                    self.player.parent_depth[nid.long(), 0])
        
        
        
    def _reward(self, rewards):
        res = rewards/rewards.sum()
        return res

    def _volumeRenderer(self):
       return VolumeRenderer(self.player, step_size=self.step_size,
                                sigma_thresh=self.sigma_thresh, stop_thresh=self.stop_thresh)

    def expand(self, idxs):
        # group expansion
        i = idxs.size(0)
        idxs = (*idxs.long().T,)
        res = self.player.refine(sel=idxs)
        
        if self.policy == 'pareto':
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


    def optim_basis_step(self, lr_sigma: float, lr_sh: float, beta: float=0.9, epsilon: float = 1e-8,
                         optim: str = 'rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        
        data = self.player.data
        
        assert (
            _C is not None and data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        
        if optim == 'rmsprop':
            if self.basis_rms is None or self.basis_rms.shape != data.shape:
                del self.basis_rms
                self.basis_rms = torch.zeros_like(data.data)
            self.basis_rms.mul_(beta).addcmul_(data.grad, data.grad, value = 1.0 - beta)
            denom = self.basis_rms.sqrt().add_(epsilon)
            data.data[...,-1].addcdiv_(data.grad[...,-1], denom[...,-1], value=-lr_sigma)
            data.data[...,:-1].addcdiv_(data.grad[...,:-1], denom[...,:-1], value=-lr_sh)
            
        elif optim == 'sgd':
            data.grad[...,-1].mul_(lr_sigma)
            data.grad[...,:-1].mul_(lr_sh)
            data.data -=data.grad
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')
        data.grad.zero_()


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


class HessianCheck():
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.pre_delta = 0
        self.hessian = None

    def __call__(self, cur, pre):
        delta = np.abs(pre - cur)
        self.hessian = np.abs(delta-self.pre_delta)
        if self.hessian < self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
                return True
        self.pre_delta = delta
        return False



# class PruneCheck():
#     def __init__(self, tolerance=5, min_delta=0):
#         self.tolerance = tolerance
#         self.min_delta = min_delta
#         self.counter = dict()
#         self.num_prune = 0

#         self.pre_frontier = None
#         self.pre_reward = None

#     def __call__(self, player, reward):
#         if self.pre_frontier is None or self.pre_reward is None:
#             self.pre_frontier = player._frontier.cpu().numpy()
#             self.pre_reward = reward
#             return

#         preFron = self.pre_frontier
#         curFron = player._frontier.cpu().numpy()
#         # only check the frontier nodes in intersection
#         nids = torch.Tensor(np.intersect1d(preFron, curFron)).long()
#         if len(nids) == 0:
#             return

#         parent_sel = (*player._unpack_index(player.parent_depth[nids, 0]).long().T,)
#         pre_reward = self.pre_reward[parent_sel]
#         cur_reward = reward[parent_sel]
#         delta = torch.abs(pre_reward-cur_reward)

#         for i, nid in enumerate(nids):
#             if delta[i] < self.min_delta:
#                 if nid in self.counter:
#                     self.counter[nid] +=1
#                     if self.counter[nid] >= self.tolerance:
#                         self.player.merge(nid == curFron)
#                         self.num_prune += 1
#                         self.counter[nid] = 0
#                 else:
#                     self.counter[nid] = 1

#         self.pre_frontier = player._frontier.cpu().numpy()
#         self.pre_reward = reward
#         return


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


def _SOFTPLUS_M1(x):
    return torch.log(torch.exp(x-1)+1)
