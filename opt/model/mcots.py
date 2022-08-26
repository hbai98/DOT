from turtle import forward
import torch.nn as nn
from svox import N3Tree, VolumeRenderer
from svox.renderer import NDCConfig, _VolumeRenderFunction, _rays_spec_from_rays

import torch
from svox.helpers import DataFormat
from warnings import warn
from einops import rearrange
import torch.nn.functional as F
from tqdm import tqdm
from svox import Rays
from svox.svox import _get_c_extension
import math
import gc
import numpy as np
from collections import namedtuple

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
    


class mcots(nn.Module):
    def __init__(self,
                 radius,
                 center,
                 step_size,
                 init_refine=0,
                 sigma_thresh=0.0,
                 stop_thresh=0.0,
                 epoch_round=5,
                 num_round=50,
                 data_dim=None,
                 explore_exploit=2.,
                 depth_limit=15,
                 device="cpu",
                 data_format="SH9",
                 dtype=torch.float32,
                 writer=None
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
        super(mcots, self).__init__()
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
        self.num_round = num_round
        self.dtype = dtype
        n_nodes = self.player.n_internal
        N = self.player.N

        self.explore_exploit = explore_exploit
        self.instant_reward = None
        self.epoch_round = epoch_round
        self.round = 0
        self.gstep_id = 0
        self.gstep_id_base = 0
        self.writer = writer
        self.basis_rms = None
        self.sigma_thresh = sigma_thresh
        self.stop_thresh = stop_thresh

        # the n_visits for mcst
        self.register_buffer("num_visits", torch.zeros(
            n_nodes, N, N, N, dtype=torch.int32, device=device))
        self.register_buffer("reward", torch.zeros(
            n_nodes, N, N, N, dtype=dtype, device=device))
        # self.init_player(device)

    def select(self, k):
        """Deep first search based on policy value: from root to the tail.
        Select the top k nodes.

        weights: shape-> [n, N, N, N] leaf nodes
        the instant reward is conputed as: weight*exp(-mse)
        """
        sel = (*self.player._all_leaves().T,)
        p_val = self.policy_puct(sel)  # leaf
        vals, idxs = torch.topk(p_val, k)
        idxs = self.player._all_leaves()[idxs]
        self.backtrace(idxs)

        return idxs.long()

    def backtrace(self, idxs):
        idxs_ = idxs.clone()
        pre = 0

        while True:
            sel = (*idxs_.long().T,)
            self.num_visits[sel] += 1
            self.reward[sel] += self.instant_reward[sel]+pre
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

        self.reward /= self.reward.sum()

    def getReward(self, rays, gt, lr_basis_func, delta_func, cuda=True, fast=False):

        render = VolumeRenderer(self.player, step_size=self.step_size,
                                sigma_thresh=self.sigma_thresh, stop_thresh=self.stop_thresh)
        total_rays = rays.origins.size(0)
        B, H, W, C = gt.shape
        batch_size = H*W*5
        batches_per_epoch = (total_rays-1)//batch_size+1
        gt = rearrange(gt, 'B H W C -> (B H W) C')
        device = self.player.data.device
        lr_factor = 1
        tol_stop = 5

        thred_mse = delta_func(self.gstep_id)

        data_stop = HessianCheck(tol_stop, thred_mse)

        while True:
            # shuffle rays and gts
            indexer = torch.randperm(total_rays)
            rays = Rays(rays.origins[indexer],
                        rays.dirs[indexer], rays.viewdirs[indexer])
            gt = gt[indexer]
            pbar = enumerate(range(0, total_rays, batch_size))

            mse = torch.zeros(1, device=device)
            pre_mse = 0
            stats = {"mse": 0.0, "psnr": 0.0, "invsqr_mse": 0.0}
            vals = torch.zeros(self.player.child.size(), device=device)

            for iter_id, batch_begin in pbar:
                self.gstep_id = iter_id + self.gstep_id_base
                batch_end = min(batch_begin + batch_size, total_rays)
                batch_origins = rays.origins[batch_begin: batch_end]
                batch_dirs = rays.dirs[batch_begin: batch_end]
                batch_viewdir = rays.viewdirs[batch_begin: batch_end]
                rgb_gt = gt[batch_begin:batch_end]
                ray = Rays(batch_origins, batch_dirs, batch_viewdir)

                lr = lr_basis_func(self.gstep_id*lr_factor)
                thred_mse = delta_func(self.gstep_id)

                with self.player.accumulate_weights(op="sum") as accum:
                    res = render.forward(ray, cuda=cuda, fast=fast)

                val = accum.value
                val /= val.sum()
                
                mse = F.mse_loss(rgb_gt, res)
                self.player.zero_grad()
                mse.backward()
                self.optim_basis_step(lr)

                vals += val
                # Stats
                mse_num: float = mse.detach().item()
                psnr = -10.0 * math.log10(mse_num)
                stats['mse'] += mse_num
                stats['psnr'] += psnr
                stats['invsqr_mse'] += 1.0 / mse_num ** 2

            self.instant_reward = vals
            stop_ = data_stop(pre_mse, stats['mse'])
            self.writer.add_scalar('train/hessian_mse',
                                   data_stop.hessian, self.gstep_id)
            self.writer.add_scalar(
                'train/mse_thred_count', data_stop.counter, self.gstep_id)

            pre_mse = stats['mse']
            self.writer.add_scalar('train/lr', lr, self.gstep_id)
            # self.writer.add_scalar('train/delta_ctb', delta_ctb, self.gstep_id)
            self.writer.add_scalar('train/thred_mse', thred_mse, self.gstep_id)
            for stat_name in stats:
                stat_val = stats[stat_name] / batches_per_epoch
                self.writer.add_scalar(
                    f'train/{stat_name}', stat_val, self.gstep_id)
                stats[stat_name] = 0

            self.gstep_id_base += batches_per_epoch

            if stop_:
                # self.evaluate()
                break

    def expand(self, idxs):
        # group expansion 
        i = idxs.size(0)
        idxs = (*idxs.long().T,)
        res = self.player.refine(sel=idxs)
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

    def run_a_round(self, rays, gt):

        lr_basis = 1e-1
        lr_basis_final = 5e-4
        lr_basis_delay_steps = 0
        lr_basis_delay_mult = 1e-2
        lr_basis_decay_steps = 1e5
        lr_basis_func = get_expon_func(lr_basis, lr_basis_final, lr_basis_delay_steps,
                                       lr_basis_delay_mult, lr_basis_decay_steps)

        delta_data_init = 5e-4
        delta_data_end = 5e-5
        delta_data_decay_steps = 1e5

        sample_rate_init = 5e-1
        sample_rate_end = 1e-1
        delta_data_decay_steps = 1e5

        explore_exploit_end = 1e-1

        delta_data_func = get_expon_func(delta_data_init, delta_data_end, lr_basis_delay_steps,
                                         lr_basis_delay_mult, delta_data_decay_steps)
        delta_sample_rate_func = get_expon_func(sample_rate_init, sample_rate_end, lr_basis_delay_steps,
                                                lr_basis_delay_mult, delta_data_decay_steps)
        delta_explore_exploit = get_expon_func(self.explore_exploit, explore_exploit_end, lr_basis_delay_steps,
                                               lr_basis_delay_mult, delta_data_decay_steps)

        self.writer.add_scalar(
            f'train/num_nodes', self.player.n_leaves, self.gstep_id)
        self.writer.add_scalar(
            f'train/depth', self.player.get_depth(), self.gstep_id)
        self.writer.add_image(
            f'train/gt', gt[0], self.gstep_id, dataformats='HWC')
        res = True
        # tol_stop = 3
        # thred_reward=1e-1
        # prune_check = PruneCheck(tol_stop, thred_reward)
        N = self.player.N
        with tqdm(total=self.player.depth_limit) as pbar:
            pbar.update(self.player.get_depth().item())
            while res:
                rate = delta_sample_rate_func(self.gstep_id)
                self.explore_exploit = delta_explore_exploit(self.gstep_id)

                k = self.player.n_leaves*rate
                k = int(max(1, k))
                # stimulate
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                self.getReward(rays, gt, lr_basis_func, delta_data_func)
                # end.record()
                # torch.cuda.synchronize()
                # print(f'getReward:{start.elapsed_time(end)}')
                
                depth = self.player.get_depth()
                
                sigma = torch.nan_to_num(_SOFTPLUS_M1(self.player[:,-1]))
                self.writer.add_histogram(f'train/sigma_dist_{self.gstep_id}', sigma , 0)
                sel = (*self.player._all_leaves().T, )
                self.writer.add_scalar(f'sigma/ctb_1', self.instant_reward[sel][sigma<1].sum()/self.instant_reward[sel].sum() , self.gstep_id)
                self.writer.add_scalar(f'sigma/ctb_1e-1', self.instant_reward[sel][sigma<0.1].sum()/self.instant_reward[sel].sum() , self.gstep_id)
                self.writer.add_scalar(f'sigma/ctb_1e-2', self.instant_reward[sel][sigma<0.01].sum()/self.instant_reward[sel].sum() , self.gstep_id)
                self.writer.add_scalar(f'sigma/num_1',(sigma<1).nonzero().size(0)/sigma.size(0), self.gstep_id)
                self.writer.add_scalar(f'sigma/num_1e-1',(sigma<0.1).nonzero().size(0)/sigma.size(0), self.gstep_id)
                self.writer.add_scalar(f'sigma/num_1e-2',(sigma<0.01).nonzero().size(0)/sigma.size(0), self.gstep_id)


                # select
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                idxs = self.select(k)
                # end.record()
                # torch.cuda.synchronize()
                # print(f'select:{start.elapsed_time(end)}')
                # expand
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                res = self.expand(idxs)
                # end.record()
                # torch.cuda.synchronize()
                # print(f'expand:{start.elapsed_time(end)}')
                # prune
                # prune_check(self.player, self.reward)

                self.writer.add_scalar(
                    f'train/num_nodes', self.player.n_leaves, self.gstep_id)
                self.writer.add_scalar(
                    f'train/depth', self.player.get_depth(), self.gstep_id)
                self.writer.add_scalar(
                    f'train/sample_weight', rate, self.gstep_id)
                self.writer.add_scalar(
                    f'train/explore_exploit', self.explore_exploit, self.gstep_id)
                # self.writer.add_scalar(f'train/prune', prune_check.num_prune, self.gstep_id)
                # log

                delta_depth = (self.player.get_depth()-depth).item()

                if delta_depth != 0:
                    # prune
                    # thred = delta_ctb_func(self.gstep_id)
                    # self.prune(thred)
                    # self.writer.add_scalar(f'train/thred_ctb',thred, self.gstep_id)
                    render = VolumeRenderer(self.player, step_size=self.step_size,
                                            sigma_thresh=self.sigma_thresh, stop_thresh=self.stop_thresh)
                    B, H, W, C = gt.shape
                    id_ = H*W
                    ray = Rays(rays.origins[:id_],
                               rays.dirs[:id_], rays.viewdirs[:id_])
                    im = rearrange(render.forward(
                        ray), '(H W) C -> H W C', H=H)
                    self.writer.add_image(
                        f'train/round_{self.round}_depth_{depth}', im, self.gstep_id, dataformats='HWC')
                    # print(self.instant_visits)
                    # print(self.num_visits)
                    pbar.update(delta_depth)

    def policy_puct(self, sel):
        """Return the policy head value to guide the sampling

        P-UCT = total_reward(s, a)+ C*instant_reward(s,a)/(1+num_visits(s))

        where s is the state, a is the action.

        Args:
            instant_reward is the sum array[n, x, y, z] of rewards after backpropagtion for node_idx
        Returns:
            p-uct value
        """

        # res = self.reward[sel] + self.explore_exploit*self.instant_reward[sel]/torch.exp((1+self.num_visits[sel]))
        res = self.reward[sel] + self.explore_exploit * \
            self.instant_reward[sel]/(1+self.num_visits[sel])
        # res = self.reward[sel] + self.explore_exploit*self.instant_reward[sel]

        return res

    def optim_basis_step(self, lr: float, beta: float = 0.9, epsilon: float = 1e-8,
                         optim: str = 'rmsprop'):
        """
        Execute RMSprop/SGD step on SH
        """
        assert (
            _C is not None and self.player.data.is_cuda
        ), "CUDA extension is currently required for optimizers"

        if optim == 'rmsprop':
            if self.basis_rms is None or self.basis_rms.shape != self.player.data.shape:
                del self.basis_rms
                self.basis_rms = torch.zeros_like(self.player.data.data)
            self.basis_rms.mul_(beta).addcmul_(
                self.player.data.grad, self.player.data.grad, value=1.0 - beta)
            denom = self.basis_rms.sqrt().add_(epsilon)
            self.player.data.data.addcdiv_(
                self.player.data.grad, denom, value=-lr)
        elif optim == 'sgd':
            self.player.data.grad.mul_(lr)
            self.player.data.data -= self.player.data.grad
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')
        self.player.data.grad.zero_()


def get_expon_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
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