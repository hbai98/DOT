import numpy as np
from svox.svox import _get_c_extension
from svox.renderer import _rays_spec_from_rays, _make_camera_spec
import torch
from skimage.filters.thresholding import threshold_li, threshold_otsu, threshold_yen, threshold_minimum, threshold_triangle
from skimage.filters._gaussian import gaussian
import torch.nn as nn
from svox import N3Tree
from svox.helpers import DataFormat
from warnings import warn

_C = _get_c_extension()

def reweight_rays(tree, rays, error, opt):
    assert error.size(0) == rays.origins.size(0)
    assert error.is_cuda 
    tree._weight_accum = None
    with tree.accumulate_weights(op="sum") as accum:
        _C.reweight_rays(tree._spec(), _rays_spec_from_rays(rays), opt, error)
    return accum.value 

def reweight_image(tree, error, c2w, opt, width=800, height=800, fx=1111.111, fy=None):
    assert error.is_cuda
    tree._weight_accum = None
    if fy is None:
        fy = fx
    with tree.accumulate_weights(op="sum") as accum:
        _C.reweight_image(tree._spec(), _make_camera_spec(c2w.to(dtype=tree.data.dtype),
                              width, height, fx, fy), opt, error
                          )
    return accum.value

def prune_func(DOT, instant_weights, 
               thresh_type='weight', 
               thresh_val=5e-3,
               thresh_tol=0.8,
               summary_writer=None,
               gstep_id = None,
               recursive=True,
               ):
    non_writer = summary_writer is None
    if not non_writer:
        assert gstep_id is not None
    with torch.no_grad():
        leaves = DOT._all_leaves()
        sel = (*leaves.long().T, )

        if thresh_type == 'sigma':
            val = DOT.data[sel][..., -1]
        elif thresh_type == 'weight':
            val = instant_weights[sel]
        # elif thresh_type == 'rweight':
        #     val = DOT.
        

        val = torch.nan_to_num(val, nan=0)
        
        thred = thresh_val
        toltal = 0 
        while True:
            # smoothed = gaussian(val.cpu().detach().numpy(), sigma=args.thresh_gaussian_sigma)   
            sel = leaves[val < thred]
            nids, counts = torch.unique(sel[:, 0], return_counts=True)
            # discover the fronts whose all children are included in sel
            mask = (counts >= int(DOT.N**3*thresh_tol)).numpy()
            sel_nids = nids[mask]
            if sel_nids.size(0) == 0 or not DOT.merge_nids(sel_nids):
                break
            parent_sel = (*DOT._unpack_index(
                DOT.parent_depth[sel_nids, 0]).long().T,)
            # if pre_sel is not None:
                # if sel_nids.size(0) == 0 or torch.equal(pre_sel, sel_nids):
                #     break
            # pre_sel = sel_nids
            # DOT.shrink_to_fit()
            n = len(sel_nids)*(DOT.N ** 3 - 1)
            toltal += n
            print(f'Prune {n}/{leaves.size(0)}')
            
            reduced = instant_weights[sel_nids].view(-1, DOT.N ** 3).sum(-1)
            instant_weights[parent_sel] = reduced

            if not recursive:
                break
            val, leaves = update_val_leaves(DOT, instant_weights)

        print(f'Purne {toltal} nodes in toltal.')
        if not non_writer:
            summary_writer.add_scalar(f'train/number_prune', toltal, gstep_id)
        return instant_weights
    
def update_val_leaves(DOT, instant_weights, thresh_type='weight'):
    leaves = DOT._all_leaves()
    if thresh_type == 'weight':
        val = instant_weights[(*leaves.long().T, )]
    elif thresh_type == 'sigma':
        val = DOT.data[(*leaves.long().T, )][..., -1]
    val = torch.nan_to_num(val, nan=0)
    return val, leaves

def sample_func(tree, sampling_rate, VAL, repeats=1, self_cp=True):
    with torch.no_grad():
        val, leaves = update_val_leaves(tree, VAL)
        sample_k = int(max(1, tree.n_leaves*sampling_rate))
        delta = sample_k*tree.N**3
        print(f'Start sampling {sample_k} nodes, and increase {delta} nodes.')
        idxs = select(sample_k, val, leaves)
        interval = idxs.size(0)//repeats
        
        for i in range(1, repeats+1):
            start = (i-1)*interval
            end = i*interval
            sel = idxs[start:end]
            expand(tree, sel, i, self_cp=self_cp)
        return idxs

def expand(tree, idxs, repeats, self_cp=True):
    # group expansion
    sel = (*idxs.long().T,)
    tree.refine(sel=sel, repeats=repeats, self_cp=self_cp)
        
def select(max_sel, reward, rw_idxs):
    p_val = reward  # only MSE
    sel = min(p_val.size(0), max_sel)
    _, idxs = torch.topk(p_val, sel)
    idxs = rw_idxs[idxs]
    return idxs

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class DOT_N3Tree(N3Tree):
    def __init__(self, N=2, data_dim=None, depth_limit=10,
                 init_reserve=1, init_refine=0, geom_resize_fact=1.0,
                 radius=0.5, center=[0.5, 0.5, 0.5],
                 data_format="SH9",
                 extra_data=None,
                 device="cuda",
                 dtype=torch.float32,
                 map_location=None,
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

        self.basis_rms = None
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
                
    def merge_nids(self, nids):
        device = self.data.device
        nids = nids.to(device)
        idxs = [f in nids for f in self._frontier]
        return self.merge(idxs)
    
    def set_depth_limit(self, depth):
        self.depth_limit = depth
    
    def optim_basis_all_step(self, lr_sigma: float, lr_sh: float, beta: float = 0.9, epsilon: float = 1e-8,
                             optim: str = 'rmsprop', rate_sel=1e1, sel=None):
        """
        Execute RMSprop/SGD step on SH
        """

        data = self.data

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
            if sel is not None:
                data.data[sel][-1].addcdiv_(data.grad[sel][-1],
                                        denom[sel][-1], value=-lr_sigma*rate_sel)
                data.data[sel][:-1].addcdiv_(data.grad[sel][:-1],
                                        denom[sel][:-1], value=-lr_sh*rate_sel) 
        elif optim == 'sgd':
            data.grad[..., -1].mul_(lr_sigma)
            data.grad[..., :-1].mul_(lr_sh)
            data.data -= data.grad
            if sel is not None:
                data.grad[sel][-1].mul_(lr_sigma)
                data.grad[sel][:-1].mul_(lr_sh)  
        else:
            raise NotImplementedError(f'Unsupported optimizer {optim}')
        
        data.grad.zero_()
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
        tree.data_format = DataFormat(z['data_format'].item()) if \
                'data_format' in z.files else None
        tree.extra_data = torch.from_numpy(z['extra_data']).to(device) if \
                          'extra_data' in z.files else None
        return tree      
    # Leaf refinement & memory management methods
    def refine(self, repeats=1, sel=None, self_cp=True):
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
                if self_cp:
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

        
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, periodic=True, per_drop=0.3
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
        if periodic:
            step = step%max_steps
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

class expon_lr():
    def __init__(self, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000, periodic=True, per_drop=0.2) -> None:
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps
        self.periodic = periodic
        self.per_drop = per_drop
        
    def step(self, step):
        if self.periodic:
            step = step%self.max_steps
            if step == 0:
                self.lr_init *= 1-self.per_drop
                self.lr_final *= 1-self.per_drop
        if step < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return delay_rate * log_lerp        
    
  
def vis_dif(path_1, path_2, out_path):
    m1 = DOT_N3Tree.load(path_1, map_location="cuda")
    m2 = DOT_N3Tree.load(path_2, map_location="cuda")
    s1 = m1.data.size(0)
    s2 = m2.data.size(0)
    
    if s1 >= s2:
        m = m1.clone("cpu")
        dif = m1.data.data - m2.data.data
    else:
        m = m2.clone("cpu")
        dif = m2.data.data - m1.data.data
    
    m = m.to("cuda")
    m.data.data = dif
    m.save(out_path)
