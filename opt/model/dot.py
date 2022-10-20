from logging import warning
import torch.nn as nn
from svox import N3Tree, VolumeRenderer, LocalIndex
from svox.renderer import NDCConfig, _rays_spec_from_rays
from torch.nn.functional import softplus
import torch
from svox.helpers import DataFormat
from warnings import warn
from svox.svox import _get_c_extension
from .utils import pareto_2d, _SOFTPLUS_M1, asoftplus
_C = _get_c_extension()

class N3Tree_(N3Tree):
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
        # nvmlInit()
        # h = nvmlDeviceGetHandleByIndex(0)
        # info = nvmlDeviceGetMemoryInfo(h)
        # need = sys.getsizeof(torch.zeros((cap_needed, *self.data.data.shape[1:]),
        #                                                 dtype=self.data.dtype,).storage())
        # if info.free < need:
        #     print('Memory runs out, and stop sampling.')
        #     return False
                
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
    

class DOT(nn.Module):
    def __init__(self,
                 radius,
                 center,
                 step_size,
                 pre_train_pth=None,
                 init_refine=0,
                 sigma_thresh=0.0,
                 stop_thresh=0.0,
                 data_dim=None,
                 depth_limit=12,
                 device="cpu",
                 data_format=None,
                 dtype=torch.float32,
                 density_softplus=True,
                 init_weight_sparsity_loss=None,
                 init_tv_sigma_loss=None,
                 init_tv_color_loss=None,
                 ):
        """Main mcts based octree data structure

        Args:
            radius: float or List[float, float, float], 1/2 side length of cube (possibly in each dim).
            center:  float or List[float, float, float], the center of cube.
            step_size: float, render step size (in voxel size units)
            data_format (str, optional): _description_. Defaults to "None".
            num_round: int, the number of round to play. 
            depth_limit: int maximum depth  of tree to stop branching/refining
                         Note that the root is at depth -1.
                         Size :code:`N^[-10]` leaves (1/1024 for octree) for example
                         are depth 9. :code:`max_depth` applies to the same
                         depth values.
        """
        super(DOT, self).__init__()
        # the octree is always kept to record the overview
        # the sh, density value are updated constantly
        self.radius = radius
        self.center = center
        self.data_format = data_format
        
        if pre_train_pth is not None:
            self.tree = N3Tree.load(pre_train_pth, device=device)
            pre_depth_lim = self.tree.depth_limit
            pre_invradius = self.tree.invradius
            pre_offset = self.tree.offset
            pre_data_dim = self.tree.data_dim
            pre_data_format = self.tree.data_format
            
            radius = torch.tensor(self.radius, dtype=dtype, device=device)
            center = torch.tensor(self.center, dtype=dtype, device=device) 
            invradius = 0.5/radius
            offset = 0.5 * (1.0 - center / radius)
            print(f'Change the depth limit from the pre_trained: {pre_depth_lim} to {depth_limit}.')
            self.tree.depth_limit = depth_limit
            
            if not torch.equal(invradius,pre_invradius) and not torch.equal(pre_offset,offset):
                warning('Not the same dataset setting.'+\
                f'Prev_invradius:{pre_invradius}'+\
                f'Prev_offset:{pre_offset}'+\
                '\n'+\
                f'invradius:{invradius}'+\
                f'offset:{offset}')  
                # self.tree.invradius = invradius
                # self.tree.offset = offset
                
            flag = True
            if data_format is not None:
                assert isinstance(data_format, str), "Please specify valid data format"
                self.tree.data_format = DataFormat(data_format)

            if data_dim != pre_data_dim and data_dim is not None:
                print(f'Change the original the data dimension from {pre_data_dim} to {data_dim}.') 
                self.tree.data_dim = data_dim
            elif pre_data_format != data_format and data_format is not None:
                print(f'Change the original the data format from {pre_data_format} to {data_format}.') 
            else:
                flag = False
            
            if flag:
                self.tree.expand(data_format, data_dim=data_dim)
        else:
            self.tree = N3Tree(radius=radius,
                            center=center,
                            data_format=data_format,
                            init_refine=init_refine,
                            data_dim=data_dim,
                            depth_limit=depth_limit,
                            dtype=dtype,
                            reload=False)
        self.step_size = step_size
        self.dtype = dtype

        self.gstep_id = 0
        self.gstep_id_base = 0
        self.basis_rms = None
        self.sigma_thresh = sigma_thresh
        self.stop_thresh = stop_thresh
        self.density_softplus = density_softplus

        if init_weight_sparsity_loss is not None:
            _ = asoftplus(torch.tensor(init_weight_sparsity_loss))
            self.w_sparsity = torch.nn.Parameter(torch.tensor([_], device=device))
        if init_tv_sigma_loss is not None:
            _ = asoftplus(torch.tensor(init_tv_sigma_loss))
            self.w_sigma_tv = torch.nn.Parameter(torch.tensor([_], device=device)) 
        if init_tv_color_loss is not None:
            _ = asoftplus(torch.tensor(init_tv_color_loss))
            self.w_color_tv = torch.nn.Parameter(torch.tensor([_], device=device))        

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

        p_val = reward  # only MSE
        sel = min(p_val.size(0), max_sel)
        _, idxs = torch.topk(p_val, sel)
        idxs = rw_idxs[idxs]
        return idxs

    def _volumeRenderer(self, ndc, orig=True):
        if orig:
            return VolumeRenderer(self.tree, ndc=ndc)
        
        return VolumeRenderer_(self.tree, step_size=self.step_size,
                              sigma_thresh=self.sigma_thresh, stop_thresh=self.stop_thresh, density_softplus=self.density_softplus,
                              ndc=ndc)

    def expand(self, idxs, repeats):
        # group expansion
        i = idxs.size(0)
        idxs = (*idxs.long().T,)
        res = self.tree.refine(sel=idxs, repeats=repeats)
        return res

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
    def get_depth(self):
        return torch.max(self.tree.parent_depth[:, 1])   
     
class VolumeRenderer_(VolumeRenderer):
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
