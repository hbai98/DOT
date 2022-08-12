import torch.nn as nn
from svox import N3Tree, VolumeRenderer
import torch
from typing import Union
from svox.helpers import  DataFormat
from warnings import warn

class SMCT(N3Tree):
    def __init__(self, record=True, N=2, data_dim=None, depth_limit=10,
            init_reserve=1, init_refine=0, geom_resize_fact=1.0,
            radius=0.5, center=[0.5, 0.5, 0.5],
            data_format="SH9",
            extra_data=None,
            device="cpu",
            dtype=torch.float32,
            map_location=None):
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
        self.N : int = N

        if map_location is not None:
            warn('map_location has been renamed to device and may be removed')
            device = map_location
        assert dtype == torch.float32 or dtype == torch.float64, 'Unsupported dtype'

        self.data_format = DataFormat(data_format) if data_format is not None else None
        self.data_dim : int = data_dim
        self._maybe_auto_data_dim()
        del data_dim

        if init_refine > 0:
            for i in range(1, init_refine + 1):
                init_reserve += (N ** i) ** 3

        # the data is used for recording purpose only 
        if record:
            self.register_buffer("data",
                torch.zeros(init_reserve, N, N, N, self.data_dim, dtype=dtype, device=device))
        else:
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
            self.register_buffer("extra_data", extra_data.to(dtype=dtype, device=device))
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
            flat = flat // self.N
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)

class mcots(nn.Module):
    def __init__(self,
                 radius,
                 center,
                 step_size,
                 round=50, 
                 data_dim=None,
                 explore_exploit=2.,
                 depth_limit=10,
                 data_format="SH9",
                 device: Union[torch.device, str] = "cpu",
                 dtype=torch.float32,
                 ):
        """Main mcts based octree data structure

        Args:
            radius: float or List[float, float, float], 1/2 side length of cube (possibly in each dim).
            center:  float or List[float, float, float], the center of cube.
            step_size: float, render step size (in voxel size units)
            data_format (str, optional): _description_. Defaults to "SH9".
            round: int, the number of round to play. 
            depth_limit: int maximum depth  of tree to stop branching/refining
                         Note that the root is at depth -1.
                         Size :code:`N^[-10]` leaves (1/1024 for octree) for example
                         are depth 9. :code:`max_depth` applies to the same
                         depth values.
            device: torch.device 
        """
        super(mcots, self).__init__()
        # the octree is always kept to record the overview
        # the sh, density value are updated constantly 
        self.octree = SMCT(radius=radius,
                             center=center,
                             data_format=data_format,
                             data_dim=data_dim,
                             depth_limit=depth_limit,
                             device=device,
                             dtype=dtype)
        self.step_size = step_size       
        self.dtype = dtype
        self.device = device 
        n_nodes = self.octree.n_internal
        N = self.octree.N
        self.explor_exploit = explore_exploit
        # the records bellow are kept throughout the rounds. 
        self.register_buffer("num_visits", torch.zeros(n_nodes, N, N, N, dtype=torch.int32, device=device))
        self.register_buffer("total_reward", torch.zeros(n_nodes, N, N, N, dtype=dtype, device=device))
        
    def select(self, weights, mse):
        """Deep first search based on policy value: from root to the tail
        
        the instant reward is conputed as: weight*exp(-mse)
        """
        bestValue = -torch.inf
        bestIdx = torch.Tensor([-1, -1, -1, -1], device=self.device)
        child = self.octree.child #[n, N, N, N]
        i = 0
        idx = (torch.ones(N, N, N)==1).nonzero()
        N = self.octree.N
        while torch.any(child[i]):
            p_val = [self.policy_puct([i, idx_], ) for idx_ in idx]
            
    
    def getReward(self, tree, rays, cuda=True, fast=False):
        render = VolumeRenderer(tree, step_size=self.step_size)
        with tree.accumulate_weights(op="sum") as accum:
            res = render.forward(rays, cuda=cuda, fast=fast)
        val = accum.value
        val /= val.sum()
        return res, val
    
    
    def backpropagate(self):
        pass
    
    def policy_puct(self, idx, instant_reward):
        """Return the policy head value to guide the sampling

        P-UCT = total_reward(s, a)+ C*instant_reward(s,a)/(1+num_visits(s))
        
        where s is the state, a is the action.
        
        Args:
            idx is [n, x, y, z]
            instant_reward is the sum array[n, x, y, z] of rewards after backpropagtion for node_idx
        Returns:
            p-uct value
        """
        return self.total_reward[idx]+self.explor_exploit*instant_reward[idx]/(1+self.num_visits)
        
        