import torch.nn as nn
from svox import N3Tree, VolumeRenderer
import torch
from svox.helpers import  DataFormat
from warnings import warn
from einops import rearrange
import torch.nn.functional as F
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
        self.record = record
        # the data is used for recording purpose only 
        if record:
            device="cpu"
            self.register_buffer("data",
                torch.zeros(init_reserve, N, N, N, self.data_dim, dtype=dtype, device=device))
            # the reward and n_visits for mcst
            self.register_buffer("num_visits", torch.zeros(init_reserve, N, N, N, dtype=torch.int32, device=device))
            self.register_buffer("total_reward", torch.zeros(init_reserve, N, N, N, dtype=dtype, device=device)) 
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
            flat = torch.div(flat, self.N, rounding_mode='trunc')
        return torch.stack((flat, t[2], t[1], t[0]), dim=-1)

    def _resize_add_cap(self, cap_needed):
        """
        Helper for increasing capacity
        """
        cap_needed = max(cap_needed, int(self.capacity * (self.geom_resize_fact - 1.0)))
        may_oom = self.capacity + cap_needed > 1e7  # My CPU Memory is limited
        if may_oom:
            # Potential OOM prevention hack
            self.data = nn.Parameter(self.data.cpu())
        # recorder
        if self.record:
            self.data = torch.cat((self.data.data,
                            torch.zeros((cap_needed, *self.data.data.shape[1:]),
                                    dtype=self.data.dtype,
                                    device=self.data.device)), dim=0)         
            self.num_visits = torch.cat((self.num_visits,
                                        torch.zeros((cap_needed, *self.num_visits.shape[1:]),
                                        dtype=self.num_visits.dtype,
                                        device=self.num_visits.device)
                                        ))
            self.total_reward = torch.cat((self.total_reward,
                                        torch.zeros((cap_needed, *self.total_reward.shape[1:]),
                                        dtype=self.total_reward.dtype,
                                        device=self.total_reward.device)
                                        ))     
        else:
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
                 num_round=50, 
                 data_dim=None,
                 explore_exploit=2.,
                 depth_limit=10,
                 device="cpu",
                 data_format="SH9",
                 dtype=torch.float32,
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
        self.recorder = SMCT(radius=radius,
                             center=center,
                             data_format=data_format,
                             data_dim=data_dim,
                             depth_limit=depth_limit,
                             device="cpu",
                             dtype=dtype)
        self.step_size = step_size    
        self.num_round = num_round   
        self.dtype = dtype
        n_nodes = self.recorder.n_internal
        N = self.recorder.N
        self.explor_exploit = explore_exploit
        self.player = None
        self.instant_reward = None
        
        self.init_player(device)
        
    def select(self):
        """Deep first search based on policy value: from root to the tail
        
        weights: shape-> [n, N, N, N] leaf nodes
        the instant reward is conputed as: weight*exp(-mse)
        """
        child = self.player.child #[n, N, N, N]
        depth = self.player.parent_depth
        N = self.player.N
        p_p= -1
        p_idx = 0
        r_idx = 0
        
        p_val = self.policy_puct() # leaf
        
        while p_p != 0:
            idx_ = torch.argmax(p_val[p_idx])
            _, u, v, z = self.recorder._unpack_index(idx_)
            # update the recorder
            self.recorder.num_visits[r_idx, u, v, z] += 1
            p_p = child[p_idx, u, v, z]
            p_r = self.recorder.child[r_idx, u, v, z]
            p_idx += p_p                            
            r_idx += p_r
            
        return [p_idx,u,v,z], r_idx, p_r
        
    def getReward(self, rays, cuda=True, fast=False):
        render = VolumeRenderer(self.player, step_size=self.step_size)
        with self.player.accumulate_weights(op="sum") as accum:
            res = render.forward(rays, cuda=cuda, fast=fast)
        val = accum.value
        val /= val.sum()
        return res, val
    
    
    def expand(self, pos, r_idx, p_r):
        # pos -> player leaf
        n, u, v, z = pos
        
        self.player._refine_at(n, (u, v, z))
        if p_r == 0:
            self.player.data[-1] += self.player.data[n, u, v, z]
        # copy physical properties clone from recorder
        else:
            self.player.data[-1] += self.recorder.data[r_idx].to(self.player.data.device)
    
    def run_a_round(self, rays, gt):
        B, H, W, _ = gt.shape 
        # stimulate
        res, weights = self.getReward(rays)
        res = rearrange(res, '(B H W) C -> B H W C', B=B, H=H)
        mse = F.mse_loss(gt, res)
        ## update the player based on mse
        mse.backward()
        # save the reward and record it at the end of the game rounds
        self._instant_reward(weights, mse)
        # select
        pos, r_idx, p_r = self.select()
        # expand
        self.expand(pos, r_idx, p_r)
        
        

    def _instant_reward(self, weights, mse):
        instant_reward = weights*torch.exp(-mse)
        # integrate the player's contributions from leafs to roots 
        depth, indexes = torch.sort(self.player.parent_depth, dim=0, descending=True)
        N = self.player.N
        total_reward = torch.zeros(self.player.n_internal, N, N, N, device=self.player.data.device)
        for d in depth:
            idx_ = d[0]
            depth = d[1]

            # internal node
            xyzi = self.player._unpack_index(idx_)    
            n, x, y, z = xyzi
            n_ = n + self.player.child[n, x, y, z]
            ins_rewards = instant_reward[n_]
            # the root node idx is not recorded!
            if depth != 0:
                total_reward[n, x, y, z] += ins_rewards.sum()
            # leaf_nodes
            total_reward[n_] += ins_rewards
        self.instant_reward = total_reward
        
    def policy_puct(self):
        """Return the policy head value to guide the sampling

        P-UCT = total_reward(s, a)+ C*instant_reward(s,a)/(1+num_visits(s))
        
        where s is the state, a is the action.
        
        Args:
            instant_reward is the sum array[n, x, y, z] of rewards after backpropagtion for node_idx
        Returns:
            p-uct value
        """
        instant_reward = self.instant_reward
        device = instant_reward.device
        total_reward = self.total_reward.to(device)
        num_visits = self.num_visits.to(device)
        return total_reward+self.explor_exploit*instant_reward/(1+num_visits)
    
    def copyFromPlayer(self):
        p_c = self.player.child.to('cpu')
        p_d = self.player.data.to('cpu')
        r_c = self.recorder.child
        r_d = self.recorder.data
        N = self.recorder.N
        instant_reward = self.instant_reward
        # iterate over the player tree
        self.copyLayer(p_d, p_c, r_d, r_c, instant_reward, N, 0, 0)
        
        

    # the recursive DFS
    def copySubFrom(self, pos, p_d, p_c, r_c,instant_reward, N, p_idx, r_idx):
        instant_reward = self.instant_reward
        n, x, y, z = pos
        self.recorder._refine_at(n, [x,y,z])
        self.recorder.data[-1] = self.player.data[p_idx].detach().cpu()
        if instant_reward:
            self.recorder.total_reward[r_idx] += instant_reward[p_idx].detach().cpu()
        
        for x, y, z in (torch.zeros(N, N, N)==0).nonzero():
            p_child = p_c[p_idx, x, y, z]
            if p_child != 0:
                r_child = r_c[r_idx, x, y, z]
                self.copySubFrom([r_idx, x, y, z], p_d, p_c, r_c, instant_reward, N, p_idx+p_child, r_idx+r_child)

    
    # the recursive DFS 
    def copyLayer(self, p_d, p_c, r_d, r_c, instant_reward, N, p_idx, r_idx):
        
        # overwrite data
        self.recorder.data[r_idx] = self.player.data[p_idx].detach().cpu()
        if instant_reward:
            self.recorder.total_reward[r_idx] += instant_reward[p_idx].detach().cpu()
        
        for x, y, z in (torch.zeros(N, N, N)==0).nonzero():
            p_child = p_c[p_idx, x, y, z]
            if p_child !=0:
                r_child = r_c[r_idx, x, y, z]
                if r_child == 0:
                    self.copySubFrom([r_idx, x, y, z], p_d, p_c, r_c, instant_reward,N, p_idx+p_child, r_idx+r_child)
                else:
                    self.copyLayer(p_d, p_c, r_d, r_c, instant_reward,N, p_idx+p_child, r_idx+r_child)
        
    def init_player(self, device):
        # if previous player exists, copy its physical properties(data) and child, depth into the recorder.
        # take the union operation on child and depth, and take the average on the data
        if self.player is not None:
            self.copyFromPlayer() 
            self.player =  SMCT(record=False, radius=self.radius, center=self.center, data_format=self.data_format, 
                                data_dim=self.recorder.data_dim, depth_limit=self.recorder.depth_limit, device=device, dtype=self.dtype)
            # initialize physical properties from recorder
            self.player.data.data = nn.Parameter(self.recorder.data.data[0, :, :, :], dtype=self.player.data.dtype, device=device)
        else:
            self.player =  SMCT(record=False, radius=self.radius, center=self.center, data_format=self.data_format, 
                                data_dim=self.recorder.data_dim, depth_limit=self.recorder.depth_limit, device=device, dtype=self.dtype)
            