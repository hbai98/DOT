from .svox2 import SparseGrid, Camera
from .defs import *
from . import utils
from typing import Union, List, NamedTuple, Optional, Tuple
import torch
from torch import nn, autograd
from tqdm import tqdm
from functools import reduce
from einops import rearrange
_C = utils._get_c_extension()

class DM_SpaseGrid(SparseGrid):
    """
    Main sparse grid data structure.
    initially it will be a dense grid of resolution <reso>.
    Only float32 is supported.

    :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
    :param radius: float or List[float, float, float], the 1/2 side length of the grid, optionally in each direction
    :param center: float or List[float, float, float], the center of the grid
    :param basis_type: int, basis type; may use svox2.BASIS_TYPE_* (1 = SH, 4 = learned 3D texture, 255 = learned MLP)
    :param basis_dim: int, size of basis / number of SH components
                           (must be square number in case of SH)
    :param basis_reso: int, resolution of grid if using BASIS_TYPE_3D_TEXTURE
    :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
    :param mlp_posenc_size: int, if using BASIS_TYPE_MLP, then enables standard axis-aligned positional encoding of
                                 given size on MLP; if 0 then does not use positional encoding
    :param mlp_width: int, if using BASIS_TYPE_MLP, specifies MLP width (hidden dimension)
    :param device: torch.device, device to store the grid
    """

    def __init__(
        self,
        reso: Union[int, List[int], Tuple[int, int, int]] = 128,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        basis_type: int = BASIS_TYPE_SH,
        basis_dim: int = 9,  # SH/learned basis size; in SH case, square number
        basis_reso: int = 16,  # Learned basis resolution (x^3 embedding grid)
        use_z_order : bool=False,
        use_sphere_bound : bool=False,
        mlp_posenc_size : int = 0,
        mlp_width : int = 16,
        background_nlayers : int = 0,  # BG MSI layers
        background_reso : int = 256,  # BG MSI cubemap face size
        use_dm: bool = True,
        dm_recursive: bool = True, 
        dm_thred_tolerence: float = 1e-1, # the boundary of threshold  (%)
        dm_step_size: int = 3, 
        dm_step_max: int = 500,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__(reso, radius, center, basis_type, basis_dim, basis_reso, use_z_order, use_sphere_bound, mlp_posenc_size, mlp_width, background_nlayers, background_reso, device)
        assert dm_thred_tolerence > 0 and dm_thred_tolerence < 1
        self.dm_thred_tolerence = dm_thred_tolerence
        self.use_dm = use_dm
        self.dm_recursive = dm_recursive
        self.dm_step_size = dm_step_size
        self.dm_step_max = dm_step_max

    def dm(self, val, thred):
        # t_l = (1-self.dm_thred_tolerence)*thred
        t_h = (1+self.dm_thred_tolerence)*thred
        bound_mask = val<=t_h # [3]
        pos_bound = torch.nonzero(bound_mask).to(torch.int)
        
        step = 0
        while True:
            step += 1
            print(f'step: {step}, total: {step*self.dm_step_size}, select {pos_bound.size(0)}({pos_bound.size(0)/torch.nonzero(val >= thred).size(0)}%) points to do dm. ')
            if pos_bound.size(0) == 0:
                print(f'Warning: No boundary is found given the thred_tolerence({self.dm_thred_tolerence}) in weights after {step} steps.')
                break
            if step > self.dm_step_max:
                print(f'Max step: {self.dm_step_max}')
                break
            # cal norm for conv/deconv direction
            pos_bound = finite_difference_march(pos_bound, val, epsilon=self.dm_step_size)
            # print(val.shape)
            val = self.conv_voxel(pos_bound, val, epsilon=self.dm_step_size)
            # filter pos_bound basd on val
            # tmp_val = val[(*pos_bound.long().T, )]
            # bound_mask = tmp_val<=t_h
            # pos_bound = pos_bound[bound_mask]
            
            if not self.dm_recursive:
                break   
            
        return val     
    def resample(
        self,
        reso: Union[int, List[int]],
        sigma_thresh: float = 5.0,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        cameras: Optional[List[Camera]] = None,
        use_z_order: bool=False,
        accelerate: bool=True,
        weight_render_stop_thresh: float = 0.2, # SHOOT, forgot to turn this off for main exps..
        max_elements:int=0
    ):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param sigma_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
                             (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
        :param accelerate: bool, if true (default), calls grid.accelerate() after resampling
                           to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold for grid weight render in [0, 1];
                                                 0.0 = no thresholding, 1.0 = hides everything.
                                                 Useful for force-cutting off
                                                 junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements in the
                upsampled grid; we will adjust the threshold to match it
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert (
                    len(reso) == 3
                ), "reso must be an integer or indexable object of 3 ints"

            if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
                print("Morton code requires a cube grid of power-of-2 size, ignoring...")
                use_z_order = False

            self.capacity: int = reduce(lambda x, y: x * y, reso)
            curr_reso = self.links.shape
            dtype = torch.float32
            reso_facts = [0.5 * curr_reso[i] / reso[i] for i in range(3)]

            use_weight_thresh = cameras is not None

            batch_size = 720720
            self.density_data.grad = None
            self.sh_data.grad = None
            self.sparse_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.sh_rms = None            
            # dm collects all information for previous resolution 
            if self.use_dm:
                val = None
                all_sample_vals_density = []
                # resample previous
                X = torch.linspace(
                    0,
                    curr_reso[0] - 1,
                    curr_reso[0],
                    dtype=dtype,
                )
                Y = torch.linspace(
                    0,
                    curr_reso[1] - 1,
                    curr_reso[1],
                    dtype=dtype,
                )
                Z = torch.linspace(
                    0,
                    curr_reso[2] - 1,
                    curr_reso[2],
                    dtype=dtype,
                )
                X, Y, Z = torch.meshgrid(X, Y, Z)
                points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

                if use_z_order:
                    morton = utils.gen_morton(reso[0], dtype=torch.long).view(-1)
                    points[morton] = points.clone()
                points = points.to(device=device)
                                    
                all_sample_vals_density = []
                for i in tqdm(range(0, len(points), batch_size)):
                    sample_vals_density, _ = self.sample(
                        points[i : i + batch_size],
                        grid_coords=True,
                        want_colors=False
                    )
                    sample_vals_density = sample_vals_density
                    all_sample_vals_density.append(sample_vals_density)
                    
                sample_vals_density = torch.cat(
                        all_sample_vals_density, dim=0).view(curr_reso)                
                del all_sample_vals_density                    
                if use_weight_thresh:
                    gsz = torch.tensor(curr_reso)
                    offset = (self._offset * gsz - 0.5).to(device=device)
                    scaling = (self._scaling * gsz).to(device=device)
                    max_wt_grid = torch.zeros(curr_reso, dtype=torch.float32, device=device)
                    print(" DM: Pre grid weight render", sample_vals_density.shape)
                    for i, cam in enumerate(cameras):
                        _C.grid_weight_render(
                            sample_vals_density, cam._to_cpp(),
                            0.5,
                            weight_render_stop_thresh,
                            #  self.opt.last_sample_opaque,
                            False,
                            offset, scaling, max_wt_grid
                        )
                    val = max_wt_grid
                    thred = weight_thresh
                else:
                    val = sample_vals_density
                    thred = sigma_thresh
                self.dm(val, thred)
                
            X = torch.linspace(
                reso_facts[0] - 0.5,
                curr_reso[0] - reso_facts[0] - 0.5,
                reso[0],
                dtype=dtype,
            )
            Y = torch.linspace(
                reso_facts[1] - 0.5,
                curr_reso[1] - reso_facts[1] - 0.5,
                reso[1],
                dtype=dtype,
            )
            Z = torch.linspace(
                reso_facts[2] - 0.5,
                curr_reso[2] - reso_facts[2] - 0.5,
                reso[2],
                dtype=dtype,
            )
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

            if use_z_order:
                morton = utils.gen_morton(reso[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points.to(device=device)
                                
            all_sample_vals_density = []
            print('Pass 1/2 (density)')
            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_density, _ = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=False
                )
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)


            sample_vals_density = torch.cat(
                    all_sample_vals_density, dim=0).view(reso)
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = torch.tensor(reso)
                offset = (self._offset * gsz - 0.5).to(device=device)
                scaling = (self._scaling * gsz).to(device=device)
                max_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                print(" Grid weight render", sample_vals_density.shape)
                for i, cam in enumerate(cameras):
                    _C.grid_weight_render(
                        sample_vals_density, cam._to_cpp(),
                        0.5,
                        weight_render_stop_thresh,
                        #  self.opt.last_sample_opaque,
                        False,
                        offset, scaling, max_wt_grid
                    )
                    #  if i % 5 == 0:
                    #      # FIXME DEBUG
                    #      tmp_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                    #      import os
                    #      os.makedirs('wmax_vol', exist_ok=True)
                    #      _C.grid_weight_render(
                    #          sample_vals_density, cam._to_cpp(),
                    #          0.5,
                    #          0.0,
                    #          self.opt.last_sample_opaque,
                    #          offset, scaling, tmp_wt_grid
                    #      )
                    #  np.save(f"wmax_vol/wmax_view{i:05d}.npy", tmp_wt_grid.detach().cpu().numpy())
                #  import sys
                #  sys.exit(0)

                        # next location for conv/deconv operations 
                        # print(torch.ceil(pos_bound + self.dm_step_size*norm) - pos_bound)
                        # print(pos_bound)
                    
                sample_vals_mask = max_wt_grid >= weight_thresh
                if max_elements > 0 and max_elements < max_wt_grid.numel() \
                                    and max_elements < torch.count_nonzero(sample_vals_mask):
                    # To bound the memory usage
                    weight_thresh_bounded = torch.topk(max_wt_grid.view(-1),
                                    k=max_elements, sorted=False).values.min().item()
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    print(' Readjusted weight thresh to fit to memory:', weight_thresh)
                    
                    sample_vals_mask = max_wt_grid >= weight_thresh
                    
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density >= sigma_thresh
                if max_elements > 0 and max_elements < sample_vals_density.numel() \
                                    and max_elements < torch.count_nonzero(sample_vals_mask):
                    # To bound the memory usage
                    sigma_thresh_bounded = torch.topk(sample_vals_density.view(-1),
                                    k=max_elements, sorted=False).values.min().item()
                    sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
                    print(' Readjusted sigma thresh to fit to memory:', sigma_thresh)
                    
                    sample_vals_mask = sample_vals_density >= sigma_thresh
                if self.opt.last_sample_opaque:
                    # Don't delete the last z layer
                    sample_vals_mask[:, :, -1] = 1
                

            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = torch.count_nonzero(sample_vals_mask).item()
            
            # Now we can get the colors for the sparse points
            points = points[sample_vals_mask]
            print('Pass 2/2 (color), eval', cnz, 'sparse pts')
            all_sample_vals_sh = []
            for i in tqdm(range(0, len(points), batch_size)):
                _, sample_vals_sh = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=True
                )
                all_sample_vals_sh.append(sample_vals_sh)

            sample_vals_sh = torch.cat(all_sample_vals_sh, dim=0) if len(all_sample_vals_sh) else torch.empty_like(self.sh_data[:0])
            del self.density_data
            del self.sh_data
            del all_sample_vals_sh

            if use_z_order:
                inv_morton = torch.empty_like(morton)
                inv_morton[morton] = torch.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = torch.full(
                    (sample_vals_mask.size(0),), fill_value=-1, dtype=torch.int32
                )
                init_links[inv_idx] = torch.arange(inv_idx.size(0), dtype=torch.int32)
            else:
                init_links = (
                    torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1
                )
                init_links[~sample_vals_mask] = -1

            self.capacity = cnz
            print(" New cap:", self.capacity)
            del sample_vals_mask
            print('density', sample_vals_density.shape, sample_vals_density.dtype)
            print('sh', sample_vals_sh.shape, sample_vals_sh.dtype)
            print('links', init_links.shape, init_links.dtype)
            self.density_data = nn.Parameter(sample_vals_density.view(-1, 1).to(device=device))
            self.sh_data = nn.Parameter(sample_vals_sh.to(device=device))
            self.links = init_links.view(reso).to(device=device)

            if accelerate and self.links.is_cuda:
                self.accelerate()
    # simple identity kernel 
    def conv_voxel(self, pos, signal, epsilon=1, deconv=False):
        
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        xl = torch.cat([torch.clamp(x-i, min=0) for i in range(1, epsilon)]) 
        xr = torch.cat([torch.clamp(x+i, max=self.links.size(0) - 1) for i in range(1, epsilon)]) 
        
        yl = torch.cat([torch.clamp(y-i, min=0) for i in range(1, epsilon)]) 
        yr = torch.cat([torch.clamp(y+i, max=self.links.size(1) - 1) for i in range(1, epsilon)]) 

        zl = torch.cat([torch.clamp(z-i, min=0) for i in range(1, epsilon)]) 
        zr = torch.cat([torch.clamp(z+i, max=self.links.size(2) - 1) for i in range(1, epsilon)]) 

        pos_near = torch.stack([torch.cat([xl, xr]), torch.cat([yl, yr]), torch.cat([zl, zr])], dim=-1)
        # 38400
        # collect signal 
        signal_center_idx = (*pos.long().T, )
        signal_near_idx = (*pos_near.long().T, )
        
        for i in range(3):
            pos_near[:, i].clamp_max_(self.links.size(i) - 2)
        pos_near = pos_near.to(torch.long)
        for i in range(3):
            pos[:, i].clamp_max_(self.links.size(i) - 1)
        pos = pos.to(torch.long)
        for i in range(3):
            pos[:, i].clamp_max_(self.links.size(i) - 2)        
        
        lx, ly, lz = pos_near.unbind(-1)
        links_near = self.links[lx, ly, lz]
        links_near = links_near[links_near>0].long()
        lx, ly, lz = pos.unbind(-1)
        links_center = self.links[lx, ly, lz]
        links_center = links_center[links_center>0].long()

        if deconv:
            sigma, rgb = self._fetch_links(links_center)
            B = links_near.size(0)
            weight = 1/B
            # TODO:split 
            self.density_data[links_near] += sigma*weight
            self.sh_data[links_near] += rgb*weight
            signal[signal_near_idx] += signal[signal_center_idx]
        else:
            sigma, rgb = self._fetch_links(links_near)
            self.density_data[links_center] += sigma.mean(dim=0)
            self.sh_data[links_center] += rgb.mean(dim=0)
            signal[signal_center_idx] += signal[signal_near_idx].mean()

        return signal
# ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
def finite_difference_march(pos, grid, epsilon=1, tolerance=1e-4):
    # x: [N, 3]
    # f(x+h, y, z), f(x, y+h, z), f(x, y, z+h) - f(x-h, y, z), f(x, y-h, z), f(x, y, z-h)
    pos = pos.long()
    # keep the grid no nan
    grid = torch.nan_to_num(grid, 0)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    
    sel_x = torch.logical_and(x>=0, x<grid.size(0))
    sel_y = torch.logical_and(y>=0, y<grid.size(1))
    sel_z = torch.logical_and(z>=0, z<grid.size(2))
    
    sel = torch.logical_and(sel_x, sel_y)
    sel = torch.logical_and(sel, sel_z)
    
    x = x[sel]
    y = y[sel]
    z = z[sel]
    pos = pos[sel]
        
    xl, xr = torch.clamp(x - epsilon, min=0), torch.clamp(x + epsilon, max=grid.size(0) - 1)
    yl, yr = torch.clamp(y - epsilon, min=0), torch.clamp(y + epsilon, max=grid.size(1) - 1)
    zl, zr = torch.clamp(z - epsilon, min=0), torch.clamp(z + epsilon, max=grid.size(2) - 1)  
       
    delta_x = 0.5 * (grid[xr, y, z] - grid[xl, y, z])
    delta_y = 0.5 * (grid[x, yr, z] - grid[x, yl, z])
    delta_z = 0.5 * (grid[x, y, zr] - grid[x, y, zl])
    
    norm = torch.stack([
        delta_x / epsilon,
        delta_y / epsilon,
        delta_z / epsilon
    ], dim=-1)
    
    # norm filter 
    mask = torch.all(torch.abs(norm)>=tolerance, dim=-1)
    norm = norm[mask]
    pos = pos[mask]
    norm = rearrange(norm, 'B D -> D B') / torch.norm(norm, dim=-1)
    norm = rearrange(norm, 'D B -> B D')
    
    # nan to 0 if there is still nan value
    norm = torch.nan_to_num(norm, 0)
    pos = torch.ceil(pos + epsilon*norm)
    return pos



if __name__ == "__main__":
    pos = torch.tensor([[0,0,0], [1, 1, 1]])
    grid = torch.tensor([[[6.,3.,5.], [4., 9., 8.]],[[6.,3.,5.], [4., 9., 8.]]])
    # a = finite_difference_normal(pos, grid)
