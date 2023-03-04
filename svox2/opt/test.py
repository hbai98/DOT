import torch
# ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
def finite_difference_normal(pos, grid, epsilon=1):
    # x: [N, 3]
    # f(x+h, y, z), f(x, y+h, z), f(x, y, z+h) - f(x-h, y, z), f(x, y-h, z), f(x, y, z-h)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    xl, xr = torch.clamp(x - 1, min=0), torch.clamp(x + 1, max=grid.size(0) - 1)
    yl, yr = torch.clamp(y - 1, min=0), torch.clamp(y + 1, max=grid.size(1) - 1)
    zl, zr = torch.clamp(z - 1, min=0), torch.clamp(z + 1, max=grid.size(2) - 1)  
       
    delta_x = 0.5 * (grid[xr, y, z] - grid[xl, y, z])
    delta_y = 0.5 * (grid[x, yr, z] - grid[x, yl, z])
    delta_z = 0.5 * (grid[x, y, zr] - grid[x, y, zl])
    
    normal = torch.stack([
        0.5 * delta_x / epsilon,
        0.5 * delta_y / epsilon,
        0.5 * delta_z / epsilon
    ], dim=-1)

    return normal

def conv_voxel(pos, grid, epsilon=1, deconv=False, inplace=True):
    pos = pos.long()
    B = pos.size(0)
    if not inplace:
        out = grid.clone() # result has the different memory location
    
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    xl, xr = torch.clamp(x - epsilon, min=0), torch.clamp(x + epsilon, max=grid.size(0) - 1)
    yl, yr = torch.clamp(y - epsilon, min=0), torch.clamp(y + epsilon, max=grid.size(1) - 1)
    zl, zr = torch.clamp(z - epsilon, min=0), torch.clamp(z + epsilon, max=grid.size(2) - 1)  
    
    if deconv:
        val = grid[x, y, z]/B
        if inplace:
            grid[torch.cat([xr, xl]), torch.cat([yl, yr]), torch.cat([zl, zr])] += torch.cat([val, val])
            return grid
        else:
            out[torch.cat([xr, xl]), torch.cat([yl, yr]), torch.cat([zl, zr])] += torch.cat([val, val])
            return out
    else:
        if inplace:
            grid[x, y, z] += grid[torch.cat(xr, xl), torch.cat(yl, yr), torch.cat(zl, zr)].mean(dim=0)
            return grid
        else:
            out[x, y, z] += grid[torch.cat(xr, xl), torch.cat(yl, yr), torch.cat(zl, zr)].mean(dim=0)
            return out 

if __name__ == "__main__":
    pos = torch.tensor([[0,0,0], [1, 1, 1]])
    grid = torch.tensor([[[6.,3.,5.], [4., 9., 8.]],[[6.,3.,5.], [4., 9., 8.]]])
    # a = finite_difference_normal(pos, grid)
    a = conv_voxel(pos, grid, deconv=True)
    print(a-grid)
    print(a.shape)

