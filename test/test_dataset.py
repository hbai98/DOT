import sys
sys.path.append('/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf')

from references.svox2.opt.util.dataset import datasets

datadir = '/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf/data/nerf_synthetic/drums'
dset = datasets["auto"](datadir,
                        split='train')
print(dset.gt)

