from math import radians
import sys
sys.path.append('/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf')
import unittest
from opt.model.mcots import mcots, SMCT
from svox import Rays
from references.svox2.opt.util.dataset import datasets
datadir = '/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf/data/nerf_synthetic/drums'
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

class TestMCOTS(unittest.TestCase):
    # python -m unittest test.test_mcots.TestMCOTS
    def setUp(self) -> None:
        self.dset = datasets["auto"](datadir, split='train')
        self.mcots = mcots(0.5, [0.5, 0.5, 0.5], 1e-5, device="cpu")
        self.rays = self.dset.rays
        directions = self.rays.dirs
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        viewdirs = directions / norms
        self.rays = Rays(self.rays.origins.cuda(), self.rays.dirs.cuda(), torch.tensor(viewdirs).cuda())
        self.gt = self.dset.gt.cuda()
        return super().setUp()
    
    def test_reward(self):
        p_tree = SMCT(record=False, radius=0.5, center=[0.5, 0.5, 0.5], device="cuda")
        p_tree._refine_at(0, [0, 0, 1])
        B, H, W, _ = self.gt.shape
        res, weights = self.mcots.getReward(p_tree, self.rays)
        res = rearrange(res, '(B H W) C -> B H W C', B=B, H=H)
        print(weights.sum())
        print(F.mse_loss(self.gt, res))
        print(weights)
        # python -m unittest test.test_mcots.TestMCOTS.test_reward
