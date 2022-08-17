from math import radians
from re import T
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
from torch.utils.tensorboard import SummaryWriter

class TestMCOTS(unittest.TestCase):
    # python -m unittest test.test_mcots.TestMCOTS
    def setUp(self) -> None:
        self.dset = datasets["auto"](datadir, split='train')
        self.writer = SummaryWriter('/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf/checkpoints/mcots')
        self.mcots = mcots(0.5, [0.5, 0.5, 0.5], 1e-5, device="cuda", writer=self.writer)
        self.rays = self.dset.rays
        directions = self.rays.dirs
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        viewdirs = directions / norms
        self.rays = Rays(self.rays.origins.cuda(), self.rays.dirs.cuda(), viewdirs.cuda())
        self.gt = self.dset.gt.cuda()
        return super().setUp()
    
    def test_reward(self):
        self.mcots.expand([0, 0, 0, 1])
        B, H, W, _ = self.gt.shape
        res, weights = self.mcots.getReward(self.rays)
        res = rearrange(res, '(B H W) C -> B H W C', B=B, H=H)
        mse = F.mse_loss(self.gt, res)
        mse.backward()
        instant_reward = self.mcots._instant_reward(weights, mse)   
        self.mcots.backpropagate(instant_reward)     
        print(weights.sum())
        print(F.mse_loss(self.gt, res))
        print(weights)
        # python -m unittest test.test_mcots.TestMCOTS.test_reward

    def test_select(self):
        self.mcots.expand([0, 0, 0, 1])
        B, H, W, _ = self.gt.shape   
        res, weights = self.mcots.getReward(self.rays)
        res = rearrange(res, '(B H W) C -> B H W C', B=B, H=H)
        mse = F.mse_loss(self.gt, res)
        self.mcots.select(weights, mse)
        # python -m unittest test.test_mcots.TestMCOTS.test_select
        
    def test_copyfromPlayer(self):
        t1 = SMCT(record=True)
        t2 = SMCT()
        
        t1._refine_at(0, [0,1,0])
        t1._refine_at(1, [0,1,0])
        
        t2._refine_at(0, [0,0,1])
        t2._refine_at(1, [0,0,0])  
        
        t2.data.data[0,0,0,1] += 1     
        
        self.mcots.recorder = t1 
        self.mcots.player = t2 
        
        self.mcots.copyFromPlayer()
        
        print(t1.parent_depth)
        print(t1.child.shape)        
        print(t1.data[0,0,0,1])
        print(t1.n_internal)
        # python -m unittest test.test_mcots.TestMCOTS.test_copyfromPlayer
    
    def test_gt(self):
        from torchvision.utils import save_image
        save_image(rearrange(self.gt[0], 'H W C -> C H W'), 'test.png')
        # python -m unittest test.test_mcots.TestMCOTS.test_gt
        
    def test_run_a_round(self):
        print(self.mcots.player)
        
        self.mcots.run_a_round(self.rays, self.gt, 0)
        
        print(self.mcots.player)
        # print(self.mcots.player.parent_depth)
        print(self.mcots.player.child)
        print(self.mcots.recorder.num_visits)
        # python -m unittest test.test_mcots.TestMCOTS.test_run_a_round
    
    def test_run(self):
        pass