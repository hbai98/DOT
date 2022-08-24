from math import radians
from re import T
import sys
sys.path.append('/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf')
import unittest
from opt.model.mcots import mcots, SMCT, get_expon_func
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
        self.writer = SummaryWriter('/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf/checkpoints/mcots/int_refine/3')
        self.mcots = mcots(self.dset.scene_radius, self.dset.scene_center, 1e-5, sigma_thresh=1e-3, device="cuda", writer=self.writer, init_refine=3)
        self.rays = self.dset.rays
        directions = self.rays.dirs
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        viewdirs = directions / norms
        self.rays = Rays(self.rays.origins.cuda(), self.rays.dirs.cuda(), viewdirs.cuda())
        self.gt = self.dset.gt.cuda()
        return super().setUp()
    
    def test_reward(self):
        self.mcots.expand([[0, 0, 0, 1]])
        B, H, W, _ = self.gt.shape
        res, weights = self.mcots.getReward(self.rays)
        res = rearrange(res, '(B H W) C -> B H W C', B=B, H=H)
        mse = F.mse_loss(self.gt, res)
        mse.backward()
        print(F.mse_loss(self.gt, res))
        print(weights)
        # python -m unittest test.test_mcots.TestMCOTS.test_reward

    def test_select(self):
        self.mcots.expand([[0, 0, 0, 1]])
        self.mcots.instant_reward = torch.rand(self.mcots.player.child.shape).cuda()
        self.mcots.select(1)
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
        
    def test_prune(self):
        self.mcots.expand([[0, 0, 0, 1]])
        weights = torch.zeros(self.mcots.player.child.shape).cuda()
        delta = 0.05
        self.mcots.prune(delta, weights)
        print(self.mcots.player)
        # python -m unittest test.test_mcots.TestMCOTS.test_prune
    
    def test_backtrace(self):
        from svox import VolumeRenderer
        self.mcots.expand([[0, 0, 0, 1]])
        render = VolumeRenderer(self.mcots.player, 1e-5)
        with self.mcots.player.accumulate_weights(op="sum") as accum:
            res = render.forward(self.rays, cuda=True, fast=False)
        self.mcots.instant_reward = accum.value
        self.mcots.instant_reward/=self.mcots.instant_reward.sum()
        print(self.mcots.player)
        idxs = torch.Tensor([[0, 0, 1, 0], [1, 0, 1, 1]]).cuda()
        self.mcots.backtrace(idxs)
        print(self.mcots.num_visits)
        # python -m unittest test.test_mcots.TestMCOTS.test_backtrace
        
    def test_run_a_round(self):
        print(self.mcots.player)
     
        self.mcots.run_a_round(self.rays, self.gt)
        
        print(self.mcots.player)
        # print(self.mcots.player.parent_depth)
        print(self.mcots.player.child)
        print(self.mcots.num_visits)
        # python -m unittest test.test_mcots.TestMCOTS.test_run_a_round
    
    
    def test_run(self):
        pass