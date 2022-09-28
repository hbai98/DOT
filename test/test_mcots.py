from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
import torch.nn.functional as F
import torch
import numpy as np
from references.svox2.opt.util.dataset import datasets
from svox import Rays
from opt.model.mcot import MCOT, SMCT, get_expon_func
import unittest
from math import radians
from re import T
import sys
sys.path.append('/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf')
datadir = '/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf/data/nerf_synthetic/drums'


class TestMCOTS(unittest.TestCase):
    # python -m unittest test.test_mcots.TestMCOTS
    def setUp(self) -> None:
        self.dset = datasets["auto"](datadir, split='train')
        self.mcots = MCOT(self.dset.scene_radius, self.dset.scene_center, 1e-5,
                          sigma_thresh=1e-3, device="cuda",  init_refine=1)
        self.rays = self.dset.rays
        directions = self.rays.dirs
        norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        viewdirs = directions / norms
        self.rays = Rays(self.rays.origins.cuda(),
                         self.rays.dirs.cuda(), viewdirs.cuda())
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
        self.mcots.expand(torch.tensor([[0, 0, 0, 1]]))
        self.mcots.instant_reward = torch.rand(
            self.mcots.tree.child.shape).cuda()
        self.mcots.select(1)
        # python -m unittest test.test_mcots.TestMCOTS.test_select

    def test_copyfromPlayer(self):
        t1 = SMCT(record=True)
        t2 = SMCT()

        t1._refine_at(0, [0, 1, 0])
        t1._refine_at(1, [0, 1, 0])

        t2._refine_at(0, [0, 0, 1])
        t2._refine_at(1, [0, 0, 0])

        t2.data.data[0, 0, 0, 1] += 1

        self.mcots.recorder = t1
        self.mcots.tree = t2

        self.mcots.copyFromPlayer()

        print(t1.parent_depth)
        print(t1.child.shape)
        print(t1.data[0, 0, 0, 1])
        print(t1.n_internal)
        # python -m unittest test.test_mcots.TestMCOTS.test_copyfromPlayer

    def test_gt(self):
        from torchvision.utils import save_image
        save_image(rearrange(self.gt[0], 'H W C -> C H W'), 'test.png')
        # python -m unittest test.test_mcots.TestMCOTS.test_gt

    def test_prune(self):
        # self.mcots.expand(torch.tensor([[0, 0, 0, 1]]))
        print(self.mcots.tree.child)
        print(self.mcots.tree.child[0, 0, 0, 1])
        print(self.mcots.tree.child[1, 0, 1, 1])

        print(self.mcots.tree.child[(*torch.tensor([0, 0, 0, 1]).long(),)])
        # weights = torch.zeros(self.mcots.player.child.shape).cuda()
        # delta = 0.05
        # self.mcots.prune(delta, weights)
        print(self.mcots.tree)
        # python -m unittest test.test_mcots.TestMCOTS.test_prune

    def test_backtrace(self):
        from svox import VolumeRenderer
        self.mcots.expand([[0, 0, 0, 1]])
        render = VolumeRenderer(self.mcots.tree, 1e-5)
        with self.mcots.tree.accumulate_weights(op="sum") as accum:
            res = render.forward(self.rays, cuda=True, fast=False)
        self.mcots.instant_reward = accum.value
        self.mcots.instant_reward /= self.mcots.instant_reward.sum()
        print(self.mcots.tree)
        idxs = torch.Tensor([[0, 0, 1, 0], [1, 0, 1, 1]]).cuda()
        self.mcots.backtrace(idxs)
        print(self.mcots.num_visits)
        # python -m unittest test.test_mcots.TestMCOTS.test_backtrace

    def test_run_a_round(self):
        print(self.mcots.tree)

        self.mcots.run_a_round(self.rays, self.gt)

        print(self.mcots.tree)
        # print(self.mcots.player.parent_depth)
        print(self.mcots.tree.child)
        print(self.mcots.num_visits)
        # python -m unittest test.test_mcots.TestMCOTS.test_run_a_round

    def test_refine(self):
        self.mcots = MCOT(self.dset.scene_radius, self.dset.scene_center, 1e-5, sigma_thresh=1e-3, depth_limit=4,
                          device="cuda", writer=self.writer, init_refine=0)
        self.mcots.tree.data.data[0, 0, 0, 1] += 2
        self.mcots.tree.data.data[0, 0, 1, 1] += 3
        sel = (*torch.Tensor([[0, 0, 0, 1], [0, 0, 1, 1]]).long().T, )
        self.mcots.tree.refine(sel=sel)
        print(self.mcots.tree.data.shape)
        print(self.mcots.tree.data)
        # python -m unittest test.test_mcots.TestMCOTS.test_refine

    def test_time(self):
        self.mcots = MCOT(self.dset.scene_radius, self.dset.scene_center, 1e-5, sigma_thresh=1e-3, depth_limit=4,
                          device="cuda", writer=self.writer, init_refine=3)
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            self.mcots.run_a_round(self.rays, self.gt)
        print(prof)
        # python -m unittest test.test_mcots.TestMCOTS.test_time

    def test_select_internal(self):
        self.mcots.expand(torch.tensor([[0, 0, 0, 1]]))
        print(self.mcots.tree.data.shape)
        print(self.mcots.tree._all_leaves().shape)
        print(self.mcots.tree._all_internals().shape)
        # python -m unittest test.test_mcots.TestMCOTS.test_select_internal

    def test_render_perspective(self):
        import svox
        import imageio
        device = 'cuda'
        t = svox.N3Tree.load(
            "/hpc/users/CONNECT/haotianbai/work_dir/AdaptiveNerf/checkpoints/mcots/test/ckpt_depth_2.npz", device=device)
        r = svox.VolumeRenderer(t)

        # Matrix copied from lego test set image 0
        c2w = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
                            [0.0, -0.7341099977493286,
                                0.6790305972099304, 2.737260103225708],
                            [0.0, 0.6790306568145752,
                                0.7341098785400391, 2.959291696548462],
                            [0.0, 0.0, 0.0, 1.0],
                            ], device=device)

        with torch.no_grad():
            im = r.render_persp(c2w, height=800, width=800,
                                fx=1111.111).clamp_(0.0, 1.0)
        imageio.imwrite('test.png', im.cpu())
        # python -m unittest test.test_mcots.TestMCOTS.test_render_perspective

    def test_policy_pareto(self):
        data = np.array([[2, 3], [1, 1], [4, 1], [5, 2], [3, 3], [2, 3]])
     # python -m unittest test.test_mcots.TestMCOTS.test_policy_pareto

    def test_priority_queue(self):
        d = dict()
        a = ['1', '2']
        for i in a:
            for j in range(int(i)+2, int(i), -1):
                d.setdefault(j, []).append(i)

        print(d)  # prints {1: ['1'], 2: ['1', '2'], 3: ['2']}
        print(d.keys())
        print(d.values())
        print(sorted(d.items()))
        # python -m unittest test.test_mcots.TestMCOTS.test_priority_queue
