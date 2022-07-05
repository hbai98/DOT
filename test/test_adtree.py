import unittest
from opt.model import AdTree
import torch
import svox

class TestSvox(unittest.TestCase):
    # python -m unittest test.test_adtree.TestSvox
    def setUp(self) -> None:
        self.t =AdTree(data_dim=32)
        return super().setUp()
    
    def test_modules(self):
        for m in self.t.modules():
            print(m)
        # python -m unittest test.test_adtree.TestSvox.test_modules
    
    def test_encode_at(self):
        self.t.expand_at(0, (0,0,1))
        self.t.expand_at(0, (0,1,1))
        self.assertEqual(len(self.t.encode_at(1)), 32)
        self.assertEqual(len(self.t.encode_at(2)), 32)
        # python -m unittest test.test_adtree.TestSvox.test_encode_at
    
    def test_reverse_order_treeConv(self):
        self.t.expand_at(0, (0,0,1))
        self.t.expand_at(1, (0,0,1))
        self.t.expand_at(0, (1,0,0))
        self.t.expand_at(0, (1,1,0))
        self.t.expand_at(1, (0,1,1))
        self.t.expand_at(5, (0,1,1))
        
        gt = torch.tensor([
                        [5, 0, 1, 1],
                        [1, 0, 1, 1],
                        [1, 0, 0, 1],
                        [0, 1, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0]
                          ], dtype=torch.int32)        
        depth, indexes = torch.sort(self.t.parent_depth, dim=0, descending=True)
        for i, d in enumerate(depth):
            idx = d[0]
            xyzi = self.t._unpack_index(idx)
            self.assertTrue(torch.equal(xyzi, gt[i]))
        # python -m unittest test.test_adtree.TestSvox.test_reverse_order_treeConv

    def test_encode(self):
        self.t.expand_at(0, (0,0,1))
        self.t.expand_at(1, (0,0,1))
        self.t.expand_at(0, (1,0,0))
        self.t.expand_at(0, (1,1,0))
        self.t.expand_at(1, (0,1,1))
        self.t.expand_at(5, (0,1,1))
        features = self.t.encode()
        self.assertTrue(features.size(), torch.Size([32]))
        # python -m unittest test.test_adtree.TestSvox.test_encode
    
    def test_init_gradient(self):
        self.t.cuda()
        orig_data = self.t.data.clone()
        
        r = svox.VolumeRenderer(self.t)

        target =  .2*torch.ones((1,31)).cuda()

        ray_ori = torch.tensor([[0.1, 0.1, -0.1]]).cuda()
        ray_dir = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
        ray = svox.Rays(origins=ray_ori, dirs=ray_dir, viewdirs=ray_dir)

        lr = 1e-2

        print('GRADIENT DESC')

        for i in range(20):
            rend = r(ray, cuda=True)
            if i % 5 == 0:
                print(rend.detach()[0].cpu().numpy())
            ((rend - target) ** 2).sum().backward()
            self.t.data.data -= lr * self.t.data.grad
            self.t.zero_grad()
        
        print('TARGET')
        print(target[0].cpu().numpy())
        latest_data = self.t.data
        
        print('Adtree')
        print(self.t.data.requires_grad)
        print(torch.abs(latest_data-orig_data))
        # python -m unittest test.test_adtree.TestSvox.test_init_gradient
        