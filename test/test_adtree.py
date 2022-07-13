import unittest
from opt.model import AdExternal_N3Tree
from svox import VolumeRenderer
import torch
import svox

class TestSvox(unittest.TestCase):
    # python -m unittest test.test_adtree.TestSvox
    def setUp(self) -> None:
        self.t =AdExternal_N3Tree(data_dim=32, device='cuda')
        return super().setUp()
    
    def test_modules(self):
        # for m in self.t.modules():
        #     print(m)
        for k,v in self.t.named_parameters():
            if v.requires_grad:
                print(k)    
        # python -m unittest test.test_adtree.TestSvox.test_modules
    
    def test_encode_at(self):
        self.t.tree._refine_at(0, (0,0,1))
        self.t.tree._refine_at(0, (0,1,1))
        self.assertEqual(len(self.t.encode_at(1)), 32)
        self.assertEqual(len(self.t.encode_at(2)), 32)
        # python -m unittest test.test_adtree.TestSvox.test_encode_at
    
    def test_reverse_order_treeConv(self):
        self.t.tree._refine_at(0, (0,0,1))
        self.t.tree._refine_at(1, (0,0,1))
        self.t.tree._refine_at(0, (1,0,0))
        self.t.tree._refine_at(0, (1,1,0))
        self.t.tree._refine_at(1, (0,1,1))
        self.t.tree._refine_at(5, (0,1,1))
        
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
        self.t.tree._refine_at(0, (0,0,1))
        self.t.tree._refine_at(1, (0,0,1))
        self.t.tree._refine_at(0, (1,0,0))
        self.t.tree._refine_at(0, (1,1,0))
        self.t.tree._refine_at(1, (0,1,1))
        self.t.tree._refine_at(5, (0,1,1))
        t_out = self.t.encode()
        self.assertTrue(t_out.data.requires_grad)
        self.assertTrue(self.t.tree.data.requires_grad)
        self.assertEqual(t_out.data.size(), torch.Size([7, 2, 2, 2, 4]))
        # python -m unittest test.test_adtree.TestSvox.test_encode
    
    def test_init_gradient(self):
        self.t.tree._refine_at(0, (0,0,1))
        self.t.tree._refine_at(1, (0,0,1))
        self.t.tree._refine_at(0, (1,0,0))
        self.t.tree._refine_at(0, (1,1,0))
        self.t.tree._refine_at(1, (0,1,1))
        self.t.tree._refine_at(5, (0,1,1))
        self.t.cuda()

        target =  torch.tensor([[0.0, 1.0, 0.5]]).cuda()
        ray_ori = torch.tensor([[0.1, 0.1, -0.1]]).cuda()
        ray_dir = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
        ray = svox.Rays(origins=ray_ori, dirs=ray_dir, viewdirs=ray_dir)
        
        lr = 1e-1

        print('GRADIENT DESC')

        leaf_val=self.t.encode()
        tree_out = self.t.out_tree(leaf_val)
        print(tree_out.data.grad_fn)
        # assert 0
        r = VolumeRenderer(tree_out)
        res = r(ray, cuda=True)
        print(res.grad_fn)
        res.sum().backward()
        
        # c2w = torch.tensor([[ -0.9999999403953552, 0.0, 0.0, 0.0 ],
        #             [ 0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708 ],
        #             [ 0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462 ],
        #             [ 0.0, 0.0, 0.0, 1.0 ],
        #      ]).cuda()
        
        # im = r.render_persp(c2w, height=800, width=800, fx=1111.111).clamp_(0.0, 1.0)
        # im.sum().backward()
        
        # print(im.grad_fn)
        # print(self.t.tree.data.grad)        
        print(self.t.depth_weight.grad)
        print(self.t.head_f.fc1.weight.grad)
        print(self.t.dict_convs['1'].conv.weight.grad)
        print(self.t.tree.data.grad)
        # python -m unittest test.test_adtree.TestSvox.test_init_gradient
        
    def test_expand_grad(self):
        self.t.tree._refine_at(0, (0,0,1))
        self.t.tree._refine_at(1, (0,0,1))
        self.t.tree._refine_at(0, (1,0,0))
        self.t.tree._refine_at(0, (1,1,0))
        self.t.tree._refine_at(1, (0,1,1))
        self.t.tree._refine_at(5, (0,1,1))
        self.t.cuda()        
        
        target =  torch.tensor([[0.0, 1.0, 0.5]]).cuda()
        ray_ori = torch.tensor([[0.1, 0.1, -0.1]]).cuda()
        ray_dir = torch.tensor([[0.0, 0.0, 1.0]]).cuda()
        ray = svox.Rays(origins=ray_ori, dirs=ray_dir, viewdirs=ray_dir)  

        leaf_val=self.t.encode()
        tree_out = self.t.out_tree(leaf_val)
        r = VolumeRenderer(tree_out)      
        res = r(ray, cuda=True)
        res.sum().backward()
        
        self.t.expand_grad()
        # python -m unittest test.test_adtree.TestSvox.test_expand_grad