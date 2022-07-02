import unittest
from model import AdTree
import torch

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
    