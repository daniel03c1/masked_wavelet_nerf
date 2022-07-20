import torch
import torch.nn as nn
import unittest
from utils import *


class UtilsTest(unittest.TestCase):
    def test_remeasure_bbox(self):
        class Density(nn.Module):
            def __init__(self):
                super().__init__()
                self.bbox_min = torch.tensor([ -1, -0.4, 0.2])
                self.bbox_max = torch.tensor([0.8,  0.9, 0.6]) 

            def forward(self, inputs):
                return ((inputs >= self.bbox_min)
                        & (inputs <= self.bbox_max)).all(-1).float()

        density_net = Density()
        bbox = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])

        new_bbox = remeasure_bbox(density_net, bbox, 21)
        torch.testing.assert_close(
            new_bbox, torch.tensor([[-1, -0.5, 0.1], [0.9, 1, 0.7]]))


if __name__ == '__main__':
    unittest.main()

