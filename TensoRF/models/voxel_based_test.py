import torch
import torch.nn as nn
import unittest
from voxel_based import *


class UtilsTest(unittest.TestCase):
    def test_PREF(self):
        inputs = torch.rand((32, 3)) # 3D coordinates
        ch, hidden_ch, out_ch = 12, 64, 27
        resolution = (64, 128, 48)

        net = PREF(resolution, ch, hidden_ch, out_ch)

        self.assertEqual(net(inputs).shape, (32, out_ch))


if __name__ == '__main__':
    unittest.main()

