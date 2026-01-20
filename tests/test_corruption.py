import unittest

import torch

from src.corruptions.keyframes import interpolate_keyframes


class TestCorruptions(unittest.TestCase):
    def test_interpolation_linear(self):
        T = 5
        x = torch.tensor([[0.0], [2.0], [4.0], [6.0], [8.0]])
        mask = torch.tensor([1, 0, 0, 0, 1], dtype=torch.bool)
        y = interpolate_keyframes(x, mask)
        self.assertTrue(torch.allclose(y, x))

    def test_keyframe_clamp(self):
        T = 6
        x = torch.randn(T, 2)
        mask = torch.tensor([1, 0, 0, 0, 0, 1], dtype=torch.bool)
        y = interpolate_keyframes(x, mask)
        r0 = x - y
        r0 = r0 * (~mask).unsqueeze(-1)
        self.assertTrue(torch.all(r0[mask] == 0))


if __name__ == "__main__":
    unittest.main()
