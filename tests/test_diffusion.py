import unittest

import torch

from src.diffusion.ddpm import ddim_step, q_sample
from src.diffusion.schedules import make_alpha_bars, make_beta_schedule


class TestDiffusion(unittest.TestCase):
    def test_q_sample_shapes(self):
        B, T, D = 2, 8, 2
        betas = make_beta_schedule("linear", 10)
        schedule = make_alpha_bars(betas)
        r0 = torch.randn(B, T, D)
        t = torch.randint(0, 10, (B,))
        rt, eps = q_sample(r0, t, schedule)
        self.assertEqual(rt.shape, r0.shape)
        self.assertEqual(eps.shape, r0.shape)

    def test_ddim_step_shapes(self):
        B, T, D = 2, 8, 2
        betas = make_beta_schedule("linear", 10)
        schedule = make_alpha_bars(betas)
        rt = torch.randn(B, T, D)
        eps = torch.randn(B, T, D)
        t = torch.full((B,), 5, dtype=torch.long)
        t_prev = torch.full((B,), 4, dtype=torch.long)
        out = ddim_step(rt, eps, t, t_prev, schedule)
        self.assertEqual(out.shape, rt.shape)


if __name__ == "__main__":
    unittest.main()
