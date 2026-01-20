import unittest

import torch

from src.corruptions.keyframes import build_nested_masks, interpolate_from_mask, sample_fixed_k_mask
from src.sample.sample_generate import get_cond_from_sample
from src.train.train_interp_levels import build_interp_level_batch


class TestInterpSystem(unittest.TestCase):
    def test_fixed_k_mask_exact_count(self):
        T = 10
        K = 4
        mask = sample_fixed_k_mask(T, K, ensure_endpoints=True)
        self.assertEqual(int(mask.sum().item()), K)
        self.assertTrue(bool(mask[0].item()))
        self.assertTrue(bool(mask[T - 1].item()))

    def test_nested_masks_are_nested_and_counts_increase(self):
        T = 16
        K_min = 3
        levels = 3
        masks = build_nested_masks(T, K_min, levels)
        self.assertEqual(len(masks), levels + 1)
        prev_count = None
        for s in range(levels, 0, -1):
            ms = masks[s]
            mprev = masks[s - 1]
            self.assertTrue(torch.all(ms <= mprev))
            if prev_count is not None:
                self.assertGreaterEqual(int(prev_count), int(ms.sum().item()))
            prev_count = mprev.sum().item()

    def test_interpolate_preserves_anchors_exactly(self):
        T = 12
        D = 2
        x = torch.randn(T, D)
        mask = sample_fixed_k_mask(T, 4, ensure_endpoints=True)
        y = interpolate_from_mask(x, mask)
        self.assertTrue(torch.allclose(y[mask], x[mask]))

    def test_stage2_training_constructs_x_s_from_x0_and_M_s(self):
        B, T, D = 2, 8, 2
        x0 = torch.randn(B, T, D)
        gen = torch.Generator().manual_seed(123)
        x_s, mask_s, _ = build_interp_level_batch(x0, K_min=3, levels=2, generator=gen)
        for b in range(B):
            y = interpolate_from_mask(x0[b], mask_s[b])
            self.assertTrue(torch.allclose(x_s[b], y))

    def test_end_to_end_generation_does_not_use_ground_truth_x(self):
        class SampleDict(dict):
            def __getitem__(self, key):
                if key == "x":
                    raise RuntimeError("x should not be accessed")
                return super().__getitem__(key)

        sample = SampleDict(cond={"occ": torch.zeros(1, 2, 2), "start_goal": torch.zeros(4)})
        cond = get_cond_from_sample(sample)
        self.assertIn("occ", cond)
        self.assertIn("start_goal", cond)


if __name__ == "__main__":
    unittest.main()
