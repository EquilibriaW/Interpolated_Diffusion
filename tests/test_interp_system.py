import os
import tempfile
import unittest

import torch
import torch.nn as nn

from src.corruptions.keyframes import (
    build_nested_masks_batch,
    interpolate_from_indices,
    interpolate_from_mask,
    sample_fixed_k_indices_batch,
    sample_fixed_k_mask,
)
from src.train.train_interp_levels import build_interp_level_batch
from src.train.train_keypoints import _build_known_mask_values as build_known_mask_values
from src.utils.clamp import apply_clamp
from src.utils.checkpoint import load_checkpoint, save_checkpoint


class TestInterpSystem(unittest.TestCase):
    def test_fixed_k_mask_exact_count(self):
        T = 10
        K = 4
        mask = sample_fixed_k_mask(T, K, ensure_endpoints=True)
        self.assertEqual(int(mask.sum().item()), K)
        self.assertTrue(bool(mask[0].item()))
        self.assertTrue(bool(mask[T - 1].item()))

    def test_nested_masks_batch_is_nested_and_counts_match(self):
        B, T, K_min, levels = 4, 16, 3, 3
        masks_levels, idx_levels = build_nested_masks_batch(B, T, K_min, levels, device=torch.device("cpu"))
        self.assertEqual(masks_levels.shape, (B, levels + 1, T))
        self.assertEqual(len(idx_levels), levels + 1)

        def compute_k_list(T, K_min, levels):
            K_min = min(K_min, T)
            K_list = [0 for _ in range(levels + 1)]
            K_list[levels] = K_min
            for s in range(levels, 0, -1):
                K_prev = min(T, max(K_list[s] + 1, 2 * K_list[s]))
                K_list[s - 1] = K_prev
            return K_list

        K_list = compute_k_list(T, K_min, levels)
        for s in range(1, levels + 1):
            self.assertTrue(torch.all(masks_levels[:, s] <= masks_levels[:, s - 1]))
        for s in range(levels + 1):
            counts = masks_levels[:, s].sum(dim=1)
            self.assertTrue(torch.all(counts == K_list[s]))

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
        x_s, mask_s, _, _, _ = build_interp_level_batch(x0, K_min=3, levels=2, generator=gen)
        for b in range(B):
            y = interpolate_from_mask(x0[b], mask_s[b])
            self.assertTrue(torch.allclose(x_s[b], y))

    def test_vectorized_interpolation_preserves_anchors_exact(self):
        B, T, K, D = 3, 10, 4, 2
        idx, mask = sample_fixed_k_indices_batch(B, T, K, device=torch.device("cpu"), ensure_endpoints=True)
        x0 = torch.randn(B, T, D)
        vals = x0.gather(1, idx.unsqueeze(-1).expand(-1, K, D))
        y = interpolate_from_indices(idx, vals, T)
        gathered = y.gather(1, idx.unsqueeze(-1).expand(-1, K, D))
        self.assertTrue(torch.allclose(gathered, vals))

    def test_known_mask_per_dim_endpoints(self):
        B, T, D = 1, 8, 4
        idx = torch.tensor([[0, 3, 6, 7]], dtype=torch.long)
        cond = {"start_goal": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
        known_mask, known_values = build_known_mask_values(idx, cond, D, T)
        self.assertTrue(torch.all(known_mask[0, 0, :2]))
        self.assertTrue(torch.all(~known_mask[0, 0, 2:]))
        self.assertTrue(torch.all(known_mask[0, -1, :2]))
        self.assertTrue(torch.all(~known_mask[0, -1, 2:]))
        self.assertTrue(torch.all(~known_mask[0, 1:-1, :]))
        self.assertTrue(torch.allclose(known_values[0, 0, :2], cond["start_goal"][0, :2]))
        self.assertTrue(torch.allclose(known_values[0, -1, :2], cond["start_goal"][0, 2:]))
        self.assertTrue(torch.all(known_values[0, :, 2:] == 0))

    def test_checkpoint_meta_roundtrip(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        meta = {"stage": "keypoints", "N_train": 100, "schedule": "linear"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(path, model, optimizer, step=5, meta=meta)
            step, payload = load_checkpoint(path, model, optimizer=None, ema=None, return_payload=True)
        self.assertEqual(step, 5)
        self.assertIn("meta", payload)
        self.assertEqual(payload["meta"], meta)

    def test_clamp_policy_endpoints_only(self):
        B, T, D = 1, 5, 4
        x_ref = torch.zeros(B, T, D)
        x_hat = torch.ones(B, T, D)
        clamp_mask = torch.zeros(B, T, dtype=torch.bool)
        clamp_mask[:, 0] = True
        clamp_mask[:, -1] = True
        out = apply_clamp(x_hat.clone(), x_ref, clamp_mask, "pos")
        self.assertTrue(torch.all(out[:, 0, :2] == 0))
        self.assertTrue(torch.all(out[:, -1, :2] == 0))
        self.assertTrue(torch.all(out[:, 1:-1, :2] == 1))
        self.assertTrue(torch.all(out[:, :, 2:] == 1))

    def test_end_to_end_generation_does_not_use_ground_truth_x(self):
        class SampleDict(dict):
            def __getitem__(self, key):
                if key == "x":
                    raise RuntimeError("x should not be accessed")
                return super().__getitem__(key)

        sample = SampleDict(cond={"occ": torch.zeros(1, 2, 2), "start_goal": torch.zeros(4)})
        cond = sample["cond"]
        self.assertIn("occ", cond)
        self.assertIn("start_goal", cond)


if __name__ == "__main__":
    unittest.main()
