import argparse
import collections
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.data.wan_synth import create_wan_synth_dataloader
from src.models.latent_straightener import load_latent_straightener
from src.models.wan_backbone import resolve_dtype
from src.selection.oracle_segment_cost import build_oracle_seg_precompute, compute_oracle_cost_seg_mse
from src.selection.epiplexity_dp import build_cost_matrix_from_segments_batch, dp_select_indices_batch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pattern", type=str, required=True)
    p.add_argument("--T", type=int, default=21)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--num_batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--cost_mode",
        type=str,
        default="z_mse",
        choices=["z_mse", "s_mse"],
        help="Oracle per-segment cost. z_mse uses raw latents; s_mse uses straightener.encode(latents).",
    )
    p.add_argument("--model_dtype", type=str, default="bf16")
    p.add_argument("--straightener_ckpt", type=str, default="")
    p.add_argument("--straightener_dtype", type=str, default="")

    # Perf knobs
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle_buffer", type=int, default=500)
    p.add_argument("--chunk_segments", type=int, default=16)
    p.add_argument("--keep_text", type=int, default=1)
    p.add_argument("--keep_text_embed", type=int, default=0)
    p.add_argument("--top_sequences", type=int, default=10)
    p.add_argument("--top_prompts", type=int, default=5)
    return p


def _format_seq(seq: Tuple[int, ...]) -> str:
    return "[" + ",".join(str(x) for x in seq) + "]"


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA required for oracle DP diagnostic.")

    model_dtype = resolve_dtype(args.model_dtype) or torch.bfloat16
    straightener = None
    if args.cost_mode == "s_mse":
        if not args.straightener_ckpt:
            raise ValueError("cost_mode=s_mse requires --straightener_ckpt")
        s_dtype = resolve_dtype(args.straightener_dtype) or model_dtype
        straightener, _ = load_latent_straightener(args.straightener_ckpt, device=device, dtype=s_dtype)
        straightener.eval()

    loader = create_wan_synth_dataloader(
        args.data_pattern,
        batch_size=args.batch,
        num_workers=args.num_workers,
        shuffle_buffer=args.shuffle_buffer,
        shuffle=True,
        shardshuffle=True,
        keep_text=bool(args.keep_text),
        keep_text_embed=bool(args.keep_text_embed),
        seed=args.seed,
    )
    it = iter(loader)
    pre = build_oracle_seg_precompute(int(args.T), device=device)

    seq_counts: Dict[Tuple[int, ...], int] = collections.Counter()
    pos_counts: List[collections.Counter] = [collections.Counter() for _ in range(int(args.K))]
    prompts_by_seq: Dict[Tuple[int, ...], List[str]] = collections.defaultdict(list)
    seqs_by_prompt: Dict[str, List[Tuple[int, ...]]] = collections.defaultdict(list)

    pbar = tqdm(range(int(args.num_batches)), dynamic_ncols=True, desc="oracle-dp")
    for _ in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        latents = batch["latents"].to(device=device, dtype=model_dtype, non_blocking=True)
        if latents.dim() != 5:
            raise ValueError("latents must be [B,T,C,H,W]")
        B, T, C, H, W = latents.shape
        if int(T) != int(args.T):
            raise ValueError(f"T mismatch: batch={T} args={args.T}")

        x = latents
        if args.cost_mode == "s_mse":
            assert straightener is not None
            flat = latents.reshape(B * T, C, H, W)
            with torch.no_grad():
                s_flat = straightener.encode(flat)
            x = s_flat.reshape(B, T, C, H, W).to(dtype=model_dtype)

        cost_seg = compute_oracle_cost_seg_mse(x, pre, chunk_segments=int(args.chunk_segments))  # [B,S]
        Cmat = build_cost_matrix_from_segments_batch(cost_seg, precomp=pre, T=int(args.T))  # type: ignore[arg-type]
        idx = dp_select_indices_batch(Cmat, int(args.K))  # [B,K]

        prompts = batch.get("text")
        if prompts is None:
            prompts = [""] * B
        for b in range(B):
            seq = tuple(int(x) for x in idx[b].detach().cpu().tolist())
            seq_counts[seq] += 1
            for k, t_sel in enumerate(seq):
                pos_counts[k][t_sel] += 1
            if isinstance(prompts, list):
                prompt = str(prompts[b])
            else:
                prompt = str(prompts[b].decode("utf-8")) if hasattr(prompts[b], "decode") else str(prompts[b])
            if prompt:
                prompts_by_seq[seq].append(prompt)
                seqs_by_prompt[prompt].append(seq)

    total = sum(seq_counts.values())
    uniq = len(seq_counts)
    print("\n=== Oracle DP Diagnostic (Wan2.1-synth) ===")
    print(f"cost_mode: {args.cost_mode}")
    if args.cost_mode == "s_mse":
        print(f"straightener_ckpt: {args.straightener_ckpt}")
    print(f"samples: {total} (batch={args.batch} x num_batches={args.num_batches})")
    print(f"T={args.T} K={args.K} segments={int(pre.seg_i.numel())}")
    print(f"unique index sequences: {uniq} ({uniq / max(1,total):.3f} fraction)")

    print(f"\nTop-{int(args.top_sequences)} sequences:")
    for seq, n in seq_counts.most_common(int(args.top_sequences)):
        frac = n / max(1, total)
        print(f"  {n:6d} ({frac:.3f}) { _format_seq(seq) }")
        if prompts_by_seq.get(seq):
            for ptxt in prompts_by_seq[seq][:3]:
                print(f"    prompt: {ptxt[:120]}")

    print("\nPer-position marginal (top-8 each):")
    for k in range(int(args.K)):
        topk = pos_counts[k].most_common(8)
        s = " ".join(f"{t}:{n}" for t, n in topk)
        print(f"  pos{k}: {s}")

    # Prompt grouping if prompts were present/repeated.
    repeated = {p: seqs for p, seqs in seqs_by_prompt.items() if len(seqs) > 1}
    if repeated:
        n_rep = len(repeated)
        uniq_counts = [len(set(seqs)) for seqs in repeated.values()]
        frac_multi = sum(1 for u in uniq_counts if u > 1) / max(1, n_rep)
        avg_uniq = float(sum(uniq_counts) / max(1, n_rep))
        avg_n = float(sum(len(seqs) for seqs in repeated.values()) / max(1, n_rep))
        print(f"\nPrompts with repeats: {n_rep}")
        print(f"  avg repeats per prompt: {avg_n:.2f}")
        print(f"  avg unique seqs per prompt: {avg_uniq:.2f}")
        print(f"  frac prompts with >1 unique seq: {frac_multi:.3f}")
        items = sorted(repeated.items(), key=lambda kv: len(kv[1]), reverse=True)[: int(args.top_prompts)]
        for ptxt, seqs in items:
            uniq_seqs = len(set(seqs))
            print(f"  n={len(seqs):3d} uniq_seqs={uniq_seqs:3d} prompt={ptxt[:120]}")
            counts = collections.Counter(seqs).most_common(3)
            for seq, n in counts:
                print(f"    {n:3d} { _format_seq(seq) }")
        # Show a few prompts where oracle DP differs within the same prompt.
        diff_items = [(p, seqs) for p, seqs in repeated.items() if len(set(seqs)) > 1]
        diff_items = sorted(diff_items, key=lambda kv: len(set(kv[1])), reverse=True)[: min(5, len(diff_items))]
        if diff_items:
            print("\nPrompts with within-prompt DP variation (top-5 by uniq seqs):")
            for ptxt, seqs in diff_items:
                counts = collections.Counter(seqs).most_common(5)
                print(f"  uniq_seqs={len(set(seqs)):3d} n={len(seqs):3d} prompt={ptxt[:120]}")
                for seq, n in counts:
                    print(f"    {n:3d} { _format_seq(seq) }")
    else:
        if bool(args.keep_text):
            print("\nNo repeated prompts observed in sampled batches.")
        else:
            print("\nPrompts not loaded (set --keep_text 1 to enable grouping).")


if __name__ == "__main__":
    main()
