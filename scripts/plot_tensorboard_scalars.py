import argparse
import os
from typing import Dict, List


def _load_scalars(log_dir: str) -> Dict[str, List[object]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tensorboard is required to read event files") from exc

    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={"scalars": 0},
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        data[tag] = ea.Scalars(tag)
    return data


def _plot_group(data: Dict[str, List[object]], tags: List[str], out_path: str, title: str) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    any_tag = False
    plt.figure(figsize=(8, 4.5))
    for tag in tags:
        if tag not in data:
            continue
        events = data[tag]
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        if not steps:
            continue
        plt.plot(steps, vals, label=tag)
        any_tag = True
    if not any_tag:
        plt.close()
        return False
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True, help="TensorBoard log directory")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for PNGs")
    args = parser.parse_args()

    log_dir = args.log_dir
    out_dir = args.out_dir or os.path.join(log_dir, "plots")

    data = _load_scalars(log_dir)

    groups = {
        "train_loss.png": [
            "train/loss",
            "train/recon_loss",
            "train/gt_loss",
            "train/uncertainty_loss",
        ],
        "train_regs.png": [
            "train/edge_loss",
            "train/ms_loss",
            "train/flow_smooth_loss",
            "train/endpoint_loss",
            "train/flow_consistency_loss",
        ],
        "train_uncertainty.png": [
            "train/uncertainty_mean",
        ],
        "val.png": [
            "val/l1",
            "val/lerp_l1",
            "val/unc_corr",
            "val/unc_low",
            "val/unc_high",
        ],
        "throughput.png": [
            "train/samples_per_sec",
            "train/step_time_sec",
            "train/max_mem_gb",
        ],
    }

    wrote_any = False
    for filename, tags in groups.items():
        out_path = os.path.join(out_dir, filename)
        wrote = _plot_group(data, tags, out_path, title=filename.replace(".png", ""))
        wrote_any = wrote_any or wrote

    if not wrote_any:
        raise RuntimeError(f"No scalar tags found in {log_dir}")


if __name__ == "__main__":
    main()
