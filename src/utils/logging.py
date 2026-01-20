import os
from typing import Optional


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def create_writer(log_dir: Optional[str]):
    if log_dir is None:
        return NullWriter()
    os.makedirs(log_dir, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir)
    except Exception:
        return NullWriter()
