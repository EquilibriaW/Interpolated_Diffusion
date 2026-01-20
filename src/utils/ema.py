from typing import Iterable

import torch


class EMA:
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float = 0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in parameters if p.requires_grad]

    def update(self, parameters: Iterable[torch.nn.Parameter]):
        i = 0
        for p in parameters:
            if not p.requires_grad:
                continue
            self.shadow[i].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            i += 1

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]):
        i = 0
        for p in parameters:
            if not p.requires_grad:
                continue
            p.data.copy_(self.shadow[i])
            i += 1

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        shadow = state["shadow"]
        self.shadow = [t.detach().clone() for t in shadow]
