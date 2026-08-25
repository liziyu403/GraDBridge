
import torch

def _sanitize(x: torch.Tensor, clip: float = 1e4):
    if not torch.is_tensor(x):
        return x
    x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return x.clamp_(-clip, clip)
