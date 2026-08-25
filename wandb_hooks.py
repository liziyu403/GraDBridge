import numpy as np
import torch
import torch.nn.functional as F
import wandb
from matplotlib import cm
from torch.nn import Module
from weakref import WeakKeyDictionary

from models.common import GNSBOperatorFusion

_G_STEPS = {}
_G_CFG: "WeakKeyDictionary[Module, dict]" = WeakKeyDictionary()

def _wandb_log(payload: dict, step: int):
    if wandb.run is None:
        return
    try:
        wandb.log(payload, step=step)
    except Exception as e:
        print(f"[wandb hook] log error: {e}")

def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for it in x:
            t = _first_tensor(it)
            if t is not None:
                return t
    return None

def _to_rgb_image(t: torch.Tensor, normalize: bool = True, max_hw: int = 512):
    if t is None:
        raise ValueError("input tensor is None")
    if t.ndim == 4:  # [B,C,H,W]
        t = t[0]
    if t.ndim == 3:  # [C,H,W] -> [H,W]
        t = t.mean(dim=0)
    if t.ndim != 2:
        raise ValueError(f"Expect 2D/3D/4D tensor, got {tuple(t.shape)}")
    t = t.detach().float()
    H, W = t.shape[-2], t.shape[-1]
    if max(H, W) > max_hw:
        s = max_hw / max(H, W)
        nh, nw = int(round(H * s)), int(round(W * s))
        t = F.interpolate(t[None, None], size=(nh, nw), mode="bilinear", align_corners=False)[0, 0]
    if normalize:
        mn, mx = float(torch.min(t)), float(torch.max(t))
        t = (t - mn) / (mx - mn + 1e-8)
    rgb = (cm.viridis(t.cpu().numpy())[..., :3] * 255).astype(np.uint8)
    return rgb

def tensor_to_heatmap(t: torch.Tensor, normalize: bool = True, max_hw: int = 512):
    return wandb.Image(_to_rgb_image(t, normalize, max_hw))

IMPORTANT_KEYS = [
    "Q_proj.weight","K_proj.weight","V_proj.weight",
    "output_proj.weight","proj_in.weight","proj_out.weight","temperature"
]

def _log_params_top(module: Module, module_name: str, step: int):
    for name, p in module.named_parameters(recurse=False):
        if any(k in name for k in IMPORTANT_KEYS):
            try:
                _wandb_log({f"{module_name}/{name}": wandb.Histogram(p.detach().cpu().numpy())}, step)
            except Exception as e:
                print(f"[wandb hook] param log error: {module_name}/{name}: {e}")

def wandb_forward_hook(module: Module, inputs, output):
    cfg = _G_CFG.get(module)
    if not cfg:
        return
    step = _G_STEPS.get(cfg["model_id"], 0)
    if step % cfg["interval"] != 0:
        return

    name, max_hw = cfg["name"], cfg["max_hw"]

    if isinstance(output, (tuple, list)) and len(output) >= 2:
        rgb_feat = _first_tensor(output[0])
        nir_feat = _first_tensor(output[1])
        if isinstance(rgb_feat, torch.Tensor):
            try:
                _wandb_log({f"{name}/rgb_feature_map": tensor_to_heatmap(rgb_feat, True, max_hw)}, step)
            except Exception as e:
                print(f"[wandb hook] rgb feature map log error @ {name}: {e}")
        if isinstance(nir_feat, torch.Tensor):
            try:
                _wandb_log({f"{name}/nir_feature_map": tensor_to_heatmap(nir_feat, True, max_hw)}, step)
            except Exception as e:
                print(f"[wandb hook] nir feature map log error @ {name}: {e}")
    else:
        feat = _first_tensor(output)
        if isinstance(feat, torch.Tensor):
            try:
                _wandb_log({f"{name}/feature_map": tensor_to_heatmap(feat, True, max_hw)}, step)
            except Exception as e:
                print(f"[wandb hook] feature map log error @ {name}: {e}")

    _log_params_top(module, name, step)

def register_wandb_hooks(model: Module, log_interval: int = 100, max_hw: int = 512):
    model_id = id(model)
    _G_STEPS.setdefault(model_id, 0)
    for name, m in model.named_modules():
        if isinstance(m, GNSBOperatorFusion):
            _G_CFG[m] = {"name": name, "interval": int(log_interval), "max_hw": int(max_hw), "model_id": model_id}
            m.register_forward_hook(wandb_forward_hook)
    print("W&B hooks registered")

def set_wandb_step(model: Module, step: int):
    _G_STEPS[id(model)] = int(step)
