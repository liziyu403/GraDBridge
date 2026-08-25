import torch

def check_tensor_stats(t, name):
    if not torch.is_tensor(t):
        return

    t = t.detach().cpu().float()
    numel = t.numel()
    num_nan = torch.isnan(t).sum().item()
    num_inf = torch.isinf(t).sum().item()

    print(f"[CHECK] {name}: shape={tuple(t.shape)} dtype={t.dtype}")
    print(f"    min={t.min().item():.4e}, max={t.max().item():.4e}, mean={t.mean().item():.4e}")
    print(f"    NaN: {num_nan}/{numel}, Inf: {num_inf}/{numel}")

def check_batch_file(file_path):
    print(f"Loading batch file: {file_path}")
    batch = torch.load(file_path, map_location="cpu")

    if isinstance(batch, dict):
        for k, v in batch.items():
            if torch.is_tensor(v):
                check_tensor_stats(v, k)
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    if torch.is_tensor(item):
                        check_tensor_stats(item, f"{k}[{i}]")
    elif isinstance(batch, (list, tuple)):
        for i, item in enumerate(batch):
            if torch.is_tensor(item):
                check_tensor_stats(item, f"[{i}]")
    elif torch.is_tensor(batch):
        check_tensor_stats(batch, "batch_tensor")
    else:
        print("Unsupported batch type:", type(batch))
