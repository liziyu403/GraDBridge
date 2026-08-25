import datetime
import logging
import math
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


logger = logging.getLogger(__name__)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        cudnn.benchmark, cudnn.deterministic = True, False


def date_modified(path=__file__):
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month}-{t.day}'


def git_describe(path=Path(__file__).parent):
    command = f'git -C {path} describe --tags --long --always'
    try:
        return subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).decode()[:-1]
    except subprocess.CalledProcessError:
        return ''


def select_device(device='', batch_size=None):
    message = f'YOLOv5 🚀 {git_describe() or date_modified()} torch {torch.__version__} '
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        count = torch.cuda.device_count()
        if count > 1 and batch_size:
            assert batch_size % count == 0, f'batch-size {batch_size} not multiple of GPU count {count}'
        indent = ' ' * len(message)
        for i, selected in enumerate(device.split(',') if device else range(count)):
            props = torch.cuda.get_device_properties(i)
            message += f"{'' if i == 0 else indent}CUDA:{selected} ({props.name}, {props.total_memory / 1024 ** 2}MB)\n"
    else:
        message += 'CPU\n'

    logger.info(message.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else message)
    return torch.device('cuda:0' if cuda else 'cpu')


def time_synchronized():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def intersect_dicts(source, target, exclude=()):
    return {k: v for k, v in source.items()
            if k in target and not any(x in k for x in exclude) and v.shape == target[k].shape}


def initialize_weights(model):
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            module.eps = 1e-3
            module.momentum = 0.03
        elif type(module) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            module.inplace = True


def model_info(model, verbose=False, img_size=640):
    parameter_count = sum(x.numel() for x in model.parameters())
    gradient_count = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' %
              ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, parameter) in enumerate(model.named_parameters()):
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, parameter.requires_grad, parameter.numel(), list(parameter.shape),
                   parameter.mean(), parameter.std()))
    logger.info(f'Model Summary: {len(list(model.modules()))} layers, '
                f'{parameter_count} parameters, {gradient_count} gradients')


def copy_attr(target, source, include=(), exclude=()):
    for key, value in source.__dict__.items():
        if (include and key not in include) or key.startswith('_') or key in exclude:
            continue
        setattr(target, key, value)


class ModelEMA:
    def __init__(self, model, decay=0.9999, updates=0):
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            decay = self.decay(self.updates)
            state = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for key, value in self.ema.state_dict().items():
                if value.dtype.is_floating_point:
                    value *= decay
                    value += (1.0 - decay) * state[key].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        copy_attr(self.ema, model, include, exclude)
