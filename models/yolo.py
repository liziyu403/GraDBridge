
import logging
from copy import deepcopy
from pathlib import Path
import math
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from models.common import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible
from utils.torch_utils import model_info, initialize_weights


class SharedGNSB(nn.Module):
    """在构图阶段注册一个共享的 GNSBOperatorFusion 实例；forward 恒等返回。"""
    _registry = {}

    def __init__(self, name: str, **cfg):
        super().__init__()
        assert isinstance(name, str) and name, "SharedGNSB requires a non-empty 'name'"
        cfg = dict(cfg)
        assert 'dim' in cfg, "SharedGNSB config must contain 'dim'"
        dim = cfg.pop('dim')
        self.name = name
        self.inner = GNSBOperatorFusion(dim, **cfg)
        SharedGNSB._registry[name] = self.inner

    def forward(self, x):
        return x


class ApplySharedGNSB(nn.Module):
    """按 name 取出共享融合器并调用；支持可选 scale_id 切换。"""
    def __init__(self, name: str, scale_id: int = None):
        super().__init__()
        assert isinstance(name, str) and name, "ApplySharedGNSB requires a non-empty 'name'"
        self.name = name
        self.scale_id = scale_id
        self.inner = None

    def _resolve(self):
        if self.inner is None:
            self.inner = SharedGNSB._registry.get(self.name, None)
            if self.inner is None:
                raise RuntimeError(
                    f"ApplySharedGNSB: shared module '{self.name}' not found. "
                    f"Add a SharedGNSB(name='{self.name}', ...) earlier in the graph."
                )

    def forward(self, xs):
        self._resolve()
        if hasattr(self.inner, "set_scale") and self.scale_id is not None:
            self.inner.set_scale(int(self.scale_id))
        return self.inner(xs)


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



class Model(nn.Module):

    def __init__(self, cfg='models/config/GNSBOperatorBiasing_SingleFusion.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        m = self.model[-1]  # Detect()

        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2):
        return self.forward_once(x, x2)

    def forward_once(self, x, x2):
        y = []
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                if m.f != -4:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if m.f == -4:
                x = m(x2)
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)




def parse_model(d, ch):  # model_dict, input_channels(3)
    def get_c(idx: int):
        """-4 代表原始 x2（NIR），其它保持不变"""
        return 3 if idx == -4 else ch[idx]
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings

        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain

        STRUCT_BLOCKS = [Conv, Bottleneck, SPP, Focus, C3, C3TR]

        if m in STRUCT_BLOCKS:
            if m is Focus:
                c1, c2 = 3, args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1 = get_c(f)  # -4 is the raw NIR input
                c2 = args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [C3, C3TR]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is nn.BatchNorm2d:
            args = [get_c(f)]

        elif m is Concat:
            c2 = sum([get_c(x) for x in f])
        elif m is Add:
            c2 = get_c(f[0])
            args = [c2]
        elif m is Add2:
            c2 = get_c(f[0])
            args = [c2, args[1]]

        elif m is SharedGNSB:
            kw = {}
            if len(args) > 0 and isinstance(args[-1], dict):
                kw = args.pop(-1)
            assert 'name' in kw, "SharedGNSB requires 'name' in config"
            name = kw.pop('name')

            c2 = get_c(f)
            args = [name, kw]
            kwargs = {}


        elif m is ApplySharedGNSB:
            kw = {}
            if len(args) > 0 and isinstance(args[-1], dict):
                kw = args.pop(-1)
            assert 'name' in kw, "ApplySharedGNSB requires 'name' in config"
            name = kw.pop('name')
            scale_id = kw.pop('scale_id', None)
            c2 = get_c(f[0]) if isinstance(f, (list, tuple)) else get_c(f)
            args = [name, scale_id]
            kwargs = {}


        elif m is GNSBOperatorFusion:
            kw = {}
            if len(args) > 0 and isinstance(args[-1], dict):
                kw = args.pop(-1)

            in_ch = ch[f[0]] if isinstance(f, list) and len(f) > 0 else ch[f]

            dim = kw.pop('dim', args[0] if len(args) > 0 else in_ch)

            if 'recon_dim' not in kw:
                kw['recon_dim'] = in_ch

            if dim != in_ch:
                print(f"[WARN] GNSBOperatorFusion layer {i}: yaml dim={dim} != in_ch={in_ch}, override dim -> {in_ch}")
                dim = in_ch

            args = [dim]
            if kw:
                args.append(kw)

            c2 = in_ch

        elif m is Detect:
            args.append([get_c(x) for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = get_c(f)

        kwargs = {}
        if len(args) > 0 and isinstance(args[-1], dict):
            kwargs = args.pop(-1)
        m_ = nn.Sequential(*[m(*args, **kwargs) for _ in range(n)]) if n > 1 else m(*args, **kwargs)

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
