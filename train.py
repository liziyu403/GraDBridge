import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import test  # import test.py to get mAP after each epoch
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader_rgb_ir
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_img_size, \
    check_requirements, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_labels, plot_results
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume
from models.yolo import Model

logger = logging.getLogger(__name__)

from models.common import *
from wandb_hooks import set_wandb_step
from utils.batch_nan import check_batch_file


def collect_all_extra_losses(model: nn.Module, device):
    loss_sum = torch.tensor(0.0, device=device)
    flat = {}
    loss_weights = {
        'loss_cfm_rgb2c': 1.0,
        'loss_cfm_nir2c': 1.0,
        'rgb2c_loss_sb_kl': 1.0,
        'nir2c_loss_sb_kl': 1.0,
        'assign_rgb_loss_attn_div': 10.0,
        'assign_nir_loss_attn_div': 10.0,
    }
    for name, m in model.named_modules():
        if not isinstance(m, GNSBOperatorFusion):
            continue
        losses = m.extra_losses()
        for key, weight in loss_weights.items():
            value = losses.get(key)
            if value is None:
                continue
            if not torch.is_tensor(value):
                value = torch.tensor(float(value), device=device)
            weighted = value * weight
            loss_sum = loss_sum + weighted
            flat[f'{name}/{key}'] = weighted
    return loss_sum, flat

def _summ(t: torch.Tensor):
    t32 = t.detach().float()
    finite = torch.isfinite(t32)
    total = t32.numel()
    bad   = int((~finite).sum().item())
    if not finite.any():
        return (f"shape={tuple(t.shape)} dtype={t.dtype} "
                f"all_nonfinite={bad}/{total}")

    tv = t32[finite]
    tmin = tv.min().item()
    tmax = tv.max().item()
    tmean = tv.mean().item()

    return (f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={tmin:.3e} max={tmax:.3e} mean={tmean:.3e} "
            f"nonfinite={bad}/{total}")

def _is_finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()

def assert_all_finite(name: str, t: torch.Tensor) -> bool:
    if not torch.is_tensor(t):
        return True
    ok = _is_finite(t)
    if not ok:
        print(f"[NaN/Inf] {name}: {_summ(t)}")
    return ok

def dump_bad_batch(save_dir, epoch, i, imgs, targets, extra=None):
    try:
        path = f"{save_dir}/nan_batch_e{epoch}_i{i}.pt"
        torch.save({
            "epoch": epoch, "iter": i,
            "imgs": imgs.detach().cpu(),
            "targets": targets.detach().cpu(),
            "extra": extra
        }, path)
        print(f"[DEBUG] 触发 NaN，已保存 batch 到: {path}")
        check_batch_file(path)
    except Exception as e:
        print(f"[DEBUG] 保存失败: {e}")
def _ddp_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()




def train_rgb_ir(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    plots = True
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict

    weights = str(weights).strip() if weights is not None else ''
    has_weight_ckpt = weights.endswith('.pt') and os.path.isfile(weights)

    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights).get('wandb_id') if has_weight_ckpt else None
        wandb_logger = WandbLogger(opt, save_dir.stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming
            weights = str(weights).strip() if weights is not None else ''
            has_weight_ckpt = weights.endswith('.pt') and os.path.isfile(weights)

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    pretrained = has_weight_ckpt
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        in_ch = 6
        anchors_cfg = hyp.get('anchors', None)
        model = Model(opt.cfg, ch=in_ch, nc=nc, anchors=anchors_cfg).to(device)
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys



        mblob = ckpt['model']
        if isinstance(mblob, dict):                    # state_dict checkpoint
            state_dict = {k: v.float() for k, v in mblob.items()}
        else:                                          # 旧格式：整模块
            state_dict = mblob.float().state_dict()

        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)


        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        in_ch      = 6
        anchors_cfg = hyp.get('anchors', None)
        model = Model(opt.cfg, ch=in_ch, nc=nc, anchors=anchors_cfg).to(device)
        logger.info('Training from scratch: no pretrained checkpoint loaded.')

    ema = ModelEMA(model) if rank in [-1, 0] else None



    if rank in [-1, 0] and loggers['wandb']:
        from wandb_hooks import register_wandb_hooks
        register_wandb_hooks(model, log_interval=100)

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path_rgb = data_dict['train_rgb']
    test_path_rgb = data_dict['val_rgb']
    train_path_ir = data_dict['train_ir']
    test_path_ir = data_dict['val_ir']

    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    logger.info(f'Optimizer: {optimizer.__class__.__name__} (use --adam to switch from default SGD)')
    del pg0, pg1, pg2

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        if ema and ckpt.get('ema'):

            if ema and ckpt.get('ema') is not None:
                eblob = ckpt['ema']
                if isinstance(eblob, dict):
                    ema.ema.load_state_dict({k: v.float() for k, v in eblob.items()})
                else:
                    ema.ema.load_state_dict(eblob.float().state_dict())

            ema.updates = ckpt['updates']

        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    dataloader, dataset = create_dataloader_rgb_ir(train_path_rgb, train_path_ir, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    _ddp_barrier()

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)


    if rank in [-1, 0]:

        testloader, testdata = create_dataloader_rgb_ir(test_path_rgb, test_path_ir, imgsz_test, batch_size, gs, opt,
                                            hyp=hyp, augment=False, cache=opt.cache_images, pad=0.5, rect=True, rank=-1,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('val: '))

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            if plots:
                plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    _ddp_barrier()
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,find_unused_parameters=True,
                    )
    _ddp_barrier()          # 只有分布式才真正 barrier
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda, init_scale=2.**10, growth_interval=1000)

    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')


    max_train_batches = opt.max_train_batches if opt.max_train_batches and opt.max_train_batches > 0 else None

    for epoch in range(start_epoch, epochs):
        model.train()

        if opt.image_weights:
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()


        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            assert targets[:, 1].max() < nc, "class id out of range"
            assert (targets[:, 2:6] >= 0).all() and (targets[:, 2:6] <= 1).all(), "bbox normalized coords out of range"

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            imgs_rgb = imgs[:, :3, :, :]
            imgs_ir = imgs[:, 3:, :, :]

            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    imgs_rgb = imgs[:, :3, :, :]
                    imgs_ir = imgs[:, 3:, :, :]

            with torch.amp.autocast('cuda', enabled=cuda):

                if rank in [-1, 0] and wandb_logger.wandb:
                    set_wandb_step(model.module if is_parallel(model) else model, ni)

                pred = model(imgs_rgb, imgs_ir)  # forward

                loss, loss_items = compute_loss(pred, targets.to(device))

                gnsb_loss_sum, gnsb_detail = collect_all_extra_losses(model, device)

                cfm_alpha = float(hyp.get('cfm_alpha', 1.0))
                loss_total = loss + cfm_alpha * gnsb_loss_sum

                box, obj, cls, total = [x.detach() for x in loss_items]
                if not assert_all_finite("loss.box", box) \
                or not assert_all_finite("loss.obj", obj) \
                or not assert_all_finite("loss.cls", cls) \
                or not assert_all_finite("loss.total", total) \
                or not assert_all_finite("loss", loss.detach()):
                    dump_bad_batch(save_dir, epoch, i, imgs, targets, extra={
                        "box": float('nan') if not torch.isfinite(box) else float(box.item()),
                        "obj": float('nan') if not torch.isfinite(obj) else float(obj.item()),
                        "cls": float('nan') if not torch.isfinite(cls) else float(cls.item()),
                        "tot": float('nan') if not torch.isfinite(total) else float(total.item()),
                    })
                    raise RuntimeError("NaN in loss")

                loss_to_bp = loss_total
                if rank != -1:
                    loss_to_bp = loss_to_bp * opt.world_size
                if opt.quad:
                    loss_to_bp = loss_to_bp * 4.0

                scaler.scale(loss_to_bp).backward()

                do_step = (ni % accumulate == 0)

                if (i % 20 == 0) and (not do_step):
                    scale_now = float(scaler.get_scale())
                    tot = 0.0
                    nonfinite = False
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        if not torch.isfinite(g).all():
                            nonfinite = True
                            break
                        tot += (g.float() / max(scale_now, 1e-6)).pow(2).sum().item()
                    gnorm = math.sqrt(tot) if (tot > 0 and not nonfinite) else float("inf")
                    if not math.isfinite(gnorm):
                        print(f"[NaN/Inf] grad_norm = {gnorm} at epoch={epoch} i={i}")
                        dump_bad_batch(save_dir, epoch, i, imgs, targets, extra=f"grad_norm={gnorm}")
                if do_step:
                    scaler.unscale_(optimizer)

                    tot = 0.0
                    bad = False
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        g = p.grad.detach()
                        if not torch.isfinite(g).all():
                            bad = True
                            break
                        tot += g.float().pow(2).sum().item()
                    gnorm = math.sqrt(tot) if tot > 0 else 0.0

                    if bad or not math.isfinite(gnorm):
                        print(f"[NaN/Inf] grad_norm = {gnorm} at epoch={epoch} i={i}")
                        dump_bad_batch(save_dir, epoch, i, imgs, targets, extra=f"grad_norm={gnorm}")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        opt.already_unscaled = False
                        continue

                    base_cap = 10.0
                    cap = base_cap * math.sqrt(accumulate)
                    if gnorm > cap:
                        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cap)
                    else:
                        total_norm = gnorm

                    if not math.isfinite(float(total_norm)):
                        print(f"[NaN/Inf] grad_norm = {float(total_norm)} at epoch={epoch} i={i}")
                        dump_bad_batch(save_dir, epoch, i, imgs, targets, extra=f"grad_norm={float(total_norm)}")
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        opt.already_unscaled = False
                        continue

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    opt.already_unscaled = False
                    if ema:
                        ema.update(model)


            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

            if rank in [-1, 0] and wandb_logger.wandb:
                ni = i + nb * epoch
                box, obj, cls, total = [x.detach() for x in loss_items]

                cur_lrs = {f"lr_batch/group{gi}": pg['lr'] for gi, pg in enumerate(optimizer.param_groups)}

                step_scalars = {
                    "train/step": int(ni),
                    "train_batch/loss_total":    float(loss_total.detach().item()),
                    "train_batch/yolo/box":      float(box.item()),
                    "train_batch/yolo/obj":      float(obj.item()),
                    "train_batch/yolo/cls":      float(cls.item()),
                    "train_batch/yolo/sum":      float(total.item()),
                    "train_batch/extra/sum":     float((cfm_alpha * gnsb_loss_sum.detach()).item()),
                    **cur_lrs,
                }

                for k, v in gnsb_detail.items():
                    step_scalars[f"train_batch/extra/{k}"] = float(v.detach().item())

                wandb_logger.log(step_scalars, step=ni)



            if max_train_batches is not None and (i + 1) >= max_train_batches:
                if rank in [-1, 0]:
                    logger.info(f"Debug limit reached: stopping training epoch after {i + 1} batch(es).")
                break


        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        if rank in [-1, 0]:
            wandb_logger.current_epoch = epoch + 1
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP

                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 max_batches=opt.max_val_batches,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss)

            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 8 % results + '\n')  # append metrics, val_loss
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.75', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch()
            if not opt.nosave or final_epoch:
                net = (model.module if is_parallel(model) else model)

                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),

                    'model': {k: v.detach().float().cpu().clone() for k, v in net.state_dict().items()},
                    'model_yaml': getattr(net, 'yaml', None),        # 便于无 cfg 恢复
                    'is_state_dict': True,

                    'ema': ({k: v.detach().float().cpu().clone() for k, v in getattr(ema, 'ema', ema).state_dict().items()}
                            if ema is not None else None),
                    'updates': int(getattr(ema, 'updates', 0)),

                    'optimizer': optimizer.state_dict(),
                    'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None
                }



                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt


    if rank in [-1, 0]:
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if wandb_logger.wandb:
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='optional initial checkpoint path; empty means train from scratch')
    parser.add_argument('--cfg', type=str, default='models/config/GNSBOperatorBiasing_SingleFusion.yaml',
                        help='GNSB SingleFusion model config')
    parser.add_argument('--data', type=str, default='./data/m3ddata.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='configs/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local-rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train_m3fd', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--disable-wandb', action='store_true', help='disable Weights & Biases logging')
    parser.add_argument('--max-train-batches', type=int, default=0, help='debug: max train batches per epoch, 0 for all')
    parser.add_argument('--max-val-batches', type=int, default=0, help='debug: max val batches per epoch, 0 for all')
    opt = parser.parse_args()

    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    if opt.disable_wandb:
        os.environ['WANDB_DISABLED'] = 'true'
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        check_requirements()

    if opt.global_rank not in (-1, 0):
        os.environ['WANDB_DISABLED'] = 'true'
        wandb_run = None              # 非主进程直接跳过 W&B resume，避免触发 data_dict['train']
    else:
        wandb_run = check_wandb_resume(opt)  # 只有主进程调用

    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = \
            '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.cfg = check_file(opt.cfg)
        opt.data = check_file(opt.data)
        opt.hyp  = check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    logger.info(opt)
    tb_writer = None
    if opt.global_rank in [-1, 0]:
        prefix = colorstr("tensorboard: ")
        logger.info(f"{prefix}Start with tensorboard --logdir {opt.project}")
        tb_writer = SummaryWriter(opt.save_dir)

    train_rgb_ir(hyp, opt, device, tb_writer)
