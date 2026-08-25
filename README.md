# GraDBridge on M3FD

本仓库仅保留 GraDBridge/GNSB 在 M3FD RGB-IR 数据集上的训练与验证代码。

## 目录

- `data/m3ddata.yaml`：M3FD 数据路径和类别定义（`data/` 中唯一的数据配置）
- `configs/hyp.scratch.yaml`：训练超参数
- `models/config/GNSBOperatorBiasing_SingleFusion.yaml`：唯一保留的单尺度 GNSB 融合模型配置
- `models/component/GNSBOperatorBiasing.py`：GNSB 融合模块实现
- `train.py`、`test.py`：训练入口与 epoch 级验证实现

M3FD 数据配置需要提供 `train_rgb`、`val_rgb`、`train_ir`、`val_ir`、`nc` 和 `names`。数据集本身放在仓库外部，并在 `data/m3ddata.yaml` 中填写绝对路径。

## 安装

```bash
pip install -r requirements.txt
```

## 训练

```bash
bash train_m3fd.sh
```

等价的完整命令为：

```bash
python train.py \
  --data data/m3ddata.yaml \
  --hyp configs/hyp.scratch.yaml \
  --cfg models/config/GNSBOperatorBiasing_SingleFusion.yaml \
  --weights yolov5l.pt \
  --batch-size 8 \
  --epochs 200 \
  --device 1
```

如需从零训练，将 `--weights` 设为空字符串。训练过程会使用相同的 M3FD 数据配置在每个 epoch 后执行验证。W&B 默认可用，可通过 `--disable-wandb` 主动关闭。
