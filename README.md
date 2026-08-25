# GraDBridge on M3FD

Official implementation of **GraDBridge / GNSB** on the **M3FD RGB-IR multispectral object detection dataset**.

This repository provides a **single-scale, single-fusion version of GraDBridge**, where the GNSB module is applied once for RGB-IR feature fusion. The code is intended to provide a compact and reproducible implementation for training and evaluation on M3FD.

## Overview

<p align="center">
  <img src="Fig/overview.png" width="95%">
</p>

<p align="center">
  <em>Overview of the GraDBridge framework.</em>
</p>

## GNSB Architecture

<p align="center">
  <img src="Fig/detail.png" width="95%">
</p>

<p align="center">
  <em>Detailed architecture of the GNSB fusion module used in the single-scale, single-fusion setting.</em>
</p>

## Visualization

<p align="center">
  <img src="Fig/viz.png" width="95%">
</p>

<p align="center">
  <em>Qualitative detection results on the M3FD dataset.</em>
</p>

## Repository Structure

```text
GraDBridge/
├── configs/
│   └── hyp.scratch.yaml
├── data/
│   └── m3ddata.yaml
├── Fig/
│   ├── detail.png
│   ├── overview.png
│   └── viz.png
├── models/
│   ├── component/
│   │   └── GNSBOperatorBiasing.py
│   └── config/
│       └── GNSBOperatorBiasing_SingleFusion.yaml
├── train.py
├── test.py
└── train_m3fd.sh
```

## Dataset

The M3FD dataset is stored outside this repository. Please specify the absolute dataset paths in:

```text
data/m3ddata.yaml
```

The configuration should contain:

```yaml
train_rgb:
val_rgb:
train_ir:
val_ir:
nc:
names:
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
bash train_m3fd.sh
```

or use the full command:

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


