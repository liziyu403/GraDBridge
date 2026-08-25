#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=4 python train.py \
  --data data/m3ddata.yaml \
  --hyp configs/hyp.scratch.yaml \
  --cfg models/config/GNSBOperatorBiasing_SingleFusion.yaml \
  --batch-size 8 \
  --epochs 200 \
  --device 1 \
  --weights yolov5l.pt
