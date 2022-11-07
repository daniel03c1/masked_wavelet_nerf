#!/bin/bash

# weak-model (lego; 1e-11)
CUDA_VISIBLE_DEVICES=3 python compress.py \
    --config=configs/chair.txt \
    --use_mask \
    --mask_weight=1e-11 \
    --grid_bit=8 \
    --use_dwt \
    --dwt_level=1 \
    --datadir=/workspace/dataset/nerf_synthetic/lego \
    --ckpt=log/lego/weak_model_lego.th \
    --compress=1 \
    --decompress=1 \
    --decompress_and_validate=1 \
    --N_vis=5

# # strong-model (chair; 1e-10)
# CUDA_VISIBLE_DEVICES=3 python compress.py \
#     --config=configs/chair.txt \
#     --use_mask \
#     --mask_weight=1e-10 \
#     --grid_bit=8 \
#     --use_dwt \
#     --dwt_level=1 \
#     --datadir=/workspace/dataset/nerf_synthetic/chair \
#     --ckpt=log/chair/strong_model_chair_lv1.th \
#     --compress=1 \
#     --decompress=1 \
#     --decompress_and_validate=1 \
#     --N_vis=5