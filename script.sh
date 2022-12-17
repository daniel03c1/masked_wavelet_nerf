#!/bin/bash

# # weak-model (lego; 1e-11)
# CUDA_VISIBLE_DEVICES=2 python compress.py \
#     --config=configs/chair.txt \
#     --use_mask \
#     --mask_weight=1e-11 \
#     --grid_bit=8 \
#     --use_dwt \
#     --dwt_level=1 \
#     --datadir=/workspace/dataset/nerf_synthetic/lego \
#     --ckpt=log/lego/weak_model_lego.th \
#     --compress=1 \
#     --decompress=1 \
#     --decompress_and_validate=1 \
#     --N_vis=-1

########################### w/o mask reconstruction ###########################

# DWT (compress, decompress)
CUDA_VISIBLE_DEVICES=0 python compress.py \
    --config=configs/chair.txt \
    --datadir=/workspace/dataset/nerf_synthetic/chair \
    --ckpt=log/chair/lv4/lv4.th \
    --reconstruct_mask=0 \
    --compress=1 \
    --decompress=1 \
    --decompress_and_validate=1 \
    --N_vis=5

# DWT (levelwise comrpess, decompress)
CUDA_VISIBLE_DEVICES=0 python compress.py \
    --config=configs/chair.txt \
    --datadir=/workspace/dataset/nerf_synthetic/chair \
    --ckpt=log/chair/lv4/lv4.th \
    --reconstruct_mask=0 \
    --compress=1 \
    --compress_levelwise=1 \
    --decompress=1 \
    --decompress_levelwise=1 \
    --decompress_and_validate=1 \
    --N_vis=5

########################### w/ mask reconstruction ###########################

# DWT (compress, decompress)
CUDA_VISIBLE_DEVICES=0 python compress.py \
    --config=configs/chair.txt \
    --datadir=/workspace/dataset/nerf_synthetic/chair \
    --ckpt=log/dwt/test.th \
    --reconstruct_mask=1 \
    --compress=1 \
    --decompress=1 \
    --decompress_and_validate=1 \
    --N_vis=5

# DWT (levelwise compress, decompress)
CUDA_VISIBLE_DEVICES=0 python compress.py \
    --config=configs/chair.txt \
    --datadir=/workspace/dataset/nerf_synthetic/chair \
    --ckpt=log/dwt/test.th \
    --reconstruct_mask=1 \
    --compress=1 \
    --compress_levelwise=1 \
    --decompress=1 \
    --decompress_levelwise=1 \
    --decompress_and_validate=1 \
    --N_vis=5

# DCT (compress, decompress)
CUDA_VISIBLE_DEVICES=0 python compress_dct.py \
    --config=configs/chair.txt \
    --datadir=/workspace/dataset/nerf_synthetic/chair \
    --ckpt=log/dct/test.th \
    --reconstruct_mask=1 \
    --compress=1 \
    --decompress=1 \
    --decompress_and_validate=1 \
    --N_vis=5
