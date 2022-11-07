#!/bin/bash

# weak-model
CUDA_VISIBLE_DEVICES=3 python compress.py \
    --config=configs/chair.txt \
    --use_mask \
    --mask_weight=1e-11 \
    --grid_bit=8 \
    --use_dwt \
    --dwt_level=1 \
    --datadir=/workspace/dataset/nerf_synthetic/lego \
    --ckpt=log/tensorf_lego_VM/gb8_um1_mw1e-11_ud1_dl1/tensorf_lego_VM.th \
    --compress=1 \
    --decompress=1 \
    --decompress_and_validate=1