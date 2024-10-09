#!/bin/bash
torchrun --standalone --nproc_per_node=1 tests/test_val_mixnmatch_imagenet.py\
    --input_val_bin "/mnt/raid/data/ffcvimagenet/val_500_0.0_100.ffcv" \
    --d_model 1024 \
    --n_layers 20 \
    --model_path "matmamba_imagenet_dmodel_1024_nlayers_20.pt" \
