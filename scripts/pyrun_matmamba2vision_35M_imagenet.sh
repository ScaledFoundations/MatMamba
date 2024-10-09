#!/bin/bash

# if you wish to train on just a single GPU, simply skip the torchrun part, i.e.
# python train_imagenet.py ... (all the other arguments the same)
torchrun --standalone --nproc_per_node=8 train_imagenet.py \
    --output_dir imagenet_matmamba_log \
    --batch_size 1024 \
    --image_size 224 \
    --patch_size 16 \
    --total_batch_size 8192 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --d_model 512 \
    --n_layers 20 \
    --num_iterations 124800 \
    --weight_decay 0.1 \
    --drop_path_rate 0.1 \
    --drop_rate 0.1 \
    --proj_drop_rate 0.1 \
    --zero_stage 0 \
    --learning_rate 0.005 \
    --beta1 0.9 \
    --beta2 0.999 \
    --warmup_iters 10000 \
    --learning_rate_decay_frac 0.0002
