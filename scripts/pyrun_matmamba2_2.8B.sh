#!/bin/bash

# if you wish to train on just a single GPU, simply skip the torchrun part, i.e.
# python train_fineweb.py ... (all the other arguments the same)
torchrun --standalone --nproc_per_node=8 train_fineweb.py \
    --input_bin "/mnt/raid/data/fineweb100B/fineweb_train_*.bin" \
    --input_val_bin "/mnt/raid/data/fineweb100B/fineweb_val_*.bin" \
    --val_loss_every 250 \
    --sample_every 0 \
    --output_dir pylog_mamba2_124M \
    --mamba_layer_type mamba2 \
    --model 2.8b \
    --batch_size 4 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 0 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 30000 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0006 \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.0 \
    --overfit_single_batch 0
