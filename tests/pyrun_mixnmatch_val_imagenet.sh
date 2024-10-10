#!/bin/bash
torchrun --standalone --nproc_per_node=1 tests/test_val_mixnmatch_imagenet.py\
    --input_val_bin "/mnt/raid/data/ffcvimagenet/val_500_0.0_100.ffcv" \
    --model 135m \
    --hf_load 1 \
