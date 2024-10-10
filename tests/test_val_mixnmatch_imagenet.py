# Copyright (c) 2024, Scaled Foundations Inc

import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, RandAugment, RandomErasing, ColorJitter
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

import timm
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import SoftTargetCrossEntropy

import torchmetrics

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

# -------------------------------------------------------------------------
# FFCV Dataloader using RandAug and RandomErasing

def create_ffcv_dataloader(ffcv_file_path, custom_transform=None, mode='train', num_workers=12, batch_size=512, distributed=True, in_memory=1, device=0, res=224):
    this_device = f'cuda:{device}'
    assert os.path.exists(ffcv_file_path), f"{ffcv_file_path} does not exist"
    assert mode in ['train', 'val'], f"mode must be 'train' or 'val'"
    
    res_tuple = (res, res)
    if mode == 'train':
        decoder = RandomResizedCropRGBImageDecoder(res_tuple)

        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            ColorJitter(jitter_prob=0.3),
            RandomErasing(prob=0.25, min_area=0.02, max_area=0.3, min_aspect=0.3, max_count=1),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        drop_last = True

    else:
        decoder = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)

        image_pipeline: List[Operation] = [
            decoder,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        order = OrderOption.SEQUENTIAL
        drop_last = False

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device), non_blocking=True)
    ]

    loader = Loader(
        ffcv_file_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        os_cache=in_memory,
        drop_last=drop_last,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline
        },
        distributed=distributed
    )

    return loader

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def val_ddp(mixnmatch_dims, debug=False):
    import time
    import argparse
    from tqdm import tqdm
    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_val_bin", type=str, default="/mnt/raid/data/ffcvimagenet/val_500_0.0_100.ffcv", help="input validation set .ffcv file")
    parser.add_argument("--model", type=str, default="35m", help="The model to use")
    parser.add_argument("--model_path", type=str, default="", help="path to model weights to load (to finetune/continue training from)")
    parser.add_argument("--num_workers", type=int, default=12, help="number of data loader workers")
    parser.add_argument("--in_memory", type=int, default=1, help="cache the dataset in memory")
    # token layout for each step of the optimization
    parser.add_argument("--image_size", type=int, default=224, help="image size (e.g. 224, 256, 384)")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size per GPU, in units of #batch dimensions")
    parser.add_argument("--patch_size", type=int, default=16, help="image patch size (e.g. 14, 16, 32)")
    parser.add_argument("--total_batch_size", type=int, default=4096, help="total desired batch size, in units of #images")
    # optimization
    parser.add_argument("--drop_rate", type=float, default=0.1, help="dropout rate for the head")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="drop path rate")
    parser.add_argument("--proj_drop_rate", type=float, default=0.1, help="dropout rate for the MLP's inside the transformer")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--tensorcores", type=int, default=1, help="use tensorcores")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # huggingface save and load args
    parser.add_argument("--hf_load", type=int, default=0, help="load model weights from huggingface hub")
    parser.add_argument("--hf_save", type=int, default=0, help="save pretrained model in huggingface hub format")

    args = parser.parse_args()

    # args error checking and convenience variables
    B, patch_size, img_size = args.batch_size, args.patch_size, args.image_size
    assert img_size % patch_size == 0, "image size must be divisible by patch size"
    T = (img_size // patch_size) ** 2
    print0("tokens per image:", T)
    total_batch_size_tokens = args.total_batch_size * T
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert total_batch_size_tokens % tokens_per_fwdbwd == 0
    grad_accum_steps = total_batch_size_tokens // tokens_per_fwdbwd
    print0(f"total desired batch size: {total_batch_size_tokens}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention and mixup
    assert args.flash in {0, 1}
    FLASH = args.flash

    # # Create the model here
    # model = timm.create_model(
    #     args.model,
    #     pretrained=False,
    #     drop_rate=args.drop_rate,
    #     drop_path_rate=args.drop_path_rate,
    #     proj_drop_rate=args.proj_drop_rate
    # )

    args.compile = 0
    from matmamba.matmamba2_vision import MatMamba2Vision, MatMamba2VisionConfig

    if args.model in ["35m", "135m"]:
        config = {
            "35m": MatMamba2VisionConfig(
                n_layer=20,
                d_model=512,
                patch_size=args.patch_size,
                drop_rate=0,
                drop_path_rate=0,
                proj_drop_rate=0,
                n_classes=1000,
            ),
            "135m": MatMamba2VisionConfig(
                n_layer=20,
                d_model=1024,
                patch_size=args.patch_size,
                drop_rate=0,
                drop_path_rate=0,
                proj_drop_rate=0,
                n_classes=1000,
            ),
        }[args.model]

    else:
        config = MatMamba2VisionConfig(
            n_layer=args.n_layers,
            d_model=args.d_model,
            patch_size=args.patch_size,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path_rate,
            proj_drop_rate=args.proj_drop_rate,
            n_classes=1000,
        )

    model = MatMamba2Vision(config)
    print0(model)

    hf_str = {
        "35m": "scaledfoundations/MatMamba-Vision-35M-ImageNet",
        "135m": "scaledfoundations/MatMamba-Vision-135M-ImageNet",
    }[args.model]

    # load model weights if provided
    if args.hf_load:
        # Load weights from Huggingface
        model = model.from_pretrained(hf_str)
        print0(f"loaded model weights from {hf_str}")
    elif args.model_path:
        # Load weights from DDP checkpoint
        model_weights = args.model_path
        print0("Loading model weights from:", model_weights)
        state_dict = torch.load(model_weights, map_location={"cuda:0": device}, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print0(f"loaded model weights from {args.model_path}")

    for layer in model.layers:
        layer.mixer.mixnmatch = True
        layer.mixer.mixnmatch_dims = mixnmatch_dims[layer.layer_idx]

    # Calculate parameters in chosen mixnmatch configuration
    mixnmatch_param_count = 0
    original_param_count = 0
    model_param_count = sum(p.numel() for p in model.parameters())
    model_dim = config.d_model
    embedding_params = sum(p.numel() for p in model.patch_embed.parameters())
    # param_dict = {pn: p for pn, p in model.named_parameters()}
    # print0(param_dict.keys())
    for layer in model.layers:
        if debug:
            print0(f"layer {layer.layer_idx} mixnmatch dims: {layer.mixer.mixnmatch_dims}")
        layer_dim = layer.mixer.mixnmatch_dims
        layer_heads = 2*layer_dim // layer.mixer.headdim
        layer_param_dict = {pn: p for pn, p in layer.named_parameters()}
        layer_running_param_sum = 0
        layer_original_param_sum = 0
        for pn, p in layer_param_dict.items():
            if debug:
                print0(f"layer {layer.layer_idx} param {pn} shape: {p.shape}")
            layer_original_param_sum += p.numel()
            if "in_proj" in pn:
                layer_running_param_sum += p.numel()
                layer_running_param_sum -= ((4*model_dim+2*layer.mixer.d_state+(2*model_dim//layer.mixer.headdim))*p.shape[1])
                layer_running_param_sum += ((4*layer_dim+2*layer.mixer.d_state+layer_heads)*p.shape[1])
            elif "out_proj" in pn:
                layer_running_param_sum += p.numel() - (2*model_dim*p.shape[0]) + (2*layer_dim*p.shape[0])
            elif "conv1d.bias" in pn:
                layer_running_param_sum += p.numel() - 2*model_dim + 2*layer_dim
            elif "conv1d.weight" in pn:
                layer_running_param_sum += p.numel() - (2*model_dim*p.shape[1]*p.shape[2]) + (2*layer_dim*p.shape[1]*p.shape[2])
            elif "A_log" in pn or "dt_bias" in pn or "D" in pn:
                layer_running_param_sum += layer_heads
            elif "norm" in pn:
                layer_running_param_sum += 2*layer_dim
            
        mixnmatch_param_count += layer_running_param_sum
        original_param_count += layer_original_param_sum
        if debug:
            print0(f"layer {layer.layer_idx}  original param count: {layer_original_param_sum:,}")
            print0(f"layer {layer.layer_idx} mixnmatch param count: {layer_running_param_sum:,}")

    if debug:
        print0(f"Total parameters in base configuration: {model_param_count:,}")
        print0(f"Parameters in embedding layer: {embedding_params:,}")
        print0(f"Parameters in LM head: {embedding_params:,}")
        print0(f"Non embedding parameters in original configuration: {original_param_count:,}")
        print0(f"Non embedding parameters in chosen mixnmatch configuration: {mixnmatch_param_count:,}")

        print0(model)

    model.to(device)

    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Dataloaders and Augmentations
    val_loader = None
    if args.input_val_bin:
        val_loader = create_ffcv_dataloader(args.input_val_bin, custom_transform=None, mode='val', num_workers=args.num_workers, batch_size=args.batch_size, distributed=ddp, in_memory=args.in_memory, device=ddp_local_rank, res=args.image_size)
    print0(f"val_loader: {len(val_loader)} batches")

    # -------------------------------------------------------------------------
    # Main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode

    val_metrics = {
        'top_1_acc': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(device),
        'top_5_acc': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(device),
    }

    best_val_metric = 0.0

    model.eval()
    with torch.no_grad():
        lossf = 0.0
        inference_time = 0.0
        for it, (images, target) in enumerate(val_loader):
            with ctx:
                x = images
                y = target
                val_start_time = time.time()
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                lossf += loss.detach()
                inference_time += time.time() - val_start_time
                for metric in val_metrics.values():
                    metric(logits, y)
        lossf /= len(val_loader)

        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
    
        stats = {k: m.compute().item() for k, m in val_metrics.items()}
        [meter.reset() for meter in val_metrics.values()]
        print0(f"val loss {lossf.item():.6f} | top-1 acc {100*stats['top_1_acc']:.6f} | top-5 acc {100*stats['top_5_acc']:.6f}")
        print0(mixnmatch_dims)
        print0(f"Total parameters in base configuration: {model_param_count:,}")
        print0(f"Embedding parameters: {embedding_params:,}")
        print0(f"Non embedding parameters in original configuration: {original_param_count:,}")
        print0(f"Non embedding parameters in chosen mixnmatch configuration: {mixnmatch_param_count:,}")
        print0(f"inference time: {inference_time}")

    if args.hf_save:
        # Save on rank 0
        if ddp_rank == 0:
            print0(f"saving model weights to {hf_str}")
            model.module.save_pretrained(hf_str)
            print0(f"saved model weights to {hf_str}")

    # barrier to ensure all processes finish
    if ddp:
        dist.barrier()

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()

    val_acc = stats['top_1_acc']
    return 100*val_acc, mixnmatch_param_count, original_param_count

if __name__ == "__main__":
    import random
    d_model = 512
    n_layers = 20

    sampling_strategy = "single"

    if sampling_strategy == "single":
        y, x, _ = val_ddp([d_model for _ in range(n_layers)], debug=False)
        print("y =", y)
        print("x =", x)
    elif sampling_strategy == "dims":
        dim_tuples = [(d_model//8, d_model//4), (d_model//4, d_model//8), (d_model//2, d_model//4), (d_model//4, d_model//2), (d_model, d_model//2), (d_model//2, d_model)]
        for dim1, dim2 in dim_tuples:
            y_matmamba = []
            x_matmamba = []
            y_mixnmatch = []
            x_mixnmatch = []
            dims = [dim1 for _ in range(n_layers)]
            idx = 0
            for dim in dims:
                idx += 1
                mixnmatch_dims = [dim for _ in range(n_layers)]
                # mixnmatch_dims[len(mixnmatch_dims)*0:] = [random.choice([dim]) for _ in range(len(mixnmatch_dims))]
                mixnmatch_dims[-idx:] = [random.choice([dim2]) for _ in range(len(mixnmatch_dims))][-idx:]
                y, x, _ = val_ddp(mixnmatch_dims, debug=False)
                
                if len(set(mixnmatch_dims)) == 1 and mixnmatch_dims[0] in [d_model//8, d_model//4, d_model//2, d_model]:
                    y_matmamba.append(y)
                    x_matmamba.append(x)
                else:
                    y_mixnmatch.append(y)
                    x_mixnmatch.append(x)

            print("y_matmamba =", y_matmamba)
            print("x_matmamba =", x_matmamba)
            print("y_mixnmatch =", y_mixnmatch)
            print("x_mixnmatch =", x_mixnmatch)