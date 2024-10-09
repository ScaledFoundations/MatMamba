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

if __name__ == '__main__':
    import time
    import argparse
    from tqdm import tqdm
    print0(f"Running pytorch {torch.version.__version__}")

    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="/mnt/raid/data/ffcvimagenet/train_500_0.0_100.ffcv", help="input .ffcv file to train on")
    parser.add_argument("--input_val_bin", type=str, default="/mnt/raid/data/ffcvimagenet/val_500_0.0_100.ffcv", help="input validation set .ffcv file")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="matmamba2vision_base_patch16_224", help="The timm model to use")
    parser.add_argument("--model_path", type=str, default="", help="path to model weights to load (to finetune/continue training from)")
    parser.add_argument("--num_workers", type=int, default=12, help="number of data loader workers")
    parser.add_argument("--in_memory", type=int, default=1, help="cache the dataset in memory")
    # model configuration
    parser.add_argument("--d_model", type=int, default=768, help="model dimension")
    parser.add_argument("--n_layers", type=int, default=24, help="number of layers")
    # token layout for each step of the optimization
    parser.add_argument("--image_size", type=int, default=224, help="image size (e.g. 224, 256, 384)")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size per GPU, in units of #batch dimensions")
    parser.add_argument("--patch_size", type=int, default=16, help="image patch size (e.g. 14, 16, 32)")
    parser.add_argument("--total_batch_size", type=int, default=4096, help="total desired batch size, in units of #images")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=31200, help="number of training steps to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="decay learning rate to this fraction of original")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    parser.add_argument("--beta1", type=float, default=0.9, help="adamw beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="adamw beta2")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="dropout rate for the head")
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="drop path rate")
    parser.add_argument("--proj_drop_rate", type=float, default=0.1, help="dropout rate for the MLP's inside the transformer")
    
    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")

    args = parser.parse_args()

    MATRYOSHKA = True
    mrl_nested_levels = [1, 2, 4, 8]

    mode = "scratch"
    if args.model_path:
        mode = "finetune"
    experiment_str = f"imagenet_mat_{MATRYOSHKA}_{args.model}_d_{args.d_model}_layers_{args.n_layers}_{mode}_steps_{args.num_iterations}_b_{args.batch_size}_btotal_{args.total_batch_size}_lr_{args.learning_rate}_wd_{args.weight_decay}_gc_{args.grad_clip}_dtype_{args.dtype}_zero_{args.zero_stage}_beta1_{args.beta1}_beta2_{args.beta2}"
    print0(f"experiment string: {experiment_str}")

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
    USE_MIXUP = True

    # # Create the model here
    # model = timm.create_model(
    #     args.model,
    #     pretrained=False,
    #     drop_rate=args.drop_rate,
    #     drop_path_rate=args.drop_path_rate,
    #     proj_drop_rate=args.proj_drop_rate
    # )
    from matmamba.matmamba2_vision import MatMamba2Vision
    args.compile = 0
    model = MatMamba2Vision(
        d_model=args.d_model,
        n_layer=args.n_layers,
        d_intermediate=0,
        n_classes=1000,
        patch_size=args.patch_size,
        drop_path_rate=args.drop_path_rate,
        proj_drop_rate=args.proj_drop_rate,
    )
    print0(model)

    model.head.bias = torch.nn.Parameter(torch.full((1000,), -6.9, requires_grad=True), requires_grad=True)

    model.train()
    model.to(device)

    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Dataloaders and Augmentations
    train_loader = create_ffcv_dataloader(args.input_bin, custom_transform=None, mode='train', num_workers=args.num_workers, batch_size=args.batch_size, distributed=ddp, in_memory=args.in_memory, device=ddp_local_rank, res=args.image_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = create_ffcv_dataloader(args.input_val_bin, custom_transform=None, mode='val', num_workers=args.num_workers, batch_size=args.batch_size, distributed=ddp, in_memory=args.in_memory, device=ddp_local_rank, res=args.image_size)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=1000
        )

    print0(f"train_loader: {len(train_loader)} batches")
    print0(f"val_loader: {len(val_loader)} batches")

    # -------------------------------------------------------------------------
    # Main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # Configure the optimizer
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in raw_model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    print0(f"using fused AdamW: {use_fused}")
    betas=(args.beta1, args.beta2)
    if zero_stage == 1:
        print0("using ZeroRedundancyOptimizer")
        optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                            lr=args.learning_rate, betas=betas, fused=use_fused)
        optimizer.add_param_group(optim_groups[1])
    else:
        print0("using regular AdamW")
        optimizer = torch.optim.AdamW(optim_groups, lr=args.learning_rate, betas=betas, fused=use_fused)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode

    if MATRYOSHKA:
        val_metrics = {}
        for level in mrl_nested_levels:
            val_metrics[level] =  {
                'top_1_acc': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(device),
                'top_5_acc': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(device),
            }
    else:
        val_metrics = {
            'top_1_acc': torchmetrics.Accuracy(task='multiclass', num_classes=1000).to(device),
            'top_5_acc': torchmetrics.Accuracy(task='multiclass', num_classes=1000, top_k=5).to(device),
        }

    if USE_MIXUP:
        mixup = mixup_fn
        criterion = SoftTargetCrossEntropy()

    num_epochs = args.num_iterations // len(train_loader) + 1
    print0(f"num_epochs: {num_epochs}")

    best_val_metric = 0.0

    for epoch in range(num_epochs+1):

        model.eval()
        with torch.no_grad():
            lossf = 0.0
            val_loss_mrl = {level: 0.0 for level in mrl_nested_levels}
            for it, (images, target) in enumerate(val_loader):
                with ctx:
                    x = images
                    y = target
                    if MATRYOSHKA:
                        for level in mrl_nested_levels:
                            logits = model(x, mrl_level=level)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                            val_loss_mrl[level] += loss.detach()
                            for metric in val_metrics[level].values():
                                metric(logits, y)
                    else:
                        logits = model(x, mrl_level=1)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                        lossf += loss.detach()
                        for metric in val_metrics.values():
                            metric(logits, y)
            lossf /= len(val_loader)
            val_loss_mrl = {level: loss / len(val_loader) for level, loss in val_loss_mrl.items()}

        write_val_checkpoint = False            
        if MATRYOSHKA:
            stats = {level: {k: m.compute().item() for k, m in val_metrics[level].items()} for level in mrl_nested_levels}
            for level in mrl_nested_levels:
                [meter.reset() for meter in val_metrics[level].values()]
                if stats[level]['top_1_acc'] > best_val_metric:
                    best_val_metric = stats[level]['top_1_acc']
                    write_val_checkpoint = True
                print0(f"level: {level} | val loss {val_loss_mrl[level]:.6f} | top-1 acc {100*stats[level]['top_1_acc']:.6f} | top-5 acc {100*stats[level]['top_5_acc']:.6f}")
                if master_process and logfile is not None:
                    with open(logfile, "a") as f:
                        f.write(f"v:{epoch} level:{level} val_loss:{val_loss_mrl[level]:.6f} top1:{stats[level]['top_1_acc']:.6f} top5:{stats[level]['top_5_acc']:.6f}\n")
        else:
            stats = {k: m.compute().item() for k, m in val_metrics.items()}
            [meter.reset() for meter in val_metrics.values()]
            if stats['top_1_acc'] > best_val_metric:
                best_val_metric = stats['top_1_acc']
                write_val_checkpoint = True
            print0(f"val loss {lossf:.6f} | top-1 acc {100*stats['top_1_acc']:.6f} | top-5 acc {100*stats['top_5_acc']:.6f}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write(f"v:{epoch} val_loss:{lossf:.6f} top1:{stats['top_1_acc']:.6f} top5:{stats['top_5_acc']:.6f}\n")

        if write_val_checkpoint:
            print0(f"saving model weights to {args.output_dir}")
            save_time = time.time()
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_{experiment_str}.pt"))
            print0(f"model save time: {time.time() - save_time:.2f}s")

        if epoch == args.num_iterations//len(train_loader)+1:
            break

        # train for one epoch
        model.train()

        for it, (images, target) in enumerate(train_loader):
            t0 = time.time()
            step = epoch * len(train_loader) + it
            lossf = 0.0
            with ctx:
                x = images
                y = target
                if USE_MIXUP:
                    x, y = mixup(x, y)
                    if MATRYOSHKA:
                        level = mrl_nested_levels[step % len(mrl_nested_levels)]
                        logits = model(x, mrl_level=level)
                        loss = criterion(logits, y)
                        loss = loss / (len(mrl_nested_levels))
                    else:
                        logits = model(x, mrl_level=1)
                        loss = criterion(logits, y)
                else: # TODO: Matryoshka without Mixup
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / (grad_accum_steps)
                lossf += loss.detach()

            # backward pass
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                micro_step = grad_accum_steps - 1
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            if not args.inference_only:
                loss.backward()

            if ddp:
                dist.all_reduce(lossf, op=dist.ReduceOp.AVG)

            lossf = lossf.item()
            lossf = lossf * len(mrl_nested_levels) if MATRYOSHKA else lossf

            if step % len(mrl_nested_levels) == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                # determine and set the learning rate for this iteration
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                # step the optimizer
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # wait on the CPU for all device work to end so we get accurate per-iteration timings below
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()

            # time and print
            t1 = time.time()
            # the 0th iteration is often an outlier (much slower) => skip logging it
            tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
            print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
            # log to logile
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d level:%d trl:%f\n" % (step, mrl_nested_levels[step % len(mrl_nested_levels)], lossf))

            # keep track of smooth timings, last 20 iterations
            if step > 0 and step > args.num_iterations - 20:
                timings.append(t1-t0)

            if step >= args.num_iterations - 1:
                break

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
    

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()