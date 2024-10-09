# Copyright (c) 2024, Scaled Foundations Inc
# Based on https://github.com/karpathy/llm.c/blob/master/train_gpt2.py

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

# using a global to toggle flash-attention
FLASH = 0

# -----------------------------------------------------------------------------
# Mamba stuff
from mamba_ssm.models.config_mamba import MambaConfig
from matmamba import MatMambaLMHeadModel

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total = int(ntok_total) + int(shard_ntok)
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse
    import tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--mamba_layer_type", type=str, default="mamba2", help="mamba2|mamba1")
    parser.add_argument("--model", type=str, default="130m", help="130m|370m|790m|1.4b|2.8b")
    parser.add_argument("--model_path", type=str, default="", help="path to model weights to load (to finetune/continue training from)")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--beta1", type=float, default=0.9, help="adamw beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="adamw beta2")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"130m", "370m", "790m", "1.4b", "2.8b"}

    
    # Matryoshka parameters
    MATRYOSHKA = True
    mrl_nested_levels = [1, 2, 4, 8] # We will divide the embedding dimension by these scalars in each forward pass

    mode = "scratch"
    if args.model_path:
        mode = "finetune"
    # experiment string for logging
    experiment_str = f"mat_{MATRYOSHKA}_{args.model}_{mode}_steps_{args.num_iterations}_b_{args.batch_size}_btotal_{args.total_batch_size}_l_{args.sequence_length}_lr_{args.learning_rate}_wd_{args.weight_decay}_gc_{args.grad_clip}_dtype_{args.dtype}_flash_{args.flash}_zero_{args.zero_stage}_beta1_{args.beta1}_beta2_{args.beta2}"
    print0(f"experiment string: {experiment_str}")

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
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
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

    # turn on/off flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash

    # init the model
    #TODO: Handle mamba layer type for mamba1 and mamba2
    model_config = {
        "130m": MambaConfig(n_layer=24, d_model=768),
        "370m": MambaConfig(n_layer=48, d_model=1024),
        "790m": MambaConfig(n_layer=48, d_model=1536),
        "1.4b": MambaConfig(n_layer=48, d_model=2048),
        "2.8b": MambaConfig(n_layer=64, d_model=2560),
    }[args.model]
    model = MatMambaLMHeadModel(model_config)

    print0(model)

    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # load model weights if provided
    if args.model_path:
        state_dict = torch.load(args.model_path, map_location=device)

        # Trim vocab to match the model
        if state_dict['lm_head.weight'].shape[0] != raw_model.lm_head.weight.shape[0]:
            state_dict['lm_head.weight'] = state_dict['lm_head.weight'][:raw_model.lm_head.weight.shape[0]]
        if state_dict['backbone.embedding.weight'].shape[0] != raw_model.backbone.embedding.weight.shape[0]:
            state_dict['backbone.embedding.weight'] = state_dict['backbone.embedding.weight'][:raw_model.backbone.embedding.weight.shape[0]]
        raw_model.load_state_dict(state_dict)
            
        print0(f"loaded model weights from {args.model_path}")


    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # main training loop

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
    # optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95))

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
    best_val_loss = float('inf')
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                if MATRYOSHKA:
                    val_loss = [0.0 for _ in range(len(mrl_nested_levels))]
                    inference_times = [0.0 for _ in range(len(mrl_nested_levels))]
                else:
                    val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    if MATRYOSHKA:
                        for level_idx in range(len(mrl_nested_levels)):
                            val_start_time = time.time()
                            logits = model(x, mrl_level=mrl_nested_levels[level_idx]).logits
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                            val_loss[level_idx] += loss.item()
                            inference_times[level_idx] += time.time() - val_start_time
                    else:
                        logits = model(x).logits
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                        val_loss += loss.item()    

                if MATRYOSHKA:
                    for val_loss_idx in range(len(val_loss)):
                        val_loss_item = val_loss[val_loss_idx]
                        val_loss_item /= args.val_max_steps
                        print0(f"val loss level {mrl_nested_levels[val_loss_idx]}: {val_loss_item:.6f}, inference time: {inference_times[val_loss_idx]:.6f}")
                        if master_process and logfile is not None:
                            with open(logfile, "a") as f:
                                f.write("s:%d level:%d tel:%f\n" % (step, mrl_nested_levels[val_loss_idx], val_loss_item))
                else:
                    val_loss /= args.val_max_steps
                    # log to console and to file
                    print0(f"val loss {val_loss}")
                    if master_process and logfile is not None:
                        with open(logfile, "a") as f:
                            f.write("s:%d tel:%f\n" % (step, val_loss))

            # if the val_loss is better than the best so far, save the model weights from rank 0
            if master_process:
                if MATRYOSHKA:
                    # if loss of any nested model is better than best seen so far
                    if min(val_loss) < best_val_loss:
                        best_val_loss = min(val_loss)
                else:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                print0(f"saving model weights to {args.output_dir}")
                save_time = time.time()
                torch.save(model.state_dict(), os.path.join(args.output_dir, f"best_model_{experiment_str}.pt"))
                print0(f"model save time: {time.time() - save_time:.2f}s")
        # once in a while perform model inference on the master process
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
            and master_process:
            # TODO: Implement eval generation
            pass

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        if not MATRYOSHKA:
            mrl_nested_levels = [1]
        for level in mrl_nested_levels:
            for micro_step in range(grad_accum_steps):
                # fetch a batch
                if not args.overfit_single_batch \
                    or (args.overfit_single_batch and step == 0 and micro_step == 0):
                    x, y = train_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                # forward pass
                with ctx:
                    logits = model(x, mrl_level=level).logits
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    # we have to scale the loss to account for gradient accumulation,
                    # because the gradients just add on each successive backward().
                    # addition of gradients corresponds to a SUM in the objective, but
                    # instead of a SUM we want MEAN, so we scale the loss here
                    loss = loss / (grad_accum_steps*len(mrl_nested_levels))
                    lossf += loss.detach() # keep track of the mean loss
                # backward pass
                if ddp:
                    # we want only the last micro-step to sync grads in a DDP model
                    # the official way to do this is with model.no_sync(), but that is a
                    # context manager that bloats the code, so we just toggle this variable
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                if not args.inference_only:
                    loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

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
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
