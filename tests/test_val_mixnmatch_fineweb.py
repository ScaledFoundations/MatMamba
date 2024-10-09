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

from mamba_ssm.models.config_mamba import MambaConfig
from matmamba.mixer_seq_simple import MatMambaLMHeadModel

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


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def val_ddp(ddp_model, mixnmatch_dims, val_loader, model_config, debug=True):

    # print("mixnmatch_dims:", mixnmatch_dims)
    # print(len(mixnmatch_dims))
    model = ddp_model.module
    for layer in model.backbone.layers:
        layer.mixer.mixnmatch = True
        layer.mixer.mixnmatch_dims = mixnmatch_dims[layer.layer_idx]

    # Calculate parameters in chosen mixnmatch configuration
    mixnmatch_param_count = 0
    original_param_count = 0
    model_param_count = sum(p.numel() for p in model.parameters())
    model_dim = model_config.d_model
    embedding_params = 50280 * model_dim
    # param_dict = {pn: p for pn, p in model.named_parameters()}
    # print0(param_dict.keys())
    for layer in model.backbone.layers:
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
    
    ddp_model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss = 0.0
        inference_time = 0.0

        for _ in range(args.val_max_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            val_start_time = time.time()
            logits = ddp_model(x).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            val_loss += loss.detach()
            inference_time += time.time() - val_start_time
        
        val_loss /= args.val_max_steps
        print0(val_loss)
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(val_loss)

        print0(mixnmatch_dims)
        print0(f"Total parameters in base configuration: {model_param_count:,}")
        print0(f"Embedding parameters: {embedding_params:,}")
        print0(f"Non embedding parameters in original configuration: {original_param_count:,}")
        print0(f"Non embedding parameters in chosen mixnmatch configuration: {mixnmatch_param_count:,}")
        print0(f"val loss {val_loss.item()}, inference time: {inference_time}")

    return val_loss.item(), mixnmatch_param_count, original_param_count

if __name__ == "__main__":

    import time
    import argparse
    import tiktoken
    # print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    parser.add_argument("--model", type=str, default="130m", help="130m|370m|790m|1.4b|2.8b")
    parser.add_argument("--model_path", type=str, default="", help="path to model weights to load (to finetune/continue training from)")
    # token layout for each step of the optimization
    parser.add_argument("--sequence_length", type=int, default=1024, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="total desired batch size, in units of #tokens")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=1, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=1, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    # huggingface hub
    parser.add_argument("--hf_load", type=int, default=0, help="load model weights from huggingface hub")
    parser.add_argument("--hf_save", type=int, default=0, help="save pretrained model in huggingface hub format")
    # mixnmatch params

    args = parser.parse_args()

    # args error checking and convenience variables
    T = args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"130m", "370m", "790m", "1.4b", "2.8b"}

    model_size = args.model
    batch_size = {
        "130m": 64,
        "370m": 32,
        "790m": 16,
        "1.4b": 8,
        "2.8b": 4,
    }[model_size]
    B = batch_size

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
        # zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        # zero_stage = 0
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
    # print0(f"total desired batch size: {args.total_batch_size}")
    # print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

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

    hf_str = {
        "130m": "scaledfoundations/MatMamba-LM-130M-FineWeb",
        "370m": "scaledfoundations/MatMamba-LM-370M-FineWeb",
        "790m": "scaledfoundations/MatMamba-LM-790M-FineWeb",
        "1.4b": "scaledfoundations/MatMamba-LM-1.4B-FineWeb",
    }

    # load model weights
    if args.hf_load:
        # Load weights from Huggingface
        model = model.from_pretrained(hf_str[args.model])
    elif args.model_path:
        # Load weights from DDP checkpoint
        model_weights = args.model_path
        print0("Loading model weights from:", model_weights)
        state_dict = torch.load(model_weights, map_location={"cuda:0": device}, weights_only=True)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print0(f"loaded model weights from {args.model_path}")

    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    else:
        raise ValueError("must provide a validation dataset")

    # MIXNMATCH BEGINS HERE
    # set up mixnmatch dimensions
    import random
    d_model = model_config.d_model
    n_layers = model_config.n_layer
    d_head = d_model // 32

    y_matmamba = []
    x_matmamba = []
    y_mixnmatch = []
    x_mixnmatch = []

    sampling_strategy = "single"

    if sampling_strategy == "single":
        mixnmatch_dims = [d_model for _ in range(n_layers)]
        y, x, _ = val_ddp(model, mixnmatch_dims, val_loader, model_config, debug=False)
        y_matmamba.append(y)
        x_matmamba.append(x)
    elif sampling_strategy == "custom":
        mixnmatch_dims = [d_model for _ in range(n_layers)]
        for idx in range(n_layers):
            mixnmatch_dims[idx] -= d_head*idx
            if mixnmatch_dims[idx] < d_model//8:
                mixnmatch_dims[idx] = d_model//8
        y, x, _ = val_ddp(model, mixnmatch_dims, val_loader, model_config, debug=False)
        y_mixnmatch.append(y)
        x_mixnmatch.append(x)
    elif sampling_strategy == "constant":
        dims = []
        for i in range(d_model//8, d_model+1, d_head):
            dims.append(i)
        print0(dims)
        for dim in dims:
            mixnmatch_dims = [dim for _ in range(n_layers)]
            y, x, _ = val_ddp(model, mixnmatch_dims, val_loader, model_config, debug=False)
            if len(set(mixnmatch_dims)) == 1 and mixnmatch_dims[0] in [d_model//8, d_model//4, d_model//2, d_model]:
                y_matmamba.append(y)
                x_matmamba.append(x)
            else:
                y_mixnmatch.append(y)
                x_mixnmatch.append(x)
    else:
        dim1, dim2 = 2048, 256
        dims = [dim1 for _ in range(n_layers)]
        idx = 0
        for dim in dims:
            idx += 1
            mixnmatch_dims = [dim for _ in range(n_layers)]
            mixnmatch_dims[-idx:] = [random.choice([dim2]) for _ in range(len(mixnmatch_dims))][-idx:]
            print0(mixnmatch_dims)
            y, x, _ = val_ddp(model, mixnmatch_dims, val_loader, model_config, debug=False)
            
            if len(set(mixnmatch_dims)) == 1 and mixnmatch_dims[0] in [d_model//8, d_model//4, d_model//2, d_model]:
                y_matmamba.append(y)
                x_matmamba.append(x)
            else:
                y_mixnmatch.append(y)
                x_mixnmatch.append(x)

    print0("y_matmamba =", y_matmamba)
    print0("x_matmamba =", x_matmamba)
    print0("y_mixnmatch =", y_mixnmatch)
    print0("x_mixnmatch =", x_mixnmatch)

    if args.hf_save:
        # Save on rank 0
        if ddp_rank == 0:
            model.module.save_pretrained(hf_str[args.model].replace("scaledfoundations/", ""))
            print0(f"saved model weights to {hf_str[args.model]}")

    # barrier to ensure all processes finish
    if ddp:
        dist.barrier()

    # clean up nice
    if ddp:
        destroy_process_group()
    