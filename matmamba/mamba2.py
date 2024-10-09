# Copyright (c) 2024, Scaled Foundations Inc
# Based on the Mamba2 code by the original authors: Tri Dao, Albert Gu.
# Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn

# from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from .tensor_parallel import ColumnParallelLinear, RowParallelLinear # Changing this to a local import from the modified tensor_parallel functions
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin


class MatMamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        proj_drop_rate=0.0,
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size
        assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = d_model//32 # Doing this because causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.mixnmatch_dims = d_model
        self.mixnmatch = False

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)
        self.drop1 = nn.Dropout(proj_drop_rate)

        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

        self.drop2 = nn.Dropout(proj_drop_rate)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None, mrl_level=1, use_pytorch_conv=False):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        dim_slice_idx = dim // mrl_level

        # print("self.mixnmatch:", self.mixnmatch)
        # print("self.mixnmatch_dims:", self.mixnmatch_dims)

        if self.mixnmatch:
            dim_slice_idx = self.mixnmatch_dims
            mrl_level = self.d_model // dim_slice_idx
        
        mrl_n_heads = 2*dim_slice_idx // self.headdim
        assert 2*dim_slice_idx % self.headdim == 0, "dim_slice_idx must be divisible by headdim"

        if mrl_n_heads % 8 != 0:
            use_pytorch_conv = True
            # This is a weird idiosyncracy because causal_conv1d with channel last layout requires strides (x.stride(0) and x.stride(2)) to be multiples of 8
            # See https://github.com/state-spaces/mamba/issues/351 and https://github.com/state-spaces/mamba/issues/345#issuecomment-2145035818

        # print("dim_slice_idx:", dim_slice_idx)
        # print("mrl_level:", mrl_level)

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out
        if self.process_group is None:
            w_z, w_x, w_B, w_C, w_dt = torch.split(
                self.in_proj.weight, [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads],
                dim=0
            )
            new_w_z = w_z[:2*dim_slice_idx, :]
            new_w_x = w_x[:2*dim_slice_idx, :]
            new_w_dt = w_dt[:mrl_n_heads, :]
            # TODO: Handle when in_proj.bias is not None
            new_w = torch.cat([new_w_z, new_w_x, w_B, w_C, new_w_dt], dim=0)
            zxbcdt = F.linear(u, new_w, self.in_proj.bias)
        else:
            zxbcdt = self.in_proj(
                u,
                dim_slice_idx=dim_slice_idx,
                mrl_n_heads=mrl_n_heads,
            )  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        zxbcdt = self.drop1(zxbcdt)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None and not use_pytorch_conv:
            # print("mrl_n_heads:", mrl_n_heads, "dim_slice_idx:", dim_slice_idx, "zxbcdt:", zxbcdt.shape)
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(torch.cat((self.conv1d.weight[:(2*dim_slice_idx), :, :], self.conv1d.weight[-2*self.d_state:, :, :]), dim=0), "d 1 w -> d w"),
                torch.cat((self.conv1d.bias[:(2*dim_slice_idx)], self.conv1d.bias[-2*self.d_state:]), dim=0),
                # rearrange(self.conv1d.weight[-((2*dim_slice_idx) + 2*self.d_state):, :, :], "d 1 w -> d w"),
                # self.conv1d.bias[-((2*dim_slice_idx) + 2*self.d_state):],
                self.dt_bias[:mrl_n_heads],
                A[:mrl_n_heads],
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D[:mrl_n_heads],
                chunk_size=self.chunk_size, # TODO: Profile with chunk_size//mrl_level,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight[:2*dim_slice_idx] if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight[:, :2*dim_slice_idx],
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            # if self.mixnmatch:
            #     raise NotImplementedError("mixnmatch not yet supported with this code path")
            # print("zxbcdt:", zxbcdt.shape)
            # print("self.dim_slice_idx:", dim_slice_idx)
            d_mlp = (zxbcdt.shape[-1] - 2 * (2*dim_slice_idx) - 2 * self.ngroups * self.d_state - mrl_n_heads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, (2*dim_slice_idx), (2*dim_slice_idx) + 2 * self.ngroups * self.d_state, mrl_n_heads],
                dim=-1
            )
            # x, B, C = torch.split(xBC, [(2*dim_slice_idx), self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            # z = z[:, :, :2*dim_slice_idx]
            # x = x[:, :, :2*dim_slice_idx]
            # xBC = torch.cat([x, B, C], dim=-1)
            # print("z0:", z0.shape)
            # print("x0:", x0.shape)
            # print("z:", z.shape)
            # print("xBC:", xBC.shape)
            # print("dt:", dt.shape)
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    # print("xBC_t:", xBC_t.shape)
                    # print("conv_state:", conv_state.shape)
                    # TODO: Fix correct indices for conv_state
                    padded_xbct = F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                    conv_state[:, - (2 * self.ngroups * self.d_state):, :].copy_(padded_xbct[:, - (2 * self.ngroups * self.d_state):, :])  # Update state (B D W)
                    conv_state[:, :2*dim_slice_idx, :].copy_(padded_xbct[:, :2*dim_slice_idx, :])
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"] or use_pytorch_conv:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                # xBC = self.act(
                #     self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, -(self.dconv - 1):]
                # )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
                xBC = self.act(
                    F.conv1d(
                        xBC.transpose(1, 2),
                        torch.cat((self.conv1d.weight[:(2*dim_slice_idx), :, :], self.conv1d.weight[-2*self.d_state:, :, :]), dim=0),
                        torch.cat((self.conv1d.bias[:(2*dim_slice_idx)], self.conv1d.bias[-2*self.d_state:]), dim=0),
                        padding=self.conv1d.padding,
                        groups=2*dim_slice_idx + 2*self.ngroups*self.d_state,
                    )[:, :, :seqlen].transpose(1, 2)
                )
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    # rearrange(self.conv1d.weight[-((2*dim_slice_idx) + 2*self.d_state):, :, :], "d 1 w -> d w"),
                    rearrange(torch.cat((self.conv1d.weight[:(2*dim_slice_idx), :, :], self.conv1d.weight[-2*self.d_state:, :, :]), dim=0), "d 1 w -> d w"),
                    bias=torch.cat((self.conv1d.bias[:(2*dim_slice_idx)], self.conv1d.bias[-2*self.d_state:]), dim=0),
                    # bias=self.conv1d.bias[-((2*dim_slice_idx) + 2*self.d_state):],
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [(2*dim_slice_idx), self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A[:mrl_n_heads],
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D[:mrl_n_heads],
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias[:mrl_n_heads],
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state[:, :last_state.shape[1], :last_state.shape[2], :].copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)
            y = rearrange(y, "b l h p -> b l (h p)")
            if self.rmsnorm:
                y = rmsnorm_fn(
                    y,
                    weight=self.norm.weight[:2*dim_slice_idx],
                    bias=self.norm.bias[:2*dim_slice_idx] if self.norm.bias is not None else None,
                    z=z,
                    eps=self.norm.eps,
                    group_size=2*dim_slice_idx // self.ngroups,
                    norm_before_gate=self.norm_before_gate
                )
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = F.linear(y, self.out_proj.weight[:, :2*dim_slice_idx], self.out_proj.bias)
        out = self.drop2(out)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        dim_slice_idx = hidden_states.shape[-1]

        if self.mixnmatch:
            dim_slice_idx = self.mixnmatch_dims
            mrl_level = self.d_model // dim_slice_idx
        
        mrl_n_heads = 2*dim_slice_idx // self.headdim
        assert 2*dim_slice_idx % self.headdim == 0, "dim_slice_idx must be divisible by headdim"

        # zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        u = hidden_states.squeeze(1)
        if self.process_group is None:
            w_z, w_x, w_B, w_C, w_dt = torch.split(
                self.in_proj.weight, [self.d_inner, self.d_inner, self.d_state, self.d_state, self.nheads],
                dim=0
            )
            new_w_z = w_z[:2*dim_slice_idx, :]
            new_w_x = w_x[:2*dim_slice_idx, :]
            new_w_dt = w_dt[:mrl_n_heads, :]
            # TODO: Handle when in_proj.bias is not None
            new_w = torch.cat([new_w_z, new_w_x, w_B, w_C,new_w_dt], dim=0)
            zxbcdt = F.linear(u, new_w, self.in_proj.bias)
        else:
            zxbcdt = self.in_proj(
                u,
                dim_slice_idx=dim_slice_idx,
            )  # (B, L, d_in_proj) or (B * L, d_in_proj)

        d_mlp = (zxbcdt.shape[-1] - 2 * (2*dim_slice_idx) - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, 2*dim_slice_idx, (2*dim_slice_idx) + 2 * self.ngroups * self.d_state, mrl_n_heads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            # conv_state[:, :, -1] = xBC
            conv_state[:, :(2*dim_slice_idx), -1] = xBC[:, :(2*dim_slice_idx)]
            conv_state[:, -2*self.ngroups*self.d_state:, -1] = xBC[:, -2*self.ngroups*self.d_state:]
            xBC = torch.sum(
                torch.cat((conv_state[:, :(2*dim_slice_idx), :], conv_state[:, - 2 * self.ngroups * self.d_state:, :]), dim=1) * rearrange(torch.cat((self.conv1d.weight[:(2*dim_slice_idx), :, :], self.conv1d.weight[-2*self.d_state:, :, :]), dim=0), "d 1 w -> d w"), 
                dim=-1
            )  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + torch.cat((self.conv1d.bias[:(2*dim_slice_idx)], self.conv1d.bias[-2*self.d_state:]), dim=0)
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            #TODO: See if there is a better way to do this than this extremely hacky conv_slice fix
            conv_slice = torch.cat((conv_state[:, :(2*dim_slice_idx), :], conv_state[:, - 2 * self.ngroups * self.d_state:, :]), dim=1)
            xBC = causal_conv1d_update(
                xBC,
                # torch.cat((conv_state[:, :(2*dim_slice_idx), :], conv_state[:, - 2 * self.ngroups * self.d_state:, :]), dim=1),
                # rearrange(self.conv1d.weight[-((2*dim_slice_idx) + 2*self.d_state):, :, :], "d 1 w -> d w"),
                # self.conv1d.bias[:(2*dim_slice_idx) + 2*self.d_state],
                conv_slice,
                rearrange(torch.cat((self.conv1d.weight[:(2*dim_slice_idx), :, :], self.conv1d.weight[-2*self.ngroups*self.d_state:, :, :]), dim=0), "d 1 w -> d w"),
                torch.cat((self.conv1d.bias[:(2*dim_slice_idx)], self.conv1d.bias[-2*self.ngroups*self.d_state:]), dim=0),
                self.activation,
            )
            conv_state[:, :2*dim_slice_idx, :] = conv_slice[:, :2*dim_slice_idx, :]
            conv_state[:, -2*self.ngroups*self.d_state:, :] = conv_slice[:, -2*self.ngroups*self.d_state:, :]

        x, B, C = torch.split(xBC, [(2*dim_slice_idx), self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())[:mrl_n_heads]  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias[:mrl_n_heads].to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D[:mrl_n_heads].to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias[:mrl_n_heads], "h -> h p", p=self.headdim)
            D = repeat(self.D[:mrl_n_heads], "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            # print("x_reshaped:", x_reshaped.shape)
            # print("ssm_state:", ssm_state.shape)
            # print("z:", z.shape)
            y = selective_state_update(
                ssm_state[:, :, :x_reshaped.shape[2], :], x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            # y = self.norm(y, z)
            y = rmsnorm_fn(
                y,
                weight=self.norm.weight[:2*dim_slice_idx],
                bias=self.norm.bias[:2*dim_slice_idx] if self.norm.bias is not None else None,
                z=z,
                eps=self.norm.eps,
                group_size=2*dim_slice_idx // self.ngroups,
                norm_before_gate=self.norm_before_gate
            )
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        # out = self.out_proj(y)
        out = F.linear(y, self.out_proj.weight[:, :2*dim_slice_idx], self.out_proj.bias)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
