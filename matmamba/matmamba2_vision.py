# Copyright (c) 2024, Scaled Foundations Inc
# Based on the implementation of Mamba2 from Albert Gu and Tri Dao, along with some ideas from the timm library

import math
from functools import partial
from typing import Optional
import json
import os
import copy

from collections import namedtuple

import torch
import torch.nn as nn
from torch import Tensor

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.modules.mamba2 import Mamba2
from matmamba import MatMamba2

from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import timm
from timm.layers import Mlp, PatchEmbed, DropPath
from dataclasses import dataclass, field

@dataclass
class MatMamba2VisionConfig:
    d_model: int
    n_layer: int
    d_intermediate: int = 0
    n_classes: int = 1000
    patch_size: int = 16
    image_size: int = 224
    ssm_cfg: Optional[dict] = field(default_factory=dict)
    attn_layer_idx: Optional[list] = field(default_factory=list)
    attn_cfg: Optional[dict] = field(default_factory=dict)
    norm_epsilon: float = 1e-5
    rms_norm: bool = False
    initializer_cfg: Optional[dict] = field(default_factory=dict)
    fused_add_norm: bool = False
    residual_in_fp32: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    proj_drop_rate: float = 0.0
    pool: str = "cls"
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        mlp_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path_rate=0.0,
        proj_drop_rate=0.0, #TODO: Add dropout to the MLP correctly
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.drop_path_rate = drop_path_rate
        self.proj_drop_rate = proj_drop_rate
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim, proj_drop_rate=proj_drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, **mixer_kwargs
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        hidden_states = self.drop_path(hidden_states)

        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    drop_path_rate=0.0,
    proj_drop_rate=0.0,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "MatMamba2") # Choose mamba1 or mamba2 here
        if ssm_layer not in ["Mamba1", "Mamba2", "MatMamba1", "MatMamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            MatMamba2 if ssm_layer == "MatMamba2" else Mamba2,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path_rate=drop_path_rate,
        proj_drop_rate=proj_drop_rate,
    )
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class MatMamba2Vision(nn.Module):
    def __init__(
        self,
        config: MatMamba2VisionConfig = None,
        d_model: int = 1024,
        n_layer: int = 20,
        d_intermediate: int = 0,
        n_classes: int = 1000,
        patch_size: int = 16,
        image_size: int = 224,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path_rate=0.0,
        proj_drop_rate=0.0,
        pool: str = "cls",
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        if config is not None:
            d_model = config.d_model
            n_layer = config.n_layer
            d_intermediate = config.d_intermediate
            n_classes = config.n_classes
            patch_size = config.patch_size
            image_size = config.image_size
            ssm_cfg = config.ssm_cfg
            attn_layer_idx = config.attn_layer_idx
            attn_cfg = config.attn_cfg
            norm_epsilon = config.norm_epsilon
            rms_norm = config.rms_norm
            initializer_cfg = config.initializer_cfg
            fused_add_norm = config.fused_add_norm
            residual_in_fp32 = config.residual_in_fp32
            drop_path_rate = config.drop_path_rate
            proj_drop_rate = config.proj_drop_rate
            pool = config.pool
            device = config.device
            dtype = config.dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.patch_size = patch_size
        self.image_size = image_size
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, embed_dim=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        n_patches = (image_size // patch_size) ** 2 #TODO: add support for non-square images
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches+1, d_model) * .02)
        self.head = nn.Linear(d_model, n_classes)
        assert pool in ["avg", "cls"], "pool type must be either 'avg' or 'cls'"
        self.pool_type = pool

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]  # stochastic depth decay rule
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path_rate=dpr[i],
                    proj_drop_rate=proj_drop_rate,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def _pos_embed(self, x):
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        x = torch.cat([x] + to_cat, dim=1)
        x = x + self.pos_embed
        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def _reshape_tensor_by_scan_order(self, hidden_states, mode=None, residual=None):
        if self.cls_token is not None:
            cur_cls_tokens = hidden_states[:, -1, :]
            hidden_states = hidden_states[:, :-1, :]
            if residual is not None:
                cur_res_cls_tokens = residual[:, -1, :]
                residual = residual[:, :-1, :]

        indices = torch.arange(0, hidden_states.shape[1]).reshape(
            self.image_size // self.patch_size,
            self.image_size // self.patch_size
        )
        if mode == "w+":
            indices = indices.flatten()
        elif mode == "w-":
            indices = indices.flatten().flip(0)
        elif mode == "h+":
            indices = indices.T.flatten()
        elif mode == "h-":
            indices = indices.T.flatten().flip(0)

        hidden_states = hidden_states[:, indices, :]
        if residual is not None:
            residual = residual[:, indices, :]

        if self.cls_token is not None:
            hidden_states = torch.cat([hidden_states, cur_cls_tokens.unsqueeze(1)], dim=1)
            if residual is not None:
                residual = torch.cat([residual, cur_res_cls_tokens.unsqueeze(1)], dim=1)

        return hidden_states, residual

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MatMamba2VisionConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

    def forward(self, x, inference_params=None, return_features=False, **mixer_kwargs):
        hidden_states = self.patch_embed(x)
        hidden_states = self._pos_embed(hidden_states)
        residual = None

        layer_idx = 0
        modes = ["w+", "w-", "h+", "h-"]
        for layer in self.layers:
            mode = modes[layer_idx % len(modes)]

            # hidden_states, _ = self._reshape_tensor_by_scan_order(hidden_states, mode=mode, residual=None)

            if "mrl_level" in mixer_kwargs:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params, mrl_level=mixer_kwargs["mrl_level"]
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

            # hidden_states, residual = self._reshape_tensor_by_scan_order(hidden_states, mode=mode, residual=residual)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        # cls token pool
        if self.pool_type == "cls":
            hidden_states = hidden_states[:, -1]
        else:
            hidden_states = hidden_states[:, :-1].mean(dim=1)
        if return_features:
            return self.head(hidden_states), hidden_states
        hidden_states = self.head(hidden_states)
        return hidden_states

if __name__ == '__main__':

    config = MatMamba2VisionConfig(
        d_model=1024,
        n_layer=20,
        d_intermediate=0,
        n_classes=1000,
        patch_size=16,
        drop_path_rate=0.1,
        proj_drop_rate=0.1,
    )
    model = MatMamba2Vision(config).cuda()

    # model = timm.create_model('vit_base_patch16_224', pretrained=False).cuda()

    print(model)

    input_tensor = torch.rand(2, 3, 224, 224).cuda()
    output = model(input_tensor)
    print(output.shape)