"""Sequential Z-Image transformer reference script.

This file keeps `inference.py` untouched and replaces only the transformer
with a single-class, fully inlined forward pass for easier porting.

Limitations:
- transformer forward only supports a single sample (`len(x) == 1`)
- only supports `patch_size == 2` and `f_patch_size == 1`
- intended for the Turbo path with `guidance <= 1.0`
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from inference import (
    ADALN_EMBED_DIM,
    BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT,
    FREQUENCY_EMBEDDING_SIZE,
    MAX_IMAGE_SEQ_LEN,
    MAX_PERIOD,
    MAX_SHIFT,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
    SEQ_MULTI_OF,
    generate_image,
    load_text_encoder,
    load_transformer as load_reference_transformer,
    load_vae,
    parse_args,
    read_json,
    resolve_device,
    resolve_dtype,
    FlowMatchEulerDiscreteScheduler,
)


class ZImageTransformer2DModel(nn.Module):
    """Single-class transformer with an unrolled forward pass."""

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=ROPE_THETA,
        t_scale=1000.0,
        axes_dims=ROPE_AXES_DIMS,
        axes_lens=ROPE_AXES_LENS,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if tuple(all_patch_size) != (2,) or tuple(all_f_patch_size) != (1,):
            raise NotImplementedError("Sequential transformer only supports patch_size=2 and f_patch_size=1")
        if n_heads != n_kv_heads:
            raise NotImplementedError("Sequential transformer only supports n_heads == n_kv_heads")
        if n_layers != 30 or n_refiner_layers != 2:
            raise NotImplementedError("Sequential transformer is generated for the released Z-Image architecture")
        if not qk_norm:
            raise NotImplementedError("Sequential transformer expects qk_norm=True")

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj_dim = self.n_heads * self.head_dim
        self.kv_proj_dim = self.n_kv_heads * self.head_dim
        self.hidden_dim = int(dim / 3 * 8)
        self.patch_size = 2
        self.f_patch_size = 1
        self.patch_dim = self.patch_size * self.patch_size * self.f_patch_size * self.in_channels
        self.output_patch_dim = self.patch_size * self.patch_size * self.f_patch_size * self.out_channels
        self.cap_feat_dim = cap_feat_dim
        self.adaln_dim = min(dim, ADALN_EMBED_DIM)
        self.frequency_embedding_size = FREQUENCY_EMBEDDING_SIZE
        self.norm_eps = norm_eps
        self.t_scale = t_scale
        self.axes_dims = list(axes_dims)
        self.axes_lens = list(axes_lens)
        self.rope_theta = rope_theta
        self.attention_scale = 1.0 / math.sqrt(self.head_dim)

        self.x_embedder_weight = nn.Parameter(torch.empty((self.dim, self.patch_dim), device=device, dtype=dtype))
        self.x_embedder_bias = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.t_mlp_w1 = nn.Parameter(torch.empty((1024, self.frequency_embedding_size), device=device, dtype=dtype))
        self.t_mlp_b1 = nn.Parameter(torch.empty((1024,), device=device, dtype=dtype))
        self.t_mlp_w2 = nn.Parameter(torch.empty((self.adaln_dim, 1024), device=device, dtype=dtype))
        self.t_mlp_b2 = nn.Parameter(torch.empty((self.adaln_dim,), device=device, dtype=dtype))
        self.cap_norm_weight = nn.Parameter(torch.empty((self.cap_feat_dim,), device=device, dtype=dtype))
        self.cap_linear_weight = nn.Parameter(torch.empty((self.dim, self.cap_feat_dim), device=device, dtype=dtype))
        self.cap_linear_bias = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.x_pad_token = nn.Parameter(torch.empty((1, self.dim), device=device, dtype=dtype))
        self.cap_pad_token = nn.Parameter(torch.empty((1, self.dim), device=device, dtype=dtype))
        self.final_linear_weight = nn.Parameter(torch.empty((self.output_patch_dim, self.dim), device=device, dtype=dtype))
        self.final_linear_bias = nn.Parameter(torch.empty((self.output_patch_dim,), device=device, dtype=dtype))
        self.final_adaln_weight = nn.Parameter(torch.empty((self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.final_adaln_bias = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr0_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.nr0_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.nr0_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.nr0_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.nr0_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.nr0_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.nr0_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.nr0_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.nr0_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.nr0_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr0_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr0_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr0_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr0_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.nr0_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.nr1_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.nr1_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.nr1_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.nr1_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.nr1_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.nr1_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.nr1_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.nr1_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.nr1_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.nr1_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr1_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr1_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr1_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.nr1_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.nr1_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.cr0_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.cr0_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.cr0_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.cr0_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.cr0_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.cr0_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.cr0_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.cr0_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.cr0_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.cr0_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.cr0_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.cr0_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.cr0_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.cr1_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.cr1_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.cr1_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.cr1_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.cr1_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.cr1_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.cr1_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.cr1_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.cr1_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.cr1_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.cr1_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.cr1_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.cr1_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer0_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer0_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer0_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer0_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer0_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer0_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer0_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer0_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer0_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer0_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer0_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer0_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer0_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer0_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer0_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer1_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer1_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer1_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer1_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer1_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer1_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer1_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer1_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer1_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer1_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer1_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer1_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer1_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer1_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer1_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer2_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer2_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer2_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer2_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer2_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer2_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer2_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer2_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer2_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer2_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer2_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer2_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer2_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer2_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer2_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer3_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer3_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer3_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer3_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer3_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer3_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer3_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer3_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer3_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer3_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer3_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer3_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer3_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer3_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer3_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer4_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer4_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer4_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer4_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer4_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer4_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer4_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer4_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer4_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer4_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer4_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer4_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer4_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer4_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer4_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer5_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer5_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer5_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer5_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer5_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer5_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer5_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer5_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer5_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer5_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer5_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer5_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer5_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer5_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer5_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer6_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer6_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer6_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer6_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer6_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer6_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer6_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer6_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer6_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer6_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer6_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer6_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer6_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer6_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer6_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer7_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer7_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer7_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer7_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer7_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer7_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer7_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer7_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer7_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer7_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer7_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer7_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer7_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer7_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer7_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer8_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer8_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer8_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer8_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer8_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer8_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer8_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer8_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer8_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer8_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer8_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer8_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer8_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer8_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer8_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer9_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer9_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer9_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer9_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer9_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer9_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer9_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer9_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer9_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer9_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer9_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer9_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer9_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer9_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer9_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer10_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer10_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer10_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer10_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer10_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer10_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer10_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer10_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer10_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer10_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer10_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer10_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer10_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer10_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer10_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer11_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer11_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer11_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer11_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer11_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer11_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer11_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer11_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer11_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer11_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer11_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer11_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer11_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer11_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer11_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer12_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer12_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer12_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer12_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer12_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer12_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer12_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer12_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer12_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer12_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer12_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer12_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer12_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer12_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer12_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer13_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer13_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer13_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer13_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer13_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer13_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer13_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer13_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer13_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer13_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer13_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer13_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer13_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer13_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer13_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer14_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer14_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer14_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer14_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer14_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer14_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer14_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer14_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer14_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer14_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer14_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer14_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer14_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer14_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer14_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer15_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer15_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer15_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer15_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer15_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer15_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer15_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer15_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer15_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer15_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer15_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer15_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer15_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer15_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer15_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer16_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer16_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer16_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer16_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer16_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer16_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer16_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer16_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer16_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer16_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer16_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer16_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer16_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer16_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer16_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer17_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer17_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer17_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer17_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer17_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer17_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer17_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer17_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer17_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer17_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer17_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer17_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer17_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer17_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer17_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer18_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer18_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer18_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer18_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer18_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer18_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer18_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer18_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer18_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer18_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer18_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer18_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer18_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer18_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer18_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer19_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer19_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer19_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer19_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer19_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer19_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer19_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer19_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer19_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer19_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer19_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer19_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer19_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer19_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer19_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer20_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer20_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer20_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer20_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer20_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer20_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer20_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer20_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer20_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer20_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer20_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer20_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer20_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer20_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer20_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer21_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer21_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer21_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer21_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer21_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer21_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer21_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer21_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer21_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer21_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer21_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer21_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer21_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer21_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer21_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer22_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer22_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer22_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer22_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer22_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer22_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer22_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer22_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer22_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer22_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer22_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer22_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer22_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer22_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer22_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer23_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer23_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer23_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer23_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer23_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer23_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer23_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer23_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer23_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer23_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer23_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer23_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer23_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer23_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer23_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer24_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer24_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer24_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer24_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer24_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer24_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer24_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer24_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer24_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer24_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer24_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer24_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer24_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer24_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer24_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer25_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer25_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer25_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer25_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer25_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer25_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer25_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer25_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer25_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer25_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer25_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer25_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer25_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer25_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer25_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer26_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer26_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer26_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer26_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer26_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer26_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer26_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer26_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer26_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer26_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer26_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer26_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer26_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer26_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer26_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer27_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer27_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer27_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer27_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer27_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer27_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer27_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer27_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer27_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer27_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer27_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer27_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer27_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer27_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer27_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer28_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer28_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer28_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer28_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer28_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer28_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer28_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer28_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer28_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer28_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer28_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer28_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer28_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer28_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer28_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))
        self.layer29_attention_to_q_weight = nn.Parameter(torch.empty((self.q_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer29_attention_to_k_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer29_attention_to_v_weight = nn.Parameter(torch.empty((self.kv_proj_dim, self.dim), device=device, dtype=dtype))
        self.layer29_attention_to_out_weight = nn.Parameter(torch.empty((self.dim, self.q_proj_dim), device=device, dtype=dtype))
        self.layer29_norm_q_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer29_norm_k_weight = nn.Parameter(torch.empty((self.head_dim,), device=device, dtype=dtype))
        self.layer29_feed_forward_w1_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer29_feed_forward_w2_weight = nn.Parameter(torch.empty((self.dim, self.hidden_dim), device=device, dtype=dtype))
        self.layer29_feed_forward_w3_weight = nn.Parameter(torch.empty((self.hidden_dim, self.dim), device=device, dtype=dtype))
        self.layer29_attention_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer29_ffn_norm1_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer29_attention_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer29_ffn_norm2_weight = nn.Parameter(torch.empty((self.dim,), device=device, dtype=dtype))
        self.layer29_adaln_weight = nn.Parameter(torch.empty((4 * self.dim, self.adaln_dim), device=device, dtype=dtype))
        self.layer29_adaln_bias = nn.Parameter(torch.empty((4 * self.dim,), device=device, dtype=dtype))

        rope_freqs = []
        for axis_dim, axis_len in zip(self.axes_dims, self.axes_lens):
            freqs = 1.0 / (self.rope_theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float64, device=device) / axis_dim))
            timestep = torch.arange(axis_len, dtype=torch.float64, device=device)
            freqs = torch.outer(timestep, freqs).float()
            rope_freqs.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        self.register_buffer("rope_freqs_0", rope_freqs[0], persistent=False)
        self.register_buffer("rope_freqs_1", rope_freqs[1], persistent=False)
        self.register_buffer("rope_freqs_2", rope_freqs[2], persistent=False)

    def forward(self, x, t, cap_feats, patch_size=2, f_patch_size=1):
        if len(x) != 1 or len(cap_feats) != 1:
            raise NotImplementedError("Sequential transformer forward only supports a single sample")
        if patch_size != 2 or f_patch_size != 1:
            raise NotImplementedError("Sequential transformer only supports patch_size=2 and f_patch_size=1")

        x0 = x[0]
        cap0 = cap_feats[0]
        if x0.dim() != 4:
            raise ValueError("Expected latent sample with shape [C, F, H, W]")

        device = x0.device

        # timestep embedding
        t_scaled = t * self.t_scale
        t_half = self.frequency_embedding_size // 2
        t_freqs = torch.exp(
            -math.log(MAX_PERIOD)
            * torch.arange(start=0, end=t_half, dtype=torch.float32, device=t_scaled.device)
            / t_half
        )
        t_args = t_scaled[:, None].float() * t_freqs[None]
        t_emb = torch.cat([torch.cos(t_args), torch.sin(t_args)], dim=-1)
        if self.frequency_embedding_size % 2:
            t_emb = torch.cat([t_emb, torch.zeros_like(t_emb[:, :1])], dim=-1)
        t_emb = t_emb.to(self.t_mlp_w1.dtype)
        t_hidden = t_emb @ self.t_mlp_w1.t() + self.t_mlp_b1
        t_hidden = F.silu(t_hidden)
        adaln_input = t_hidden @ self.t_mlp_w2.t() + self.t_mlp_b2

        # patchify latent image tokens
        channels, frames, height, width = x0.shape
        frame_tokens = frames // self.f_patch_size
        height_tokens = height // self.patch_size
        width_tokens = width // self.patch_size
        image_tokens = x0.view(
            channels,
            frame_tokens,
            self.f_patch_size,
            height_tokens,
            self.patch_size,
            width_tokens,
            self.patch_size,
        )
        image_tokens = image_tokens.permute(1, 3, 5, 2, 4, 6, 0).reshape(-1, self.patch_dim)
        image_original_len = image_tokens.shape[0]
        image_pad = (-image_original_len) % SEQ_MULTI_OF
        if image_pad > 0:
            image_tokens = torch.cat([image_tokens, image_tokens[-1:].repeat(image_pad, 1)], dim=0)

        # pad caption tokens to the same multiple-of-32 layout as the original model
        cap_original_len = cap0.shape[0]
        cap_pad = (-cap_original_len) % SEQ_MULTI_OF
        if cap_pad > 0:
            cap_tokens = torch.cat([cap0, cap0[-1:].repeat(cap_pad, 1)], dim=0)
        else:
            cap_tokens = cap0

        # positional ids and rotary frequencies for caption tokens
        cap_positions = torch.arange(1, cap_tokens.shape[0] + 1, dtype=torch.long, device=device)
        cap_zero = torch.zeros_like(cap_positions)
        cap_pos_ids = torch.stack([cap_positions, cap_zero, cap_zero], dim=-1)
        cap_freqs = torch.cat(
            [
                self.rope_freqs_0[cap_pos_ids[:, 0]],
                self.rope_freqs_1[cap_pos_ids[:, 1]],
                self.rope_freqs_2[cap_pos_ids[:, 2]],
            ],
            dim=-1,
        ).unsqueeze(0)

        # positional ids and rotary frequencies for latent image tokens
        image_frame_pos = torch.arange(
            cap_tokens.shape[0] + 1,
            cap_tokens.shape[0] + 1 + frame_tokens,
            dtype=torch.long,
            device=device,
        )
        image_height_pos = torch.arange(height_tokens, dtype=torch.long, device=device)
        image_width_pos = torch.arange(width_tokens, dtype=torch.long, device=device)
        image_pos_ids = torch.stack(
            torch.meshgrid(image_frame_pos, image_height_pos, image_width_pos, indexing="ij"),
            dim=-1,
        ).reshape(-1, 3)
        if image_pad > 0:
            image_pos_ids = torch.cat([image_pos_ids, torch.zeros((image_pad, 3), dtype=torch.long, device=device)], dim=0)
        x_freqs = torch.cat(
            [
                self.rope_freqs_0[image_pos_ids[:, 0]],
                self.rope_freqs_1[image_pos_ids[:, 1]],
                self.rope_freqs_2[image_pos_ids[:, 2]],
            ],
            dim=-1,
        ).unsqueeze(0)

        # input embeddings
        x_state = image_tokens @ self.x_embedder_weight.t() + self.x_embedder_bias
        if image_pad > 0:
            x_state[image_original_len:] = self.x_pad_token
        x_state = x_state.unsqueeze(0)

        cap_norm = cap_tokens * torch.rsqrt(cap_tokens.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cap_norm_weight
        cap_state = cap_norm @ self.cap_linear_weight.t() + self.cap_linear_bias
        if cap_pad > 0:
            cap_state[cap_original_len:] = self.cap_pad_token
        cap_state = cap_state.unsqueeze(0)

        # noise refiner
        # nr0
        nr0_mod = adaln_input @ self.nr0_adaln_weight.t() + self.nr0_adaln_bias
        nr0_mod = nr0_mod.unsqueeze(1)
        nr0_scale_msa, nr0_gate_msa, nr0_scale_mlp, nr0_gate_mlp = nr0_mod.chunk(4, dim=2)
        nr0_gate_msa = torch.tanh(nr0_gate_msa)
        nr0_gate_mlp = torch.tanh(nr0_gate_mlp)
        nr0_scale_msa = 1.0 + nr0_scale_msa
        nr0_scale_mlp = 1.0 + nr0_scale_mlp
        nr0_attn_in = x_state * torch.rsqrt(x_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr0_attention_norm1_weight
        nr0_attn_in = nr0_attn_in * nr0_scale_msa
        nr0_q = nr0_attn_in @ self.nr0_attention_to_q_weight.t()
        nr0_k = nr0_attn_in @ self.nr0_attention_to_k_weight.t()
        nr0_v = nr0_attn_in @ self.nr0_attention_to_v_weight.t()
        nr0_q = nr0_q.view(1, x_state.shape[1], self.n_heads, self.head_dim)
        nr0_k = nr0_k.view(1, x_state.shape[1], self.n_kv_heads, self.head_dim)
        nr0_v = nr0_v.view(1, x_state.shape[1], self.n_kv_heads, self.head_dim)
        nr0_q = nr0_q * torch.rsqrt(nr0_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr0_norm_q_weight.view(1, 1, 1, -1)
        nr0_k = nr0_k * torch.rsqrt(nr0_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr0_norm_k_weight.view(1, 1, 1, -1)
        nr0_q_complex = torch.view_as_complex(nr0_q.float().reshape(1, x_state.shape[1], self.n_heads, -1, 2))
        nr0_k_complex = torch.view_as_complex(nr0_k.float().reshape(1, x_state.shape[1], self.n_kv_heads, -1, 2))
        nr0_q = torch.view_as_real(nr0_q_complex * x_freqs.unsqueeze(2)).flatten(3).to(nr0_attn_in.dtype)
        nr0_k = torch.view_as_real(nr0_k_complex * x_freqs.unsqueeze(2)).flatten(3).to(nr0_attn_in.dtype)
        nr0_q = nr0_q.transpose(1, 2)
        nr0_k = nr0_k.transpose(1, 2)
        nr0_v = nr0_v.transpose(1, 2)
        nr0_scores = torch.matmul(nr0_q.float(), nr0_k.float().transpose(-2, -1)) * self.attention_scale
        nr0_probs = torch.softmax(nr0_scores, dim=-1).to(nr0_v.dtype)
        nr0_attn = torch.matmul(nr0_probs, nr0_v)
        nr0_attn = nr0_attn.transpose(1, 2).reshape(1, x_state.shape[1], self.q_proj_dim)
        nr0_attn = nr0_attn @ self.nr0_attention_to_out_weight.t()
        nr0_attn = nr0_attn * torch.rsqrt(nr0_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr0_attention_norm2_weight
        x_state = x_state + nr0_gate_msa * nr0_attn
        nr0_ffn_in = x_state * torch.rsqrt(x_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr0_ffn_norm1_weight
        nr0_ffn_in = nr0_ffn_in * nr0_scale_mlp
        nr0_ffn_a = nr0_ffn_in @ self.nr0_feed_forward_w1_weight.t()
        nr0_ffn_b = nr0_ffn_in @ self.nr0_feed_forward_w3_weight.t()
        nr0_ffn = F.silu(nr0_ffn_a) * nr0_ffn_b
        nr0_ffn = nr0_ffn @ self.nr0_feed_forward_w2_weight.t()
        nr0_ffn = nr0_ffn * torch.rsqrt(nr0_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr0_ffn_norm2_weight
        x_state = x_state + nr0_gate_mlp * nr0_ffn

        # nr1
        nr1_mod = adaln_input @ self.nr1_adaln_weight.t() + self.nr1_adaln_bias
        nr1_mod = nr1_mod.unsqueeze(1)
        nr1_scale_msa, nr1_gate_msa, nr1_scale_mlp, nr1_gate_mlp = nr1_mod.chunk(4, dim=2)
        nr1_gate_msa = torch.tanh(nr1_gate_msa)
        nr1_gate_mlp = torch.tanh(nr1_gate_mlp)
        nr1_scale_msa = 1.0 + nr1_scale_msa
        nr1_scale_mlp = 1.0 + nr1_scale_mlp
        nr1_attn_in = x_state * torch.rsqrt(x_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr1_attention_norm1_weight
        nr1_attn_in = nr1_attn_in * nr1_scale_msa
        nr1_q = nr1_attn_in @ self.nr1_attention_to_q_weight.t()
        nr1_k = nr1_attn_in @ self.nr1_attention_to_k_weight.t()
        nr1_v = nr1_attn_in @ self.nr1_attention_to_v_weight.t()
        nr1_q = nr1_q.view(1, x_state.shape[1], self.n_heads, self.head_dim)
        nr1_k = nr1_k.view(1, x_state.shape[1], self.n_kv_heads, self.head_dim)
        nr1_v = nr1_v.view(1, x_state.shape[1], self.n_kv_heads, self.head_dim)
        nr1_q = nr1_q * torch.rsqrt(nr1_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr1_norm_q_weight.view(1, 1, 1, -1)
        nr1_k = nr1_k * torch.rsqrt(nr1_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr1_norm_k_weight.view(1, 1, 1, -1)
        nr1_q_complex = torch.view_as_complex(nr1_q.float().reshape(1, x_state.shape[1], self.n_heads, -1, 2))
        nr1_k_complex = torch.view_as_complex(nr1_k.float().reshape(1, x_state.shape[1], self.n_kv_heads, -1, 2))
        nr1_q = torch.view_as_real(nr1_q_complex * x_freqs.unsqueeze(2)).flatten(3).to(nr1_attn_in.dtype)
        nr1_k = torch.view_as_real(nr1_k_complex * x_freqs.unsqueeze(2)).flatten(3).to(nr1_attn_in.dtype)
        nr1_q = nr1_q.transpose(1, 2)
        nr1_k = nr1_k.transpose(1, 2)
        nr1_v = nr1_v.transpose(1, 2)
        nr1_scores = torch.matmul(nr1_q.float(), nr1_k.float().transpose(-2, -1)) * self.attention_scale
        nr1_probs = torch.softmax(nr1_scores, dim=-1).to(nr1_v.dtype)
        nr1_attn = torch.matmul(nr1_probs, nr1_v)
        nr1_attn = nr1_attn.transpose(1, 2).reshape(1, x_state.shape[1], self.q_proj_dim)
        nr1_attn = nr1_attn @ self.nr1_attention_to_out_weight.t()
        nr1_attn = nr1_attn * torch.rsqrt(nr1_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr1_attention_norm2_weight
        x_state = x_state + nr1_gate_msa * nr1_attn
        nr1_ffn_in = x_state * torch.rsqrt(x_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr1_ffn_norm1_weight
        nr1_ffn_in = nr1_ffn_in * nr1_scale_mlp
        nr1_ffn_a = nr1_ffn_in @ self.nr1_feed_forward_w1_weight.t()
        nr1_ffn_b = nr1_ffn_in @ self.nr1_feed_forward_w3_weight.t()
        nr1_ffn = F.silu(nr1_ffn_a) * nr1_ffn_b
        nr1_ffn = nr1_ffn @ self.nr1_feed_forward_w2_weight.t()
        nr1_ffn = nr1_ffn * torch.rsqrt(nr1_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.nr1_ffn_norm2_weight
        x_state = x_state + nr1_gate_mlp * nr1_ffn

        # context refiner
        # cr0
        cr0_attn_in = cap_state * torch.rsqrt(cap_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr0_attention_norm1_weight
        cr0_q = cr0_attn_in @ self.cr0_attention_to_q_weight.t()
        cr0_k = cr0_attn_in @ self.cr0_attention_to_k_weight.t()
        cr0_v = cr0_attn_in @ self.cr0_attention_to_v_weight.t()
        cr0_q = cr0_q.view(1, cap_state.shape[1], self.n_heads, self.head_dim)
        cr0_k = cr0_k.view(1, cap_state.shape[1], self.n_kv_heads, self.head_dim)
        cr0_v = cr0_v.view(1, cap_state.shape[1], self.n_kv_heads, self.head_dim)
        cr0_q = cr0_q * torch.rsqrt(cr0_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr0_norm_q_weight.view(1, 1, 1, -1)
        cr0_k = cr0_k * torch.rsqrt(cr0_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr0_norm_k_weight.view(1, 1, 1, -1)
        cr0_q_complex = torch.view_as_complex(cr0_q.float().reshape(1, cap_state.shape[1], self.n_heads, -1, 2))
        cr0_k_complex = torch.view_as_complex(cr0_k.float().reshape(1, cap_state.shape[1], self.n_kv_heads, -1, 2))
        cr0_q = torch.view_as_real(cr0_q_complex * cap_freqs.unsqueeze(2)).flatten(3).to(cr0_attn_in.dtype)
        cr0_k = torch.view_as_real(cr0_k_complex * cap_freqs.unsqueeze(2)).flatten(3).to(cr0_attn_in.dtype)
        cr0_q = cr0_q.transpose(1, 2)
        cr0_k = cr0_k.transpose(1, 2)
        cr0_v = cr0_v.transpose(1, 2)
        cr0_scores = torch.matmul(cr0_q.float(), cr0_k.float().transpose(-2, -1)) * self.attention_scale
        cr0_probs = torch.softmax(cr0_scores, dim=-1).to(cr0_v.dtype)
        cr0_attn = torch.matmul(cr0_probs, cr0_v)
        cr0_attn = cr0_attn.transpose(1, 2).reshape(1, cap_state.shape[1], self.q_proj_dim)
        cr0_attn = cr0_attn @ self.cr0_attention_to_out_weight.t()
        cr0_attn = cr0_attn * torch.rsqrt(cr0_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr0_attention_norm2_weight
        cap_state = cap_state + cr0_attn
        cr0_ffn_in = cap_state * torch.rsqrt(cap_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr0_ffn_norm1_weight
        cr0_ffn_a = cr0_ffn_in @ self.cr0_feed_forward_w1_weight.t()
        cr0_ffn_b = cr0_ffn_in @ self.cr0_feed_forward_w3_weight.t()
        cr0_ffn = F.silu(cr0_ffn_a) * cr0_ffn_b
        cr0_ffn = cr0_ffn @ self.cr0_feed_forward_w2_weight.t()
        cr0_ffn = cr0_ffn * torch.rsqrt(cr0_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr0_ffn_norm2_weight
        cap_state = cap_state + cr0_ffn

        # cr1
        cr1_attn_in = cap_state * torch.rsqrt(cap_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr1_attention_norm1_weight
        cr1_q = cr1_attn_in @ self.cr1_attention_to_q_weight.t()
        cr1_k = cr1_attn_in @ self.cr1_attention_to_k_weight.t()
        cr1_v = cr1_attn_in @ self.cr1_attention_to_v_weight.t()
        cr1_q = cr1_q.view(1, cap_state.shape[1], self.n_heads, self.head_dim)
        cr1_k = cr1_k.view(1, cap_state.shape[1], self.n_kv_heads, self.head_dim)
        cr1_v = cr1_v.view(1, cap_state.shape[1], self.n_kv_heads, self.head_dim)
        cr1_q = cr1_q * torch.rsqrt(cr1_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr1_norm_q_weight.view(1, 1, 1, -1)
        cr1_k = cr1_k * torch.rsqrt(cr1_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr1_norm_k_weight.view(1, 1, 1, -1)
        cr1_q_complex = torch.view_as_complex(cr1_q.float().reshape(1, cap_state.shape[1], self.n_heads, -1, 2))
        cr1_k_complex = torch.view_as_complex(cr1_k.float().reshape(1, cap_state.shape[1], self.n_kv_heads, -1, 2))
        cr1_q = torch.view_as_real(cr1_q_complex * cap_freqs.unsqueeze(2)).flatten(3).to(cr1_attn_in.dtype)
        cr1_k = torch.view_as_real(cr1_k_complex * cap_freqs.unsqueeze(2)).flatten(3).to(cr1_attn_in.dtype)
        cr1_q = cr1_q.transpose(1, 2)
        cr1_k = cr1_k.transpose(1, 2)
        cr1_v = cr1_v.transpose(1, 2)
        cr1_scores = torch.matmul(cr1_q.float(), cr1_k.float().transpose(-2, -1)) * self.attention_scale
        cr1_probs = torch.softmax(cr1_scores, dim=-1).to(cr1_v.dtype)
        cr1_attn = torch.matmul(cr1_probs, cr1_v)
        cr1_attn = cr1_attn.transpose(1, 2).reshape(1, cap_state.shape[1], self.q_proj_dim)
        cr1_attn = cr1_attn @ self.cr1_attention_to_out_weight.t()
        cr1_attn = cr1_attn * torch.rsqrt(cr1_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr1_attention_norm2_weight
        cap_state = cap_state + cr1_attn
        cr1_ffn_in = cap_state * torch.rsqrt(cap_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr1_ffn_norm1_weight
        cr1_ffn_a = cr1_ffn_in @ self.cr1_feed_forward_w1_weight.t()
        cr1_ffn_b = cr1_ffn_in @ self.cr1_feed_forward_w3_weight.t()
        cr1_ffn = F.silu(cr1_ffn_a) * cr1_ffn_b
        cr1_ffn = cr1_ffn @ self.cr1_feed_forward_w2_weight.t()
        cr1_ffn = cr1_ffn * torch.rsqrt(cr1_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.cr1_ffn_norm2_weight
        cap_state = cap_state + cr1_ffn

        # unify image and caption streams
        unified_state = torch.cat([x_state, cap_state], dim=1)
        unified_freqs = torch.cat([x_freqs, cap_freqs], dim=1)

        # main transformer layers
        # layer0
        layer0_mod = adaln_input @ self.layer0_adaln_weight.t() + self.layer0_adaln_bias
        layer0_mod = layer0_mod.unsqueeze(1)
        layer0_scale_msa, layer0_gate_msa, layer0_scale_mlp, layer0_gate_mlp = layer0_mod.chunk(4, dim=2)
        layer0_gate_msa = torch.tanh(layer0_gate_msa)
        layer0_gate_mlp = torch.tanh(layer0_gate_mlp)
        layer0_scale_msa = 1.0 + layer0_scale_msa
        layer0_scale_mlp = 1.0 + layer0_scale_mlp
        layer0_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer0_attention_norm1_weight
        layer0_attn_in = layer0_attn_in * layer0_scale_msa
        layer0_q = layer0_attn_in @ self.layer0_attention_to_q_weight.t()
        layer0_k = layer0_attn_in @ self.layer0_attention_to_k_weight.t()
        layer0_v = layer0_attn_in @ self.layer0_attention_to_v_weight.t()
        layer0_q = layer0_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer0_k = layer0_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer0_v = layer0_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer0_q = layer0_q * torch.rsqrt(layer0_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer0_norm_q_weight.view(1, 1, 1, -1)
        layer0_k = layer0_k * torch.rsqrt(layer0_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer0_norm_k_weight.view(1, 1, 1, -1)
        layer0_q_complex = torch.view_as_complex(layer0_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer0_k_complex = torch.view_as_complex(layer0_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer0_q = torch.view_as_real(layer0_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer0_attn_in.dtype)
        layer0_k = torch.view_as_real(layer0_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer0_attn_in.dtype)
        layer0_q = layer0_q.transpose(1, 2)
        layer0_k = layer0_k.transpose(1, 2)
        layer0_v = layer0_v.transpose(1, 2)
        layer0_scores = torch.matmul(layer0_q.float(), layer0_k.float().transpose(-2, -1)) * self.attention_scale
        layer0_probs = torch.softmax(layer0_scores, dim=-1).to(layer0_v.dtype)
        layer0_attn = torch.matmul(layer0_probs, layer0_v)
        layer0_attn = layer0_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer0_attn = layer0_attn @ self.layer0_attention_to_out_weight.t()
        layer0_attn = layer0_attn * torch.rsqrt(layer0_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer0_attention_norm2_weight
        unified_state = unified_state + layer0_gate_msa * layer0_attn
        layer0_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer0_ffn_norm1_weight
        layer0_ffn_in = layer0_ffn_in * layer0_scale_mlp
        layer0_ffn_a = layer0_ffn_in @ self.layer0_feed_forward_w1_weight.t()
        layer0_ffn_b = layer0_ffn_in @ self.layer0_feed_forward_w3_weight.t()
        layer0_ffn = F.silu(layer0_ffn_a) * layer0_ffn_b
        layer0_ffn = layer0_ffn @ self.layer0_feed_forward_w2_weight.t()
        layer0_ffn = layer0_ffn * torch.rsqrt(layer0_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer0_ffn_norm2_weight
        unified_state = unified_state + layer0_gate_mlp * layer0_ffn

        # layer1
        layer1_mod = adaln_input @ self.layer1_adaln_weight.t() + self.layer1_adaln_bias
        layer1_mod = layer1_mod.unsqueeze(1)
        layer1_scale_msa, layer1_gate_msa, layer1_scale_mlp, layer1_gate_mlp = layer1_mod.chunk(4, dim=2)
        layer1_gate_msa = torch.tanh(layer1_gate_msa)
        layer1_gate_mlp = torch.tanh(layer1_gate_mlp)
        layer1_scale_msa = 1.0 + layer1_scale_msa
        layer1_scale_mlp = 1.0 + layer1_scale_mlp
        layer1_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer1_attention_norm1_weight
        layer1_attn_in = layer1_attn_in * layer1_scale_msa
        layer1_q = layer1_attn_in @ self.layer1_attention_to_q_weight.t()
        layer1_k = layer1_attn_in @ self.layer1_attention_to_k_weight.t()
        layer1_v = layer1_attn_in @ self.layer1_attention_to_v_weight.t()
        layer1_q = layer1_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer1_k = layer1_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer1_v = layer1_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer1_q = layer1_q * torch.rsqrt(layer1_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer1_norm_q_weight.view(1, 1, 1, -1)
        layer1_k = layer1_k * torch.rsqrt(layer1_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer1_norm_k_weight.view(1, 1, 1, -1)
        layer1_q_complex = torch.view_as_complex(layer1_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer1_k_complex = torch.view_as_complex(layer1_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer1_q = torch.view_as_real(layer1_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer1_attn_in.dtype)
        layer1_k = torch.view_as_real(layer1_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer1_attn_in.dtype)
        layer1_q = layer1_q.transpose(1, 2)
        layer1_k = layer1_k.transpose(1, 2)
        layer1_v = layer1_v.transpose(1, 2)
        layer1_scores = torch.matmul(layer1_q.float(), layer1_k.float().transpose(-2, -1)) * self.attention_scale
        layer1_probs = torch.softmax(layer1_scores, dim=-1).to(layer1_v.dtype)
        layer1_attn = torch.matmul(layer1_probs, layer1_v)
        layer1_attn = layer1_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer1_attn = layer1_attn @ self.layer1_attention_to_out_weight.t()
        layer1_attn = layer1_attn * torch.rsqrt(layer1_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer1_attention_norm2_weight
        unified_state = unified_state + layer1_gate_msa * layer1_attn
        layer1_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer1_ffn_norm1_weight
        layer1_ffn_in = layer1_ffn_in * layer1_scale_mlp
        layer1_ffn_a = layer1_ffn_in @ self.layer1_feed_forward_w1_weight.t()
        layer1_ffn_b = layer1_ffn_in @ self.layer1_feed_forward_w3_weight.t()
        layer1_ffn = F.silu(layer1_ffn_a) * layer1_ffn_b
        layer1_ffn = layer1_ffn @ self.layer1_feed_forward_w2_weight.t()
        layer1_ffn = layer1_ffn * torch.rsqrt(layer1_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer1_ffn_norm2_weight
        unified_state = unified_state + layer1_gate_mlp * layer1_ffn

        # layer2
        layer2_mod = adaln_input @ self.layer2_adaln_weight.t() + self.layer2_adaln_bias
        layer2_mod = layer2_mod.unsqueeze(1)
        layer2_scale_msa, layer2_gate_msa, layer2_scale_mlp, layer2_gate_mlp = layer2_mod.chunk(4, dim=2)
        layer2_gate_msa = torch.tanh(layer2_gate_msa)
        layer2_gate_mlp = torch.tanh(layer2_gate_mlp)
        layer2_scale_msa = 1.0 + layer2_scale_msa
        layer2_scale_mlp = 1.0 + layer2_scale_mlp
        layer2_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer2_attention_norm1_weight
        layer2_attn_in = layer2_attn_in * layer2_scale_msa
        layer2_q = layer2_attn_in @ self.layer2_attention_to_q_weight.t()
        layer2_k = layer2_attn_in @ self.layer2_attention_to_k_weight.t()
        layer2_v = layer2_attn_in @ self.layer2_attention_to_v_weight.t()
        layer2_q = layer2_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer2_k = layer2_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer2_v = layer2_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer2_q = layer2_q * torch.rsqrt(layer2_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer2_norm_q_weight.view(1, 1, 1, -1)
        layer2_k = layer2_k * torch.rsqrt(layer2_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer2_norm_k_weight.view(1, 1, 1, -1)
        layer2_q_complex = torch.view_as_complex(layer2_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer2_k_complex = torch.view_as_complex(layer2_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer2_q = torch.view_as_real(layer2_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer2_attn_in.dtype)
        layer2_k = torch.view_as_real(layer2_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer2_attn_in.dtype)
        layer2_q = layer2_q.transpose(1, 2)
        layer2_k = layer2_k.transpose(1, 2)
        layer2_v = layer2_v.transpose(1, 2)
        layer2_scores = torch.matmul(layer2_q.float(), layer2_k.float().transpose(-2, -1)) * self.attention_scale
        layer2_probs = torch.softmax(layer2_scores, dim=-1).to(layer2_v.dtype)
        layer2_attn = torch.matmul(layer2_probs, layer2_v)
        layer2_attn = layer2_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer2_attn = layer2_attn @ self.layer2_attention_to_out_weight.t()
        layer2_attn = layer2_attn * torch.rsqrt(layer2_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer2_attention_norm2_weight
        unified_state = unified_state + layer2_gate_msa * layer2_attn
        layer2_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer2_ffn_norm1_weight
        layer2_ffn_in = layer2_ffn_in * layer2_scale_mlp
        layer2_ffn_a = layer2_ffn_in @ self.layer2_feed_forward_w1_weight.t()
        layer2_ffn_b = layer2_ffn_in @ self.layer2_feed_forward_w3_weight.t()
        layer2_ffn = F.silu(layer2_ffn_a) * layer2_ffn_b
        layer2_ffn = layer2_ffn @ self.layer2_feed_forward_w2_weight.t()
        layer2_ffn = layer2_ffn * torch.rsqrt(layer2_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer2_ffn_norm2_weight
        unified_state = unified_state + layer2_gate_mlp * layer2_ffn

        # layer3
        layer3_mod = adaln_input @ self.layer3_adaln_weight.t() + self.layer3_adaln_bias
        layer3_mod = layer3_mod.unsqueeze(1)
        layer3_scale_msa, layer3_gate_msa, layer3_scale_mlp, layer3_gate_mlp = layer3_mod.chunk(4, dim=2)
        layer3_gate_msa = torch.tanh(layer3_gate_msa)
        layer3_gate_mlp = torch.tanh(layer3_gate_mlp)
        layer3_scale_msa = 1.0 + layer3_scale_msa
        layer3_scale_mlp = 1.0 + layer3_scale_mlp
        layer3_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer3_attention_norm1_weight
        layer3_attn_in = layer3_attn_in * layer3_scale_msa
        layer3_q = layer3_attn_in @ self.layer3_attention_to_q_weight.t()
        layer3_k = layer3_attn_in @ self.layer3_attention_to_k_weight.t()
        layer3_v = layer3_attn_in @ self.layer3_attention_to_v_weight.t()
        layer3_q = layer3_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer3_k = layer3_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer3_v = layer3_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer3_q = layer3_q * torch.rsqrt(layer3_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer3_norm_q_weight.view(1, 1, 1, -1)
        layer3_k = layer3_k * torch.rsqrt(layer3_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer3_norm_k_weight.view(1, 1, 1, -1)
        layer3_q_complex = torch.view_as_complex(layer3_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer3_k_complex = torch.view_as_complex(layer3_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer3_q = torch.view_as_real(layer3_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer3_attn_in.dtype)
        layer3_k = torch.view_as_real(layer3_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer3_attn_in.dtype)
        layer3_q = layer3_q.transpose(1, 2)
        layer3_k = layer3_k.transpose(1, 2)
        layer3_v = layer3_v.transpose(1, 2)
        layer3_scores = torch.matmul(layer3_q.float(), layer3_k.float().transpose(-2, -1)) * self.attention_scale
        layer3_probs = torch.softmax(layer3_scores, dim=-1).to(layer3_v.dtype)
        layer3_attn = torch.matmul(layer3_probs, layer3_v)
        layer3_attn = layer3_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer3_attn = layer3_attn @ self.layer3_attention_to_out_weight.t()
        layer3_attn = layer3_attn * torch.rsqrt(layer3_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer3_attention_norm2_weight
        unified_state = unified_state + layer3_gate_msa * layer3_attn
        layer3_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer3_ffn_norm1_weight
        layer3_ffn_in = layer3_ffn_in * layer3_scale_mlp
        layer3_ffn_a = layer3_ffn_in @ self.layer3_feed_forward_w1_weight.t()
        layer3_ffn_b = layer3_ffn_in @ self.layer3_feed_forward_w3_weight.t()
        layer3_ffn = F.silu(layer3_ffn_a) * layer3_ffn_b
        layer3_ffn = layer3_ffn @ self.layer3_feed_forward_w2_weight.t()
        layer3_ffn = layer3_ffn * torch.rsqrt(layer3_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer3_ffn_norm2_weight
        unified_state = unified_state + layer3_gate_mlp * layer3_ffn

        # layer4
        layer4_mod = adaln_input @ self.layer4_adaln_weight.t() + self.layer4_adaln_bias
        layer4_mod = layer4_mod.unsqueeze(1)
        layer4_scale_msa, layer4_gate_msa, layer4_scale_mlp, layer4_gate_mlp = layer4_mod.chunk(4, dim=2)
        layer4_gate_msa = torch.tanh(layer4_gate_msa)
        layer4_gate_mlp = torch.tanh(layer4_gate_mlp)
        layer4_scale_msa = 1.0 + layer4_scale_msa
        layer4_scale_mlp = 1.0 + layer4_scale_mlp
        layer4_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer4_attention_norm1_weight
        layer4_attn_in = layer4_attn_in * layer4_scale_msa
        layer4_q = layer4_attn_in @ self.layer4_attention_to_q_weight.t()
        layer4_k = layer4_attn_in @ self.layer4_attention_to_k_weight.t()
        layer4_v = layer4_attn_in @ self.layer4_attention_to_v_weight.t()
        layer4_q = layer4_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer4_k = layer4_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer4_v = layer4_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer4_q = layer4_q * torch.rsqrt(layer4_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer4_norm_q_weight.view(1, 1, 1, -1)
        layer4_k = layer4_k * torch.rsqrt(layer4_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer4_norm_k_weight.view(1, 1, 1, -1)
        layer4_q_complex = torch.view_as_complex(layer4_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer4_k_complex = torch.view_as_complex(layer4_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer4_q = torch.view_as_real(layer4_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer4_attn_in.dtype)
        layer4_k = torch.view_as_real(layer4_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer4_attn_in.dtype)
        layer4_q = layer4_q.transpose(1, 2)
        layer4_k = layer4_k.transpose(1, 2)
        layer4_v = layer4_v.transpose(1, 2)
        layer4_scores = torch.matmul(layer4_q.float(), layer4_k.float().transpose(-2, -1)) * self.attention_scale
        layer4_probs = torch.softmax(layer4_scores, dim=-1).to(layer4_v.dtype)
        layer4_attn = torch.matmul(layer4_probs, layer4_v)
        layer4_attn = layer4_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer4_attn = layer4_attn @ self.layer4_attention_to_out_weight.t()
        layer4_attn = layer4_attn * torch.rsqrt(layer4_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer4_attention_norm2_weight
        unified_state = unified_state + layer4_gate_msa * layer4_attn
        layer4_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer4_ffn_norm1_weight
        layer4_ffn_in = layer4_ffn_in * layer4_scale_mlp
        layer4_ffn_a = layer4_ffn_in @ self.layer4_feed_forward_w1_weight.t()
        layer4_ffn_b = layer4_ffn_in @ self.layer4_feed_forward_w3_weight.t()
        layer4_ffn = F.silu(layer4_ffn_a) * layer4_ffn_b
        layer4_ffn = layer4_ffn @ self.layer4_feed_forward_w2_weight.t()
        layer4_ffn = layer4_ffn * torch.rsqrt(layer4_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer4_ffn_norm2_weight
        unified_state = unified_state + layer4_gate_mlp * layer4_ffn

        # layer5
        layer5_mod = adaln_input @ self.layer5_adaln_weight.t() + self.layer5_adaln_bias
        layer5_mod = layer5_mod.unsqueeze(1)
        layer5_scale_msa, layer5_gate_msa, layer5_scale_mlp, layer5_gate_mlp = layer5_mod.chunk(4, dim=2)
        layer5_gate_msa = torch.tanh(layer5_gate_msa)
        layer5_gate_mlp = torch.tanh(layer5_gate_mlp)
        layer5_scale_msa = 1.0 + layer5_scale_msa
        layer5_scale_mlp = 1.0 + layer5_scale_mlp
        layer5_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer5_attention_norm1_weight
        layer5_attn_in = layer5_attn_in * layer5_scale_msa
        layer5_q = layer5_attn_in @ self.layer5_attention_to_q_weight.t()
        layer5_k = layer5_attn_in @ self.layer5_attention_to_k_weight.t()
        layer5_v = layer5_attn_in @ self.layer5_attention_to_v_weight.t()
        layer5_q = layer5_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer5_k = layer5_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer5_v = layer5_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer5_q = layer5_q * torch.rsqrt(layer5_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer5_norm_q_weight.view(1, 1, 1, -1)
        layer5_k = layer5_k * torch.rsqrt(layer5_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer5_norm_k_weight.view(1, 1, 1, -1)
        layer5_q_complex = torch.view_as_complex(layer5_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer5_k_complex = torch.view_as_complex(layer5_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer5_q = torch.view_as_real(layer5_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer5_attn_in.dtype)
        layer5_k = torch.view_as_real(layer5_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer5_attn_in.dtype)
        layer5_q = layer5_q.transpose(1, 2)
        layer5_k = layer5_k.transpose(1, 2)
        layer5_v = layer5_v.transpose(1, 2)
        layer5_scores = torch.matmul(layer5_q.float(), layer5_k.float().transpose(-2, -1)) * self.attention_scale
        layer5_probs = torch.softmax(layer5_scores, dim=-1).to(layer5_v.dtype)
        layer5_attn = torch.matmul(layer5_probs, layer5_v)
        layer5_attn = layer5_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer5_attn = layer5_attn @ self.layer5_attention_to_out_weight.t()
        layer5_attn = layer5_attn * torch.rsqrt(layer5_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer5_attention_norm2_weight
        unified_state = unified_state + layer5_gate_msa * layer5_attn
        layer5_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer5_ffn_norm1_weight
        layer5_ffn_in = layer5_ffn_in * layer5_scale_mlp
        layer5_ffn_a = layer5_ffn_in @ self.layer5_feed_forward_w1_weight.t()
        layer5_ffn_b = layer5_ffn_in @ self.layer5_feed_forward_w3_weight.t()
        layer5_ffn = F.silu(layer5_ffn_a) * layer5_ffn_b
        layer5_ffn = layer5_ffn @ self.layer5_feed_forward_w2_weight.t()
        layer5_ffn = layer5_ffn * torch.rsqrt(layer5_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer5_ffn_norm2_weight
        unified_state = unified_state + layer5_gate_mlp * layer5_ffn

        # layer6
        layer6_mod = adaln_input @ self.layer6_adaln_weight.t() + self.layer6_adaln_bias
        layer6_mod = layer6_mod.unsqueeze(1)
        layer6_scale_msa, layer6_gate_msa, layer6_scale_mlp, layer6_gate_mlp = layer6_mod.chunk(4, dim=2)
        layer6_gate_msa = torch.tanh(layer6_gate_msa)
        layer6_gate_mlp = torch.tanh(layer6_gate_mlp)
        layer6_scale_msa = 1.0 + layer6_scale_msa
        layer6_scale_mlp = 1.0 + layer6_scale_mlp
        layer6_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer6_attention_norm1_weight
        layer6_attn_in = layer6_attn_in * layer6_scale_msa
        layer6_q = layer6_attn_in @ self.layer6_attention_to_q_weight.t()
        layer6_k = layer6_attn_in @ self.layer6_attention_to_k_weight.t()
        layer6_v = layer6_attn_in @ self.layer6_attention_to_v_weight.t()
        layer6_q = layer6_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer6_k = layer6_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer6_v = layer6_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer6_q = layer6_q * torch.rsqrt(layer6_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer6_norm_q_weight.view(1, 1, 1, -1)
        layer6_k = layer6_k * torch.rsqrt(layer6_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer6_norm_k_weight.view(1, 1, 1, -1)
        layer6_q_complex = torch.view_as_complex(layer6_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer6_k_complex = torch.view_as_complex(layer6_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer6_q = torch.view_as_real(layer6_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer6_attn_in.dtype)
        layer6_k = torch.view_as_real(layer6_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer6_attn_in.dtype)
        layer6_q = layer6_q.transpose(1, 2)
        layer6_k = layer6_k.transpose(1, 2)
        layer6_v = layer6_v.transpose(1, 2)
        layer6_scores = torch.matmul(layer6_q.float(), layer6_k.float().transpose(-2, -1)) * self.attention_scale
        layer6_probs = torch.softmax(layer6_scores, dim=-1).to(layer6_v.dtype)
        layer6_attn = torch.matmul(layer6_probs, layer6_v)
        layer6_attn = layer6_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer6_attn = layer6_attn @ self.layer6_attention_to_out_weight.t()
        layer6_attn = layer6_attn * torch.rsqrt(layer6_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer6_attention_norm2_weight
        unified_state = unified_state + layer6_gate_msa * layer6_attn
        layer6_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer6_ffn_norm1_weight
        layer6_ffn_in = layer6_ffn_in * layer6_scale_mlp
        layer6_ffn_a = layer6_ffn_in @ self.layer6_feed_forward_w1_weight.t()
        layer6_ffn_b = layer6_ffn_in @ self.layer6_feed_forward_w3_weight.t()
        layer6_ffn = F.silu(layer6_ffn_a) * layer6_ffn_b
        layer6_ffn = layer6_ffn @ self.layer6_feed_forward_w2_weight.t()
        layer6_ffn = layer6_ffn * torch.rsqrt(layer6_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer6_ffn_norm2_weight
        unified_state = unified_state + layer6_gate_mlp * layer6_ffn

        # layer7
        layer7_mod = adaln_input @ self.layer7_adaln_weight.t() + self.layer7_adaln_bias
        layer7_mod = layer7_mod.unsqueeze(1)
        layer7_scale_msa, layer7_gate_msa, layer7_scale_mlp, layer7_gate_mlp = layer7_mod.chunk(4, dim=2)
        layer7_gate_msa = torch.tanh(layer7_gate_msa)
        layer7_gate_mlp = torch.tanh(layer7_gate_mlp)
        layer7_scale_msa = 1.0 + layer7_scale_msa
        layer7_scale_mlp = 1.0 + layer7_scale_mlp
        layer7_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer7_attention_norm1_weight
        layer7_attn_in = layer7_attn_in * layer7_scale_msa
        layer7_q = layer7_attn_in @ self.layer7_attention_to_q_weight.t()
        layer7_k = layer7_attn_in @ self.layer7_attention_to_k_weight.t()
        layer7_v = layer7_attn_in @ self.layer7_attention_to_v_weight.t()
        layer7_q = layer7_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer7_k = layer7_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer7_v = layer7_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer7_q = layer7_q * torch.rsqrt(layer7_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer7_norm_q_weight.view(1, 1, 1, -1)
        layer7_k = layer7_k * torch.rsqrt(layer7_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer7_norm_k_weight.view(1, 1, 1, -1)
        layer7_q_complex = torch.view_as_complex(layer7_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer7_k_complex = torch.view_as_complex(layer7_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer7_q = torch.view_as_real(layer7_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer7_attn_in.dtype)
        layer7_k = torch.view_as_real(layer7_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer7_attn_in.dtype)
        layer7_q = layer7_q.transpose(1, 2)
        layer7_k = layer7_k.transpose(1, 2)
        layer7_v = layer7_v.transpose(1, 2)
        layer7_scores = torch.matmul(layer7_q.float(), layer7_k.float().transpose(-2, -1)) * self.attention_scale
        layer7_probs = torch.softmax(layer7_scores, dim=-1).to(layer7_v.dtype)
        layer7_attn = torch.matmul(layer7_probs, layer7_v)
        layer7_attn = layer7_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer7_attn = layer7_attn @ self.layer7_attention_to_out_weight.t()
        layer7_attn = layer7_attn * torch.rsqrt(layer7_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer7_attention_norm2_weight
        unified_state = unified_state + layer7_gate_msa * layer7_attn
        layer7_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer7_ffn_norm1_weight
        layer7_ffn_in = layer7_ffn_in * layer7_scale_mlp
        layer7_ffn_a = layer7_ffn_in @ self.layer7_feed_forward_w1_weight.t()
        layer7_ffn_b = layer7_ffn_in @ self.layer7_feed_forward_w3_weight.t()
        layer7_ffn = F.silu(layer7_ffn_a) * layer7_ffn_b
        layer7_ffn = layer7_ffn @ self.layer7_feed_forward_w2_weight.t()
        layer7_ffn = layer7_ffn * torch.rsqrt(layer7_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer7_ffn_norm2_weight
        unified_state = unified_state + layer7_gate_mlp * layer7_ffn

        # layer8
        layer8_mod = adaln_input @ self.layer8_adaln_weight.t() + self.layer8_adaln_bias
        layer8_mod = layer8_mod.unsqueeze(1)
        layer8_scale_msa, layer8_gate_msa, layer8_scale_mlp, layer8_gate_mlp = layer8_mod.chunk(4, dim=2)
        layer8_gate_msa = torch.tanh(layer8_gate_msa)
        layer8_gate_mlp = torch.tanh(layer8_gate_mlp)
        layer8_scale_msa = 1.0 + layer8_scale_msa
        layer8_scale_mlp = 1.0 + layer8_scale_mlp
        layer8_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer8_attention_norm1_weight
        layer8_attn_in = layer8_attn_in * layer8_scale_msa
        layer8_q = layer8_attn_in @ self.layer8_attention_to_q_weight.t()
        layer8_k = layer8_attn_in @ self.layer8_attention_to_k_weight.t()
        layer8_v = layer8_attn_in @ self.layer8_attention_to_v_weight.t()
        layer8_q = layer8_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer8_k = layer8_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer8_v = layer8_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer8_q = layer8_q * torch.rsqrt(layer8_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer8_norm_q_weight.view(1, 1, 1, -1)
        layer8_k = layer8_k * torch.rsqrt(layer8_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer8_norm_k_weight.view(1, 1, 1, -1)
        layer8_q_complex = torch.view_as_complex(layer8_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer8_k_complex = torch.view_as_complex(layer8_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer8_q = torch.view_as_real(layer8_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer8_attn_in.dtype)
        layer8_k = torch.view_as_real(layer8_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer8_attn_in.dtype)
        layer8_q = layer8_q.transpose(1, 2)
        layer8_k = layer8_k.transpose(1, 2)
        layer8_v = layer8_v.transpose(1, 2)
        layer8_scores = torch.matmul(layer8_q.float(), layer8_k.float().transpose(-2, -1)) * self.attention_scale
        layer8_probs = torch.softmax(layer8_scores, dim=-1).to(layer8_v.dtype)
        layer8_attn = torch.matmul(layer8_probs, layer8_v)
        layer8_attn = layer8_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer8_attn = layer8_attn @ self.layer8_attention_to_out_weight.t()
        layer8_attn = layer8_attn * torch.rsqrt(layer8_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer8_attention_norm2_weight
        unified_state = unified_state + layer8_gate_msa * layer8_attn
        layer8_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer8_ffn_norm1_weight
        layer8_ffn_in = layer8_ffn_in * layer8_scale_mlp
        layer8_ffn_a = layer8_ffn_in @ self.layer8_feed_forward_w1_weight.t()
        layer8_ffn_b = layer8_ffn_in @ self.layer8_feed_forward_w3_weight.t()
        layer8_ffn = F.silu(layer8_ffn_a) * layer8_ffn_b
        layer8_ffn = layer8_ffn @ self.layer8_feed_forward_w2_weight.t()
        layer8_ffn = layer8_ffn * torch.rsqrt(layer8_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer8_ffn_norm2_weight
        unified_state = unified_state + layer8_gate_mlp * layer8_ffn

        # layer9
        layer9_mod = adaln_input @ self.layer9_adaln_weight.t() + self.layer9_adaln_bias
        layer9_mod = layer9_mod.unsqueeze(1)
        layer9_scale_msa, layer9_gate_msa, layer9_scale_mlp, layer9_gate_mlp = layer9_mod.chunk(4, dim=2)
        layer9_gate_msa = torch.tanh(layer9_gate_msa)
        layer9_gate_mlp = torch.tanh(layer9_gate_mlp)
        layer9_scale_msa = 1.0 + layer9_scale_msa
        layer9_scale_mlp = 1.0 + layer9_scale_mlp
        layer9_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer9_attention_norm1_weight
        layer9_attn_in = layer9_attn_in * layer9_scale_msa
        layer9_q = layer9_attn_in @ self.layer9_attention_to_q_weight.t()
        layer9_k = layer9_attn_in @ self.layer9_attention_to_k_weight.t()
        layer9_v = layer9_attn_in @ self.layer9_attention_to_v_weight.t()
        layer9_q = layer9_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer9_k = layer9_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer9_v = layer9_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer9_q = layer9_q * torch.rsqrt(layer9_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer9_norm_q_weight.view(1, 1, 1, -1)
        layer9_k = layer9_k * torch.rsqrt(layer9_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer9_norm_k_weight.view(1, 1, 1, -1)
        layer9_q_complex = torch.view_as_complex(layer9_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer9_k_complex = torch.view_as_complex(layer9_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer9_q = torch.view_as_real(layer9_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer9_attn_in.dtype)
        layer9_k = torch.view_as_real(layer9_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer9_attn_in.dtype)
        layer9_q = layer9_q.transpose(1, 2)
        layer9_k = layer9_k.transpose(1, 2)
        layer9_v = layer9_v.transpose(1, 2)
        layer9_scores = torch.matmul(layer9_q.float(), layer9_k.float().transpose(-2, -1)) * self.attention_scale
        layer9_probs = torch.softmax(layer9_scores, dim=-1).to(layer9_v.dtype)
        layer9_attn = torch.matmul(layer9_probs, layer9_v)
        layer9_attn = layer9_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer9_attn = layer9_attn @ self.layer9_attention_to_out_weight.t()
        layer9_attn = layer9_attn * torch.rsqrt(layer9_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer9_attention_norm2_weight
        unified_state = unified_state + layer9_gate_msa * layer9_attn
        layer9_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer9_ffn_norm1_weight
        layer9_ffn_in = layer9_ffn_in * layer9_scale_mlp
        layer9_ffn_a = layer9_ffn_in @ self.layer9_feed_forward_w1_weight.t()
        layer9_ffn_b = layer9_ffn_in @ self.layer9_feed_forward_w3_weight.t()
        layer9_ffn = F.silu(layer9_ffn_a) * layer9_ffn_b
        layer9_ffn = layer9_ffn @ self.layer9_feed_forward_w2_weight.t()
        layer9_ffn = layer9_ffn * torch.rsqrt(layer9_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer9_ffn_norm2_weight
        unified_state = unified_state + layer9_gate_mlp * layer9_ffn

        # layer10
        layer10_mod = adaln_input @ self.layer10_adaln_weight.t() + self.layer10_adaln_bias
        layer10_mod = layer10_mod.unsqueeze(1)
        layer10_scale_msa, layer10_gate_msa, layer10_scale_mlp, layer10_gate_mlp = layer10_mod.chunk(4, dim=2)
        layer10_gate_msa = torch.tanh(layer10_gate_msa)
        layer10_gate_mlp = torch.tanh(layer10_gate_mlp)
        layer10_scale_msa = 1.0 + layer10_scale_msa
        layer10_scale_mlp = 1.0 + layer10_scale_mlp
        layer10_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer10_attention_norm1_weight
        layer10_attn_in = layer10_attn_in * layer10_scale_msa
        layer10_q = layer10_attn_in @ self.layer10_attention_to_q_weight.t()
        layer10_k = layer10_attn_in @ self.layer10_attention_to_k_weight.t()
        layer10_v = layer10_attn_in @ self.layer10_attention_to_v_weight.t()
        layer10_q = layer10_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer10_k = layer10_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer10_v = layer10_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer10_q = layer10_q * torch.rsqrt(layer10_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer10_norm_q_weight.view(1, 1, 1, -1)
        layer10_k = layer10_k * torch.rsqrt(layer10_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer10_norm_k_weight.view(1, 1, 1, -1)
        layer10_q_complex = torch.view_as_complex(layer10_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer10_k_complex = torch.view_as_complex(layer10_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer10_q = torch.view_as_real(layer10_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer10_attn_in.dtype)
        layer10_k = torch.view_as_real(layer10_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer10_attn_in.dtype)
        layer10_q = layer10_q.transpose(1, 2)
        layer10_k = layer10_k.transpose(1, 2)
        layer10_v = layer10_v.transpose(1, 2)
        layer10_scores = torch.matmul(layer10_q.float(), layer10_k.float().transpose(-2, -1)) * self.attention_scale
        layer10_probs = torch.softmax(layer10_scores, dim=-1).to(layer10_v.dtype)
        layer10_attn = torch.matmul(layer10_probs, layer10_v)
        layer10_attn = layer10_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer10_attn = layer10_attn @ self.layer10_attention_to_out_weight.t()
        layer10_attn = layer10_attn * torch.rsqrt(layer10_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer10_attention_norm2_weight
        unified_state = unified_state + layer10_gate_msa * layer10_attn
        layer10_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer10_ffn_norm1_weight
        layer10_ffn_in = layer10_ffn_in * layer10_scale_mlp
        layer10_ffn_a = layer10_ffn_in @ self.layer10_feed_forward_w1_weight.t()
        layer10_ffn_b = layer10_ffn_in @ self.layer10_feed_forward_w3_weight.t()
        layer10_ffn = F.silu(layer10_ffn_a) * layer10_ffn_b
        layer10_ffn = layer10_ffn @ self.layer10_feed_forward_w2_weight.t()
        layer10_ffn = layer10_ffn * torch.rsqrt(layer10_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer10_ffn_norm2_weight
        unified_state = unified_state + layer10_gate_mlp * layer10_ffn

        # layer11
        layer11_mod = adaln_input @ self.layer11_adaln_weight.t() + self.layer11_adaln_bias
        layer11_mod = layer11_mod.unsqueeze(1)
        layer11_scale_msa, layer11_gate_msa, layer11_scale_mlp, layer11_gate_mlp = layer11_mod.chunk(4, dim=2)
        layer11_gate_msa = torch.tanh(layer11_gate_msa)
        layer11_gate_mlp = torch.tanh(layer11_gate_mlp)
        layer11_scale_msa = 1.0 + layer11_scale_msa
        layer11_scale_mlp = 1.0 + layer11_scale_mlp
        layer11_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer11_attention_norm1_weight
        layer11_attn_in = layer11_attn_in * layer11_scale_msa
        layer11_q = layer11_attn_in @ self.layer11_attention_to_q_weight.t()
        layer11_k = layer11_attn_in @ self.layer11_attention_to_k_weight.t()
        layer11_v = layer11_attn_in @ self.layer11_attention_to_v_weight.t()
        layer11_q = layer11_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer11_k = layer11_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer11_v = layer11_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer11_q = layer11_q * torch.rsqrt(layer11_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer11_norm_q_weight.view(1, 1, 1, -1)
        layer11_k = layer11_k * torch.rsqrt(layer11_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer11_norm_k_weight.view(1, 1, 1, -1)
        layer11_q_complex = torch.view_as_complex(layer11_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer11_k_complex = torch.view_as_complex(layer11_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer11_q = torch.view_as_real(layer11_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer11_attn_in.dtype)
        layer11_k = torch.view_as_real(layer11_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer11_attn_in.dtype)
        layer11_q = layer11_q.transpose(1, 2)
        layer11_k = layer11_k.transpose(1, 2)
        layer11_v = layer11_v.transpose(1, 2)
        layer11_scores = torch.matmul(layer11_q.float(), layer11_k.float().transpose(-2, -1)) * self.attention_scale
        layer11_probs = torch.softmax(layer11_scores, dim=-1).to(layer11_v.dtype)
        layer11_attn = torch.matmul(layer11_probs, layer11_v)
        layer11_attn = layer11_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer11_attn = layer11_attn @ self.layer11_attention_to_out_weight.t()
        layer11_attn = layer11_attn * torch.rsqrt(layer11_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer11_attention_norm2_weight
        unified_state = unified_state + layer11_gate_msa * layer11_attn
        layer11_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer11_ffn_norm1_weight
        layer11_ffn_in = layer11_ffn_in * layer11_scale_mlp
        layer11_ffn_a = layer11_ffn_in @ self.layer11_feed_forward_w1_weight.t()
        layer11_ffn_b = layer11_ffn_in @ self.layer11_feed_forward_w3_weight.t()
        layer11_ffn = F.silu(layer11_ffn_a) * layer11_ffn_b
        layer11_ffn = layer11_ffn @ self.layer11_feed_forward_w2_weight.t()
        layer11_ffn = layer11_ffn * torch.rsqrt(layer11_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer11_ffn_norm2_weight
        unified_state = unified_state + layer11_gate_mlp * layer11_ffn

        # layer12
        layer12_mod = adaln_input @ self.layer12_adaln_weight.t() + self.layer12_adaln_bias
        layer12_mod = layer12_mod.unsqueeze(1)
        layer12_scale_msa, layer12_gate_msa, layer12_scale_mlp, layer12_gate_mlp = layer12_mod.chunk(4, dim=2)
        layer12_gate_msa = torch.tanh(layer12_gate_msa)
        layer12_gate_mlp = torch.tanh(layer12_gate_mlp)
        layer12_scale_msa = 1.0 + layer12_scale_msa
        layer12_scale_mlp = 1.0 + layer12_scale_mlp
        layer12_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer12_attention_norm1_weight
        layer12_attn_in = layer12_attn_in * layer12_scale_msa
        layer12_q = layer12_attn_in @ self.layer12_attention_to_q_weight.t()
        layer12_k = layer12_attn_in @ self.layer12_attention_to_k_weight.t()
        layer12_v = layer12_attn_in @ self.layer12_attention_to_v_weight.t()
        layer12_q = layer12_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer12_k = layer12_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer12_v = layer12_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer12_q = layer12_q * torch.rsqrt(layer12_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer12_norm_q_weight.view(1, 1, 1, -1)
        layer12_k = layer12_k * torch.rsqrt(layer12_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer12_norm_k_weight.view(1, 1, 1, -1)
        layer12_q_complex = torch.view_as_complex(layer12_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer12_k_complex = torch.view_as_complex(layer12_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer12_q = torch.view_as_real(layer12_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer12_attn_in.dtype)
        layer12_k = torch.view_as_real(layer12_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer12_attn_in.dtype)
        layer12_q = layer12_q.transpose(1, 2)
        layer12_k = layer12_k.transpose(1, 2)
        layer12_v = layer12_v.transpose(1, 2)
        layer12_scores = torch.matmul(layer12_q.float(), layer12_k.float().transpose(-2, -1)) * self.attention_scale
        layer12_probs = torch.softmax(layer12_scores, dim=-1).to(layer12_v.dtype)
        layer12_attn = torch.matmul(layer12_probs, layer12_v)
        layer12_attn = layer12_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer12_attn = layer12_attn @ self.layer12_attention_to_out_weight.t()
        layer12_attn = layer12_attn * torch.rsqrt(layer12_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer12_attention_norm2_weight
        unified_state = unified_state + layer12_gate_msa * layer12_attn
        layer12_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer12_ffn_norm1_weight
        layer12_ffn_in = layer12_ffn_in * layer12_scale_mlp
        layer12_ffn_a = layer12_ffn_in @ self.layer12_feed_forward_w1_weight.t()
        layer12_ffn_b = layer12_ffn_in @ self.layer12_feed_forward_w3_weight.t()
        layer12_ffn = F.silu(layer12_ffn_a) * layer12_ffn_b
        layer12_ffn = layer12_ffn @ self.layer12_feed_forward_w2_weight.t()
        layer12_ffn = layer12_ffn * torch.rsqrt(layer12_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer12_ffn_norm2_weight
        unified_state = unified_state + layer12_gate_mlp * layer12_ffn

        # layer13
        layer13_mod = adaln_input @ self.layer13_adaln_weight.t() + self.layer13_adaln_bias
        layer13_mod = layer13_mod.unsqueeze(1)
        layer13_scale_msa, layer13_gate_msa, layer13_scale_mlp, layer13_gate_mlp = layer13_mod.chunk(4, dim=2)
        layer13_gate_msa = torch.tanh(layer13_gate_msa)
        layer13_gate_mlp = torch.tanh(layer13_gate_mlp)
        layer13_scale_msa = 1.0 + layer13_scale_msa
        layer13_scale_mlp = 1.0 + layer13_scale_mlp
        layer13_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer13_attention_norm1_weight
        layer13_attn_in = layer13_attn_in * layer13_scale_msa
        layer13_q = layer13_attn_in @ self.layer13_attention_to_q_weight.t()
        layer13_k = layer13_attn_in @ self.layer13_attention_to_k_weight.t()
        layer13_v = layer13_attn_in @ self.layer13_attention_to_v_weight.t()
        layer13_q = layer13_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer13_k = layer13_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer13_v = layer13_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer13_q = layer13_q * torch.rsqrt(layer13_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer13_norm_q_weight.view(1, 1, 1, -1)
        layer13_k = layer13_k * torch.rsqrt(layer13_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer13_norm_k_weight.view(1, 1, 1, -1)
        layer13_q_complex = torch.view_as_complex(layer13_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer13_k_complex = torch.view_as_complex(layer13_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer13_q = torch.view_as_real(layer13_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer13_attn_in.dtype)
        layer13_k = torch.view_as_real(layer13_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer13_attn_in.dtype)
        layer13_q = layer13_q.transpose(1, 2)
        layer13_k = layer13_k.transpose(1, 2)
        layer13_v = layer13_v.transpose(1, 2)
        layer13_scores = torch.matmul(layer13_q.float(), layer13_k.float().transpose(-2, -1)) * self.attention_scale
        layer13_probs = torch.softmax(layer13_scores, dim=-1).to(layer13_v.dtype)
        layer13_attn = torch.matmul(layer13_probs, layer13_v)
        layer13_attn = layer13_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer13_attn = layer13_attn @ self.layer13_attention_to_out_weight.t()
        layer13_attn = layer13_attn * torch.rsqrt(layer13_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer13_attention_norm2_weight
        unified_state = unified_state + layer13_gate_msa * layer13_attn
        layer13_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer13_ffn_norm1_weight
        layer13_ffn_in = layer13_ffn_in * layer13_scale_mlp
        layer13_ffn_a = layer13_ffn_in @ self.layer13_feed_forward_w1_weight.t()
        layer13_ffn_b = layer13_ffn_in @ self.layer13_feed_forward_w3_weight.t()
        layer13_ffn = F.silu(layer13_ffn_a) * layer13_ffn_b
        layer13_ffn = layer13_ffn @ self.layer13_feed_forward_w2_weight.t()
        layer13_ffn = layer13_ffn * torch.rsqrt(layer13_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer13_ffn_norm2_weight
        unified_state = unified_state + layer13_gate_mlp * layer13_ffn

        # layer14
        layer14_mod = adaln_input @ self.layer14_adaln_weight.t() + self.layer14_adaln_bias
        layer14_mod = layer14_mod.unsqueeze(1)
        layer14_scale_msa, layer14_gate_msa, layer14_scale_mlp, layer14_gate_mlp = layer14_mod.chunk(4, dim=2)
        layer14_gate_msa = torch.tanh(layer14_gate_msa)
        layer14_gate_mlp = torch.tanh(layer14_gate_mlp)
        layer14_scale_msa = 1.0 + layer14_scale_msa
        layer14_scale_mlp = 1.0 + layer14_scale_mlp
        layer14_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer14_attention_norm1_weight
        layer14_attn_in = layer14_attn_in * layer14_scale_msa
        layer14_q = layer14_attn_in @ self.layer14_attention_to_q_weight.t()
        layer14_k = layer14_attn_in @ self.layer14_attention_to_k_weight.t()
        layer14_v = layer14_attn_in @ self.layer14_attention_to_v_weight.t()
        layer14_q = layer14_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer14_k = layer14_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer14_v = layer14_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer14_q = layer14_q * torch.rsqrt(layer14_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer14_norm_q_weight.view(1, 1, 1, -1)
        layer14_k = layer14_k * torch.rsqrt(layer14_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer14_norm_k_weight.view(1, 1, 1, -1)
        layer14_q_complex = torch.view_as_complex(layer14_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer14_k_complex = torch.view_as_complex(layer14_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer14_q = torch.view_as_real(layer14_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer14_attn_in.dtype)
        layer14_k = torch.view_as_real(layer14_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer14_attn_in.dtype)
        layer14_q = layer14_q.transpose(1, 2)
        layer14_k = layer14_k.transpose(1, 2)
        layer14_v = layer14_v.transpose(1, 2)
        layer14_scores = torch.matmul(layer14_q.float(), layer14_k.float().transpose(-2, -1)) * self.attention_scale
        layer14_probs = torch.softmax(layer14_scores, dim=-1).to(layer14_v.dtype)
        layer14_attn = torch.matmul(layer14_probs, layer14_v)
        layer14_attn = layer14_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer14_attn = layer14_attn @ self.layer14_attention_to_out_weight.t()
        layer14_attn = layer14_attn * torch.rsqrt(layer14_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer14_attention_norm2_weight
        unified_state = unified_state + layer14_gate_msa * layer14_attn
        layer14_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer14_ffn_norm1_weight
        layer14_ffn_in = layer14_ffn_in * layer14_scale_mlp
        layer14_ffn_a = layer14_ffn_in @ self.layer14_feed_forward_w1_weight.t()
        layer14_ffn_b = layer14_ffn_in @ self.layer14_feed_forward_w3_weight.t()
        layer14_ffn = F.silu(layer14_ffn_a) * layer14_ffn_b
        layer14_ffn = layer14_ffn @ self.layer14_feed_forward_w2_weight.t()
        layer14_ffn = layer14_ffn * torch.rsqrt(layer14_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer14_ffn_norm2_weight
        unified_state = unified_state + layer14_gate_mlp * layer14_ffn

        # layer15
        layer15_mod = adaln_input @ self.layer15_adaln_weight.t() + self.layer15_adaln_bias
        layer15_mod = layer15_mod.unsqueeze(1)
        layer15_scale_msa, layer15_gate_msa, layer15_scale_mlp, layer15_gate_mlp = layer15_mod.chunk(4, dim=2)
        layer15_gate_msa = torch.tanh(layer15_gate_msa)
        layer15_gate_mlp = torch.tanh(layer15_gate_mlp)
        layer15_scale_msa = 1.0 + layer15_scale_msa
        layer15_scale_mlp = 1.0 + layer15_scale_mlp
        layer15_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer15_attention_norm1_weight
        layer15_attn_in = layer15_attn_in * layer15_scale_msa
        layer15_q = layer15_attn_in @ self.layer15_attention_to_q_weight.t()
        layer15_k = layer15_attn_in @ self.layer15_attention_to_k_weight.t()
        layer15_v = layer15_attn_in @ self.layer15_attention_to_v_weight.t()
        layer15_q = layer15_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer15_k = layer15_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer15_v = layer15_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer15_q = layer15_q * torch.rsqrt(layer15_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer15_norm_q_weight.view(1, 1, 1, -1)
        layer15_k = layer15_k * torch.rsqrt(layer15_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer15_norm_k_weight.view(1, 1, 1, -1)
        layer15_q_complex = torch.view_as_complex(layer15_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer15_k_complex = torch.view_as_complex(layer15_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer15_q = torch.view_as_real(layer15_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer15_attn_in.dtype)
        layer15_k = torch.view_as_real(layer15_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer15_attn_in.dtype)
        layer15_q = layer15_q.transpose(1, 2)
        layer15_k = layer15_k.transpose(1, 2)
        layer15_v = layer15_v.transpose(1, 2)
        layer15_scores = torch.matmul(layer15_q.float(), layer15_k.float().transpose(-2, -1)) * self.attention_scale
        layer15_probs = torch.softmax(layer15_scores, dim=-1).to(layer15_v.dtype)
        layer15_attn = torch.matmul(layer15_probs, layer15_v)
        layer15_attn = layer15_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer15_attn = layer15_attn @ self.layer15_attention_to_out_weight.t()
        layer15_attn = layer15_attn * torch.rsqrt(layer15_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer15_attention_norm2_weight
        unified_state = unified_state + layer15_gate_msa * layer15_attn
        layer15_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer15_ffn_norm1_weight
        layer15_ffn_in = layer15_ffn_in * layer15_scale_mlp
        layer15_ffn_a = layer15_ffn_in @ self.layer15_feed_forward_w1_weight.t()
        layer15_ffn_b = layer15_ffn_in @ self.layer15_feed_forward_w3_weight.t()
        layer15_ffn = F.silu(layer15_ffn_a) * layer15_ffn_b
        layer15_ffn = layer15_ffn @ self.layer15_feed_forward_w2_weight.t()
        layer15_ffn = layer15_ffn * torch.rsqrt(layer15_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer15_ffn_norm2_weight
        unified_state = unified_state + layer15_gate_mlp * layer15_ffn

        # layer16
        layer16_mod = adaln_input @ self.layer16_adaln_weight.t() + self.layer16_adaln_bias
        layer16_mod = layer16_mod.unsqueeze(1)
        layer16_scale_msa, layer16_gate_msa, layer16_scale_mlp, layer16_gate_mlp = layer16_mod.chunk(4, dim=2)
        layer16_gate_msa = torch.tanh(layer16_gate_msa)
        layer16_gate_mlp = torch.tanh(layer16_gate_mlp)
        layer16_scale_msa = 1.0 + layer16_scale_msa
        layer16_scale_mlp = 1.0 + layer16_scale_mlp
        layer16_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer16_attention_norm1_weight
        layer16_attn_in = layer16_attn_in * layer16_scale_msa
        layer16_q = layer16_attn_in @ self.layer16_attention_to_q_weight.t()
        layer16_k = layer16_attn_in @ self.layer16_attention_to_k_weight.t()
        layer16_v = layer16_attn_in @ self.layer16_attention_to_v_weight.t()
        layer16_q = layer16_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer16_k = layer16_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer16_v = layer16_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer16_q = layer16_q * torch.rsqrt(layer16_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer16_norm_q_weight.view(1, 1, 1, -1)
        layer16_k = layer16_k * torch.rsqrt(layer16_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer16_norm_k_weight.view(1, 1, 1, -1)
        layer16_q_complex = torch.view_as_complex(layer16_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer16_k_complex = torch.view_as_complex(layer16_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer16_q = torch.view_as_real(layer16_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer16_attn_in.dtype)
        layer16_k = torch.view_as_real(layer16_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer16_attn_in.dtype)
        layer16_q = layer16_q.transpose(1, 2)
        layer16_k = layer16_k.transpose(1, 2)
        layer16_v = layer16_v.transpose(1, 2)
        layer16_scores = torch.matmul(layer16_q.float(), layer16_k.float().transpose(-2, -1)) * self.attention_scale
        layer16_probs = torch.softmax(layer16_scores, dim=-1).to(layer16_v.dtype)
        layer16_attn = torch.matmul(layer16_probs, layer16_v)
        layer16_attn = layer16_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer16_attn = layer16_attn @ self.layer16_attention_to_out_weight.t()
        layer16_attn = layer16_attn * torch.rsqrt(layer16_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer16_attention_norm2_weight
        unified_state = unified_state + layer16_gate_msa * layer16_attn
        layer16_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer16_ffn_norm1_weight
        layer16_ffn_in = layer16_ffn_in * layer16_scale_mlp
        layer16_ffn_a = layer16_ffn_in @ self.layer16_feed_forward_w1_weight.t()
        layer16_ffn_b = layer16_ffn_in @ self.layer16_feed_forward_w3_weight.t()
        layer16_ffn = F.silu(layer16_ffn_a) * layer16_ffn_b
        layer16_ffn = layer16_ffn @ self.layer16_feed_forward_w2_weight.t()
        layer16_ffn = layer16_ffn * torch.rsqrt(layer16_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer16_ffn_norm2_weight
        unified_state = unified_state + layer16_gate_mlp * layer16_ffn

        # layer17
        layer17_mod = adaln_input @ self.layer17_adaln_weight.t() + self.layer17_adaln_bias
        layer17_mod = layer17_mod.unsqueeze(1)
        layer17_scale_msa, layer17_gate_msa, layer17_scale_mlp, layer17_gate_mlp = layer17_mod.chunk(4, dim=2)
        layer17_gate_msa = torch.tanh(layer17_gate_msa)
        layer17_gate_mlp = torch.tanh(layer17_gate_mlp)
        layer17_scale_msa = 1.0 + layer17_scale_msa
        layer17_scale_mlp = 1.0 + layer17_scale_mlp
        layer17_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer17_attention_norm1_weight
        layer17_attn_in = layer17_attn_in * layer17_scale_msa
        layer17_q = layer17_attn_in @ self.layer17_attention_to_q_weight.t()
        layer17_k = layer17_attn_in @ self.layer17_attention_to_k_weight.t()
        layer17_v = layer17_attn_in @ self.layer17_attention_to_v_weight.t()
        layer17_q = layer17_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer17_k = layer17_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer17_v = layer17_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer17_q = layer17_q * torch.rsqrt(layer17_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer17_norm_q_weight.view(1, 1, 1, -1)
        layer17_k = layer17_k * torch.rsqrt(layer17_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer17_norm_k_weight.view(1, 1, 1, -1)
        layer17_q_complex = torch.view_as_complex(layer17_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer17_k_complex = torch.view_as_complex(layer17_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer17_q = torch.view_as_real(layer17_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer17_attn_in.dtype)
        layer17_k = torch.view_as_real(layer17_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer17_attn_in.dtype)
        layer17_q = layer17_q.transpose(1, 2)
        layer17_k = layer17_k.transpose(1, 2)
        layer17_v = layer17_v.transpose(1, 2)
        layer17_scores = torch.matmul(layer17_q.float(), layer17_k.float().transpose(-2, -1)) * self.attention_scale
        layer17_probs = torch.softmax(layer17_scores, dim=-1).to(layer17_v.dtype)
        layer17_attn = torch.matmul(layer17_probs, layer17_v)
        layer17_attn = layer17_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer17_attn = layer17_attn @ self.layer17_attention_to_out_weight.t()
        layer17_attn = layer17_attn * torch.rsqrt(layer17_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer17_attention_norm2_weight
        unified_state = unified_state + layer17_gate_msa * layer17_attn
        layer17_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer17_ffn_norm1_weight
        layer17_ffn_in = layer17_ffn_in * layer17_scale_mlp
        layer17_ffn_a = layer17_ffn_in @ self.layer17_feed_forward_w1_weight.t()
        layer17_ffn_b = layer17_ffn_in @ self.layer17_feed_forward_w3_weight.t()
        layer17_ffn = F.silu(layer17_ffn_a) * layer17_ffn_b
        layer17_ffn = layer17_ffn @ self.layer17_feed_forward_w2_weight.t()
        layer17_ffn = layer17_ffn * torch.rsqrt(layer17_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer17_ffn_norm2_weight
        unified_state = unified_state + layer17_gate_mlp * layer17_ffn

        # layer18
        layer18_mod = adaln_input @ self.layer18_adaln_weight.t() + self.layer18_adaln_bias
        layer18_mod = layer18_mod.unsqueeze(1)
        layer18_scale_msa, layer18_gate_msa, layer18_scale_mlp, layer18_gate_mlp = layer18_mod.chunk(4, dim=2)
        layer18_gate_msa = torch.tanh(layer18_gate_msa)
        layer18_gate_mlp = torch.tanh(layer18_gate_mlp)
        layer18_scale_msa = 1.0 + layer18_scale_msa
        layer18_scale_mlp = 1.0 + layer18_scale_mlp
        layer18_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer18_attention_norm1_weight
        layer18_attn_in = layer18_attn_in * layer18_scale_msa
        layer18_q = layer18_attn_in @ self.layer18_attention_to_q_weight.t()
        layer18_k = layer18_attn_in @ self.layer18_attention_to_k_weight.t()
        layer18_v = layer18_attn_in @ self.layer18_attention_to_v_weight.t()
        layer18_q = layer18_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer18_k = layer18_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer18_v = layer18_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer18_q = layer18_q * torch.rsqrt(layer18_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer18_norm_q_weight.view(1, 1, 1, -1)
        layer18_k = layer18_k * torch.rsqrt(layer18_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer18_norm_k_weight.view(1, 1, 1, -1)
        layer18_q_complex = torch.view_as_complex(layer18_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer18_k_complex = torch.view_as_complex(layer18_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer18_q = torch.view_as_real(layer18_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer18_attn_in.dtype)
        layer18_k = torch.view_as_real(layer18_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer18_attn_in.dtype)
        layer18_q = layer18_q.transpose(1, 2)
        layer18_k = layer18_k.transpose(1, 2)
        layer18_v = layer18_v.transpose(1, 2)
        layer18_scores = torch.matmul(layer18_q.float(), layer18_k.float().transpose(-2, -1)) * self.attention_scale
        layer18_probs = torch.softmax(layer18_scores, dim=-1).to(layer18_v.dtype)
        layer18_attn = torch.matmul(layer18_probs, layer18_v)
        layer18_attn = layer18_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer18_attn = layer18_attn @ self.layer18_attention_to_out_weight.t()
        layer18_attn = layer18_attn * torch.rsqrt(layer18_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer18_attention_norm2_weight
        unified_state = unified_state + layer18_gate_msa * layer18_attn
        layer18_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer18_ffn_norm1_weight
        layer18_ffn_in = layer18_ffn_in * layer18_scale_mlp
        layer18_ffn_a = layer18_ffn_in @ self.layer18_feed_forward_w1_weight.t()
        layer18_ffn_b = layer18_ffn_in @ self.layer18_feed_forward_w3_weight.t()
        layer18_ffn = F.silu(layer18_ffn_a) * layer18_ffn_b
        layer18_ffn = layer18_ffn @ self.layer18_feed_forward_w2_weight.t()
        layer18_ffn = layer18_ffn * torch.rsqrt(layer18_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer18_ffn_norm2_weight
        unified_state = unified_state + layer18_gate_mlp * layer18_ffn

        # layer19
        layer19_mod = adaln_input @ self.layer19_adaln_weight.t() + self.layer19_adaln_bias
        layer19_mod = layer19_mod.unsqueeze(1)
        layer19_scale_msa, layer19_gate_msa, layer19_scale_mlp, layer19_gate_mlp = layer19_mod.chunk(4, dim=2)
        layer19_gate_msa = torch.tanh(layer19_gate_msa)
        layer19_gate_mlp = torch.tanh(layer19_gate_mlp)
        layer19_scale_msa = 1.0 + layer19_scale_msa
        layer19_scale_mlp = 1.0 + layer19_scale_mlp
        layer19_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer19_attention_norm1_weight
        layer19_attn_in = layer19_attn_in * layer19_scale_msa
        layer19_q = layer19_attn_in @ self.layer19_attention_to_q_weight.t()
        layer19_k = layer19_attn_in @ self.layer19_attention_to_k_weight.t()
        layer19_v = layer19_attn_in @ self.layer19_attention_to_v_weight.t()
        layer19_q = layer19_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer19_k = layer19_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer19_v = layer19_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer19_q = layer19_q * torch.rsqrt(layer19_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer19_norm_q_weight.view(1, 1, 1, -1)
        layer19_k = layer19_k * torch.rsqrt(layer19_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer19_norm_k_weight.view(1, 1, 1, -1)
        layer19_q_complex = torch.view_as_complex(layer19_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer19_k_complex = torch.view_as_complex(layer19_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer19_q = torch.view_as_real(layer19_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer19_attn_in.dtype)
        layer19_k = torch.view_as_real(layer19_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer19_attn_in.dtype)
        layer19_q = layer19_q.transpose(1, 2)
        layer19_k = layer19_k.transpose(1, 2)
        layer19_v = layer19_v.transpose(1, 2)
        layer19_scores = torch.matmul(layer19_q.float(), layer19_k.float().transpose(-2, -1)) * self.attention_scale
        layer19_probs = torch.softmax(layer19_scores, dim=-1).to(layer19_v.dtype)
        layer19_attn = torch.matmul(layer19_probs, layer19_v)
        layer19_attn = layer19_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer19_attn = layer19_attn @ self.layer19_attention_to_out_weight.t()
        layer19_attn = layer19_attn * torch.rsqrt(layer19_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer19_attention_norm2_weight
        unified_state = unified_state + layer19_gate_msa * layer19_attn
        layer19_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer19_ffn_norm1_weight
        layer19_ffn_in = layer19_ffn_in * layer19_scale_mlp
        layer19_ffn_a = layer19_ffn_in @ self.layer19_feed_forward_w1_weight.t()
        layer19_ffn_b = layer19_ffn_in @ self.layer19_feed_forward_w3_weight.t()
        layer19_ffn = F.silu(layer19_ffn_a) * layer19_ffn_b
        layer19_ffn = layer19_ffn @ self.layer19_feed_forward_w2_weight.t()
        layer19_ffn = layer19_ffn * torch.rsqrt(layer19_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer19_ffn_norm2_weight
        unified_state = unified_state + layer19_gate_mlp * layer19_ffn

        # layer20
        layer20_mod = adaln_input @ self.layer20_adaln_weight.t() + self.layer20_adaln_bias
        layer20_mod = layer20_mod.unsqueeze(1)
        layer20_scale_msa, layer20_gate_msa, layer20_scale_mlp, layer20_gate_mlp = layer20_mod.chunk(4, dim=2)
        layer20_gate_msa = torch.tanh(layer20_gate_msa)
        layer20_gate_mlp = torch.tanh(layer20_gate_mlp)
        layer20_scale_msa = 1.0 + layer20_scale_msa
        layer20_scale_mlp = 1.0 + layer20_scale_mlp
        layer20_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer20_attention_norm1_weight
        layer20_attn_in = layer20_attn_in * layer20_scale_msa
        layer20_q = layer20_attn_in @ self.layer20_attention_to_q_weight.t()
        layer20_k = layer20_attn_in @ self.layer20_attention_to_k_weight.t()
        layer20_v = layer20_attn_in @ self.layer20_attention_to_v_weight.t()
        layer20_q = layer20_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer20_k = layer20_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer20_v = layer20_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer20_q = layer20_q * torch.rsqrt(layer20_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer20_norm_q_weight.view(1, 1, 1, -1)
        layer20_k = layer20_k * torch.rsqrt(layer20_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer20_norm_k_weight.view(1, 1, 1, -1)
        layer20_q_complex = torch.view_as_complex(layer20_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer20_k_complex = torch.view_as_complex(layer20_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer20_q = torch.view_as_real(layer20_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer20_attn_in.dtype)
        layer20_k = torch.view_as_real(layer20_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer20_attn_in.dtype)
        layer20_q = layer20_q.transpose(1, 2)
        layer20_k = layer20_k.transpose(1, 2)
        layer20_v = layer20_v.transpose(1, 2)
        layer20_scores = torch.matmul(layer20_q.float(), layer20_k.float().transpose(-2, -1)) * self.attention_scale
        layer20_probs = torch.softmax(layer20_scores, dim=-1).to(layer20_v.dtype)
        layer20_attn = torch.matmul(layer20_probs, layer20_v)
        layer20_attn = layer20_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer20_attn = layer20_attn @ self.layer20_attention_to_out_weight.t()
        layer20_attn = layer20_attn * torch.rsqrt(layer20_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer20_attention_norm2_weight
        unified_state = unified_state + layer20_gate_msa * layer20_attn
        layer20_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer20_ffn_norm1_weight
        layer20_ffn_in = layer20_ffn_in * layer20_scale_mlp
        layer20_ffn_a = layer20_ffn_in @ self.layer20_feed_forward_w1_weight.t()
        layer20_ffn_b = layer20_ffn_in @ self.layer20_feed_forward_w3_weight.t()
        layer20_ffn = F.silu(layer20_ffn_a) * layer20_ffn_b
        layer20_ffn = layer20_ffn @ self.layer20_feed_forward_w2_weight.t()
        layer20_ffn = layer20_ffn * torch.rsqrt(layer20_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer20_ffn_norm2_weight
        unified_state = unified_state + layer20_gate_mlp * layer20_ffn

        # layer21
        layer21_mod = adaln_input @ self.layer21_adaln_weight.t() + self.layer21_adaln_bias
        layer21_mod = layer21_mod.unsqueeze(1)
        layer21_scale_msa, layer21_gate_msa, layer21_scale_mlp, layer21_gate_mlp = layer21_mod.chunk(4, dim=2)
        layer21_gate_msa = torch.tanh(layer21_gate_msa)
        layer21_gate_mlp = torch.tanh(layer21_gate_mlp)
        layer21_scale_msa = 1.0 + layer21_scale_msa
        layer21_scale_mlp = 1.0 + layer21_scale_mlp
        layer21_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer21_attention_norm1_weight
        layer21_attn_in = layer21_attn_in * layer21_scale_msa
        layer21_q = layer21_attn_in @ self.layer21_attention_to_q_weight.t()
        layer21_k = layer21_attn_in @ self.layer21_attention_to_k_weight.t()
        layer21_v = layer21_attn_in @ self.layer21_attention_to_v_weight.t()
        layer21_q = layer21_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer21_k = layer21_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer21_v = layer21_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer21_q = layer21_q * torch.rsqrt(layer21_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer21_norm_q_weight.view(1, 1, 1, -1)
        layer21_k = layer21_k * torch.rsqrt(layer21_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer21_norm_k_weight.view(1, 1, 1, -1)
        layer21_q_complex = torch.view_as_complex(layer21_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer21_k_complex = torch.view_as_complex(layer21_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer21_q = torch.view_as_real(layer21_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer21_attn_in.dtype)
        layer21_k = torch.view_as_real(layer21_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer21_attn_in.dtype)
        layer21_q = layer21_q.transpose(1, 2)
        layer21_k = layer21_k.transpose(1, 2)
        layer21_v = layer21_v.transpose(1, 2)
        layer21_scores = torch.matmul(layer21_q.float(), layer21_k.float().transpose(-2, -1)) * self.attention_scale
        layer21_probs = torch.softmax(layer21_scores, dim=-1).to(layer21_v.dtype)
        layer21_attn = torch.matmul(layer21_probs, layer21_v)
        layer21_attn = layer21_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer21_attn = layer21_attn @ self.layer21_attention_to_out_weight.t()
        layer21_attn = layer21_attn * torch.rsqrt(layer21_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer21_attention_norm2_weight
        unified_state = unified_state + layer21_gate_msa * layer21_attn
        layer21_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer21_ffn_norm1_weight
        layer21_ffn_in = layer21_ffn_in * layer21_scale_mlp
        layer21_ffn_a = layer21_ffn_in @ self.layer21_feed_forward_w1_weight.t()
        layer21_ffn_b = layer21_ffn_in @ self.layer21_feed_forward_w3_weight.t()
        layer21_ffn = F.silu(layer21_ffn_a) * layer21_ffn_b
        layer21_ffn = layer21_ffn @ self.layer21_feed_forward_w2_weight.t()
        layer21_ffn = layer21_ffn * torch.rsqrt(layer21_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer21_ffn_norm2_weight
        unified_state = unified_state + layer21_gate_mlp * layer21_ffn

        # layer22
        layer22_mod = adaln_input @ self.layer22_adaln_weight.t() + self.layer22_adaln_bias
        layer22_mod = layer22_mod.unsqueeze(1)
        layer22_scale_msa, layer22_gate_msa, layer22_scale_mlp, layer22_gate_mlp = layer22_mod.chunk(4, dim=2)
        layer22_gate_msa = torch.tanh(layer22_gate_msa)
        layer22_gate_mlp = torch.tanh(layer22_gate_mlp)
        layer22_scale_msa = 1.0 + layer22_scale_msa
        layer22_scale_mlp = 1.0 + layer22_scale_mlp
        layer22_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer22_attention_norm1_weight
        layer22_attn_in = layer22_attn_in * layer22_scale_msa
        layer22_q = layer22_attn_in @ self.layer22_attention_to_q_weight.t()
        layer22_k = layer22_attn_in @ self.layer22_attention_to_k_weight.t()
        layer22_v = layer22_attn_in @ self.layer22_attention_to_v_weight.t()
        layer22_q = layer22_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer22_k = layer22_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer22_v = layer22_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer22_q = layer22_q * torch.rsqrt(layer22_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer22_norm_q_weight.view(1, 1, 1, -1)
        layer22_k = layer22_k * torch.rsqrt(layer22_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer22_norm_k_weight.view(1, 1, 1, -1)
        layer22_q_complex = torch.view_as_complex(layer22_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer22_k_complex = torch.view_as_complex(layer22_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer22_q = torch.view_as_real(layer22_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer22_attn_in.dtype)
        layer22_k = torch.view_as_real(layer22_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer22_attn_in.dtype)
        layer22_q = layer22_q.transpose(1, 2)
        layer22_k = layer22_k.transpose(1, 2)
        layer22_v = layer22_v.transpose(1, 2)
        layer22_scores = torch.matmul(layer22_q.float(), layer22_k.float().transpose(-2, -1)) * self.attention_scale
        layer22_probs = torch.softmax(layer22_scores, dim=-1).to(layer22_v.dtype)
        layer22_attn = torch.matmul(layer22_probs, layer22_v)
        layer22_attn = layer22_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer22_attn = layer22_attn @ self.layer22_attention_to_out_weight.t()
        layer22_attn = layer22_attn * torch.rsqrt(layer22_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer22_attention_norm2_weight
        unified_state = unified_state + layer22_gate_msa * layer22_attn
        layer22_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer22_ffn_norm1_weight
        layer22_ffn_in = layer22_ffn_in * layer22_scale_mlp
        layer22_ffn_a = layer22_ffn_in @ self.layer22_feed_forward_w1_weight.t()
        layer22_ffn_b = layer22_ffn_in @ self.layer22_feed_forward_w3_weight.t()
        layer22_ffn = F.silu(layer22_ffn_a) * layer22_ffn_b
        layer22_ffn = layer22_ffn @ self.layer22_feed_forward_w2_weight.t()
        layer22_ffn = layer22_ffn * torch.rsqrt(layer22_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer22_ffn_norm2_weight
        unified_state = unified_state + layer22_gate_mlp * layer22_ffn

        # layer23
        layer23_mod = adaln_input @ self.layer23_adaln_weight.t() + self.layer23_adaln_bias
        layer23_mod = layer23_mod.unsqueeze(1)
        layer23_scale_msa, layer23_gate_msa, layer23_scale_mlp, layer23_gate_mlp = layer23_mod.chunk(4, dim=2)
        layer23_gate_msa = torch.tanh(layer23_gate_msa)
        layer23_gate_mlp = torch.tanh(layer23_gate_mlp)
        layer23_scale_msa = 1.0 + layer23_scale_msa
        layer23_scale_mlp = 1.0 + layer23_scale_mlp
        layer23_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer23_attention_norm1_weight
        layer23_attn_in = layer23_attn_in * layer23_scale_msa
        layer23_q = layer23_attn_in @ self.layer23_attention_to_q_weight.t()
        layer23_k = layer23_attn_in @ self.layer23_attention_to_k_weight.t()
        layer23_v = layer23_attn_in @ self.layer23_attention_to_v_weight.t()
        layer23_q = layer23_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer23_k = layer23_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer23_v = layer23_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer23_q = layer23_q * torch.rsqrt(layer23_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer23_norm_q_weight.view(1, 1, 1, -1)
        layer23_k = layer23_k * torch.rsqrt(layer23_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer23_norm_k_weight.view(1, 1, 1, -1)
        layer23_q_complex = torch.view_as_complex(layer23_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer23_k_complex = torch.view_as_complex(layer23_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer23_q = torch.view_as_real(layer23_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer23_attn_in.dtype)
        layer23_k = torch.view_as_real(layer23_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer23_attn_in.dtype)
        layer23_q = layer23_q.transpose(1, 2)
        layer23_k = layer23_k.transpose(1, 2)
        layer23_v = layer23_v.transpose(1, 2)
        layer23_scores = torch.matmul(layer23_q.float(), layer23_k.float().transpose(-2, -1)) * self.attention_scale
        layer23_probs = torch.softmax(layer23_scores, dim=-1).to(layer23_v.dtype)
        layer23_attn = torch.matmul(layer23_probs, layer23_v)
        layer23_attn = layer23_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer23_attn = layer23_attn @ self.layer23_attention_to_out_weight.t()
        layer23_attn = layer23_attn * torch.rsqrt(layer23_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer23_attention_norm2_weight
        unified_state = unified_state + layer23_gate_msa * layer23_attn
        layer23_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer23_ffn_norm1_weight
        layer23_ffn_in = layer23_ffn_in * layer23_scale_mlp
        layer23_ffn_a = layer23_ffn_in @ self.layer23_feed_forward_w1_weight.t()
        layer23_ffn_b = layer23_ffn_in @ self.layer23_feed_forward_w3_weight.t()
        layer23_ffn = F.silu(layer23_ffn_a) * layer23_ffn_b
        layer23_ffn = layer23_ffn @ self.layer23_feed_forward_w2_weight.t()
        layer23_ffn = layer23_ffn * torch.rsqrt(layer23_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer23_ffn_norm2_weight
        unified_state = unified_state + layer23_gate_mlp * layer23_ffn

        # layer24
        layer24_mod = adaln_input @ self.layer24_adaln_weight.t() + self.layer24_adaln_bias
        layer24_mod = layer24_mod.unsqueeze(1)
        layer24_scale_msa, layer24_gate_msa, layer24_scale_mlp, layer24_gate_mlp = layer24_mod.chunk(4, dim=2)
        layer24_gate_msa = torch.tanh(layer24_gate_msa)
        layer24_gate_mlp = torch.tanh(layer24_gate_mlp)
        layer24_scale_msa = 1.0 + layer24_scale_msa
        layer24_scale_mlp = 1.0 + layer24_scale_mlp
        layer24_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer24_attention_norm1_weight
        layer24_attn_in = layer24_attn_in * layer24_scale_msa
        layer24_q = layer24_attn_in @ self.layer24_attention_to_q_weight.t()
        layer24_k = layer24_attn_in @ self.layer24_attention_to_k_weight.t()
        layer24_v = layer24_attn_in @ self.layer24_attention_to_v_weight.t()
        layer24_q = layer24_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer24_k = layer24_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer24_v = layer24_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer24_q = layer24_q * torch.rsqrt(layer24_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer24_norm_q_weight.view(1, 1, 1, -1)
        layer24_k = layer24_k * torch.rsqrt(layer24_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer24_norm_k_weight.view(1, 1, 1, -1)
        layer24_q_complex = torch.view_as_complex(layer24_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer24_k_complex = torch.view_as_complex(layer24_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer24_q = torch.view_as_real(layer24_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer24_attn_in.dtype)
        layer24_k = torch.view_as_real(layer24_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer24_attn_in.dtype)
        layer24_q = layer24_q.transpose(1, 2)
        layer24_k = layer24_k.transpose(1, 2)
        layer24_v = layer24_v.transpose(1, 2)
        layer24_scores = torch.matmul(layer24_q.float(), layer24_k.float().transpose(-2, -1)) * self.attention_scale
        layer24_probs = torch.softmax(layer24_scores, dim=-1).to(layer24_v.dtype)
        layer24_attn = torch.matmul(layer24_probs, layer24_v)
        layer24_attn = layer24_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer24_attn = layer24_attn @ self.layer24_attention_to_out_weight.t()
        layer24_attn = layer24_attn * torch.rsqrt(layer24_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer24_attention_norm2_weight
        unified_state = unified_state + layer24_gate_msa * layer24_attn
        layer24_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer24_ffn_norm1_weight
        layer24_ffn_in = layer24_ffn_in * layer24_scale_mlp
        layer24_ffn_a = layer24_ffn_in @ self.layer24_feed_forward_w1_weight.t()
        layer24_ffn_b = layer24_ffn_in @ self.layer24_feed_forward_w3_weight.t()
        layer24_ffn = F.silu(layer24_ffn_a) * layer24_ffn_b
        layer24_ffn = layer24_ffn @ self.layer24_feed_forward_w2_weight.t()
        layer24_ffn = layer24_ffn * torch.rsqrt(layer24_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer24_ffn_norm2_weight
        unified_state = unified_state + layer24_gate_mlp * layer24_ffn

        # layer25
        layer25_mod = adaln_input @ self.layer25_adaln_weight.t() + self.layer25_adaln_bias
        layer25_mod = layer25_mod.unsqueeze(1)
        layer25_scale_msa, layer25_gate_msa, layer25_scale_mlp, layer25_gate_mlp = layer25_mod.chunk(4, dim=2)
        layer25_gate_msa = torch.tanh(layer25_gate_msa)
        layer25_gate_mlp = torch.tanh(layer25_gate_mlp)
        layer25_scale_msa = 1.0 + layer25_scale_msa
        layer25_scale_mlp = 1.0 + layer25_scale_mlp
        layer25_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer25_attention_norm1_weight
        layer25_attn_in = layer25_attn_in * layer25_scale_msa
        layer25_q = layer25_attn_in @ self.layer25_attention_to_q_weight.t()
        layer25_k = layer25_attn_in @ self.layer25_attention_to_k_weight.t()
        layer25_v = layer25_attn_in @ self.layer25_attention_to_v_weight.t()
        layer25_q = layer25_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer25_k = layer25_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer25_v = layer25_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer25_q = layer25_q * torch.rsqrt(layer25_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer25_norm_q_weight.view(1, 1, 1, -1)
        layer25_k = layer25_k * torch.rsqrt(layer25_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer25_norm_k_weight.view(1, 1, 1, -1)
        layer25_q_complex = torch.view_as_complex(layer25_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer25_k_complex = torch.view_as_complex(layer25_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer25_q = torch.view_as_real(layer25_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer25_attn_in.dtype)
        layer25_k = torch.view_as_real(layer25_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer25_attn_in.dtype)
        layer25_q = layer25_q.transpose(1, 2)
        layer25_k = layer25_k.transpose(1, 2)
        layer25_v = layer25_v.transpose(1, 2)
        layer25_scores = torch.matmul(layer25_q.float(), layer25_k.float().transpose(-2, -1)) * self.attention_scale
        layer25_probs = torch.softmax(layer25_scores, dim=-1).to(layer25_v.dtype)
        layer25_attn = torch.matmul(layer25_probs, layer25_v)
        layer25_attn = layer25_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer25_attn = layer25_attn @ self.layer25_attention_to_out_weight.t()
        layer25_attn = layer25_attn * torch.rsqrt(layer25_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer25_attention_norm2_weight
        unified_state = unified_state + layer25_gate_msa * layer25_attn
        layer25_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer25_ffn_norm1_weight
        layer25_ffn_in = layer25_ffn_in * layer25_scale_mlp
        layer25_ffn_a = layer25_ffn_in @ self.layer25_feed_forward_w1_weight.t()
        layer25_ffn_b = layer25_ffn_in @ self.layer25_feed_forward_w3_weight.t()
        layer25_ffn = F.silu(layer25_ffn_a) * layer25_ffn_b
        layer25_ffn = layer25_ffn @ self.layer25_feed_forward_w2_weight.t()
        layer25_ffn = layer25_ffn * torch.rsqrt(layer25_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer25_ffn_norm2_weight
        unified_state = unified_state + layer25_gate_mlp * layer25_ffn

        # layer26
        layer26_mod = adaln_input @ self.layer26_adaln_weight.t() + self.layer26_adaln_bias
        layer26_mod = layer26_mod.unsqueeze(1)
        layer26_scale_msa, layer26_gate_msa, layer26_scale_mlp, layer26_gate_mlp = layer26_mod.chunk(4, dim=2)
        layer26_gate_msa = torch.tanh(layer26_gate_msa)
        layer26_gate_mlp = torch.tanh(layer26_gate_mlp)
        layer26_scale_msa = 1.0 + layer26_scale_msa
        layer26_scale_mlp = 1.0 + layer26_scale_mlp
        layer26_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer26_attention_norm1_weight
        layer26_attn_in = layer26_attn_in * layer26_scale_msa
        layer26_q = layer26_attn_in @ self.layer26_attention_to_q_weight.t()
        layer26_k = layer26_attn_in @ self.layer26_attention_to_k_weight.t()
        layer26_v = layer26_attn_in @ self.layer26_attention_to_v_weight.t()
        layer26_q = layer26_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer26_k = layer26_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer26_v = layer26_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer26_q = layer26_q * torch.rsqrt(layer26_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer26_norm_q_weight.view(1, 1, 1, -1)
        layer26_k = layer26_k * torch.rsqrt(layer26_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer26_norm_k_weight.view(1, 1, 1, -1)
        layer26_q_complex = torch.view_as_complex(layer26_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer26_k_complex = torch.view_as_complex(layer26_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer26_q = torch.view_as_real(layer26_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer26_attn_in.dtype)
        layer26_k = torch.view_as_real(layer26_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer26_attn_in.dtype)
        layer26_q = layer26_q.transpose(1, 2)
        layer26_k = layer26_k.transpose(1, 2)
        layer26_v = layer26_v.transpose(1, 2)
        layer26_scores = torch.matmul(layer26_q.float(), layer26_k.float().transpose(-2, -1)) * self.attention_scale
        layer26_probs = torch.softmax(layer26_scores, dim=-1).to(layer26_v.dtype)
        layer26_attn = torch.matmul(layer26_probs, layer26_v)
        layer26_attn = layer26_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer26_attn = layer26_attn @ self.layer26_attention_to_out_weight.t()
        layer26_attn = layer26_attn * torch.rsqrt(layer26_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer26_attention_norm2_weight
        unified_state = unified_state + layer26_gate_msa * layer26_attn
        layer26_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer26_ffn_norm1_weight
        layer26_ffn_in = layer26_ffn_in * layer26_scale_mlp
        layer26_ffn_a = layer26_ffn_in @ self.layer26_feed_forward_w1_weight.t()
        layer26_ffn_b = layer26_ffn_in @ self.layer26_feed_forward_w3_weight.t()
        layer26_ffn = F.silu(layer26_ffn_a) * layer26_ffn_b
        layer26_ffn = layer26_ffn @ self.layer26_feed_forward_w2_weight.t()
        layer26_ffn = layer26_ffn * torch.rsqrt(layer26_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer26_ffn_norm2_weight
        unified_state = unified_state + layer26_gate_mlp * layer26_ffn

        # layer27
        layer27_mod = adaln_input @ self.layer27_adaln_weight.t() + self.layer27_adaln_bias
        layer27_mod = layer27_mod.unsqueeze(1)
        layer27_scale_msa, layer27_gate_msa, layer27_scale_mlp, layer27_gate_mlp = layer27_mod.chunk(4, dim=2)
        layer27_gate_msa = torch.tanh(layer27_gate_msa)
        layer27_gate_mlp = torch.tanh(layer27_gate_mlp)
        layer27_scale_msa = 1.0 + layer27_scale_msa
        layer27_scale_mlp = 1.0 + layer27_scale_mlp
        layer27_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer27_attention_norm1_weight
        layer27_attn_in = layer27_attn_in * layer27_scale_msa
        layer27_q = layer27_attn_in @ self.layer27_attention_to_q_weight.t()
        layer27_k = layer27_attn_in @ self.layer27_attention_to_k_weight.t()
        layer27_v = layer27_attn_in @ self.layer27_attention_to_v_weight.t()
        layer27_q = layer27_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer27_k = layer27_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer27_v = layer27_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer27_q = layer27_q * torch.rsqrt(layer27_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer27_norm_q_weight.view(1, 1, 1, -1)
        layer27_k = layer27_k * torch.rsqrt(layer27_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer27_norm_k_weight.view(1, 1, 1, -1)
        layer27_q_complex = torch.view_as_complex(layer27_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer27_k_complex = torch.view_as_complex(layer27_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer27_q = torch.view_as_real(layer27_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer27_attn_in.dtype)
        layer27_k = torch.view_as_real(layer27_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer27_attn_in.dtype)
        layer27_q = layer27_q.transpose(1, 2)
        layer27_k = layer27_k.transpose(1, 2)
        layer27_v = layer27_v.transpose(1, 2)
        layer27_scores = torch.matmul(layer27_q.float(), layer27_k.float().transpose(-2, -1)) * self.attention_scale
        layer27_probs = torch.softmax(layer27_scores, dim=-1).to(layer27_v.dtype)
        layer27_attn = torch.matmul(layer27_probs, layer27_v)
        layer27_attn = layer27_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer27_attn = layer27_attn @ self.layer27_attention_to_out_weight.t()
        layer27_attn = layer27_attn * torch.rsqrt(layer27_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer27_attention_norm2_weight
        unified_state = unified_state + layer27_gate_msa * layer27_attn
        layer27_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer27_ffn_norm1_weight
        layer27_ffn_in = layer27_ffn_in * layer27_scale_mlp
        layer27_ffn_a = layer27_ffn_in @ self.layer27_feed_forward_w1_weight.t()
        layer27_ffn_b = layer27_ffn_in @ self.layer27_feed_forward_w3_weight.t()
        layer27_ffn = F.silu(layer27_ffn_a) * layer27_ffn_b
        layer27_ffn = layer27_ffn @ self.layer27_feed_forward_w2_weight.t()
        layer27_ffn = layer27_ffn * torch.rsqrt(layer27_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer27_ffn_norm2_weight
        unified_state = unified_state + layer27_gate_mlp * layer27_ffn

        # layer28
        layer28_mod = adaln_input @ self.layer28_adaln_weight.t() + self.layer28_adaln_bias
        layer28_mod = layer28_mod.unsqueeze(1)
        layer28_scale_msa, layer28_gate_msa, layer28_scale_mlp, layer28_gate_mlp = layer28_mod.chunk(4, dim=2)
        layer28_gate_msa = torch.tanh(layer28_gate_msa)
        layer28_gate_mlp = torch.tanh(layer28_gate_mlp)
        layer28_scale_msa = 1.0 + layer28_scale_msa
        layer28_scale_mlp = 1.0 + layer28_scale_mlp
        layer28_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer28_attention_norm1_weight
        layer28_attn_in = layer28_attn_in * layer28_scale_msa
        layer28_q = layer28_attn_in @ self.layer28_attention_to_q_weight.t()
        layer28_k = layer28_attn_in @ self.layer28_attention_to_k_weight.t()
        layer28_v = layer28_attn_in @ self.layer28_attention_to_v_weight.t()
        layer28_q = layer28_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer28_k = layer28_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer28_v = layer28_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer28_q = layer28_q * torch.rsqrt(layer28_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer28_norm_q_weight.view(1, 1, 1, -1)
        layer28_k = layer28_k * torch.rsqrt(layer28_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer28_norm_k_weight.view(1, 1, 1, -1)
        layer28_q_complex = torch.view_as_complex(layer28_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer28_k_complex = torch.view_as_complex(layer28_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer28_q = torch.view_as_real(layer28_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer28_attn_in.dtype)
        layer28_k = torch.view_as_real(layer28_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer28_attn_in.dtype)
        layer28_q = layer28_q.transpose(1, 2)
        layer28_k = layer28_k.transpose(1, 2)
        layer28_v = layer28_v.transpose(1, 2)
        layer28_scores = torch.matmul(layer28_q.float(), layer28_k.float().transpose(-2, -1)) * self.attention_scale
        layer28_probs = torch.softmax(layer28_scores, dim=-1).to(layer28_v.dtype)
        layer28_attn = torch.matmul(layer28_probs, layer28_v)
        layer28_attn = layer28_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer28_attn = layer28_attn @ self.layer28_attention_to_out_weight.t()
        layer28_attn = layer28_attn * torch.rsqrt(layer28_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer28_attention_norm2_weight
        unified_state = unified_state + layer28_gate_msa * layer28_attn
        layer28_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer28_ffn_norm1_weight
        layer28_ffn_in = layer28_ffn_in * layer28_scale_mlp
        layer28_ffn_a = layer28_ffn_in @ self.layer28_feed_forward_w1_weight.t()
        layer28_ffn_b = layer28_ffn_in @ self.layer28_feed_forward_w3_weight.t()
        layer28_ffn = F.silu(layer28_ffn_a) * layer28_ffn_b
        layer28_ffn = layer28_ffn @ self.layer28_feed_forward_w2_weight.t()
        layer28_ffn = layer28_ffn * torch.rsqrt(layer28_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer28_ffn_norm2_weight
        unified_state = unified_state + layer28_gate_mlp * layer28_ffn

        # layer29
        layer29_mod = adaln_input @ self.layer29_adaln_weight.t() + self.layer29_adaln_bias
        layer29_mod = layer29_mod.unsqueeze(1)
        layer29_scale_msa, layer29_gate_msa, layer29_scale_mlp, layer29_gate_mlp = layer29_mod.chunk(4, dim=2)
        layer29_gate_msa = torch.tanh(layer29_gate_msa)
        layer29_gate_mlp = torch.tanh(layer29_gate_mlp)
        layer29_scale_msa = 1.0 + layer29_scale_msa
        layer29_scale_mlp = 1.0 + layer29_scale_mlp
        layer29_attn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer29_attention_norm1_weight
        layer29_attn_in = layer29_attn_in * layer29_scale_msa
        layer29_q = layer29_attn_in @ self.layer29_attention_to_q_weight.t()
        layer29_k = layer29_attn_in @ self.layer29_attention_to_k_weight.t()
        layer29_v = layer29_attn_in @ self.layer29_attention_to_v_weight.t()
        layer29_q = layer29_q.view(1, unified_state.shape[1], self.n_heads, self.head_dim)
        layer29_k = layer29_k.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer29_v = layer29_v.view(1, unified_state.shape[1], self.n_kv_heads, self.head_dim)
        layer29_q = layer29_q * torch.rsqrt(layer29_q.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer29_norm_q_weight.view(1, 1, 1, -1)
        layer29_k = layer29_k * torch.rsqrt(layer29_k.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer29_norm_k_weight.view(1, 1, 1, -1)
        layer29_q_complex = torch.view_as_complex(layer29_q.float().reshape(1, unified_state.shape[1], self.n_heads, -1, 2))
        layer29_k_complex = torch.view_as_complex(layer29_k.float().reshape(1, unified_state.shape[1], self.n_kv_heads, -1, 2))
        layer29_q = torch.view_as_real(layer29_q_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer29_attn_in.dtype)
        layer29_k = torch.view_as_real(layer29_k_complex * unified_freqs.unsqueeze(2)).flatten(3).to(layer29_attn_in.dtype)
        layer29_q = layer29_q.transpose(1, 2)
        layer29_k = layer29_k.transpose(1, 2)
        layer29_v = layer29_v.transpose(1, 2)
        layer29_scores = torch.matmul(layer29_q.float(), layer29_k.float().transpose(-2, -1)) * self.attention_scale
        layer29_probs = torch.softmax(layer29_scores, dim=-1).to(layer29_v.dtype)
        layer29_attn = torch.matmul(layer29_probs, layer29_v)
        layer29_attn = layer29_attn.transpose(1, 2).reshape(1, unified_state.shape[1], self.q_proj_dim)
        layer29_attn = layer29_attn @ self.layer29_attention_to_out_weight.t()
        layer29_attn = layer29_attn * torch.rsqrt(layer29_attn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer29_attention_norm2_weight
        unified_state = unified_state + layer29_gate_msa * layer29_attn
        layer29_ffn_in = unified_state * torch.rsqrt(unified_state.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer29_ffn_norm1_weight
        layer29_ffn_in = layer29_ffn_in * layer29_scale_mlp
        layer29_ffn_a = layer29_ffn_in @ self.layer29_feed_forward_w1_weight.t()
        layer29_ffn_b = layer29_ffn_in @ self.layer29_feed_forward_w3_weight.t()
        layer29_ffn = F.silu(layer29_ffn_a) * layer29_ffn_b
        layer29_ffn = layer29_ffn @ self.layer29_feed_forward_w2_weight.t()
        layer29_ffn = layer29_ffn * torch.rsqrt(layer29_ffn.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.layer29_ffn_norm2_weight
        unified_state = unified_state + layer29_gate_mlp * layer29_ffn

        # final projection back to latent patches
        final_scale = F.silu(adaln_input) @ self.final_adaln_weight.t() + self.final_adaln_bias
        final_scale = 1.0 + final_scale
        unified_state = F.layer_norm(unified_state.float(), (self.dim,), None, None, 1e-6).to(adaln_input.dtype)
        unified_state = unified_state * final_scale.unsqueeze(1)
        unified_state = unified_state @ self.final_linear_weight.t() + self.final_linear_bias

        # unpatchify only the image token prefix back to [C, F, H, W]
        image_out = unified_state[0, :image_original_len, :]
        image_out = image_out.view(
            frame_tokens,
            height_tokens,
            width_tokens,
            self.f_patch_size,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        image_out = image_out.permute(6, 0, 3, 1, 4, 2, 5).reshape(self.out_channels, frames, height, width)
        return [image_out], {}


def load_transformer(path: Path, device: torch.device, dtype: torch.dtype) -> ZImageTransformer2DModel:
    reference = load_reference_transformer(path, device, dtype)
    model = ZImageTransformer2DModel(
        all_patch_size=tuple(reference.all_patch_size),
        all_f_patch_size=tuple(reference.all_f_patch_size),
        in_channels=reference.in_channels,
        dim=reference.dim,
        n_layers=len(reference.layers),
        n_refiner_layers=len(reference.noise_refiner),
        n_heads=reference.n_heads,
        n_kv_heads=reference.noise_refiner[0].attention.n_kv_heads,
        norm_eps=reference.noise_refiner[0].attention_norm1.eps,
        qk_norm=reference.noise_refiner[0].attention.norm_q is not None,
        cap_feat_dim=reference.cap_embedder[0].weight.shape[0],
        rope_theta=reference.rope_theta,
        t_scale=reference.t_scale,
        axes_dims=reference.axes_dims,
        axes_lens=reference.axes_lens,
        device=device,
        dtype=dtype,
    )

    model.x_embedder_weight.data.copy_(reference.all_x_embedder["2-1"].weight.data)
    model.x_embedder_bias.data.copy_(reference.all_x_embedder["2-1"].bias.data)
    model.t_mlp_w1.data.copy_(reference.t_embedder.mlp[0].weight.data)
    model.t_mlp_b1.data.copy_(reference.t_embedder.mlp[0].bias.data)
    model.t_mlp_w2.data.copy_(reference.t_embedder.mlp[2].weight.data)
    model.t_mlp_b2.data.copy_(reference.t_embedder.mlp[2].bias.data)
    model.cap_norm_weight.data.copy_(reference.cap_embedder[0].weight.data)
    model.cap_linear_weight.data.copy_(reference.cap_embedder[1].weight.data)
    model.cap_linear_bias.data.copy_(reference.cap_embedder[1].bias.data)
    model.x_pad_token.data.copy_(reference.x_pad_token.data)
    model.cap_pad_token.data.copy_(reference.cap_pad_token.data)
    model.final_linear_weight.data.copy_(reference.all_final_layer["2-1"].linear.weight.data)
    model.final_linear_bias.data.copy_(reference.all_final_layer["2-1"].linear.bias.data)
    model.final_adaln_weight.data.copy_(reference.all_final_layer["2-1"].adaLN_modulation[1].weight.data)
    model.final_adaln_bias.data.copy_(reference.all_final_layer["2-1"].adaLN_modulation[1].bias.data)
    model.nr0_attention_to_q_weight.data.copy_(reference.noise_refiner[0].attention.to_q.weight.data)
    model.nr0_attention_to_k_weight.data.copy_(reference.noise_refiner[0].attention.to_k.weight.data)
    model.nr0_attention_to_v_weight.data.copy_(reference.noise_refiner[0].attention.to_v.weight.data)
    model.nr0_attention_to_out_weight.data.copy_(reference.noise_refiner[0].attention.to_out[0].weight.data)
    model.nr0_norm_q_weight.data.copy_(reference.noise_refiner[0].attention.norm_q.weight.data)
    model.nr0_norm_k_weight.data.copy_(reference.noise_refiner[0].attention.norm_k.weight.data)
    model.nr0_feed_forward_w1_weight.data.copy_(reference.noise_refiner[0].feed_forward.w1.weight.data)
    model.nr0_feed_forward_w2_weight.data.copy_(reference.noise_refiner[0].feed_forward.w2.weight.data)
    model.nr0_feed_forward_w3_weight.data.copy_(reference.noise_refiner[0].feed_forward.w3.weight.data)
    model.nr0_attention_norm1_weight.data.copy_(reference.noise_refiner[0].attention_norm1.weight.data)
    model.nr0_ffn_norm1_weight.data.copy_(reference.noise_refiner[0].ffn_norm1.weight.data)
    model.nr0_attention_norm2_weight.data.copy_(reference.noise_refiner[0].attention_norm2.weight.data)
    model.nr0_ffn_norm2_weight.data.copy_(reference.noise_refiner[0].ffn_norm2.weight.data)
    model.nr0_adaln_weight.data.copy_(reference.noise_refiner[0].adaLN_modulation[0].weight.data)
    model.nr0_adaln_bias.data.copy_(reference.noise_refiner[0].adaLN_modulation[0].bias.data)
    model.nr1_attention_to_q_weight.data.copy_(reference.noise_refiner[1].attention.to_q.weight.data)
    model.nr1_attention_to_k_weight.data.copy_(reference.noise_refiner[1].attention.to_k.weight.data)
    model.nr1_attention_to_v_weight.data.copy_(reference.noise_refiner[1].attention.to_v.weight.data)
    model.nr1_attention_to_out_weight.data.copy_(reference.noise_refiner[1].attention.to_out[0].weight.data)
    model.nr1_norm_q_weight.data.copy_(reference.noise_refiner[1].attention.norm_q.weight.data)
    model.nr1_norm_k_weight.data.copy_(reference.noise_refiner[1].attention.norm_k.weight.data)
    model.nr1_feed_forward_w1_weight.data.copy_(reference.noise_refiner[1].feed_forward.w1.weight.data)
    model.nr1_feed_forward_w2_weight.data.copy_(reference.noise_refiner[1].feed_forward.w2.weight.data)
    model.nr1_feed_forward_w3_weight.data.copy_(reference.noise_refiner[1].feed_forward.w3.weight.data)
    model.nr1_attention_norm1_weight.data.copy_(reference.noise_refiner[1].attention_norm1.weight.data)
    model.nr1_ffn_norm1_weight.data.copy_(reference.noise_refiner[1].ffn_norm1.weight.data)
    model.nr1_attention_norm2_weight.data.copy_(reference.noise_refiner[1].attention_norm2.weight.data)
    model.nr1_ffn_norm2_weight.data.copy_(reference.noise_refiner[1].ffn_norm2.weight.data)
    model.nr1_adaln_weight.data.copy_(reference.noise_refiner[1].adaLN_modulation[0].weight.data)
    model.nr1_adaln_bias.data.copy_(reference.noise_refiner[1].adaLN_modulation[0].bias.data)
    model.cr0_attention_to_q_weight.data.copy_(reference.context_refiner[0].attention.to_q.weight.data)
    model.cr0_attention_to_k_weight.data.copy_(reference.context_refiner[0].attention.to_k.weight.data)
    model.cr0_attention_to_v_weight.data.copy_(reference.context_refiner[0].attention.to_v.weight.data)
    model.cr0_attention_to_out_weight.data.copy_(reference.context_refiner[0].attention.to_out[0].weight.data)
    model.cr0_norm_q_weight.data.copy_(reference.context_refiner[0].attention.norm_q.weight.data)
    model.cr0_norm_k_weight.data.copy_(reference.context_refiner[0].attention.norm_k.weight.data)
    model.cr0_feed_forward_w1_weight.data.copy_(reference.context_refiner[0].feed_forward.w1.weight.data)
    model.cr0_feed_forward_w2_weight.data.copy_(reference.context_refiner[0].feed_forward.w2.weight.data)
    model.cr0_feed_forward_w3_weight.data.copy_(reference.context_refiner[0].feed_forward.w3.weight.data)
    model.cr0_attention_norm1_weight.data.copy_(reference.context_refiner[0].attention_norm1.weight.data)
    model.cr0_ffn_norm1_weight.data.copy_(reference.context_refiner[0].ffn_norm1.weight.data)
    model.cr0_attention_norm2_weight.data.copy_(reference.context_refiner[0].attention_norm2.weight.data)
    model.cr0_ffn_norm2_weight.data.copy_(reference.context_refiner[0].ffn_norm2.weight.data)
    model.cr1_attention_to_q_weight.data.copy_(reference.context_refiner[1].attention.to_q.weight.data)
    model.cr1_attention_to_k_weight.data.copy_(reference.context_refiner[1].attention.to_k.weight.data)
    model.cr1_attention_to_v_weight.data.copy_(reference.context_refiner[1].attention.to_v.weight.data)
    model.cr1_attention_to_out_weight.data.copy_(reference.context_refiner[1].attention.to_out[0].weight.data)
    model.cr1_norm_q_weight.data.copy_(reference.context_refiner[1].attention.norm_q.weight.data)
    model.cr1_norm_k_weight.data.copy_(reference.context_refiner[1].attention.norm_k.weight.data)
    model.cr1_feed_forward_w1_weight.data.copy_(reference.context_refiner[1].feed_forward.w1.weight.data)
    model.cr1_feed_forward_w2_weight.data.copy_(reference.context_refiner[1].feed_forward.w2.weight.data)
    model.cr1_feed_forward_w3_weight.data.copy_(reference.context_refiner[1].feed_forward.w3.weight.data)
    model.cr1_attention_norm1_weight.data.copy_(reference.context_refiner[1].attention_norm1.weight.data)
    model.cr1_ffn_norm1_weight.data.copy_(reference.context_refiner[1].ffn_norm1.weight.data)
    model.cr1_attention_norm2_weight.data.copy_(reference.context_refiner[1].attention_norm2.weight.data)
    model.cr1_ffn_norm2_weight.data.copy_(reference.context_refiner[1].ffn_norm2.weight.data)
    model.layer0_attention_to_q_weight.data.copy_(reference.layers[0].attention.to_q.weight.data)
    model.layer0_attention_to_k_weight.data.copy_(reference.layers[0].attention.to_k.weight.data)
    model.layer0_attention_to_v_weight.data.copy_(reference.layers[0].attention.to_v.weight.data)
    model.layer0_attention_to_out_weight.data.copy_(reference.layers[0].attention.to_out[0].weight.data)
    model.layer0_norm_q_weight.data.copy_(reference.layers[0].attention.norm_q.weight.data)
    model.layer0_norm_k_weight.data.copy_(reference.layers[0].attention.norm_k.weight.data)
    model.layer0_feed_forward_w1_weight.data.copy_(reference.layers[0].feed_forward.w1.weight.data)
    model.layer0_feed_forward_w2_weight.data.copy_(reference.layers[0].feed_forward.w2.weight.data)
    model.layer0_feed_forward_w3_weight.data.copy_(reference.layers[0].feed_forward.w3.weight.data)
    model.layer0_attention_norm1_weight.data.copy_(reference.layers[0].attention_norm1.weight.data)
    model.layer0_ffn_norm1_weight.data.copy_(reference.layers[0].ffn_norm1.weight.data)
    model.layer0_attention_norm2_weight.data.copy_(reference.layers[0].attention_norm2.weight.data)
    model.layer0_ffn_norm2_weight.data.copy_(reference.layers[0].ffn_norm2.weight.data)
    model.layer0_adaln_weight.data.copy_(reference.layers[0].adaLN_modulation[0].weight.data)
    model.layer0_adaln_bias.data.copy_(reference.layers[0].adaLN_modulation[0].bias.data)
    model.layer1_attention_to_q_weight.data.copy_(reference.layers[1].attention.to_q.weight.data)
    model.layer1_attention_to_k_weight.data.copy_(reference.layers[1].attention.to_k.weight.data)
    model.layer1_attention_to_v_weight.data.copy_(reference.layers[1].attention.to_v.weight.data)
    model.layer1_attention_to_out_weight.data.copy_(reference.layers[1].attention.to_out[0].weight.data)
    model.layer1_norm_q_weight.data.copy_(reference.layers[1].attention.norm_q.weight.data)
    model.layer1_norm_k_weight.data.copy_(reference.layers[1].attention.norm_k.weight.data)
    model.layer1_feed_forward_w1_weight.data.copy_(reference.layers[1].feed_forward.w1.weight.data)
    model.layer1_feed_forward_w2_weight.data.copy_(reference.layers[1].feed_forward.w2.weight.data)
    model.layer1_feed_forward_w3_weight.data.copy_(reference.layers[1].feed_forward.w3.weight.data)
    model.layer1_attention_norm1_weight.data.copy_(reference.layers[1].attention_norm1.weight.data)
    model.layer1_ffn_norm1_weight.data.copy_(reference.layers[1].ffn_norm1.weight.data)
    model.layer1_attention_norm2_weight.data.copy_(reference.layers[1].attention_norm2.weight.data)
    model.layer1_ffn_norm2_weight.data.copy_(reference.layers[1].ffn_norm2.weight.data)
    model.layer1_adaln_weight.data.copy_(reference.layers[1].adaLN_modulation[0].weight.data)
    model.layer1_adaln_bias.data.copy_(reference.layers[1].adaLN_modulation[0].bias.data)
    model.layer2_attention_to_q_weight.data.copy_(reference.layers[2].attention.to_q.weight.data)
    model.layer2_attention_to_k_weight.data.copy_(reference.layers[2].attention.to_k.weight.data)
    model.layer2_attention_to_v_weight.data.copy_(reference.layers[2].attention.to_v.weight.data)
    model.layer2_attention_to_out_weight.data.copy_(reference.layers[2].attention.to_out[0].weight.data)
    model.layer2_norm_q_weight.data.copy_(reference.layers[2].attention.norm_q.weight.data)
    model.layer2_norm_k_weight.data.copy_(reference.layers[2].attention.norm_k.weight.data)
    model.layer2_feed_forward_w1_weight.data.copy_(reference.layers[2].feed_forward.w1.weight.data)
    model.layer2_feed_forward_w2_weight.data.copy_(reference.layers[2].feed_forward.w2.weight.data)
    model.layer2_feed_forward_w3_weight.data.copy_(reference.layers[2].feed_forward.w3.weight.data)
    model.layer2_attention_norm1_weight.data.copy_(reference.layers[2].attention_norm1.weight.data)
    model.layer2_ffn_norm1_weight.data.copy_(reference.layers[2].ffn_norm1.weight.data)
    model.layer2_attention_norm2_weight.data.copy_(reference.layers[2].attention_norm2.weight.data)
    model.layer2_ffn_norm2_weight.data.copy_(reference.layers[2].ffn_norm2.weight.data)
    model.layer2_adaln_weight.data.copy_(reference.layers[2].adaLN_modulation[0].weight.data)
    model.layer2_adaln_bias.data.copy_(reference.layers[2].adaLN_modulation[0].bias.data)
    model.layer3_attention_to_q_weight.data.copy_(reference.layers[3].attention.to_q.weight.data)
    model.layer3_attention_to_k_weight.data.copy_(reference.layers[3].attention.to_k.weight.data)
    model.layer3_attention_to_v_weight.data.copy_(reference.layers[3].attention.to_v.weight.data)
    model.layer3_attention_to_out_weight.data.copy_(reference.layers[3].attention.to_out[0].weight.data)
    model.layer3_norm_q_weight.data.copy_(reference.layers[3].attention.norm_q.weight.data)
    model.layer3_norm_k_weight.data.copy_(reference.layers[3].attention.norm_k.weight.data)
    model.layer3_feed_forward_w1_weight.data.copy_(reference.layers[3].feed_forward.w1.weight.data)
    model.layer3_feed_forward_w2_weight.data.copy_(reference.layers[3].feed_forward.w2.weight.data)
    model.layer3_feed_forward_w3_weight.data.copy_(reference.layers[3].feed_forward.w3.weight.data)
    model.layer3_attention_norm1_weight.data.copy_(reference.layers[3].attention_norm1.weight.data)
    model.layer3_ffn_norm1_weight.data.copy_(reference.layers[3].ffn_norm1.weight.data)
    model.layer3_attention_norm2_weight.data.copy_(reference.layers[3].attention_norm2.weight.data)
    model.layer3_ffn_norm2_weight.data.copy_(reference.layers[3].ffn_norm2.weight.data)
    model.layer3_adaln_weight.data.copy_(reference.layers[3].adaLN_modulation[0].weight.data)
    model.layer3_adaln_bias.data.copy_(reference.layers[3].adaLN_modulation[0].bias.data)
    model.layer4_attention_to_q_weight.data.copy_(reference.layers[4].attention.to_q.weight.data)
    model.layer4_attention_to_k_weight.data.copy_(reference.layers[4].attention.to_k.weight.data)
    model.layer4_attention_to_v_weight.data.copy_(reference.layers[4].attention.to_v.weight.data)
    model.layer4_attention_to_out_weight.data.copy_(reference.layers[4].attention.to_out[0].weight.data)
    model.layer4_norm_q_weight.data.copy_(reference.layers[4].attention.norm_q.weight.data)
    model.layer4_norm_k_weight.data.copy_(reference.layers[4].attention.norm_k.weight.data)
    model.layer4_feed_forward_w1_weight.data.copy_(reference.layers[4].feed_forward.w1.weight.data)
    model.layer4_feed_forward_w2_weight.data.copy_(reference.layers[4].feed_forward.w2.weight.data)
    model.layer4_feed_forward_w3_weight.data.copy_(reference.layers[4].feed_forward.w3.weight.data)
    model.layer4_attention_norm1_weight.data.copy_(reference.layers[4].attention_norm1.weight.data)
    model.layer4_ffn_norm1_weight.data.copy_(reference.layers[4].ffn_norm1.weight.data)
    model.layer4_attention_norm2_weight.data.copy_(reference.layers[4].attention_norm2.weight.data)
    model.layer4_ffn_norm2_weight.data.copy_(reference.layers[4].ffn_norm2.weight.data)
    model.layer4_adaln_weight.data.copy_(reference.layers[4].adaLN_modulation[0].weight.data)
    model.layer4_adaln_bias.data.copy_(reference.layers[4].adaLN_modulation[0].bias.data)
    model.layer5_attention_to_q_weight.data.copy_(reference.layers[5].attention.to_q.weight.data)
    model.layer5_attention_to_k_weight.data.copy_(reference.layers[5].attention.to_k.weight.data)
    model.layer5_attention_to_v_weight.data.copy_(reference.layers[5].attention.to_v.weight.data)
    model.layer5_attention_to_out_weight.data.copy_(reference.layers[5].attention.to_out[0].weight.data)
    model.layer5_norm_q_weight.data.copy_(reference.layers[5].attention.norm_q.weight.data)
    model.layer5_norm_k_weight.data.copy_(reference.layers[5].attention.norm_k.weight.data)
    model.layer5_feed_forward_w1_weight.data.copy_(reference.layers[5].feed_forward.w1.weight.data)
    model.layer5_feed_forward_w2_weight.data.copy_(reference.layers[5].feed_forward.w2.weight.data)
    model.layer5_feed_forward_w3_weight.data.copy_(reference.layers[5].feed_forward.w3.weight.data)
    model.layer5_attention_norm1_weight.data.copy_(reference.layers[5].attention_norm1.weight.data)
    model.layer5_ffn_norm1_weight.data.copy_(reference.layers[5].ffn_norm1.weight.data)
    model.layer5_attention_norm2_weight.data.copy_(reference.layers[5].attention_norm2.weight.data)
    model.layer5_ffn_norm2_weight.data.copy_(reference.layers[5].ffn_norm2.weight.data)
    model.layer5_adaln_weight.data.copy_(reference.layers[5].adaLN_modulation[0].weight.data)
    model.layer5_adaln_bias.data.copy_(reference.layers[5].adaLN_modulation[0].bias.data)
    model.layer6_attention_to_q_weight.data.copy_(reference.layers[6].attention.to_q.weight.data)
    model.layer6_attention_to_k_weight.data.copy_(reference.layers[6].attention.to_k.weight.data)
    model.layer6_attention_to_v_weight.data.copy_(reference.layers[6].attention.to_v.weight.data)
    model.layer6_attention_to_out_weight.data.copy_(reference.layers[6].attention.to_out[0].weight.data)
    model.layer6_norm_q_weight.data.copy_(reference.layers[6].attention.norm_q.weight.data)
    model.layer6_norm_k_weight.data.copy_(reference.layers[6].attention.norm_k.weight.data)
    model.layer6_feed_forward_w1_weight.data.copy_(reference.layers[6].feed_forward.w1.weight.data)
    model.layer6_feed_forward_w2_weight.data.copy_(reference.layers[6].feed_forward.w2.weight.data)
    model.layer6_feed_forward_w3_weight.data.copy_(reference.layers[6].feed_forward.w3.weight.data)
    model.layer6_attention_norm1_weight.data.copy_(reference.layers[6].attention_norm1.weight.data)
    model.layer6_ffn_norm1_weight.data.copy_(reference.layers[6].ffn_norm1.weight.data)
    model.layer6_attention_norm2_weight.data.copy_(reference.layers[6].attention_norm2.weight.data)
    model.layer6_ffn_norm2_weight.data.copy_(reference.layers[6].ffn_norm2.weight.data)
    model.layer6_adaln_weight.data.copy_(reference.layers[6].adaLN_modulation[0].weight.data)
    model.layer6_adaln_bias.data.copy_(reference.layers[6].adaLN_modulation[0].bias.data)
    model.layer7_attention_to_q_weight.data.copy_(reference.layers[7].attention.to_q.weight.data)
    model.layer7_attention_to_k_weight.data.copy_(reference.layers[7].attention.to_k.weight.data)
    model.layer7_attention_to_v_weight.data.copy_(reference.layers[7].attention.to_v.weight.data)
    model.layer7_attention_to_out_weight.data.copy_(reference.layers[7].attention.to_out[0].weight.data)
    model.layer7_norm_q_weight.data.copy_(reference.layers[7].attention.norm_q.weight.data)
    model.layer7_norm_k_weight.data.copy_(reference.layers[7].attention.norm_k.weight.data)
    model.layer7_feed_forward_w1_weight.data.copy_(reference.layers[7].feed_forward.w1.weight.data)
    model.layer7_feed_forward_w2_weight.data.copy_(reference.layers[7].feed_forward.w2.weight.data)
    model.layer7_feed_forward_w3_weight.data.copy_(reference.layers[7].feed_forward.w3.weight.data)
    model.layer7_attention_norm1_weight.data.copy_(reference.layers[7].attention_norm1.weight.data)
    model.layer7_ffn_norm1_weight.data.copy_(reference.layers[7].ffn_norm1.weight.data)
    model.layer7_attention_norm2_weight.data.copy_(reference.layers[7].attention_norm2.weight.data)
    model.layer7_ffn_norm2_weight.data.copy_(reference.layers[7].ffn_norm2.weight.data)
    model.layer7_adaln_weight.data.copy_(reference.layers[7].adaLN_modulation[0].weight.data)
    model.layer7_adaln_bias.data.copy_(reference.layers[7].adaLN_modulation[0].bias.data)
    model.layer8_attention_to_q_weight.data.copy_(reference.layers[8].attention.to_q.weight.data)
    model.layer8_attention_to_k_weight.data.copy_(reference.layers[8].attention.to_k.weight.data)
    model.layer8_attention_to_v_weight.data.copy_(reference.layers[8].attention.to_v.weight.data)
    model.layer8_attention_to_out_weight.data.copy_(reference.layers[8].attention.to_out[0].weight.data)
    model.layer8_norm_q_weight.data.copy_(reference.layers[8].attention.norm_q.weight.data)
    model.layer8_norm_k_weight.data.copy_(reference.layers[8].attention.norm_k.weight.data)
    model.layer8_feed_forward_w1_weight.data.copy_(reference.layers[8].feed_forward.w1.weight.data)
    model.layer8_feed_forward_w2_weight.data.copy_(reference.layers[8].feed_forward.w2.weight.data)
    model.layer8_feed_forward_w3_weight.data.copy_(reference.layers[8].feed_forward.w3.weight.data)
    model.layer8_attention_norm1_weight.data.copy_(reference.layers[8].attention_norm1.weight.data)
    model.layer8_ffn_norm1_weight.data.copy_(reference.layers[8].ffn_norm1.weight.data)
    model.layer8_attention_norm2_weight.data.copy_(reference.layers[8].attention_norm2.weight.data)
    model.layer8_ffn_norm2_weight.data.copy_(reference.layers[8].ffn_norm2.weight.data)
    model.layer8_adaln_weight.data.copy_(reference.layers[8].adaLN_modulation[0].weight.data)
    model.layer8_adaln_bias.data.copy_(reference.layers[8].adaLN_modulation[0].bias.data)
    model.layer9_attention_to_q_weight.data.copy_(reference.layers[9].attention.to_q.weight.data)
    model.layer9_attention_to_k_weight.data.copy_(reference.layers[9].attention.to_k.weight.data)
    model.layer9_attention_to_v_weight.data.copy_(reference.layers[9].attention.to_v.weight.data)
    model.layer9_attention_to_out_weight.data.copy_(reference.layers[9].attention.to_out[0].weight.data)
    model.layer9_norm_q_weight.data.copy_(reference.layers[9].attention.norm_q.weight.data)
    model.layer9_norm_k_weight.data.copy_(reference.layers[9].attention.norm_k.weight.data)
    model.layer9_feed_forward_w1_weight.data.copy_(reference.layers[9].feed_forward.w1.weight.data)
    model.layer9_feed_forward_w2_weight.data.copy_(reference.layers[9].feed_forward.w2.weight.data)
    model.layer9_feed_forward_w3_weight.data.copy_(reference.layers[9].feed_forward.w3.weight.data)
    model.layer9_attention_norm1_weight.data.copy_(reference.layers[9].attention_norm1.weight.data)
    model.layer9_ffn_norm1_weight.data.copy_(reference.layers[9].ffn_norm1.weight.data)
    model.layer9_attention_norm2_weight.data.copy_(reference.layers[9].attention_norm2.weight.data)
    model.layer9_ffn_norm2_weight.data.copy_(reference.layers[9].ffn_norm2.weight.data)
    model.layer9_adaln_weight.data.copy_(reference.layers[9].adaLN_modulation[0].weight.data)
    model.layer9_adaln_bias.data.copy_(reference.layers[9].adaLN_modulation[0].bias.data)
    model.layer10_attention_to_q_weight.data.copy_(reference.layers[10].attention.to_q.weight.data)
    model.layer10_attention_to_k_weight.data.copy_(reference.layers[10].attention.to_k.weight.data)
    model.layer10_attention_to_v_weight.data.copy_(reference.layers[10].attention.to_v.weight.data)
    model.layer10_attention_to_out_weight.data.copy_(reference.layers[10].attention.to_out[0].weight.data)
    model.layer10_norm_q_weight.data.copy_(reference.layers[10].attention.norm_q.weight.data)
    model.layer10_norm_k_weight.data.copy_(reference.layers[10].attention.norm_k.weight.data)
    model.layer10_feed_forward_w1_weight.data.copy_(reference.layers[10].feed_forward.w1.weight.data)
    model.layer10_feed_forward_w2_weight.data.copy_(reference.layers[10].feed_forward.w2.weight.data)
    model.layer10_feed_forward_w3_weight.data.copy_(reference.layers[10].feed_forward.w3.weight.data)
    model.layer10_attention_norm1_weight.data.copy_(reference.layers[10].attention_norm1.weight.data)
    model.layer10_ffn_norm1_weight.data.copy_(reference.layers[10].ffn_norm1.weight.data)
    model.layer10_attention_norm2_weight.data.copy_(reference.layers[10].attention_norm2.weight.data)
    model.layer10_ffn_norm2_weight.data.copy_(reference.layers[10].ffn_norm2.weight.data)
    model.layer10_adaln_weight.data.copy_(reference.layers[10].adaLN_modulation[0].weight.data)
    model.layer10_adaln_bias.data.copy_(reference.layers[10].adaLN_modulation[0].bias.data)
    model.layer11_attention_to_q_weight.data.copy_(reference.layers[11].attention.to_q.weight.data)
    model.layer11_attention_to_k_weight.data.copy_(reference.layers[11].attention.to_k.weight.data)
    model.layer11_attention_to_v_weight.data.copy_(reference.layers[11].attention.to_v.weight.data)
    model.layer11_attention_to_out_weight.data.copy_(reference.layers[11].attention.to_out[0].weight.data)
    model.layer11_norm_q_weight.data.copy_(reference.layers[11].attention.norm_q.weight.data)
    model.layer11_norm_k_weight.data.copy_(reference.layers[11].attention.norm_k.weight.data)
    model.layer11_feed_forward_w1_weight.data.copy_(reference.layers[11].feed_forward.w1.weight.data)
    model.layer11_feed_forward_w2_weight.data.copy_(reference.layers[11].feed_forward.w2.weight.data)
    model.layer11_feed_forward_w3_weight.data.copy_(reference.layers[11].feed_forward.w3.weight.data)
    model.layer11_attention_norm1_weight.data.copy_(reference.layers[11].attention_norm1.weight.data)
    model.layer11_ffn_norm1_weight.data.copy_(reference.layers[11].ffn_norm1.weight.data)
    model.layer11_attention_norm2_weight.data.copy_(reference.layers[11].attention_norm2.weight.data)
    model.layer11_ffn_norm2_weight.data.copy_(reference.layers[11].ffn_norm2.weight.data)
    model.layer11_adaln_weight.data.copy_(reference.layers[11].adaLN_modulation[0].weight.data)
    model.layer11_adaln_bias.data.copy_(reference.layers[11].adaLN_modulation[0].bias.data)
    model.layer12_attention_to_q_weight.data.copy_(reference.layers[12].attention.to_q.weight.data)
    model.layer12_attention_to_k_weight.data.copy_(reference.layers[12].attention.to_k.weight.data)
    model.layer12_attention_to_v_weight.data.copy_(reference.layers[12].attention.to_v.weight.data)
    model.layer12_attention_to_out_weight.data.copy_(reference.layers[12].attention.to_out[0].weight.data)
    model.layer12_norm_q_weight.data.copy_(reference.layers[12].attention.norm_q.weight.data)
    model.layer12_norm_k_weight.data.copy_(reference.layers[12].attention.norm_k.weight.data)
    model.layer12_feed_forward_w1_weight.data.copy_(reference.layers[12].feed_forward.w1.weight.data)
    model.layer12_feed_forward_w2_weight.data.copy_(reference.layers[12].feed_forward.w2.weight.data)
    model.layer12_feed_forward_w3_weight.data.copy_(reference.layers[12].feed_forward.w3.weight.data)
    model.layer12_attention_norm1_weight.data.copy_(reference.layers[12].attention_norm1.weight.data)
    model.layer12_ffn_norm1_weight.data.copy_(reference.layers[12].ffn_norm1.weight.data)
    model.layer12_attention_norm2_weight.data.copy_(reference.layers[12].attention_norm2.weight.data)
    model.layer12_ffn_norm2_weight.data.copy_(reference.layers[12].ffn_norm2.weight.data)
    model.layer12_adaln_weight.data.copy_(reference.layers[12].adaLN_modulation[0].weight.data)
    model.layer12_adaln_bias.data.copy_(reference.layers[12].adaLN_modulation[0].bias.data)
    model.layer13_attention_to_q_weight.data.copy_(reference.layers[13].attention.to_q.weight.data)
    model.layer13_attention_to_k_weight.data.copy_(reference.layers[13].attention.to_k.weight.data)
    model.layer13_attention_to_v_weight.data.copy_(reference.layers[13].attention.to_v.weight.data)
    model.layer13_attention_to_out_weight.data.copy_(reference.layers[13].attention.to_out[0].weight.data)
    model.layer13_norm_q_weight.data.copy_(reference.layers[13].attention.norm_q.weight.data)
    model.layer13_norm_k_weight.data.copy_(reference.layers[13].attention.norm_k.weight.data)
    model.layer13_feed_forward_w1_weight.data.copy_(reference.layers[13].feed_forward.w1.weight.data)
    model.layer13_feed_forward_w2_weight.data.copy_(reference.layers[13].feed_forward.w2.weight.data)
    model.layer13_feed_forward_w3_weight.data.copy_(reference.layers[13].feed_forward.w3.weight.data)
    model.layer13_attention_norm1_weight.data.copy_(reference.layers[13].attention_norm1.weight.data)
    model.layer13_ffn_norm1_weight.data.copy_(reference.layers[13].ffn_norm1.weight.data)
    model.layer13_attention_norm2_weight.data.copy_(reference.layers[13].attention_norm2.weight.data)
    model.layer13_ffn_norm2_weight.data.copy_(reference.layers[13].ffn_norm2.weight.data)
    model.layer13_adaln_weight.data.copy_(reference.layers[13].adaLN_modulation[0].weight.data)
    model.layer13_adaln_bias.data.copy_(reference.layers[13].adaLN_modulation[0].bias.data)
    model.layer14_attention_to_q_weight.data.copy_(reference.layers[14].attention.to_q.weight.data)
    model.layer14_attention_to_k_weight.data.copy_(reference.layers[14].attention.to_k.weight.data)
    model.layer14_attention_to_v_weight.data.copy_(reference.layers[14].attention.to_v.weight.data)
    model.layer14_attention_to_out_weight.data.copy_(reference.layers[14].attention.to_out[0].weight.data)
    model.layer14_norm_q_weight.data.copy_(reference.layers[14].attention.norm_q.weight.data)
    model.layer14_norm_k_weight.data.copy_(reference.layers[14].attention.norm_k.weight.data)
    model.layer14_feed_forward_w1_weight.data.copy_(reference.layers[14].feed_forward.w1.weight.data)
    model.layer14_feed_forward_w2_weight.data.copy_(reference.layers[14].feed_forward.w2.weight.data)
    model.layer14_feed_forward_w3_weight.data.copy_(reference.layers[14].feed_forward.w3.weight.data)
    model.layer14_attention_norm1_weight.data.copy_(reference.layers[14].attention_norm1.weight.data)
    model.layer14_ffn_norm1_weight.data.copy_(reference.layers[14].ffn_norm1.weight.data)
    model.layer14_attention_norm2_weight.data.copy_(reference.layers[14].attention_norm2.weight.data)
    model.layer14_ffn_norm2_weight.data.copy_(reference.layers[14].ffn_norm2.weight.data)
    model.layer14_adaln_weight.data.copy_(reference.layers[14].adaLN_modulation[0].weight.data)
    model.layer14_adaln_bias.data.copy_(reference.layers[14].adaLN_modulation[0].bias.data)
    model.layer15_attention_to_q_weight.data.copy_(reference.layers[15].attention.to_q.weight.data)
    model.layer15_attention_to_k_weight.data.copy_(reference.layers[15].attention.to_k.weight.data)
    model.layer15_attention_to_v_weight.data.copy_(reference.layers[15].attention.to_v.weight.data)
    model.layer15_attention_to_out_weight.data.copy_(reference.layers[15].attention.to_out[0].weight.data)
    model.layer15_norm_q_weight.data.copy_(reference.layers[15].attention.norm_q.weight.data)
    model.layer15_norm_k_weight.data.copy_(reference.layers[15].attention.norm_k.weight.data)
    model.layer15_feed_forward_w1_weight.data.copy_(reference.layers[15].feed_forward.w1.weight.data)
    model.layer15_feed_forward_w2_weight.data.copy_(reference.layers[15].feed_forward.w2.weight.data)
    model.layer15_feed_forward_w3_weight.data.copy_(reference.layers[15].feed_forward.w3.weight.data)
    model.layer15_attention_norm1_weight.data.copy_(reference.layers[15].attention_norm1.weight.data)
    model.layer15_ffn_norm1_weight.data.copy_(reference.layers[15].ffn_norm1.weight.data)
    model.layer15_attention_norm2_weight.data.copy_(reference.layers[15].attention_norm2.weight.data)
    model.layer15_ffn_norm2_weight.data.copy_(reference.layers[15].ffn_norm2.weight.data)
    model.layer15_adaln_weight.data.copy_(reference.layers[15].adaLN_modulation[0].weight.data)
    model.layer15_adaln_bias.data.copy_(reference.layers[15].adaLN_modulation[0].bias.data)
    model.layer16_attention_to_q_weight.data.copy_(reference.layers[16].attention.to_q.weight.data)
    model.layer16_attention_to_k_weight.data.copy_(reference.layers[16].attention.to_k.weight.data)
    model.layer16_attention_to_v_weight.data.copy_(reference.layers[16].attention.to_v.weight.data)
    model.layer16_attention_to_out_weight.data.copy_(reference.layers[16].attention.to_out[0].weight.data)
    model.layer16_norm_q_weight.data.copy_(reference.layers[16].attention.norm_q.weight.data)
    model.layer16_norm_k_weight.data.copy_(reference.layers[16].attention.norm_k.weight.data)
    model.layer16_feed_forward_w1_weight.data.copy_(reference.layers[16].feed_forward.w1.weight.data)
    model.layer16_feed_forward_w2_weight.data.copy_(reference.layers[16].feed_forward.w2.weight.data)
    model.layer16_feed_forward_w3_weight.data.copy_(reference.layers[16].feed_forward.w3.weight.data)
    model.layer16_attention_norm1_weight.data.copy_(reference.layers[16].attention_norm1.weight.data)
    model.layer16_ffn_norm1_weight.data.copy_(reference.layers[16].ffn_norm1.weight.data)
    model.layer16_attention_norm2_weight.data.copy_(reference.layers[16].attention_norm2.weight.data)
    model.layer16_ffn_norm2_weight.data.copy_(reference.layers[16].ffn_norm2.weight.data)
    model.layer16_adaln_weight.data.copy_(reference.layers[16].adaLN_modulation[0].weight.data)
    model.layer16_adaln_bias.data.copy_(reference.layers[16].adaLN_modulation[0].bias.data)
    model.layer17_attention_to_q_weight.data.copy_(reference.layers[17].attention.to_q.weight.data)
    model.layer17_attention_to_k_weight.data.copy_(reference.layers[17].attention.to_k.weight.data)
    model.layer17_attention_to_v_weight.data.copy_(reference.layers[17].attention.to_v.weight.data)
    model.layer17_attention_to_out_weight.data.copy_(reference.layers[17].attention.to_out[0].weight.data)
    model.layer17_norm_q_weight.data.copy_(reference.layers[17].attention.norm_q.weight.data)
    model.layer17_norm_k_weight.data.copy_(reference.layers[17].attention.norm_k.weight.data)
    model.layer17_feed_forward_w1_weight.data.copy_(reference.layers[17].feed_forward.w1.weight.data)
    model.layer17_feed_forward_w2_weight.data.copy_(reference.layers[17].feed_forward.w2.weight.data)
    model.layer17_feed_forward_w3_weight.data.copy_(reference.layers[17].feed_forward.w3.weight.data)
    model.layer17_attention_norm1_weight.data.copy_(reference.layers[17].attention_norm1.weight.data)
    model.layer17_ffn_norm1_weight.data.copy_(reference.layers[17].ffn_norm1.weight.data)
    model.layer17_attention_norm2_weight.data.copy_(reference.layers[17].attention_norm2.weight.data)
    model.layer17_ffn_norm2_weight.data.copy_(reference.layers[17].ffn_norm2.weight.data)
    model.layer17_adaln_weight.data.copy_(reference.layers[17].adaLN_modulation[0].weight.data)
    model.layer17_adaln_bias.data.copy_(reference.layers[17].adaLN_modulation[0].bias.data)
    model.layer18_attention_to_q_weight.data.copy_(reference.layers[18].attention.to_q.weight.data)
    model.layer18_attention_to_k_weight.data.copy_(reference.layers[18].attention.to_k.weight.data)
    model.layer18_attention_to_v_weight.data.copy_(reference.layers[18].attention.to_v.weight.data)
    model.layer18_attention_to_out_weight.data.copy_(reference.layers[18].attention.to_out[0].weight.data)
    model.layer18_norm_q_weight.data.copy_(reference.layers[18].attention.norm_q.weight.data)
    model.layer18_norm_k_weight.data.copy_(reference.layers[18].attention.norm_k.weight.data)
    model.layer18_feed_forward_w1_weight.data.copy_(reference.layers[18].feed_forward.w1.weight.data)
    model.layer18_feed_forward_w2_weight.data.copy_(reference.layers[18].feed_forward.w2.weight.data)
    model.layer18_feed_forward_w3_weight.data.copy_(reference.layers[18].feed_forward.w3.weight.data)
    model.layer18_attention_norm1_weight.data.copy_(reference.layers[18].attention_norm1.weight.data)
    model.layer18_ffn_norm1_weight.data.copy_(reference.layers[18].ffn_norm1.weight.data)
    model.layer18_attention_norm2_weight.data.copy_(reference.layers[18].attention_norm2.weight.data)
    model.layer18_ffn_norm2_weight.data.copy_(reference.layers[18].ffn_norm2.weight.data)
    model.layer18_adaln_weight.data.copy_(reference.layers[18].adaLN_modulation[0].weight.data)
    model.layer18_adaln_bias.data.copy_(reference.layers[18].adaLN_modulation[0].bias.data)
    model.layer19_attention_to_q_weight.data.copy_(reference.layers[19].attention.to_q.weight.data)
    model.layer19_attention_to_k_weight.data.copy_(reference.layers[19].attention.to_k.weight.data)
    model.layer19_attention_to_v_weight.data.copy_(reference.layers[19].attention.to_v.weight.data)
    model.layer19_attention_to_out_weight.data.copy_(reference.layers[19].attention.to_out[0].weight.data)
    model.layer19_norm_q_weight.data.copy_(reference.layers[19].attention.norm_q.weight.data)
    model.layer19_norm_k_weight.data.copy_(reference.layers[19].attention.norm_k.weight.data)
    model.layer19_feed_forward_w1_weight.data.copy_(reference.layers[19].feed_forward.w1.weight.data)
    model.layer19_feed_forward_w2_weight.data.copy_(reference.layers[19].feed_forward.w2.weight.data)
    model.layer19_feed_forward_w3_weight.data.copy_(reference.layers[19].feed_forward.w3.weight.data)
    model.layer19_attention_norm1_weight.data.copy_(reference.layers[19].attention_norm1.weight.data)
    model.layer19_ffn_norm1_weight.data.copy_(reference.layers[19].ffn_norm1.weight.data)
    model.layer19_attention_norm2_weight.data.copy_(reference.layers[19].attention_norm2.weight.data)
    model.layer19_ffn_norm2_weight.data.copy_(reference.layers[19].ffn_norm2.weight.data)
    model.layer19_adaln_weight.data.copy_(reference.layers[19].adaLN_modulation[0].weight.data)
    model.layer19_adaln_bias.data.copy_(reference.layers[19].adaLN_modulation[0].bias.data)
    model.layer20_attention_to_q_weight.data.copy_(reference.layers[20].attention.to_q.weight.data)
    model.layer20_attention_to_k_weight.data.copy_(reference.layers[20].attention.to_k.weight.data)
    model.layer20_attention_to_v_weight.data.copy_(reference.layers[20].attention.to_v.weight.data)
    model.layer20_attention_to_out_weight.data.copy_(reference.layers[20].attention.to_out[0].weight.data)
    model.layer20_norm_q_weight.data.copy_(reference.layers[20].attention.norm_q.weight.data)
    model.layer20_norm_k_weight.data.copy_(reference.layers[20].attention.norm_k.weight.data)
    model.layer20_feed_forward_w1_weight.data.copy_(reference.layers[20].feed_forward.w1.weight.data)
    model.layer20_feed_forward_w2_weight.data.copy_(reference.layers[20].feed_forward.w2.weight.data)
    model.layer20_feed_forward_w3_weight.data.copy_(reference.layers[20].feed_forward.w3.weight.data)
    model.layer20_attention_norm1_weight.data.copy_(reference.layers[20].attention_norm1.weight.data)
    model.layer20_ffn_norm1_weight.data.copy_(reference.layers[20].ffn_norm1.weight.data)
    model.layer20_attention_norm2_weight.data.copy_(reference.layers[20].attention_norm2.weight.data)
    model.layer20_ffn_norm2_weight.data.copy_(reference.layers[20].ffn_norm2.weight.data)
    model.layer20_adaln_weight.data.copy_(reference.layers[20].adaLN_modulation[0].weight.data)
    model.layer20_adaln_bias.data.copy_(reference.layers[20].adaLN_modulation[0].bias.data)
    model.layer21_attention_to_q_weight.data.copy_(reference.layers[21].attention.to_q.weight.data)
    model.layer21_attention_to_k_weight.data.copy_(reference.layers[21].attention.to_k.weight.data)
    model.layer21_attention_to_v_weight.data.copy_(reference.layers[21].attention.to_v.weight.data)
    model.layer21_attention_to_out_weight.data.copy_(reference.layers[21].attention.to_out[0].weight.data)
    model.layer21_norm_q_weight.data.copy_(reference.layers[21].attention.norm_q.weight.data)
    model.layer21_norm_k_weight.data.copy_(reference.layers[21].attention.norm_k.weight.data)
    model.layer21_feed_forward_w1_weight.data.copy_(reference.layers[21].feed_forward.w1.weight.data)
    model.layer21_feed_forward_w2_weight.data.copy_(reference.layers[21].feed_forward.w2.weight.data)
    model.layer21_feed_forward_w3_weight.data.copy_(reference.layers[21].feed_forward.w3.weight.data)
    model.layer21_attention_norm1_weight.data.copy_(reference.layers[21].attention_norm1.weight.data)
    model.layer21_ffn_norm1_weight.data.copy_(reference.layers[21].ffn_norm1.weight.data)
    model.layer21_attention_norm2_weight.data.copy_(reference.layers[21].attention_norm2.weight.data)
    model.layer21_ffn_norm2_weight.data.copy_(reference.layers[21].ffn_norm2.weight.data)
    model.layer21_adaln_weight.data.copy_(reference.layers[21].adaLN_modulation[0].weight.data)
    model.layer21_adaln_bias.data.copy_(reference.layers[21].adaLN_modulation[0].bias.data)
    model.layer22_attention_to_q_weight.data.copy_(reference.layers[22].attention.to_q.weight.data)
    model.layer22_attention_to_k_weight.data.copy_(reference.layers[22].attention.to_k.weight.data)
    model.layer22_attention_to_v_weight.data.copy_(reference.layers[22].attention.to_v.weight.data)
    model.layer22_attention_to_out_weight.data.copy_(reference.layers[22].attention.to_out[0].weight.data)
    model.layer22_norm_q_weight.data.copy_(reference.layers[22].attention.norm_q.weight.data)
    model.layer22_norm_k_weight.data.copy_(reference.layers[22].attention.norm_k.weight.data)
    model.layer22_feed_forward_w1_weight.data.copy_(reference.layers[22].feed_forward.w1.weight.data)
    model.layer22_feed_forward_w2_weight.data.copy_(reference.layers[22].feed_forward.w2.weight.data)
    model.layer22_feed_forward_w3_weight.data.copy_(reference.layers[22].feed_forward.w3.weight.data)
    model.layer22_attention_norm1_weight.data.copy_(reference.layers[22].attention_norm1.weight.data)
    model.layer22_ffn_norm1_weight.data.copy_(reference.layers[22].ffn_norm1.weight.data)
    model.layer22_attention_norm2_weight.data.copy_(reference.layers[22].attention_norm2.weight.data)
    model.layer22_ffn_norm2_weight.data.copy_(reference.layers[22].ffn_norm2.weight.data)
    model.layer22_adaln_weight.data.copy_(reference.layers[22].adaLN_modulation[0].weight.data)
    model.layer22_adaln_bias.data.copy_(reference.layers[22].adaLN_modulation[0].bias.data)
    model.layer23_attention_to_q_weight.data.copy_(reference.layers[23].attention.to_q.weight.data)
    model.layer23_attention_to_k_weight.data.copy_(reference.layers[23].attention.to_k.weight.data)
    model.layer23_attention_to_v_weight.data.copy_(reference.layers[23].attention.to_v.weight.data)
    model.layer23_attention_to_out_weight.data.copy_(reference.layers[23].attention.to_out[0].weight.data)
    model.layer23_norm_q_weight.data.copy_(reference.layers[23].attention.norm_q.weight.data)
    model.layer23_norm_k_weight.data.copy_(reference.layers[23].attention.norm_k.weight.data)
    model.layer23_feed_forward_w1_weight.data.copy_(reference.layers[23].feed_forward.w1.weight.data)
    model.layer23_feed_forward_w2_weight.data.copy_(reference.layers[23].feed_forward.w2.weight.data)
    model.layer23_feed_forward_w3_weight.data.copy_(reference.layers[23].feed_forward.w3.weight.data)
    model.layer23_attention_norm1_weight.data.copy_(reference.layers[23].attention_norm1.weight.data)
    model.layer23_ffn_norm1_weight.data.copy_(reference.layers[23].ffn_norm1.weight.data)
    model.layer23_attention_norm2_weight.data.copy_(reference.layers[23].attention_norm2.weight.data)
    model.layer23_ffn_norm2_weight.data.copy_(reference.layers[23].ffn_norm2.weight.data)
    model.layer23_adaln_weight.data.copy_(reference.layers[23].adaLN_modulation[0].weight.data)
    model.layer23_adaln_bias.data.copy_(reference.layers[23].adaLN_modulation[0].bias.data)
    model.layer24_attention_to_q_weight.data.copy_(reference.layers[24].attention.to_q.weight.data)
    model.layer24_attention_to_k_weight.data.copy_(reference.layers[24].attention.to_k.weight.data)
    model.layer24_attention_to_v_weight.data.copy_(reference.layers[24].attention.to_v.weight.data)
    model.layer24_attention_to_out_weight.data.copy_(reference.layers[24].attention.to_out[0].weight.data)
    model.layer24_norm_q_weight.data.copy_(reference.layers[24].attention.norm_q.weight.data)
    model.layer24_norm_k_weight.data.copy_(reference.layers[24].attention.norm_k.weight.data)
    model.layer24_feed_forward_w1_weight.data.copy_(reference.layers[24].feed_forward.w1.weight.data)
    model.layer24_feed_forward_w2_weight.data.copy_(reference.layers[24].feed_forward.w2.weight.data)
    model.layer24_feed_forward_w3_weight.data.copy_(reference.layers[24].feed_forward.w3.weight.data)
    model.layer24_attention_norm1_weight.data.copy_(reference.layers[24].attention_norm1.weight.data)
    model.layer24_ffn_norm1_weight.data.copy_(reference.layers[24].ffn_norm1.weight.data)
    model.layer24_attention_norm2_weight.data.copy_(reference.layers[24].attention_norm2.weight.data)
    model.layer24_ffn_norm2_weight.data.copy_(reference.layers[24].ffn_norm2.weight.data)
    model.layer24_adaln_weight.data.copy_(reference.layers[24].adaLN_modulation[0].weight.data)
    model.layer24_adaln_bias.data.copy_(reference.layers[24].adaLN_modulation[0].bias.data)
    model.layer25_attention_to_q_weight.data.copy_(reference.layers[25].attention.to_q.weight.data)
    model.layer25_attention_to_k_weight.data.copy_(reference.layers[25].attention.to_k.weight.data)
    model.layer25_attention_to_v_weight.data.copy_(reference.layers[25].attention.to_v.weight.data)
    model.layer25_attention_to_out_weight.data.copy_(reference.layers[25].attention.to_out[0].weight.data)
    model.layer25_norm_q_weight.data.copy_(reference.layers[25].attention.norm_q.weight.data)
    model.layer25_norm_k_weight.data.copy_(reference.layers[25].attention.norm_k.weight.data)
    model.layer25_feed_forward_w1_weight.data.copy_(reference.layers[25].feed_forward.w1.weight.data)
    model.layer25_feed_forward_w2_weight.data.copy_(reference.layers[25].feed_forward.w2.weight.data)
    model.layer25_feed_forward_w3_weight.data.copy_(reference.layers[25].feed_forward.w3.weight.data)
    model.layer25_attention_norm1_weight.data.copy_(reference.layers[25].attention_norm1.weight.data)
    model.layer25_ffn_norm1_weight.data.copy_(reference.layers[25].ffn_norm1.weight.data)
    model.layer25_attention_norm2_weight.data.copy_(reference.layers[25].attention_norm2.weight.data)
    model.layer25_ffn_norm2_weight.data.copy_(reference.layers[25].ffn_norm2.weight.data)
    model.layer25_adaln_weight.data.copy_(reference.layers[25].adaLN_modulation[0].weight.data)
    model.layer25_adaln_bias.data.copy_(reference.layers[25].adaLN_modulation[0].bias.data)
    model.layer26_attention_to_q_weight.data.copy_(reference.layers[26].attention.to_q.weight.data)
    model.layer26_attention_to_k_weight.data.copy_(reference.layers[26].attention.to_k.weight.data)
    model.layer26_attention_to_v_weight.data.copy_(reference.layers[26].attention.to_v.weight.data)
    model.layer26_attention_to_out_weight.data.copy_(reference.layers[26].attention.to_out[0].weight.data)
    model.layer26_norm_q_weight.data.copy_(reference.layers[26].attention.norm_q.weight.data)
    model.layer26_norm_k_weight.data.copy_(reference.layers[26].attention.norm_k.weight.data)
    model.layer26_feed_forward_w1_weight.data.copy_(reference.layers[26].feed_forward.w1.weight.data)
    model.layer26_feed_forward_w2_weight.data.copy_(reference.layers[26].feed_forward.w2.weight.data)
    model.layer26_feed_forward_w3_weight.data.copy_(reference.layers[26].feed_forward.w3.weight.data)
    model.layer26_attention_norm1_weight.data.copy_(reference.layers[26].attention_norm1.weight.data)
    model.layer26_ffn_norm1_weight.data.copy_(reference.layers[26].ffn_norm1.weight.data)
    model.layer26_attention_norm2_weight.data.copy_(reference.layers[26].attention_norm2.weight.data)
    model.layer26_ffn_norm2_weight.data.copy_(reference.layers[26].ffn_norm2.weight.data)
    model.layer26_adaln_weight.data.copy_(reference.layers[26].adaLN_modulation[0].weight.data)
    model.layer26_adaln_bias.data.copy_(reference.layers[26].adaLN_modulation[0].bias.data)
    model.layer27_attention_to_q_weight.data.copy_(reference.layers[27].attention.to_q.weight.data)
    model.layer27_attention_to_k_weight.data.copy_(reference.layers[27].attention.to_k.weight.data)
    model.layer27_attention_to_v_weight.data.copy_(reference.layers[27].attention.to_v.weight.data)
    model.layer27_attention_to_out_weight.data.copy_(reference.layers[27].attention.to_out[0].weight.data)
    model.layer27_norm_q_weight.data.copy_(reference.layers[27].attention.norm_q.weight.data)
    model.layer27_norm_k_weight.data.copy_(reference.layers[27].attention.norm_k.weight.data)
    model.layer27_feed_forward_w1_weight.data.copy_(reference.layers[27].feed_forward.w1.weight.data)
    model.layer27_feed_forward_w2_weight.data.copy_(reference.layers[27].feed_forward.w2.weight.data)
    model.layer27_feed_forward_w3_weight.data.copy_(reference.layers[27].feed_forward.w3.weight.data)
    model.layer27_attention_norm1_weight.data.copy_(reference.layers[27].attention_norm1.weight.data)
    model.layer27_ffn_norm1_weight.data.copy_(reference.layers[27].ffn_norm1.weight.data)
    model.layer27_attention_norm2_weight.data.copy_(reference.layers[27].attention_norm2.weight.data)
    model.layer27_ffn_norm2_weight.data.copy_(reference.layers[27].ffn_norm2.weight.data)
    model.layer27_adaln_weight.data.copy_(reference.layers[27].adaLN_modulation[0].weight.data)
    model.layer27_adaln_bias.data.copy_(reference.layers[27].adaLN_modulation[0].bias.data)
    model.layer28_attention_to_q_weight.data.copy_(reference.layers[28].attention.to_q.weight.data)
    model.layer28_attention_to_k_weight.data.copy_(reference.layers[28].attention.to_k.weight.data)
    model.layer28_attention_to_v_weight.data.copy_(reference.layers[28].attention.to_v.weight.data)
    model.layer28_attention_to_out_weight.data.copy_(reference.layers[28].attention.to_out[0].weight.data)
    model.layer28_norm_q_weight.data.copy_(reference.layers[28].attention.norm_q.weight.data)
    model.layer28_norm_k_weight.data.copy_(reference.layers[28].attention.norm_k.weight.data)
    model.layer28_feed_forward_w1_weight.data.copy_(reference.layers[28].feed_forward.w1.weight.data)
    model.layer28_feed_forward_w2_weight.data.copy_(reference.layers[28].feed_forward.w2.weight.data)
    model.layer28_feed_forward_w3_weight.data.copy_(reference.layers[28].feed_forward.w3.weight.data)
    model.layer28_attention_norm1_weight.data.copy_(reference.layers[28].attention_norm1.weight.data)
    model.layer28_ffn_norm1_weight.data.copy_(reference.layers[28].ffn_norm1.weight.data)
    model.layer28_attention_norm2_weight.data.copy_(reference.layers[28].attention_norm2.weight.data)
    model.layer28_ffn_norm2_weight.data.copy_(reference.layers[28].ffn_norm2.weight.data)
    model.layer28_adaln_weight.data.copy_(reference.layers[28].adaLN_modulation[0].weight.data)
    model.layer28_adaln_bias.data.copy_(reference.layers[28].adaLN_modulation[0].bias.data)
    model.layer29_attention_to_q_weight.data.copy_(reference.layers[29].attention.to_q.weight.data)
    model.layer29_attention_to_k_weight.data.copy_(reference.layers[29].attention.to_k.weight.data)
    model.layer29_attention_to_v_weight.data.copy_(reference.layers[29].attention.to_v.weight.data)
    model.layer29_attention_to_out_weight.data.copy_(reference.layers[29].attention.to_out[0].weight.data)
    model.layer29_norm_q_weight.data.copy_(reference.layers[29].attention.norm_q.weight.data)
    model.layer29_norm_k_weight.data.copy_(reference.layers[29].attention.norm_k.weight.data)
    model.layer29_feed_forward_w1_weight.data.copy_(reference.layers[29].feed_forward.w1.weight.data)
    model.layer29_feed_forward_w2_weight.data.copy_(reference.layers[29].feed_forward.w2.weight.data)
    model.layer29_feed_forward_w3_weight.data.copy_(reference.layers[29].feed_forward.w3.weight.data)
    model.layer29_attention_norm1_weight.data.copy_(reference.layers[29].attention_norm1.weight.data)
    model.layer29_ffn_norm1_weight.data.copy_(reference.layers[29].ffn_norm1.weight.data)
    model.layer29_attention_norm2_weight.data.copy_(reference.layers[29].attention_norm2.weight.data)
    model.layer29_ffn_norm2_weight.data.copy_(reference.layers[29].ffn_norm2.weight.data)
    model.layer29_adaln_weight.data.copy_(reference.layers[29].adaLN_modulation[0].weight.data)
    model.layer29_adaln_bias.data.copy_(reference.layers[29].adaLN_modulation[0].bias.data)

    model.eval()
    del reference
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model


def main() -> None:
    args = parse_args()
    if args.guidance > 1.0:
        raise ValueError("inference-sequential.py only supports guidance <= 1.0")

    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    start_time = time.time()

    transformer = load_transformer(model_dir / "transformer", device, dtype)
    vae = load_vae(model_dir / "vae", device)
    text_encoder = load_text_encoder(model_dir / "text_encoder", device, dtype)
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir if tokenizer_dir.exists() else model_dir / "text_encoder"),
        trust_remote_code=True,
    )

    scheduler_config = read_json(model_dir / "scheduler" / "scheduler_config.json")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=scheduler_config.get("num_train_timesteps", 1000),
        shift=scheduler_config.get("shift", 3.0),
        use_dynamic_shifting=scheduler_config.get("use_dynamic_shifting", False),
    )

    image = generate_image(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        max_length=args.max_length,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    elapsed = time.time() - start_time
    print(f"saved {output_path} in {elapsed:.2f}s on {device} with {dtype}")


if __name__ == "__main__":
    main()
