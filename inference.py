"""Standalone local text-to-image inference for Z-Image.

This file intentionally keeps the full runtime in one place:
- scheduler
- VAE
- transformer
- checkpoint loading
- prompt encoding
- sampling loop
- CLI
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard library imports
# -----------------------------------------------------------------------------

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# -----------------------------------------------------------------------------
# Third-party imports
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

# -----------------------------------------------------------------------------
# Model constants
# -----------------------------------------------------------------------------

ADALN_EMBED_DIM = 256
FREQUENCY_EMBEDDING_SIZE = 256
MAX_PERIOD = 10000
ROPE_THETA = 256.0
ROPE_AXES_DIMS = [32, 48, 48]
ROPE_AXES_LENS = [1536, 512, 512]
SEQ_MULTI_OF = 32

BASE_IMAGE_SEQ_LEN = 256
MAX_IMAGE_SEQ_LEN = 4096
BASE_SHIFT = 0.5
MAX_SHIFT = 1.15


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------

class FlowMatchEulerDiscreteScheduler:
    """Minimal Euler scheduler used by Z-Image."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        use_dynamic_shifting: bool = False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting

        base = torch.linspace(1, num_train_timesteps, num_train_timesteps, dtype=torch.float32).flip(0)
        sigmas = base / num_train_timesteps
        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = sigmas.cpu()
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()
        self._step_index: int | None = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device, mu: float | None = None) -> None:
        timesteps = torch.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1, dtype=torch.float32)[:-1]
        sigmas = timesteps.clone()

        if self.use_dynamic_shifting:
            mu_tensor = torch.tensor(mu, dtype=torch.float32)
            sigmas = torch.exp(mu_tensor) / (torch.exp(mu_tensor) + (1 / sigmas - 1))
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.timesteps = sigmas.to(device) * self.num_train_timesteps
        self.sigmas = torch.cat([sigmas.to(device), torch.zeros(1, device=device)])
        self._step_index = None

    def _index_for_timestep(self, timestep: torch.Tensor) -> int:
        matches = (self.timesteps == timestep).nonzero()
        return matches[1 if len(matches) > 1 else 0].item()

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        if self._step_index is None:
            self._step_index = self._index_for_timestep(timestep)

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        self._step_index += 1
        return (sample.float() + (sigma_next - sigma) * model_output.float()).to(model_output.dtype)


# -----------------------------------------------------------------------------
# VAE building blocks
# -----------------------------------------------------------------------------

@dataclass
class AutoencoderKLOutput:
    sample: torch.Tensor


class AutoencoderConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getattr__(self, name):
        return self.__dict__.get(name)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, groups=32, eps=1e-6):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = swish

        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, input_tensor, temb=None):
        hidden_states = self.norm1(input_tensor)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        return input_tensor + hidden_states


class VAEAttention(nn.Module):
    def __init__(self, in_channels, groups=32, eps=1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.to_q = nn.Linear(in_channels, in_channels)
        self.to_k = nn.Linear(in_channels, in_channels)
        self.to_v = nn.Linear(in_channels, in_channels)
        self.to_out = nn.ModuleList([nn.Linear(in_channels, in_channels)])

    def forward(self, hidden_states):
        batch, channels, height, width = hidden_states.shape
        residual = hidden_states
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channels, -1).transpose(1, 2)

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = hidden_states.transpose(1, 2).view(batch, channels, height, width)
        return residual + hidden_states


class Downsample2D(nn.Module):
    def __init__(self, channels, with_conv=True, out_channels=None, padding=1):
        super().__init__()
        out_channels = out_channels or channels
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=padding)

    def forward(self, hidden_states):
        if self.with_conv:
            return self.conv(hidden_states)
        return F.avg_pool2d(hidden_states, kernel_size=2, stride=2)


class Upsample2D(nn.Module):
    def __init__(self, channels, with_conv=True, out_channels=None):
        super().__init__()
        out_channels = out_channels or channels
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, resnet_eps=1e-6, resnet_groups=32, add_downsample=True):
        super().__init__()
        resnets = []
        for index in range(num_layers):
            in_c = in_channels if index == 0 else out_channels
            resnets.append(ResnetBlock2D(in_c, out_channels, eps=resnet_eps, groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [Downsample2D(out_channels, with_conv=True, out_channels=out_channels, padding=0)]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = F.pad(hidden_states, (0, 1, 0, 1), mode="constant", value=0)
                hidden_states = downsampler(hidden_states)

        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, resnet_eps=1e-6, resnet_groups=32, add_upsample=True):
        super().__init__()
        resnets = []
        for index in range(num_layers):
            in_c = in_channels if index == 0 else out_channels
            resnets.append(ResnetBlock2D(in_c, out_channels, eps=resnet_eps, groups=resnet_groups))
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, with_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(self, in_channels, resnet_eps=1e-6, resnet_groups=32):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, in_channels, eps=resnet_eps, groups=resnet_groups),
                ResnetBlock2D(in_channels, in_channels, eps=resnet_eps, groups=resnet_groups),
            ]
        )
        self.attentions = nn.ModuleList([VAEAttention(in_channels, groups=resnet_groups, eps=resnet_eps)])

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        for attn in self.attentions:
            hidden_states = attn(hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        double_z=True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for index, block_out_channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            is_final_block = index == len(block_out_channels) - 1
            block = DownEncoderBlock2D(
                input_channel,
                output_channel,
                num_layers=layers_per_block,
                resnet_groups=norm_num_groups,
                add_downsample=not is_final_block,
            )
            self.down_blocks.append(block)

        self.mid_block = UNetMidBlock2D(block_out_channels[-1], resnet_groups=norm_num_groups)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = UNetMidBlock2D(block_out_channels[-1], resnet_groups=norm_num_groups)
        self.up_blocks = nn.ModuleList([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for index, block_out_channel in enumerate(reversed_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            is_final_block = index == len(block_out_channels) - 1
            block = UpDecoderBlock2D(
                input_channel,
                output_channel,
                num_layers=layers_per_block + 1,
                resnet_groups=norm_num_groups,
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(block)

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        return x


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: Optional[float] = None,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.config = AutoencoderConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            latent_channels=latent_channels,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)
        if not return_dict:
            return (dec,)
        return AutoencoderKLOutput(sample=dec)


# -----------------------------------------------------------------------------
# Transformer building blocks
# -----------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=FREQUENCY_EMBEDDING_SIZE):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=MAX_PERIOD):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
            )
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states).unflatten(-1, (self.n_heads, -1))
        key = self.to_k(hidden_states).unflatten(-1, (self.n_kv_heads, -1))
        value = self.to_v(hidden_states).unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)
        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        return self.to_out[0](hidden_states)


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation: bool = True,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa = gate_msa.tanh()
            gate_mlp = gate_mlp.tanh()
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp

            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
            return x

        attn_out = self.attention(
            self.attention_norm1(x),
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,
        )
        x = x + self.attention_norm2(attn_out)
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        return self.linear(x)


class RopeEmbedder:
    def __init__(
        self,
        theta: float = ROPE_THETA,
        axes_dims: List[int] = ROPE_AXES_DIMS,
        axes_lens: List[int] = ROPE_AXES_LENS,
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = ROPE_THETA):
        with torch.device("cpu"):
            result = []
            for axis_dim, axis_len in zip(dim, end):
                freqs = 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float64, device="cpu") / axis_dim))
                timestep = torch.arange(axis_len, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                result.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
            return result

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
        if self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs.to(device) for freqs in self.freqs_cis]

        result = []
        for index in range(len(self.axes_dims)):
            result.append(self.freqs_cis[index][ids[:, index]])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(nn.Module):
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            embed_dim = f_patch_size * patch_size * patch_size * in_channels
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = nn.Linear(embed_dim, dim, bias=True)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = FinalLayer(
                dim,
                patch_size * patch_size * f_patch_size * self.out_channels,
            )

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True)
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False)
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )
        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        assert dim // n_heads == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        p_h = p_w = patch_size
        p_f = f_patch_size
        for index, sample in enumerate(x):
            frames, height, width = size[index]
            original_len = (frames // p_f) * (height // p_h) * (width // p_w)
            x[index] = (
                sample[:original_len]
                .view(frames // p_f, height // p_h, width // p_w, p_f, p_h, p_w, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, frames, height, width)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        p_h = p_w = patch_size
        p_f = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for image, cap_feat in zip(all_image, all_cap_feats):
            cap_len = len(cap_feat)
            cap_pad = (-cap_len) % SEQ_MULTI_OF
            cap_pos = self.create_coordinate_grid(
                size=(cap_len + cap_pad, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_pos)
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_pad,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
                if cap_pad > 0
                else torch.zeros((cap_len,), dtype=torch.bool, device=device)
            )
            all_cap_feats_out.append(
                torch.cat([cap_feat, cap_feat[-1:].repeat(cap_pad, 1)], dim=0) if cap_pad > 0 else cap_feat
            )

            channels, frames, height, width = image.size()
            all_image_size.append((frames, height, width))
            frame_tokens = frames // p_f
            height_tokens = height // p_h
            width_tokens = width // p_w

            image = image.view(channels, frame_tokens, p_f, height_tokens, p_h, width_tokens, p_w)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(
                frame_tokens * height_tokens * width_tokens,
                p_f * p_h * p_w * channels,
            )

            image_len = len(image)
            image_pad = (-image_len) % SEQ_MULTI_OF
            image_pos = self.create_coordinate_grid(
                size=(frame_tokens, height_tokens, width_tokens),
                start=(cap_len + cap_pad + 1, 0, 0),
                device=device,
            ).flatten(0, 2)

            if image_pad > 0:
                image_pos = torch.cat(
                    [
                        image_pos,
                        self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                        .flatten(0, 2)
                        .repeat(image_pad, 1),
                    ],
                    dim=0,
                )
            all_image_pos_ids.append(image_pos)

            image_mask = torch.cat(
                [
                    torch.zeros((image_len,), dtype=torch.bool, device=device),
                    torch.ones((image_pad,), dtype=torch.bool, device=device),
                ],
                dim=0,
            )
            all_image_pad_mask.append(
                image_mask if image_pad > 0 else torch.zeros((image_len,), dtype=torch.bool, device=device)
            )

            all_image_out.append(torch.cat([image, image[-1:].repeat(image_pad, 1)], dim=0) if image_pad > 0 else image)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        bsz = len(x)
        device = x[0].device
        t = self.t_embedder(t * self.t_scale)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        x_item_seqlens = [len(item) for item in x]
        x_max_item_seqlen = max(x_item_seqlens)
        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(item) for item in x_pos_ids], dim=0))
        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)[:, : x.shape[1]]

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for index, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[index, :seq_len] = 1

        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

        cap_item_seqlens = [len(item) for item in cap_feats]
        cap_max_item_seqlen = max(cap_item_seqlens)
        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(
            self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(item) for item in cap_pos_ids], dim=0)
        )
        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)[:, : cap_feats.shape[1]]

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for index, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[index, :seq_len] = 1

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

        unified = []
        unified_freqs_cis = []
        for index in range(bsz):
            x_len = x_item_seqlens[index]
            cap_len = cap_item_seqlens[index]
            unified.append(torch.cat([x[index][:x_len], cap_feats[index][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[index][:x_len], cap_freqs_cis[index][:cap_len]]))

        unified_item_seqlens = [cap_len + x_len for cap_len, x_len in zip(cap_item_seqlens, x_item_seqlens)]
        unified_max_item_seqlen = max(unified_item_seqlens)
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for index, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[index, :seq_len] = 1

        for layer in self.layers:
            unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        return self.unpatchify(unified, x_size, patch_size, f_patch_size), {}


# -----------------------------------------------------------------------------
# Checkpoint loading helpers
# -----------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_safetensors_dir(path: Path, device: str = "cpu", dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    index_files = list(path.glob("*.safetensors.index.json"))
    if index_files:
        index = read_json(index_files[0])
        shard_names = sorted(set(index["weight_map"].values()))
        state_dict: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            state_dict.update(load_file(str(path / shard_name), device=device))
    else:
        shard_files = sorted(path.glob("*.safetensors"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")
        state_dict = load_file(str(shard_files[0]), device=device)

    if dtype is None:
        return state_dict
    return {key: value.to(dtype) if value.is_floating_point() else value for key, value in state_dict.items()}


def load_transformer(path: Path, device: torch.device, dtype: torch.dtype) -> ZImageTransformer2DModel:
    config = read_json(path / "config.json")
    with torch.device("meta"):
        model = ZImageTransformer2DModel(
            all_patch_size=tuple(config.get("all_patch_size", (2,))),
            all_f_patch_size=tuple(config.get("all_f_patch_size", (1,))),
            in_channels=config.get("in_channels", 16),
            dim=config.get("dim", 3840),
            n_layers=config.get("n_layers", 30),
            n_refiner_layers=config.get("n_refiner_layers", 2),
            n_heads=config.get("n_heads", 30),
            n_kv_heads=config.get("n_kv_heads", 30),
            norm_eps=config.get("norm_eps", 1e-5),
            qk_norm=config.get("qk_norm", True),
            cap_feat_dim=config.get("cap_feat_dim", 2560),
            rope_theta=config.get("rope_theta", ROPE_THETA),
            t_scale=config.get("t_scale", 1000.0),
            axes_dims=config.get("axes_dims", ROPE_AXES_DIMS),
            axes_lens=config.get("axes_lens", ROPE_AXES_LENS),
        ).to(dtype)

    model.load_state_dict(load_safetensors_dir(path, dtype=dtype), strict=False, assign=True)
    return model.to(device).eval()


def load_vae(path: Path, device: torch.device) -> AutoencoderKL:
    config = read_json(path / "config.json")
    model = AutoencoderKL(
        in_channels=config.get("in_channels", 3),
        out_channels=config.get("out_channels", 3),
        down_block_types=tuple(config.get("down_block_types", ("DownEncoderBlock2D",))),
        up_block_types=tuple(config.get("up_block_types", ("UpDecoderBlock2D",))),
        block_out_channels=tuple(config.get("block_out_channels", (64,))),
        layers_per_block=config.get("layers_per_block", 1),
        latent_channels=config.get("latent_channels", 16),
        norm_num_groups=config.get("norm_num_groups", 32),
        scaling_factor=config.get("scaling_factor", 0.3611),
        shift_factor=config.get("shift_factor"),
        use_quant_conv=config.get("use_quant_conv", True),
        use_post_quant_conv=config.get("use_post_quant_conv", True),
        mid_block_add_attention=config.get("mid_block_add_attention", True),
    )
    model.load_state_dict(load_safetensors_dir(path), strict=False)
    return model.to(device=device, dtype=torch.float32).eval()


def load_text_encoder(path: Path, device: torch.device, dtype: torch.dtype) -> AutoModel:
    kwargs = {"trust_remote_code": True}
    try:
        model = AutoModel.from_pretrained(str(path), dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(str(path), torch_dtype=dtype, **kwargs)
    return model.to(device).eval()


# -----------------------------------------------------------------------------
# Prompt encoding helpers
# -----------------------------------------------------------------------------

def format_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    return prompt


@torch.no_grad()
def encode_prompt(
    prompt: str,
    tokenizer: AutoTokenizer,
    text_encoder: AutoModel,
    device: torch.device,
    max_length: int,
) -> list[torch.Tensor]:
    tokens = tokenizer(
        [format_prompt(tokenizer, prompt)],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device).bool()
    hidden = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-2][0]
    return [hidden[attention_mask[0]]]


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------

def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + (base_shift - slope * base_seq_len)


@torch.no_grad()
def generate_image(
    transformer: ZImageTransformer2DModel,
    vae: AutoencoderKL,
    text_encoder: AutoModel,
    tokenizer: AutoTokenizer,
    scheduler: FlowMatchEulerDiscreteScheduler,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    seed: int,
    max_length: int,
) -> Image.Image:
    # The VAE downsamples by 8 and the transformer operates on a 2x latent grid.
    vae_scale = 2 ** len(vae.config.block_out_channels)
    if height % vae_scale != 0 or width % vae_scale != 0:
        raise ValueError(f"Height and width must be divisible by {vae_scale}")

    device = next(transformer.parameters()).device
    transformer_dtype = next(transformer.parameters()).dtype
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    latent_height = 2 * (height // vae_scale)
    latent_width = 2 * (width // vae_scale)
    latents = torch.randn((1, transformer.in_channels, latent_height, latent_width), device=device, dtype=torch.float32)

    prompt_embeds = [embed.to(transformer_dtype) for embed in encode_prompt(prompt, tokenizer, text_encoder, device, max_length)]
    use_cfg = guidance > 1.0
    negative_embeds = (
        [embed.to(transformer_dtype) for embed in encode_prompt(negative_prompt, tokenizer, text_encoder, device, max_length)]
        if use_cfg
        else None
    )

    image_seq_len = (latent_height // 2) * (latent_width // 2)
    scheduler.set_timesteps(steps, device=device, mu=calculate_shift(image_seq_len))

    for index, timestep in enumerate(scheduler.timesteps):
        if index == len(scheduler.timesteps) - 1 and float(timestep.detach().cpu()) == 0.0:
            continue

        timestep_input = (1000 - timestep.expand(latents.shape[0])) / 1000
        if use_cfg:
            model_latents = latents.repeat(2, 1, 1, 1).to(transformer_dtype)
            model_timestep = timestep_input.repeat(2)
            model_embeds = prompt_embeds + negative_embeds
        else:
            model_latents = latents.to(transformer_dtype)
            model_timestep = timestep_input
            model_embeds = prompt_embeds

        outputs = transformer(list(model_latents.unsqueeze(2).unbind(0)), model_timestep, model_embeds)[0]
        if use_cfg:
            positive = outputs[0].float()
            negative = outputs[1].float()
            noise_pred = -(positive + guidance * (positive - negative)).squeeze(1).unsqueeze(0)
        else:
            noise_pred = -outputs[0].float().squeeze(1).unsqueeze(0)

        latents = scheduler.step(noise_pred, timestep, latents)

    shift = vae.config.shift_factor or 0.0
    image = vae.decode((latents.to(vae.dtype) / vae.config.scaling_factor) + shift, return_dict=False)[0]
    image = ((image[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().float().numpy() * 255).round().astype("uint8")
    return Image.fromarray(image)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Text prompt")
    parser.add_argument("--model", default="ckpts/Z-Image-Turbo", help="Local model directory")
    parser.add_argument("--output", default="output.png", help="Output image path")
    parser.add_argument("--negative-prompt", default="", help="Negative prompt for CFG models")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    start_time = time.time()

    # Load all components from the local checkpoint layout.
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
