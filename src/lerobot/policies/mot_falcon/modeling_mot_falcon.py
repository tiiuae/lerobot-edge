#!/usr/bin/env python

# Copyright 2025 TII and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# MOT-Falcon (Mixture of Transformers – Falcon3-VLM backbone)
#
# Mirrors mot0/modeling_mot0.py but replaces:
#   PaliGemma (SigLIP + Gemma LM)  →  Falcon3-VLM (ViT + Falcon3/Llama LM)
#   Gemma action expert             →  LlamaForCausalLM with Falcon3 config
# =============================================================================

import builtins
import copy
import gc
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModelForImageTextToText, LlamaConfig, LlamaForCausalLM
else:
    AutoModelForImageTextToText = None
    LlamaConfig = None
    LlamaForCausalLM = None

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

from .configuration_mot_falcon import DEFAULT_IMAGE_SIZE, MOTFalconConfig
from ..pretrained import PreTrainedPolicy, T
from ..rtc.modeling_rtc import RTCProcessor

from ..mot0.modeling_mot0 import (
    get_safe_dtype,
    create_sinusoidal_pos_embedding,
    sample_beta,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad_torch,
)
from ..pi_gemma import PiGemmaRMSNorm, _gated_residual

# =============================================================================
FALCON3_PATCH_SIZE = 14
FALCON3_MERGE_SIZE = 2
FALCON3_TEMPORAL_PS = 2

# ImageNet normalisation (from preprocessor_config.json)
FALCON3_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FALCON3_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


# =============================================================================
# RoPE helpers
# =============================================================================

def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the last dimension of x by half."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """Apply rotary position embeddings.

    Args:
        q, k : [B, num_heads, seq, head_dim]
        cos, sin : [B, seq, head_dim]  (will be unsqueezed over heads)
    """
    cos = cos.unsqueeze(1)  # [B, 1, seq, head_dim]
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


def _get_rope_cos_sin(
    position_ids: Tensor,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE cos/sin for the given position_ids.

    Args:
        position_ids : [B, seq]
        head_dim, rope_theta : from the loaded VLM's own config (see
            FalconVLMWithExpertModel.__init__) -- deliberately no default, so
            call sites can't silently fall back to a stale hardcoded size.
    Returns:
        cos, sin : [B, seq, head_dim]
    """
    half = head_dim // 2
    device = position_ids.device
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    # [B, seq, half]
    freqs = position_ids.float().unsqueeze(-1) * inv_freq[None, None, :]
    emb = torch.cat([freqs, freqs], dim=-1)  # [B, seq, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


# =============================================================================
# AdaRMS dispatch helper
#
# The action expert's norms are replaced with PiGemmaRMSNorm (see
# FalconVLMWithExpertModel._install_adarms_expert_norms) so the diffusion
# timestep re-conditions every expert block via a learned, zero-init gate --
# mirroring how mot0/pi_gemma.py treats its Gemma-based action expert. The VLM
# stream keeps its stock, pretrained LlamaRMSNorm untouched and is never
# time-conditioned (mirrors mot0's `use_adarms=[False, True]`).
# =============================================================================

def _apply_adarms_norm(
    norm_module: nn.Module, x: Tensor, cond: Tensor | None
) -> tuple[Tensor, Tensor | None]:
    """Apply a norm, returning (normed, gate).

    `gate` is a real, learned tensor for `PiGemmaRMSNorm` (expert stream) and
    `None` for a stock `LlamaRMSNorm` (VLM stream), in which case
    `_gated_residual` degenerates to an ordinary residual add.
    """
    if isinstance(norm_module, PiGemmaRMSNorm):
        return norm_module(x, cond=cond)
    return norm_module(x).to(x.dtype), None


# =============================================================================
# Layer-wise Mixture-of-Transformers mixing
# =============================================================================

def compute_falcon_layer(
    layer_idx: int,
    inputs_embeds: list[Tensor],
    attention_mask: Tensor,
    position_ids: Tensor,
    adarms_cond: list[Tensor | None],  # [vlm_cond (always None), expert_cond]
    falcon_vlm,
    falcon_expert,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
) -> tuple[Tensor, Tensor]:
    """Run one transformer layer for both streams with shared attention (MoT).

    Both VLM and expert Q/K/V are concatenated along the sequence dimension so
    that prefix and suffix tokens attend to each other fully.  The combined
    output is then split and each stream gets its own output projection + MLP.

    Args:
        layer_idx      : which decoder layer to process
        inputs_embeds  : [vlm_hidden [B,Sv,H], expert_hidden [B,Se,H]]
        attention_mask : [B, 1, Sv+Se, Sv+Se]  (additive, 0=attend, large-neg=mask)
        position_ids   : [B, Sv+Se]
        falcon_vlm     : Falcon3VLForConditionalGeneration (loaded with trust_remote_code)
        falcon_expert  : LlamaForCausalLM (action expert)
        num_heads, num_kv_heads, head_dim, rope_theta : from the loaded VLM's
            own config (see FalconVLMWithExpertModel.__init__); the action
            expert is built to share these same values (see the comment in
            FalconVLMWithExpertModel.__init__ on why).

    Returns:
        (vlm_out [B,Sv,H], expert_out [B,Se,H])
    """
    vlm_layer = falcon_vlm.model.layers[layer_idx]
    exp_layer = falcon_expert.model.layers[layer_idx]

    vlm_h = inputs_embeds[0]
    exp_h = inputs_embeds[1]
    B, Sv, _ = vlm_h.shape
    Se = exp_h.shape[1]

    # ---- input layer norms (expert stream conditioned on the diffusion timestep) ----
    vlm_n, vlm_gate1 = _apply_adarms_norm(vlm_layer.input_layernorm, vlm_h, adarms_cond[0])
    exp_n, exp_gate1 = _apply_adarms_norm(exp_layer.input_layernorm, exp_h, adarms_cond[1])

    # ---- QKV projections ----
    q_v = vlm_layer.self_attn.q_proj(vlm_n).view(B, Sv, num_heads, head_dim).transpose(1, 2)
    k_v = vlm_layer.self_attn.k_proj(vlm_n).view(B, Sv, num_kv_heads, head_dim).transpose(1, 2)
    v_v = vlm_layer.self_attn.v_proj(vlm_n).view(B, Sv, num_kv_heads, head_dim).transpose(1, 2)

    q_e = exp_layer.self_attn.q_proj(exp_n).view(B, Se, num_heads, head_dim).transpose(1, 2)
    k_e = exp_layer.self_attn.k_proj(exp_n).view(B, Se, num_kv_heads, head_dim).transpose(1, 2)
    v_e = exp_layer.self_attn.v_proj(exp_n).view(B, Se, num_kv_heads, head_dim).transpose(1, 2)

    # ---- concatenate along sequence dim ----
    q = torch.cat([q_v, q_e], dim=2)  # [B, num_heads, Sv+Se, head_dim]
    k = torch.cat([k_v, k_e], dim=2)  # [B, num_kv_heads, Sv+Se, head_dim]
    v = torch.cat([v_v, v_e], dim=2)

    # ---- RoPE ----
    cos, sin = _get_rope_cos_sin(position_ids, head_dim=head_dim, rope_theta=rope_theta, dtype=q.dtype)
    q, k = _apply_rotary_emb(q, k, cos, sin)

    # ---- GQA expansion: num_kv_heads → num_heads ----
    n_rep = num_heads // num_kv_heads
    k = k.repeat_interleave(n_rep, dim=1)  # [B, num_heads, Sv+Se, head_dim]
    v = v.repeat_interleave(n_rep, dim=1)

    # ---- scaled dot-product attention ----
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    # [B, num_heads, Sv+Se, head_dim]

    # ---- split, project, MLP for each stream ----
    def _stream_post_attn(layer, residual, attn_slice, seq_len, attn_gate, cond):
        out = attn_slice.transpose(1, 2).reshape(B, seq_len, num_heads * head_dim)
        out = layer.self_attn.o_proj(out.to(layer.self_attn.o_proj.weight.dtype))
        h = _gated_residual(residual, out.to(residual.dtype), attn_gate)
        res2 = h
        h, mlp_gate = _apply_adarms_norm(layer.post_attention_layernorm, h, cond)
        w_dtype = layer.mlp.gate_proj.weight.dtype
        gate_proj_out = layer.mlp.gate_proj(h.to(w_dtype))
        up = layer.mlp.up_proj(h.to(w_dtype))
        h = layer.mlp.down_proj(F.silu(gate_proj_out) * up)
        return _gated_residual(res2, h.to(res2.dtype), mlp_gate)

    vlm_out = _stream_post_attn(vlm_layer, vlm_h, attn_out[:, :, :Sv, :], Sv, vlm_gate1, adarms_cond[0])
    exp_out = _stream_post_attn(exp_layer, exp_h, attn_out[:, :, Sv:, :], Se, exp_gate1, adarms_cond[1])

    return vlm_out, exp_out


# =============================================================================
# FalconVLMWithExpertModel
# =============================================================================

class FalconVLMWithExpertModel(nn.Module):
    """Falcon3-VLM + Llama-based action expert, wired for MoT training."""

    def __init__(
        self,
        falcon_vlm_model_id: str,
        precision: Literal["bfloat16", "float32"] = "float32",
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        embedding_scaling: bool = True,
        action_expert_hidden_size: int = 1024,
        action_expert_intermediate_size: int = 4096,
    ):
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.embedding_scaling = embedding_scaling
        self.expert_hidden_size = action_expert_hidden_size

        # ---- Load Falcon3-VLM ----
        from transformers import AutoModelForImageTextToText

        self.falcon_vlm = AutoModelForImageTextToText.from_pretrained(
            falcon_vlm_model_id,
            torch_dtype=torch.float32,
        )

        vlm_cfg = self.falcon_vlm.config
        self.vlm_hidden_size = vlm_cfg.hidden_size
        self.num_layers = vlm_cfg.num_hidden_layers
        self.num_heads = vlm_cfg.num_attention_heads
        self.num_kv_heads = vlm_cfg.num_key_value_heads
        self.head_dim = vlm_cfg.head_dim
        self.vocab_size = vlm_cfg.vocab_size
        self.rope_theta = vlm_cfg.rope_theta
        self.rms_norm_eps = vlm_cfg.rms_norm_eps
        self.max_position_embeddings = vlm_cfg.max_position_embeddings

        # ---- Initialise action expert as LlamaForCausalLM with Falcon3 LLM config ----
        from transformers import LlamaConfig, LlamaForCausalLM

        # The expert's residual-stream width (hidden_size) is independent of its
        # attention head geometry: num_attention_heads/num_key_value_heads/head_dim
        # MUST stay pinned to the VLM's own values because compute_falcon_layer
        # / _llm_layer_forward concatenate the two streams' Q/K/V tensors directly
        # (torch.cat requires identical num_heads and head_dim), and num_hidden_layers
        # MUST stay pinned because the MoT mixing loop pairs VLM-layer-i with
        # expert-layer-i one-to-one. Only hidden_size/intermediate_size are free to
        # shrink -- mirrors mot0 pairing a `gemma_300m` expert with a `gemma_2b` VLM,
        # where both presets share the same num_heads/head_dim but differ in width.
        #
        # transformers' LlamaConfig validator requires hidden_size % num_attention_heads
        # == 0 even with an explicit head_dim override, so we construct with a
        # validator-satisfying placeholder and patch afterward; LlamaForCausalLM reads
        # cfg.hidden_size when sizing q/k/v/o_proj and the MLP, so the patched value
        # (not the placeholder) is what the model actually gets built with.
        expert_cfg = LlamaConfig(
            hidden_size=self.num_heads * self.head_dim,
            intermediate_size=action_expert_intermediate_size,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            hidden_act="silu",
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            rope_theta=self.rope_theta,
            attention_bias=False,
            mlp_bias=False,
        )
        expert_cfg.hidden_size = action_expert_hidden_size
        self.falcon_expert = LlamaForCausalLM(expert_cfg)
        # We always supply inputs_embeds so embed_tokens is never called.
        self.falcon_expert.model.embed_tokens = None
        # The expert has no language modeling objective; free the lm_head memory.
        del self.falcon_expert.lm_head
        self.falcon_expert.lm_head = None

        self._install_adarms_expert_norms(expert_cfg)

        self._set_precision(precision)
        self._set_requires_grad()

    def _install_adarms_expert_norms(self, expert_cfg) -> None:
        """Swap the action expert's (freshly-initialised) LlamaRMSNorm layers for
        AdaRMS norms so the diffusion timestep re-conditions every expert block via
        a learned, zero-init gate -- mirroring how mot0/pi_gemma.py treats its
        Gemma-based action expert. The VLM stream keeps its stock, pretrained
        LlamaRMSNorm untouched (mirrors mot0's `use_adarms=[False, True]`).
        """
        cond_dim = expert_cfg.hidden_size
        for layer in self.falcon_expert.model.layers:
            layer.input_layernorm = PiGemmaRMSNorm(
                expert_cfg.hidden_size, eps=expert_cfg.rms_norm_eps, cond_dim=cond_dim
            )
            layer.post_attention_layernorm = PiGemmaRMSNorm(
                expert_cfg.hidden_size, eps=expert_cfg.rms_norm_eps, cond_dim=cond_dim
            )
        self.falcon_expert.model.norm = PiGemmaRMSNorm(
            expert_cfg.hidden_size, eps=expert_cfg.rms_norm_eps, cond_dim=cond_dim
        )

    # ------------------------------------------------------------------
    # Precision / grad helpers
    # ------------------------------------------------------------------
    def _set_precision(self, precision: str):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
            # Keep vision encoder and layer norms in float32 for stability.
            float32_selectors = [
                "falcon_vlm.visual",
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            ]
            for name, param in self.named_parameters():
                if any(s in name for s in float32_selectors):
                    param.data = param.data.to(torch.float32)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.falcon_vlm.visual.eval()
            for p in self.falcon_vlm.visual.parameters():
                p.requires_grad = False
        if self.train_expert_only:
            self.falcon_vlm.eval()
            for p in self.falcon_vlm.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.falcon_vlm.visual.eval()
        if self.train_expert_only:
            self.falcon_vlm.eval()

    # ------------------------------------------------------------------
    # Image embedding
    # ------------------------------------------------------------------
    def embed_image(self, image: Tensor) -> Tensor:
        """Convert a batch of images to VLM token embeddings.

        Args:
            image : [B, C, H, W] in [0, 1] range (NOT ImageNet-normalised)
        Returns:
            [B, num_tokens, vlm_hidden_size]  (64 tokens per 224×224 image)
        """
        B, C, H, W = image.shape
        device = image.device

        # 1. ImageNet normalisation
        mean = torch.tensor(FALCON3_IMAGE_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(FALCON3_IMAGE_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        img_norm = (image.float() - mean) / std  # [B, C, H, W]

        ps = FALCON3_PATCH_SIZE   # 14
        tps = FALCON3_TEMPORAL_PS  # 2
        ms = FALCON3_MERGE_SIZE    # 2

        grid_h = H // ps  # 16 for 224×224
        grid_w = W // ps  # 16 for 224×224
        grid_t = 1        # static image → 1 temporal group

        # 2. Duplicate along temporal dim and extract patches.
        #    Mirrors Falcon3VLImageProcessor._preprocess:
        #      patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        #    on shape [grid_t, tps, C, gh//ms, ms, ps, gw//ms, ms, ps]
        img_t = img_norm.unsqueeze(1).repeat(1, tps, 1, 1, 1)
        # img_t: [B, tps=2, C=3, H=224, W=224]

        patches = img_t.reshape(
            B, grid_t, tps, C,
            grid_h // ms, ms, ps,
            grid_w // ms, ms, ps,
        )
        # [B, 1, 2, 3, 8, 2, 14, 8, 2, 14]

        # Permute: numpy axes (0,3,6,4,7,2,1,5,8) + leading B → (0,1,4,7,5,8,3,2,6,9)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).contiguous()
        # [B, 1, 8, 8, 2, 2, 3, 2, 14, 14]

        patches = patches.reshape(B, grid_t * grid_h * grid_w, C * tps * ps * ps)
        # [B, 256, 1176]

        # 3. Flatten for the visual encoder: [B*256, 1176]
        pixel_values = patches.reshape(B * grid_t * grid_h * grid_w, C * tps * ps * ps)

        # 4. grid_thw: [B, 3]
        grid_thw = torch.tensor(
            [[grid_t, grid_h, grid_w]] * B,
            dtype=torch.long,
            device=device,
        )

        # 5. Cast to visual encoder dtype and call it.
        vis_dtype = next(self.falcon_vlm.visual.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vis_dtype)

        # visual encoder: [B*256, 1176] → [B*out_tokens, 3072]
        image_embeds = self.falcon_vlm.visual(pixel_values, grid_thw, patch_size=ps)

        # 6. Reshape to [B, out_tokens, 3072]
        out_tokens = image_embeds.shape[0] // B
        image_embeds = image_embeds.view(B, out_tokens, self.vlm_hidden_size)

        if self.embedding_scaling:
            image_embeds = image_embeds * math.sqrt(self.vlm_hidden_size)

        return image_embeds

    # ------------------------------------------------------------------
    # Language token embedding
    # ------------------------------------------------------------------
    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        """Embed token IDs using Falcon3-VLM's shared embedding table.

        Args:
            tokens : [B, seq] int64
        Returns:
            [B, seq, vlm_hidden_size]
        """
        lang_emb = self.falcon_vlm.model.embed_tokens(tokens)
        if self.embedding_scaling:
            lang_emb = lang_emb * math.sqrt(self.vlm_hidden_size)
        return lang_emb

    # ------------------------------------------------------------------
    # Single-layer helper (used for Mode A and Mode B)
    # ------------------------------------------------------------------
    def _llm_layer_forward(
        self,
        layer,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
        adarms_cond: Tensor | None = None,
    ) -> tuple[Tensor, tuple | None]:
        """Forward one LLM decoder layer with optional KV-cache read/write.

        `adarms_cond` only has an effect when `layer`'s norms are `PiGemmaRMSNorm`
        (the action expert); for the VLM's stock `LlamaRMSNorm` it is ignored.

        Returns (hidden_states, new_kv) where new_kv is None if use_cache=False.
        """
        B, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # --- pre-norm (AdaRMS-conditioned for the expert) ---
        hidden_states, attn_gate = _apply_adarms_norm(layer.input_layernorm, hidden_states, adarms_cond)

        # --- QKV ---
        q = layer.self_attn.q_proj(hidden_states).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(hidden_states).view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(hidden_states).view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # --- RoPE ---
        cos, sin = _get_rope_cos_sin(
            position_ids, head_dim=self.head_dim, rope_theta=self.rope_theta, dtype=hidden_states.dtype
        )
        q, k = _apply_rotary_emb(q, k, cos, sin)

        # --- append prefix KV from cache (Mode B) ---
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k.detach(), v.detach()) if use_cache else None

        # --- GQA expansion ---
        n_rep = self.num_heads // self.num_kv_heads
        k_exp = k.repeat_interleave(n_rep, dim=1)
        v_exp = v.repeat_interleave(n_rep, dim=1)

        # --- SDPA ---
        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attention_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, seq_len, self.num_heads * self.head_dim)

        # --- output projection + first residual (gated for the expert) ---
        hidden_states = layer.self_attn.o_proj(attn_out.to(layer.self_attn.o_proj.weight.dtype))
        hidden_states = _gated_residual(residual, hidden_states.to(residual.dtype), attn_gate)

        # --- MLP + second residual (gated for the expert) ---
        residual = hidden_states
        hidden_states, mlp_gate = _apply_adarms_norm(layer.post_attention_layernorm, hidden_states, adarms_cond)
        w_dtype = layer.mlp.gate_proj.weight.dtype
        gate_proj_out = layer.mlp.gate_proj(hidden_states.to(w_dtype))
        up = layer.mlp.up_proj(hidden_states.to(w_dtype))
        hidden_states = layer.mlp.down_proj(F.silu(gate_proj_out) * up)
        hidden_states = _gated_residual(residual, hidden_states.to(residual.dtype), mlp_gate)

        return hidden_states, new_kv

    # ------------------------------------------------------------------
    # Forward: three calling modes
    # ------------------------------------------------------------------
    def forward(
        self,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values=None,
        inputs_embeds: list[Tensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[Tensor | None] | None = None,  # [vlm_cond (always None), expert_cond]
    ):
        """Three calling modes (mirrors PaliGemmaWithExpertModel.forward in mot0):

        Mode A  [prefix, None]  – VLM builds KV cache from prefix (inference prefill)
        Mode B  [None, suffix]  – Expert denoises suffix using prefix KV cache
        Mode C  [prefix, suffix]– Full MoT mixing (training)

        Returns:
            ([prefix_out, suffix_out], past_key_values)
        """
        if inputs_embeds[1] is None:
            # ---- Mode A: prefix-only (KV cache pre-fill) ----
            hidden = inputs_embeds[0]
            all_kvs = []
            for layer in self.falcon_vlm.model.layers:
                hidden, kv = self._llm_layer_forward(
                    layer, hidden, attention_mask, position_ids, use_cache=True
                )
                all_kvs.append(kv)
            prefix_out, _ = _apply_adarms_norm(self.falcon_vlm.model.norm, hidden, None)
            return [prefix_out, None], all_kvs

        elif inputs_embeds[0] is None:
            # ---- Mode B: suffix-only (denoising step with prefix KV cache) ----
            hidden = inputs_embeds[1]
            expert_cond = adarms_cond[1] if adarms_cond is not None else None
            for layer_idx, layer in enumerate(self.falcon_expert.model.layers):
                past_kv = past_key_values[layer_idx] if past_key_values is not None else None
                hidden, _ = self._llm_layer_forward(
                    layer, hidden, attention_mask, position_ids,
                    past_kv=past_kv, use_cache=False, adarms_cond=expert_cond,
                )
            suffix_out, _ = _apply_adarms_norm(self.falcon_expert.model.norm, hidden, expert_cond)
            return [None, suffix_out], None

        else:
            # ---- Mode C: both streams – full MoT layer mixing (training) ----
            vlm_h = inputs_embeds[0]
            exp_h = inputs_embeds[1]
            cond = adarms_cond if adarms_cond is not None else [None, None]
            for layer_idx in range(self.num_layers):
                vlm_h, exp_h = compute_falcon_layer(
                    layer_idx, [vlm_h, exp_h],
                    attention_mask, position_ids,
                    cond,
                    self.falcon_vlm, self.falcon_expert,
                    num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim, rope_theta=self.rope_theta,
                )
            prefix_out, _ = _apply_adarms_norm(self.falcon_vlm.model.norm, vlm_h, None)
            suffix_out, _ = _apply_adarms_norm(self.falcon_expert.model.norm, exp_h, cond[1])
            return [prefix_out, suffix_out], None


# =============================================================================
# MOTFalconPytorch – flow-matching core (model-agnostic)
# =============================================================================

class MOTFalconPytorch(nn.Module):
    """Flow-matching MoT-Falcon model.

    Identical flow-matching logic as MOT0Pytorch; only the backbone differs.
    """

    def __init__(self, config: MOTFalconConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        # The expert's residual-stream width is independent of the VLM's hidden
        # size (see FalconVLMWithExpertModel.__init__) -- only its attention head
        # geometry and depth are pinned to the VLM's (read from the loaded
        # checkpoint's own config). Defaults to config.action_expert_hidden_size
        # (gemma_300m-scale by default), so action_in_proj/time_mlp_* below must
        # match whatever width the expert actually ends up with.
        expert_hidden = config.action_expert_hidden_size

        self.falcon_with_expert = FalconVLMWithExpertModel(
            falcon_vlm_model_id=config.falcon_vlm_model_id,
            precision=config.dtype,
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
            embedding_scaling=config.embedding_scaling,
            action_expert_hidden_size=config.action_expert_hidden_size,
            action_expert_intermediate_size=config.action_expert_intermediate_size,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, config.max_action_dim)
        self.time_mlp_in = nn.Linear(expert_hidden, expert_hidden)
        self.time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        # Enable gradient checkpointing on Falcon3 submodules.
        if hasattr(self.falcon_with_expert.falcon_vlm, "gradient_checkpointing_enable"):
            self.falcon_with_expert.falcon_vlm.gradient_checkpointing_enable()
        if hasattr(self.falcon_with_expert.falcon_expert, "gradient_checkpointing_enable"):
            self.falcon_with_expert.falcon_expert.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        mask = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        dtype = next(self.falcon_with_expert.falcon_vlm.model.layers[0].parameters()).dtype
        return mask.to(dtype)

    # ------------------------------------------------------------------
    # Noise / time sampling
    # ------------------------------------------------------------------
    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # Prefix embedding
    # ------------------------------------------------------------------
    def embed_prefix(self, images, img_masks, tokens, masks):
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.falcon_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        lang_emb = self._apply_checkpoint(self.falcon_with_expert.embed_language_tokens, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        bsize = pad_masks.shape[0]
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_prefix_fast(
        self, images, img_masks, tokens, masks,
        fast_action_tokens=None, fast_action_masks=None,
    ):
        """Embed images, language tokens, and FAST discrete action tokens."""
        embs, pad_masks, att_mask_segments = [], [], []
        total_t_images, num_fast_embs = 0, 0

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.falcon_with_expert.embed_image, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_mask_segments.append(("image", num_img_embs))
            total_t_images += num_img_embs

        lang_emb = self._apply_checkpoint(self.falcon_with_expert.embed_language_tokens, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_mask_segments.append(("language", lang_emb.shape[1]))

        if fast_action_tokens is not None:
            fast_emb = self._apply_checkpoint(
                self.falcon_with_expert.embed_language_tokens, fast_action_tokens
            )
            embs.append(fast_emb)
            num_fast_embs = fast_action_tokens.shape[1]
            pad_masks.append(fast_action_masks)
            att_mask_segments.append(("fast", num_fast_embs))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = self._create_custom_attention_mask_fast(att_mask_segments, pad_masks, bsize)

        return embs, pad_masks, att_masks, total_t_images, num_fast_embs

    def _create_custom_attention_mask_fast(self, att_mask_segments, pad_masks, bsize):
        total_len = sum(length for _, length in att_mask_segments)
        device = pad_masks.device
        att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)

        positions, current_pos = [], 0
        for seg_type, seg_len in att_mask_segments:
            positions.append((seg_type, current_pos, current_pos + seg_len))
            current_pos += seg_len

        for query_type, query_start, query_end in positions:
            for key_type, key_start, key_end in positions:
                if (
                    query_type in ["image", "language"] and key_type in ["image", "language"]
                    or query_type == "fast" and key_type in ["image", "language"]
                ):
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True
                elif query_type == "fast" and key_type == "fast":
                    fast_len = query_end - query_start
                    causal_mask = torch.tril(
                        torch.ones(fast_len, fast_len, dtype=torch.bool, device=device)
                    )
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None]

        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    # ------------------------------------------------------------------
    # Suffix embedding (noisy actions + timestep)
    # ------------------------------------------------------------------
    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor):
        """Embed noisy actions; the timestep conditions the expert via AdaRMS
        (a learned per-layer scale/shift/gate, see `_install_adarms_expert_norms`)
        rather than being added into the action embedding. Mirrors mot0's
        `embed_suffix`, where `action_time_emb` is just `action_emb` and the
        timestep only ever flows through `adarms_cond`.
        """
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.to(dtype=timestep.dtype)

        action_emb = self._apply_checkpoint(self.action_in_proj, noisy_actions)

        def time_mlp_func(t):
            x = self.time_mlp_in(t)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        adarms_cond = time_emb

        bsize, action_time_dim = action_emb.shape[:2]
        pad_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        att_mask = torch.tensor(
            [1] + [0] * (self.config.chunk_size - 1),
            dtype=action_emb.dtype,
            device=action_emb.device,
        )
        att_mask = att_mask[None, :].expand(bsize, len(att_mask))

        return action_emb, pad_mask, att_mask, adarms_cond

    # ------------------------------------------------------------------
    # FAST action cross-entropy loss
    # ------------------------------------------------------------------
    def fast_action_loss(self, prefix_out: Tensor, fast_targets: Tensor, fast_action_masks: Tensor):
        fast_hidden = prefix_out[:, -fast_targets.shape[1]:, :]
        fast_logits = self.falcon_with_expert.falcon_vlm.lm_head(fast_hidden)

        fast_logits = fast_logits[:, :-1, :]
        fast_targets = fast_targets[:, 1:]
        fast_action_masks = fast_action_masks[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        fast_loss_per_token = loss_fct(
            fast_logits.reshape(-1, fast_logits.size(-1)),
            fast_targets.reshape(-1),
        ).reshape(fast_targets.shape)

        masked_loss = fast_loss_per_token * fast_action_masks.float()
        return masked_loss.sum(dim=1) / fast_action_masks.sum(dim=1).clamp(min=1)

    # ------------------------------------------------------------------
    # Full attention-mask builder
    # ------------------------------------------------------------------
    def build_full_mask(
        self,
        prefix_att_masks,
        suffix_att_masks,
        prefix_pad_masks,
        suffix_pad_masks,
        num_fast_embs,
    ):
        B, Lp, _ = prefix_att_masks.shape
        _, Ls, _ = suffix_att_masks.shape
        device, dtype = prefix_att_masks.device, prefix_att_masks.dtype

        prefix_to_suffix = torch.zeros((B, Lp, Ls), dtype=dtype, device=device)
        suffix_to_prefix = torch.zeros((B, Ls, Lp), dtype=dtype, device=device)
        # The flow-matching suffix may attend to images and language but not to target FAST tokens,
        # which would leak the answer. With FAST loss disabled, num_fast_embs is zero and the full
        # image+language prefix remains visible.
        suffix_to_prefix[:, :, :Lp - num_fast_embs] = 1

        top = torch.cat([prefix_att_masks, prefix_to_suffix], dim=2)
        bot = torch.cat([suffix_to_prefix, suffix_att_masks], dim=2)
        full_att_masks = torch.cat([top, bot], dim=1)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        full_att_masks = full_att_masks * pad_masks[:, :, None] * pad_masks[:, None, :]
        return full_att_masks, pad_masks

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        images, img_masks, tokens, masks,
        actions, fast_action_tokens, fast_action_masks,
        noise=None, time=None,
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        num_fast_embs = 0
        if self.config.use_fast_action_loss:
            prefix_embs, prefix_pad_masks, prefix_2d_att_masks, _, num_fast_embs = (
                self.embed_prefix_fast(
                    images, img_masks, tokens, masks,
                    fast_action_tokens=fast_action_tokens,
                    fast_action_masks=fast_action_masks,
                )
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, tokens, masks
            )
            prefix_2d_att_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)
        cumsum = torch.cumsum(suffix_att_masks, dim=1)
        suffix_2d_att_masks = cumsum[:, None, :] <= cumsum[:, :, None]

        att_2d_masks, pad_masks = self.build_full_mask(
            prefix_2d_att_masks,
            suffix_2d_att_masks,
            prefix_pad_masks,
            suffix_pad_masks,
            num_fast_embs,
        )

        # Cast embeddings to VLM weight dtype if needed.
        vlm_dtype = next(self.falcon_with_expert.falcon_vlm.model.layers[0].parameters()).dtype
        suffix_embs = suffix_embs.to(vlm_dtype)
        prefix_embs = prefix_embs.to(vlm_dtype)

        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (prefix_out, suffix_out), _ = self.falcon_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.chunk_size:].to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        if self.config.use_fast_action_loss:
            ce_loss_per_sample = self.fast_action_loss(prefix_out, fast_action_tokens, fast_action_masks)
        else:
            ce_loss_per_sample = torch.zeros(v_t.shape[0], device=v_t.device, dtype=v_t.dtype)

        return F.mse_loss(u_t, v_t, reduction="none"), ce_loss_per_sample

    # ------------------------------------------------------------------
    # Inference: sample actions
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        images, img_masks, tokens, masks,
        noise=None, num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim), device
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        # Cast embeddings to VLM weight dtype if needed (mirrors the training-time
        # forward() cast above -- embed_prefix returns float32 regardless of model
        # precision).
        vlm_dtype = next(self.falcon_with_expert.falcon_vlm.model.layers[0].parameters()).dtype
        prefix_embs = prefix_embs.to(vlm_dtype)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        _, past_key_values = self.falcon_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial(input_x_t, t=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=t,
                )

            if self._rtc_enabled():
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
                    inference_delay=kwargs.get("inference_delay"),
                    time=time,
                    original_denoise_step_partial=denoise_step_partial,
                    execution_horizon=kwargs.get("execution_horizon"),
                )
            else:
                v_t = denoise_step_partial(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        # Cast embeddings to VLM weight dtype if needed (see sample_actions).
        vlm_dtype = next(self.falcon_with_expert.falcon_vlm.model.layers[0].parameters()).dtype
        suffix_embs = suffix_embs.to(vlm_dtype)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.falcon_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1][:, -self.config.chunk_size:].to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


# =============================================================================
# MOTFalconPolicy
# =============================================================================

class MOTFalconPolicy(PreTrainedPolicy):
    """MOT-Falcon Policy: Mixture of Transformers with Falcon3-VLM backbone."""

    config_class = MOTFalconConfig
    name = "mot_falcon"

    def supports_rtc(self) -> bool:
        return True

    def __init__(self, config: MOTFalconConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.init_rtc_processor()
        self.model = MOTFalconPytorch(config, rtc_processor=self.rtc_processor)

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model = cls(config, **kwargs)

        try:
            from transformers.utils import cached_file
            from safetensors.torch import load_file

            resolved_file = cached_file(
                pretrained_name_or_path,
                "model.safetensors",
                cache_dir=kwargs.get("cache_dir"),
                force_download=kwargs.get("force_download", False),
                resume_download=kwargs.get("resume_download"),
                proxies=kwargs.get("proxies"),
                token=kwargs.get("token"),
                revision=kwargs.get("revision"),
                local_files_only=kwargs.get("local_files_only", False),
            )
            original_state_dict = load_file(resolved_file)

            remapped = {
                (f"model.{k}" if not k.startswith("model.") else k): v
                for k, v in original_state_dict.items()
            }
            missing, unexpected = model.load_state_dict(remapped, strict=strict)
            if missing:
                logging.warning(f"Missing keys ({len(missing)}): {missing[:5]}")
            if unexpected:
                logging.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        except Exception as e:
            raise RuntimeError(
                f"Could not load MOT-Falcon state dict from {pretrained_name_or_path!r}"
            ) from e

        return model

    def get_optim_params(self) -> dict:
        other_params: list[Tensor] = []
        action_expert_params: list[Tensor] = []
        vlm_params: list[Tensor] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "falcon_with_expert.falcon_expert" in name:
                action_expert_params.append(param)
            elif "falcon_with_expert.falcon_vlm" in name:
                vlm_params.append(param)
            else:
                other_params.append(param)

        vlm_lr = self.config.optimizer_lr * self.config.optimizer_lr_multiper_vlm
        action_lr = self.config.optimizer_lr * self.config.optimizer_lr_multiper_action
        other_lr = self.config.optimizer_lr

        groups: list[dict[str, Any]] = []
        if other_params:
            groups.append({"params": other_params, "lr": other_lr})
        if vlm_params:
            groups.append({"params": vlm_params, "lr": vlm_lr})
        if action_expert_params:
            groups.append({"params": action_expert_params, "lr": action_lr})

        return groups

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def init_rtc_processor(self):
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_images(self, batch: dict[str, Tensor]):
        """Resize images to target resolution and return in [0, 1] range.

        ImageNet normalisation is applied later inside embed_image so that
        the Falcon3 visual encoder receives correctly scaled input.
        """
        images, img_masks = [], []
        device = next(self.parameters()).device

        present_img_keys = [k for k in self.config.image_features if k in batch]
        missing_img_keys = [k for k in self.config.image_features if k not in batch]

        if not present_img_keys:
            raise ValueError(
                f"All image features missing from batch. "
                f"Expected one of: {self.config.image_features}"
            )

        for key in present_img_keys:
            img = batch[key].to(device=device, dtype=torch.float32)

            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)  # → [B, H, W, C]

            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Keep in [0, 1]; Falcon3 uses ImageNet normalisation (handled in embed_image).
            img = img.clamp(0.0, 1.0)

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # → [B, C, H, W]

            mask_key = f"{key}_mask"
            if mask_key in batch:
                mask = batch[mask_key].to(device=device, dtype=torch.bool)
            else:
                mask = torch.ones(img.shape[0], dtype=torch.bool, device=device)

            images.append(img)
            img_masks.append(mask)

        for _ in range(len(missing_img_keys)):
            img = torch.ones_like(img) * 0.5  # neutral grey placeholder
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action; use predict_action_chunk instead."
        )
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    def merge_fast_vlm_labels(
        self, vlm_answer, vlm_answer_mask, fast_action_tokens, fast_action_masks, sample_types
    ):
        if vlm_answer is None:
            return fast_action_tokens, fast_action_masks

        is_robotic = torch.tensor(
            [s == "robotic" for s in sample_types],
            device=vlm_answer.device,
            dtype=torch.bool,
        ).unsqueeze(1)

        discrete_target_labels = torch.where(is_robotic, fast_action_tokens, vlm_answer)
        discrete_target_mask = torch.where(is_robotic, fast_action_masks, vlm_answer_mask)
        return discrete_target_labels, discrete_target_mask

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean"):
        images, img_masks = self._preprocess_images(batch)
        vlm_answer = batch.get("answer.tokens")
        vlm_answer_mask = batch.get("answer.attention_mask")
        sample_types = batch.get("sample_type")
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        fast_action_tokens = batch.get(ACTION_TOKENS)
        fast_action_masks = batch.get(ACTION_TOKEN_MASK)

        actions = self.prepare_action(batch)
        discrete_target_tokens, discrete_target_mask = self.merge_fast_vlm_labels(
            vlm_answer, vlm_answer_mask, fast_action_tokens, fast_action_masks, sample_types
        )

        mse_loss, ce_loss = self.model.forward(
            images, img_masks, tokens, masks, actions, discrete_target_tokens, discrete_target_mask
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        mse_loss = mse_loss[:, :, :original_action_dim]

        robotic_sample_mask = torch.tensor(
            [1.0 if s == "robotic" else 0.0 for s in sample_types],
            device=mse_loss.device,
            dtype=mse_loss.dtype,
        )
        masked_mse_loss = mse_loss * robotic_sample_mask[:, None, None]
        normalizer = robotic_sample_mask.sum().clamp_min(1.0) * mse_loss.shape[1]
        mse_per_dim = masked_mse_loss.sum(dim=(0, 1)) / normalizer

        loss_dict = {f"mse_loss_dim_{i}": mse_per_dim[i].item() for i in range(mse_per_dim.shape[0])}

        if reduction == "none":
            mse_per_sample = mse_loss.mean(dim=(1, 2))
            loss_dict["mse_loss_per_sample"] = mse_per_sample * robotic_sample_mask
            loss_dict["action_ce_loss_per_sample"] = ce_loss * robotic_sample_mask
            loss_dict["vlm_ce_loss_per_sample"] = ce_loss * (1.0 - robotic_sample_mask)
            loss_per_sample = (
                loss_dict["mse_loss_per_sample"]
                + self.config.action_ce_loss_weight * loss_dict["action_ce_loss_per_sample"]
                + self.config.vlm_ce_loss_weight * loss_dict["vlm_ce_loss_per_sample"]
            )
            return loss_per_sample, loss_dict

        mse_loss_scalar = masked_mse_loss.sum() / (
            robotic_sample_mask.sum().clamp_min(1.0) * mse_loss.shape[1] * mse_loss.shape[2]
        )
        robotic_mask = robotic_sample_mask.bool()
        zero = torch.zeros((), device=ce_loss.device, dtype=ce_loss.dtype)
        action_ce_loss = ce_loss[robotic_mask].mean() if robotic_mask.any() else zero
        vlm_ce_loss = ce_loss[~robotic_mask].mean() if (~robotic_mask).any() else zero

        loss_dict["mse_loss"] = mse_loss_scalar.item()
        loss_dict["action_ce_loss"] = action_ce_loss.item()
        loss_dict["vlm_ce_loss"] = vlm_ce_loss.item()
        loss = (
            mse_loss_scalar
            + self.config.action_ce_loss_weight * action_ce_loss
            + self.config.vlm_ce_loss_weight * vlm_ce_loss
        )
        return loss, loss_dict

    def _get_default_peft_targets(self) -> dict:
        common_projections = "action_in_proj|action_out_proj|time_mlp_in|time_mlp_out"
        target_modules = rf"(.*\.falcon_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {"target_modules": target_modules, "modules_to_save": []}
