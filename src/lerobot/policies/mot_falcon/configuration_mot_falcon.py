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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

# Falcon3-VLM uses patch_size=14, merge_size=2 → factor=28; 224 is divisible by 28.
DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("mot_falcon")
@dataclass
class MOTFalconConfig(PreTrainedConfig):
    # ---------------------------------------------------------------------------
    # Falcon3-VLM model path (local checkpoint or HF model ID)
    # ---------------------------------------------------------------------------
    falcon_vlm_model_id: str = "/lustre1/tier2/projects/falcon-vla/vla_training/base_models/startVLA/playground/Pretrained_models/Falcon3-Vision-3B-Instruct"

    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"
    embedding_scaling: bool = True  # Scale embeddings by sqrt(hidden_size) before feeding into LLM

    action_expert_hidden_size: int = 1024
    action_expert_intermediate_size: int = 4096

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters (same as MOT0 / PI0)
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Relative actions: converts absolute actions to relative (relative to state).
    use_relative_actions: bool = False
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    # Populated at runtime from dataset metadata by make_policy.
    action_feature_names: list[str] | None = None

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    # Falcon3-VLM uses factor=28 for smart_resize; 224 is the smallest valid resolution.
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)

    # Add empty images when no image features are present.
    empty_cameras: int = 0

    # Tokenizer settings
    tokenizer_max_length: int = 200

    # FAST action tokenizer (discrete action tokens)
    max_action_tokens: int = 200
    action_tokenizer_name: str = "lerobot/fast-action-tokenizer"
    fast_skip_tokens: int = 128
    validate_action_token_prefix: bool = True
    use_kv_cache: bool = True

    action_ce_loss_weight: float = 1.0
    vlm_ce_loss_weight: float = 1.0
    # False uses an image+language prefix and skips the discrete FAST action-token loss entirely,
    # matching PI0.5's flow-only training structure. True restores MOT-Falcon's FAST-token prefix
    # and auxiliary cross-entropy loss. This is structurally different from setting
    # action_ce_loss_weight=0: a zero weight still embeds and attends to the target FAST tokens.
    use_fast_action_loss: bool = False
    load_pretrained_pipeline: bool = True

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "default"
    device: str | None = None

    # Finetuning settings
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    # Optimizer settings
    optimizer_lr: float = 2.5e-5
    optimizer_lr_multiper_vlm: float = 1.0  # lr for vlm = optimizer_lr_multiper_vlm * optimizer_lr
    optimizer_lr_multiper_action: float = 1.0  # lr for action = optimizer_lr_multiper_action * optimizer_lr
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_start_step: int = 5000

    # Scheduler settings
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
