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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from .configuration_mot_falcon import MOTFalconConfig
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    FalconActionTokenizerProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

# Falcon3-VLM's shipped chat_template.json renders an assistant turn as
# "Falcon:{content}|<end_of_turn>|\n". We only need the closing half of that (the
# opening "Falcon:" comes from applying the template with add_generation_prompt=True),
# so it's reproduced here as a literal rather than re-invoking the template for a
# single-field append.
_FALCON_TURN_END = "|<end_of_turn>|"

# Read-only compatibility alias: lerobot-internal (where mot_falcon checkpoints are trained)
# registers RelativeActionsProcessorStep as "delta_actions_processor"; this lerobot checkout
# registers the same (functionally equivalent) class as "relative_actions_processor" instead.
# A mot_falcon checkpoint's saved policy_preprocessor.json therefore references
# "delta_actions_processor" by name, which PolicyProcessorPipeline.from_pretrained can't resolve
# without this. Inserted directly into the registry dict (not via ProcessorStepRegistry.register,
# which would overwrite RelativeActionsProcessorStep._registry_name and change what name any
# *newly saved* pipeline in this repo serializes under) so this is purely additive: it only
# widens what names can be read, and never affects this repo's own canonical write-side name.
ProcessorStepRegistry._registry.setdefault("delta_actions_processor", RelativeActionsProcessorStep)


@ProcessorStepRegistry.register(name="mot_falcon_prepare_state_tokenizer_processor_step")
@dataclass
class MOTFalconPrepareStateTokenizerProcessorStep(ProcessorStep):
    """Prepare the discretised state and build the language prompt for Falcon3-VLM.

    Robotic samples use the same bare "Task/State/Action" prefix-suffix convention
    pi0/pi0_fast use for PaliGemma (itself a non-chat prefix-LM, so no chat formatting
    applies there). VLM co-training samples (`sample_type == "vlm"`), however, exist to
    lean on Falcon3-VLM's own pretrained captioning/grounding ability, so they're
    formatted through Falcon3's real chat template instead of as a bare string.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    falcon_vlm_model_id: str = ""

    # Internal tokenizer instance, only needed to render the chat template for vlm samples.
    _chat_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.falcon_vlm_model_id:
            from transformers import AutoTokenizer

            self._chat_tokenizer = AutoTokenizer.from_pretrained(
                self.falcon_vlm_model_id, trust_remote_code=True
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        sample_types = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("sample_type")

        if state is None:
            raise ValueError("State is required for MOT-Falcon")

        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")
        if sample_types is None:
            sample_types = ["robotic"] * len(tasks)

        state = deepcopy(state)

        # State is already normalised to [-1, 1] by NormalizerProcessorStep.
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        answers = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get("answer")

        full_prompts = []
        full_answers = list(answers) if answers is not None else None
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            if sample_types[i] == "vlm":
                if self._chat_tokenizer is None:
                    raise ValueError(
                        "MOTFalconPrepareStateTokenizerProcessorStep received a 'vlm' sample "
                        "but falcon_vlm_model_id was not set, so Falcon3's chat template "
                        "can't be applied."
                    )
                full_prompt = self._chat_tokenizer.apply_chat_template(
                    [{"role": "user", "content": cleaned_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if full_answers is not None and full_answers[i] is not None:
                    full_answers[i] = f"{full_answers[i]}{_FALCON_TURN_END}"
            else:
                state_str = " ".join(map(str, discretized_states[i]))
                # Robotic samples follow pi0/pi0_fast's PaliGemma-style prefix/suffix
                # convention (see class docstring), not Falcon3's chat template.
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        if full_answers is not None:
            transition[TransitionKey.COMPLEMENTARY_DATA]["answer"] = full_answers
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_mot_falcon_pre_post_processors(
    config: MOTFalconConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre- and post-processor pipelines for MOT-Falcon.

    Structure mirrors make_mot0_pre_post_processors.

    Pre-processing:
      1. Rename observations (no-op by default)
      2. Add batch dimension
      3. Relative actions (if enabled)
      4. Normalise (quantiles for state/action, identity for visual)
      5. Discretise state → build prompt string
      6. Tokenise prompt with Falcon3-VLM tokenizer
      7. Tokenise actions with FAST action tokenizer
      8. Move to device

    Post-processing:
      1. Unnormalise outputs
      2. Absolute actions (if relative was enabled)
      3. Move to CPU
    """
    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
    )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        relative_step,
        # NormalizerProcessorStep MUST come before MOTFalconPrepareStateTokenizerProcessorStep
        # because the tokeniser step expects normalised state in [-1, 1] for discretisation.
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        MOTFalconPrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            falcon_vlm_model_id=config.falcon_vlm_model_id,
        ),
        TokenizerProcessorStep(
            tokenizer_name=config.falcon_vlm_model_id,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        # FalconActionTokenizerProcessorStep maps action vectors to discrete tokens in
        # the tail of Falcon3's vocabulary (same FAST scheme as pi0/pi0_fast use for
        # PaliGemma), patching in Falcon3's `<|startoftext|>` token as its BOS-equivalent
        # since Falcon3's tokenizer has no `bos_token`.
        FalconActionTokenizerProcessorStep(
            action_tokenizer_name=config.action_tokenizer_name,
            max_action_tokens=config.max_action_tokens,
            fast_skip_tokens=config.fast_skip_tokens,
            vlm_tokenizer_name=config.falcon_vlm_model_id,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        AbsoluteActionsProcessorStep(
            enabled=config.use_relative_actions,
            relative_step=relative_step,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
