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

"""
Falcon3-VLM action tokenizer processor step.

`ActionTokenizerProcessorStep` (in `tokenizer_processor.py`) was written for PaliGemma-style
tokenizers (used by pi0/pi0_fast): it requires a real `bos_token_id` and maps FAST action
tokens into the tail of that tokenizer's vocabulary. Falcon3's tokenizer has no `bos_token`
(its BOS-equivalent is the `<|startoftext|>` special token) and a different vocabulary, so
mot_falcon can't reuse that step as-is. This module adapts it for Falcon3 rather than
overloading `paligemma_tokenizer_name` with a non-PaliGemma tokenizer id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pipeline import ProcessorStepRegistry
from .tokenizer_processor import ActionTokenizerProcessorStep

_FALCON_BOS_TOKEN = "<|startoftext|>"


@dataclass
@ProcessorStepRegistry.register(name="falcon_action_tokenizer_processor")
class FalconActionTokenizerProcessorStep(ActionTokenizerProcessorStep):
    """
    `ActionTokenizerProcessorStep` variant for Falcon3-VLM tokenizers.

    Falcon3's tokenizer has `bos_token_id is None`, so this resolves the `<|startoftext|>`
    special token as the BOS-equivalent. Once that's patched onto the loaded tokenizer, the
    inherited FAST-to-vocab-tail tokenization logic (shared with pi0/pi0_fast) works unmodified.

    Attributes:
        vlm_tokenizer_name: The Falcon3-VLM model id/path to load the tokenizer from
            (e.g. `config.falcon_vlm_model_id`).
    """

    vlm_tokenizer_name: str = ""

    def __post_init__(self):
        self.paligemma_tokenizer_name = self.vlm_tokenizer_name
        super().__post_init__()

        if self._paligemma_tokenizer.bos_token_id is None:
            start_id = self._paligemma_tokenizer.convert_tokens_to_ids(_FALCON_BOS_TOKEN)
            if start_id is None or start_id == self._paligemma_tokenizer.unk_token_id:
                raise ValueError(
                    f"Tokenizer {self.vlm_tokenizer_name!r} has no bos_token and no "
                    f"{_FALCON_BOS_TOKEN!r} special token; cannot determine a BOS-equivalent "
                    "for Falcon3 action tokenization."
                )
            self._paligemma_tokenizer.bos_token_id = start_id

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["vlm_tokenizer_name"] = self.vlm_tokenizer_name
        return config
