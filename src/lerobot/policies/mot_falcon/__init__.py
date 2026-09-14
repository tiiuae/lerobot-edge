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

from .configuration_mot_falcon import MOTFalconConfig

# Eager import (not lazy like the rest of this module) so the
# @ProcessorStepRegistry.register decorators inside processor_mot_falcon.py
# always run. factory.make_pre_post_processors loads a pretrained
# MOTFalconConfig's pipeline generically by step name via
# PolicyProcessorPipeline.from_pretrained, without importing this module
# first -- if these steps aren't already registered by then, that lookup
# raises KeyError.
from . import processor_mot_falcon  # noqa: F401

__all__ = ["MOTFalconConfig", "MOTFalconPolicy", "make_mot_falcon_pre_post_processors"]


def __getattr__(name: str):
    if name == "MOTFalconPolicy":
        from .modeling_mot_falcon import MOTFalconPolicy

        return MOTFalconPolicy
    if name == "make_mot_falcon_pre_post_processors":
        from .processor_mot_falcon import make_mot_falcon_pre_post_processors

        return make_mot_falcon_pre_post_processors
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
