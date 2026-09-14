#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Lets mot_falcon (needs the vendored Falcon3-VL `transformers` fork) and pi0.5/mot0 (need the
real, current `transformers`) construct in the same process.

Background: `FalconVLMWithExpertModel.__init__` (`policies/mot_falcon/modeling_mot_falcon.py`)
does `from transformers import AutoModelForImageTextToText` and calls
`AutoModelForImageTextToText.from_pretrained(falcon_vlm_model_id, ...)` with no
``trust_remote_code`` argument. This only resolves if ``"falcon3_vl"`` is a model type
*natively* registered in whatever package `transformers` currently refers to -- true only for
the vendored fork (package name `transformers_falcon`, distinct from real `transformers` so both
can be installed side by side), not real transformers (confirmed empirically: real transformers
raises a "not a valid model type"/trust_remote_code prompt, and even with
``trust_remote_code=True`` forced, the checkpoint's own `falcon_vlm_model_id` directory does not
ship the required `modeling_falcon3_vl.py` custom-code file -- only the fork carries that code,
as part of its own package).

Python only cares what `sys.modules['transformers']` points to at the moment an import statement
executes -- and once a class is defined, its base classes are fixed, not re-looked-up later -- so
temporarily registering the fork as `transformers` in `sys.modules` only while mot_falcon's
policy is being constructed, then restoring the real one immediately after, is enough. No changes
to mot_falcon/pi05/mot0 source needed.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

# Policy types whose construction needs the vendored Falcon3-VL transformers fork instead of the
# real transformers package. Add to this set if another Falcon3-VL-backed policy lands
# (e.g. mot_falcon_pi05expert, once ported).
TYPES_NEEDING_TRANSFORMERS_FALCON = {"mot_falcon"}


@contextmanager
def masquerade_transformers_falcon():
    """While active, `sys.modules['transformers']` is the vendored fork (`transformers_falcon`
    package), not the real `transformers`. Restores whatever was registered before on exit
    (including "nothing", if `transformers` hadn't been imported at all yet).

    Requires the fork installed as its own distinctly-named distribution, e.g.:
        uv pip install /path/to/transformers_falcon-*.whl
    (see deployment/DEPLOYMENT.md for where this repo's wheel currently lives).
    """
    import transformers_falcon  # the vendored fork, a separate distribution from `transformers`

    had_real = "transformers" in sys.modules
    real_transformers = sys.modules.get("transformers")
    sys.modules["transformers"] = transformers_falcon
    try:
        yield
    finally:
        if had_real:
            sys.modules["transformers"] = real_transformers
        else:
            del sys.modules["transformers"]
